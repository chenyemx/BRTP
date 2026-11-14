"""Prepare fresh, hash-bound reproductions using frozen experiment loops.

Preparation never runs training. Frozen runners remain byte-identical; the
launcher supplies portable paths and the simplified model implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import sys
import time
import types
import zipfile

from brtp.utils.artifacts import file_sha256 as sha

PACKAGE = Path(__file__).resolve().parent


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def inventory() -> dict:
    return read(PACKAGE / "recipes.json")


def path_map(value: dict) -> dict:
    """Allow Linux targets when preparing on Windows, and reject ambiguous prefixes."""
    if not isinstance(value, dict):
        raise ValueError("Path map must be a JSON object of absolute old-prefix: new-prefix")
    result = {}
    for old, new in value.items():
        if any(
            not isinstance(p, str)
            or not (PurePosixPath(p).is_absolute() or PureWindowsPath(p).is_absolute())
            for p in (old, new)
        ):
            raise ValueError("Path map prefixes must be absolute paths")
        old = old.replace("\\", "/").rstrip("/") or "/"
        new = new.replace("\\", "/").rstrip("/") or "/"
        if old in result and result[old] != new:
            raise ValueError("Conflicting normalized path prefixes: " + old)
        result[old] = new
    return result


def mapped_image_path(original: str, mappings: dict) -> str:
    normalized = original.replace("\\", "/")
    for old in sorted(mappings, key=len, reverse=True):
        if normalized == old or normalized.startswith(old.rstrip("/") + "/"):
            suffix = normalized[len(old) :].lstrip("/")
            target = mappings[old].rstrip("/")
            return target + "/" + suffix if suffix or not target or target.endswith(":") else target
    return original


def copy_public_inputs(output: Path, recipe: dict, supplied_bundle: Path = None) -> dict:
    """Copy the pinned configuration. A local archive can be supplied explicitly."""
    manifest = read(PACKAGE / "public_inputs.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("recipe") != "Full"
        or manifest.get("bundle") != "protocols/full_sgd50_inputs.zip"
    ):
        raise ValueError("Unsupported public input manifest")
    bundle = supplied_bundle.resolve() if supplied_bundle else PACKAGE.parent / manifest["bundle"]
    if not bundle.is_file():
        raise FileNotFoundError("Supply your local Full input ZIP with --protocol-archive.")
    if sha(bundle) != manifest["sha256"]:
        raise ValueError("Public input bundle SHA mismatch")
    required = {
        "frozen_runner.py",
        "data_index.json",
        "protocol/BRTP_protocol.json",
        "protocol/experiment_plan.json",
    }
    files = manifest["files"]
    if (
        not required.issubset(files)
        or not any(p.startswith("paired_support/") for p in files)
        or files["frozen_runner.py"] != recipe["runner_sha256"]
    ):
        raise ValueError("Public inputs do not match the Full execution contract")
    for name in files:
        if (
            name not in required
            and not (name.startswith("paired_support/") and name.endswith(".json"))
            or "\\" in name
            or ":" in name
            or ".." in PurePosixPath(name).parts
            or not within((output / name).resolve(), output)
        ):
            raise ValueError("Unexpected public input path: " + name)
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(files):
            raise ValueError("Public input bundle inventory mismatch")
        for name, expected in files.items():
            data = archive.read(name)  # Includes CRC verification, even for unused members.
            if digest(data) != expected:
                raise ValueError("Public input member SHA mismatch: " + name)
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    return {
        "bundle": manifest["bundle"],
        "sha256": manifest["sha256"],
        "manifest_sha256": sha(PACKAGE / "public_inputs.json"),
        "files": files,
    }


def archive_path(root: Path, relative: str) -> Path:
    """Resolve a relative archive path within the chosen input directory."""
    path = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if not within(path, root):
        raise ValueError("Archive escapes the specified archive root")
    return path


class Archives:
    """Verify each archive before extracting selected, explicitly named members."""

    def __init__(self, root: Path, pins: dict, output: Path):
        self.root, self.pins, self.output, self.opened = root, pins, output, {}

    def get(self, relative: str) -> zipfile.ZipFile:
        if relative not in self.opened:
            path = archive_path(self.root, relative)
            if sha(path) != self.pins[relative]:
                raise ValueError("Archive SHA mismatch: " + relative)
            self.opened[relative] = zipfile.ZipFile(path)
        return self.opened[relative]

    def copy(self, archive: str, member: str, destination: Path) -> None:
        if not within(destination.resolve(), self.output):
            raise ValueError("Archive input escapes the reproduction directory")
        data = self.get(archive).read(member)  # ZipFile also checks member CRC.
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError("Conflicting input: " + str(destination))
        destination.write_bytes(data)

    def close(self) -> None:
        for archive in self.opened.values():
            archive.close()


def copy_core(destination: Path) -> None:
    for relative in ("brtp", "CLIP"):
        source = PACKAGE.parent / relative
        shutil.copytree(
            source, destination / relative, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )


def prepare(args: argparse.Namespace) -> dict:
    catalog = inventory()
    recipe = catalog["recipes"][args.recipe]
    root = args.archive_root.resolve() if args.archive_root is not None else None
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("A reproduction requires a new output directory: " + str(output))
    if root is not None and (within(output, root) or within(root, output)):
        raise ValueError("Reproduction outputs must be outside the evidence archive")
    if any(within(output, source.resolve()) for source in (PACKAGE, PACKAGE.parent / "CLIP")):
        raise ValueError("Output cannot be inside a source directory copied into the reproduction")
    if root is None and args.recipe != "Full":
        raise ValueError("Only Full has bundled inputs; historical recipes require --archive-root")
    local_bundle = getattr(args, "protocol_archive", None)
    if root and (local_bundle or recipe.get("local_protocol")):
        raise ValueError("Use --protocol-archive for the Full software configuration.")
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    weights_sha256 = sha(weights)
    if weights_sha256 != recipe["weights_sha256"]:
        raise ValueError("CLIP weight SHA does not match recipe backbone " + recipe["backbone"])
    mappings = path_map(read(args.path_map) if args.path_map else {})
    output.mkdir(parents=True)
    archives = Archives(root, catalog["archives"], output) if root is not None else None
    family = recipe["family"]
    path_aliases: dict[str, str] = {}
    try:
        runner = output / "frozen_runner.py"
        public_inputs = (
            copy_public_inputs(output, recipe, local_bundle) if archives is None else None
        )
        if archives is not None:
            archives.copy(recipe["runner_archive"], recipe["runner_member"], runner)
        if sha(runner) != recipe["runner_sha256"]:
            raise ValueError("Runner does not match the original execution contract")
        if family in ("full", "control"):
            if archives is not None:
                z = archives.get(recipe["archive"])
                prefix = recipe["prefix"]
                for member in z.namelist():
                    name = member[len(prefix) :] if member.startswith(prefix) else member
                    if member.startswith(prefix) and (
                        name
                        in (
                            "data_index.json",
                            "protocol/BRTP_protocol.json",
                            "protocol/experiment_plan.json",
                        )
                        or name.startswith(("paired_support/", "support/"))
                    ):
                        archives.copy(recipe["archive"], member, output / name)
            if family == "control":
                reference = recipe["reference_archive"]
                archives.copy(reference, "data_index.json", output / "data_index.json")
                for name in archives.get(reference).namelist():
                    if name.startswith("paired_support/"):
                        archives.copy(reference, name, output / name)
            plan = read(output / "protocol/experiment_plan.json")
            spec = read(output / "protocol/BRTP_protocol.json")
            # A pristine receipts directory is required by original preflight.
            (output / "receipts").mkdir()
            radius = 0.25 if family == "control" else spec["method"]["radius"]
        else:
            control = recipe["runner_archive"]
            archives.copy(control, "common.py", output / "common.py")
            if family == "160":
                archives.copy(control, recipe["plan_member"], output / "plan.json")
                plan = read(output / "plan.json")
                archives.copy(control, "data_index.json", output / "data_index.json")
                support_dir = plan.get("support_dir", "support")
                for job in plan["main_training"]:
                    name = support_dir + "/" + job["source_job_id"] + ".json"
                    archives.copy(control, name, output / name)
            else:
                archives.copy(recipe["archives"][0], "plan.json", output / "plan.json")
                plan = read(output / "plan.json")
                candidates = ("data_index.json", "external44/combined_index.json")
                index_member = next(
                    n
                    for n in candidates
                    if digest(archives.get(control).read(n)) == plan["shared_reference_index_sha"]
                )
                archives.copy(control, index_member, output / "data_index.json")
                for archive in recipe["archives"]:
                    z = archives.get(archive)
                    if digest(z.read("plan.json")) != recipe["plan_sha256"]:
                        raise ValueError("Supplement job plans differ")
                    job = json.loads(z.read("job.json"))
                    job_id = job["source_job_id"]
                    archives.copy(
                        archive,
                        "support.json",
                        output / plan.get("support_dir", "support") / (job_id + ".json"),
                    )
                    if "checkpoint_path" in job:
                        cp = output / "reused_inputs" / job_id / "checkpoint.pt"
                        archives.copy(archive, "checkpoint.pt", cp)
                        if sha(cp) != job["checkpoint_sha256"]:
                            raise ValueError("Reused checkpoint SHA mismatch")
                        path_aliases[job["checkpoint_path"]] = str(cp.relative_to(output))
                    for ev in plan["main_evaluation"]:
                        if ev["source_job_id"] != job_id:
                            continue
                        for kind, basename in (
                            ("metrics", "metrics.json"),
                            ("predictions", "predictions.npz"),
                        ):
                            key = "reuse_" + kind + "_path"
                            if key not in ev:
                                continue
                            member = "evaluations/" + ev["evaluation_job_id"] + "/" + basename
                            expected = ev["reuse_" + kind + "_sha256"]
                            choices = [(archive, member)]
                            # Original Full reuse is bound to its exact original files.
                            full = catalog["recipes"]["Full"]["archive"]
                            choices.append((full, member))
                            found = None
                            for arc, name in choices:
                                az = archives.get(arc)
                                if name in az.namelist() and digest(az.read(name)) == expected:
                                    found = (arc, name)
                                    break
                            if found is None:
                                raise ValueError(
                                    "Cannot resolve hash-bound reused input: "
                                    + key
                                    + " / "
                                    + ev["evaluation_job_id"]
                                )
                            dest = (
                                output
                                / "reused_inputs"
                                / job_id
                                / ev["evaluation_job_id"]
                                / basename
                            )
                            archives.copy(*found, dest)
                            path_aliases[ev[key]] = str(dest.relative_to(output))
            if sha(output / "data_index.json") != plan["shared_reference_index_sha"]:
                raise ValueError("Index binding differs from the frozen plan")
            for job in plan["main_training"]:
                if job["source_job_id"] not in recipe["jobs"]:
                    continue
                support = (
                    output / plan.get("support_dir", "support") / (job["source_job_id"] + ".json")
                )
                if sha(support) != plan["support_map"][job["source_job_id"]]["sha256"]:
                    raise ValueError("Support SHA mismatch: " + str(support))
            radius = float(plan["parameters"].get("radius", 0.25))
            shim = output / "brtp_impl"
            shim.mkdir()
            (shim / "__init__.py").write_text("", encoding="utf-8")
        copy_core(output / "BRTP")
        if family in ("n", "160"):
            core = {
                p.relative_to(output / "BRTP").as_posix(): sha(p)
                for p in (output / "BRTP/brtp").rglob("*.py")
            }
            core_sha = digest(json.dumps(core, sort_keys=True).encode())
            for name in ("adapter", "residual_transport"):
                (output / "brtp_impl" / (name + ".py")).write_text(
                    "# Simplified core SHA256: "
                    + core_sha
                    + "\nfrom brtp.models."
                    + name
                    + " import *\n",
                    encoding="utf-8",
                )
        (output / "launch.py").write_text(
            "from pathlib import Path\nimport sys\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parent / 'BRTP'))\n"
            "from brtp.reproduce import launch\n"
            "if __name__ == '__main__': launch(Path(__file__).resolve().parent)\n",
            encoding="utf-8",
        )
        binding = dict(
            recipe=args.recipe,
            family=family,
            configuration=recipe["configuration"],
            runner_sha256=recipe["runner_sha256"],
            jobs=recipe["jobs"],
            archive_sha256={k: catalog["archives"][k] for k in archives.opened} if archives else {},
            public_inputs=public_inputs,
            weights=str(weights),
            weights_sha256=weights_sha256,
            path_map=mappings,
            original_path_aliases=path_aliases,
            legacy_radius_guard=radius,
            training_started=False,
            adaptation="Hash-checked runner and inputs; local paths are mapped at execution time",
            files={
                p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()
            },
        )
        if family in ("n", "160"):
            eligible = [j for j in plan["main_training"] if j["source_job_id"] in recipe["jobs"]]
            binding["evaluation_only_jobs"] = [
                j["source_job_id"] for j in eligible if j.get("checkpoint_path")
            ]
            binding["smoke_jobs"] = {
                task: max(
                    (j for j in eligible if j["task"] == task and not j.get("checkpoint_path")),
                    key=lambda j: len(j["native_class_ids"]),
                )["source_job_id"]
                for task in sorted({j["task"] for j in eligible if not j.get("checkpoint_path")})
            }
        write(output / "PREPARED.json", binding)
        return binding
    except Exception as exc:
        write(output / "PREPARATION_FAILED.json", {"error": repr(exc), "training_started": False})
        raise
    finally:
        if archives is not None:
            archives.close()


def verify_prepared(root: Path) -> dict:
    binding = read(root / "PREPARED.json")
    for relative, expected in binding["files"].items():
        path = (root / relative).resolve()
        if not within(path, root) or sha(path) != expected:
            raise ValueError("Prepared source/input changed: " + relative)
    for directory in ("BRTP", "brtp_impl"):
        expected = {p for p in binding["files"] if p.startswith(directory + "/")}
        actual = {
            p.relative_to(root).as_posix()
            for p in (root / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        }
        if actual != expected:
            raise ValueError("Prepared source inventory changed: " + directory)
    source_suffixes = {".py", ".pyc", ".pyd", ".so", ".pth"}
    expected_sources = {p for p in binding["files"] if Path(p).suffix in source_suffixes}
    actual_sources = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix in source_suffixes and "__pycache__" not in p.parts
    }
    if actual_sources != expected_sources:
        raise ValueError("Prepared importable source inventory changed")
    if sha(Path(binding["weights"])) != binding["weights_sha256"]:
        raise ValueError("CLIP weights changed after preparation")
    return binding


def require_assertions() -> None:
    """Frozen runners use assertions for experiment-integrity gates."""
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE", "0") not in ("", "0"):
        raise RuntimeError("Frozen execution requires assertions; remove -O/-OO and PYTHONOPTIMIZE")


def runtime_environment() -> dict:
    """Check the supported release environment before creating execution artifacts."""
    import torch
    import torchvision
    import numpy
    import PIL

    if (
        sys.version_info[:2] != (3, 10)
        or torch.__version__ != "2.1.2+cu118"
        or torchvision.__version__ != "0.16.2+cu118"
    ):
        raise RuntimeError(
            "Frozen execution requires Python 3.10, torch 2.1.2+cu118 and torchvision 0.16.2+cu118; see docs/INSTALL.md"
        )
    if numpy.__version__ != "1.26.4" or PIL.__version__ != "10.3.0":
        raise RuntimeError(
            "Release execution requires numpy 1.26.4 and Pillow 10.3.0; install requirements.txt"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen execution requires an available CUDA GPU")
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }


def execution_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "smoke", "run", "eval_only"), required=True)
    parser.add_argument("--job-id")
    return parser.parse_args()


def validate_execution(root: Path, args: argparse.Namespace, binding: dict) -> None:
    """Reject invalid requests and missing qualifications without poisoning a run."""
    family = binding["family"]
    if family in ("n", "160"):
        if args.job_id not in binding["jobs"] or args.mode == "preflight":
            raise ValueError("Select a listed job-id and smoke/run/eval_only")
        plan = read(root / "plan.json")
        job = next(j for j in plan["main_training"] if j["source_job_id"] == args.job_id)
        result = root / "results" / binding["configuration"]
        folder = result / "runs" / args.job_id
        if args.mode == "run" and folder.exists():
            raise FileExistsError("Existing source fit preserved; no implicit rerun/resume")
        if bool(job.get("checkpoint_path")) != (args.mode == "eval_only"):
            raise ValueError("Frozen reused jobs require eval_only; new fits require smoke/run")
        if args.mode == "run":
            receipt_path = result / "smoke_receipts" / (job["task"] + ".json")
            if not receipt_path.is_file():
                raise ValueError(
                    "Run the qualified task smoke job first: " + binding["smoke_jobs"][job["task"]]
                )
            gate = read(receipt_path)
            expected = {
                "passed": True,
                "runner_sha": binding["runner_sha256"],
                "adapter_sha": sha(root / "brtp_impl/adapter.py"),
                "model_sha": sha(root / "brtp_impl/residual_transport.py"),
                "plan_sha": sha(root / "plan.json"),
                "training": plan["training"],
                "parameters": plan["parameters"],
            }
            if any(gate.get(k) != v for k, v in expected.items()) or gate.get(
                "source_class_count", 0
            ) < len(job["native_class_ids"]):
                raise ValueError("Task smoke receipt does not match the prepared execution")
    elif args.job_id or args.mode == "eval_only":
        raise ValueError("Full/control runners use their complete 18-job frozen plans")
    elif args.mode == "run":
        if (root / "runs").exists():
            raise FileExistsError("Existing fits preserved; prepare a new reproduction directory")
        paths = [root / "receipts" / (name + ".json") for name in ("preflight", "smoke")]
        if not all(path.is_file() for path in paths):
            raise ValueError("Run preflight and smoke successfully before formal execution")
        preflight, smoke = map(read, paths)
        if (
            preflight.get("passed") is not True
            or smoke.get("passed") is not True
            or preflight.get("protocol_sha") != sha(root / "protocol/BRTP_protocol.json")
            or smoke.get("contract", {}).get("runner_sha") != binding["runner_sha256"]
        ):
            raise ValueError("Preflight/smoke receipts do not match the prepared execution")


def _launch(root: Path, args: argparse.Namespace, binding: dict, environment: dict) -> None:
    """Dispatch a validated request in the exclusively locked Python process."""
    family = binding["family"]
    os.environ["BRTP_CONFIG_ROOT"] = str(root)
    os.environ["BRTP_REFERENCE"] = str(root)
    sys.path.insert(0, str(root))
    import brtp.config

    brtp.config.get_cfg = lambda: types.SimpleNamespace(
        TRAINER=types.SimpleNamespace(
            CODE8=types.SimpleNamespace(MAX_RESIDUAL_NORM=binding["legacy_radius_guard"])
        )
    )
    spec = importlib.util.spec_from_file_location("brtp_frozen_runner", root / "frozen_runner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = root
    module.REFERENCE = root
    module.WEIGHTS = module.WEIGHT = Path(binding["weights"])
    index = read(root / "data_index.json")
    original_hashes = {}
    for rows in index.values():
        for row in rows:
            if row["path"] in original_hashes and original_hashes[row["path"]] != row["sha256"]:
                raise ValueError("Conflicting image SHA in frozen index: " + row["path"])
            original_hashes[row["path"]] = row["sha256"]
    mappings = path_map(binding["path_map"])
    checked: set[str] = set()
    from PIL import Image

    def open_image(path: str):
        original = str(path)
        mapped = mapped_image_path(original, mappings)
        if original not in original_hashes:
            raise ValueError("Image is outside the frozen index: " + original)
        if original not in checked:
            if sha(Path(mapped)) != original_hashes[original]:
                raise ValueError("Image SHA mismatch: " + original)
            checked.add(original)
        return Image.open(mapped)

    module.Image = types.SimpleNamespace(open=open_image)
    aliases = binding["original_path_aliases"]
    module.Path = lambda p: root / aliases[str(p)] if str(p) in aliases else Path(p)
    record = dict(
        prepared_sha256=sha(root / "PREPARED.json"),
        mode=args.mode,
        job_id=args.job_id,
        frozen_runner_sha256=binding["runner_sha256"],
        runtime_environment=environment,
        model_package_files={k: v for k, v in binding["files"].items() if k.startswith("BRTP/")},
        original_records_unchanged=True,
        image_sha_checked_on_first_open=True,
    )
    write(root / "EXECUTION_BINDING.json", record)
    write(root / "execution_records" / (str(time.time_ns()) + ".json"), record)
    if family in ("full", "control"):
        module.run(args.mode)
    else:
        sys.argv = [
            str(root / "frozen_runner.py"),
            "--plan",
            str(root / "plan.json"),
            "--index",
            str(root / "data_index.json"),
            "--job-id",
            args.job_id,
            "--mode",
            args.mode,
        ]
        module.main()


def reject_completed_execution(root: Path) -> None:
    """Keep finished Full/control runs read-only, including their run binding."""
    done = root / "DONE.json"
    if (
        done.exists()
        and read(root / "PREPARED.json")["family"] in ("full", "control")
        and read(done).get("status") == "completed"
    ):
        raise RuntimeError(
            "Completed reproduction preserved; use verify or summarize instead of execute"
        )


def launch(root: Path) -> None:
    """Keep failures and an exclusive process lock; never clear a stale lock."""
    require_assertions()
    args = execution_arguments()
    reject_completed_execution(root)
    if (root / "EXECUTION_FAILED.json").exists():
        raise RuntimeError(
            "Previous execution failed; preserve it and diagnose before creating a new run"
        )
    binding = verify_prepared(root)
    validate_execution(root, args, binding)
    if sys.platform != "linux":
        raise RuntimeError(
            "Frozen DataLoader execution is supported on Linux/CUDA; prepare and audits also run on Windows"
        )
    environment = runtime_environment()
    lock = root / "EXECUTION.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "time": time.time()}, stream)
    # Another invocation may have finished between validation and lock acquisition.
    # Recheck its outputs under the lock without writing a new failure into them.
    try:
        reject_completed_execution(root)
        if (root / "EXECUTION_FAILED.json").exists():
            raise RuntimeError(
                "Previous execution failed; preserve it and diagnose before creating a new run"
            )
        validate_execution(root, args, binding)
    except BaseException:
        lock.unlink()
        raise
    try:
        _launch(root, args, binding, environment)
    except BaseException as exc:
        write(
            root / "EXECUTION_FAILED.json",
            {
                "error": repr(exc),
                "pid": os.getpid(),
                "time": time.time(),
                "arguments": sys.argv[1:],
                "prepared_sha256": sha(root / "PREPARED.json"),
            },
        )
        raise
    else:
        write(
            root / "execution_records" / (str(time.time_ns()) + "_complete.json"),
            {
                "passed": True,
                "arguments": sys.argv[1:],
                "prepared_sha256": sha(root / "PREPARED.json"),
            },
        )
    finally:
        lock.unlink()


def main() -> None:
    from brtp import __version__

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", action="version", version="BRTP " + __version__)
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="List retained BRTP/CoOp-control recipes; never starts training")
    p = sub.add_parser(
        "prepare", help="Extract inputs into a new directory, preserving frozen runner bytes"
    )
    p.add_argument("--recipe", choices=sorted(inventory()["recipes"]), default="Full")
    p.add_argument(
        "--archive-root", type=Path, help="Optional for Full; required for historical recipes"
    )
    p.add_argument(
        "--protocol-archive", type=Path, help="Local Full protocol ZIP with matching SHA-256"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--path-map", type=Path)
    e = sub.add_parser("execute", help="Explicitly execute a prepared Linux/CUDA reproduction")
    e.add_argument("--run-dir", type=Path, required=True)
    e.add_argument("--mode", choices=("preflight", "smoke", "run", "eval_only"), required=True)
    e.add_argument("--job-id")
    v = sub.add_parser("verify", help="Verify prepared source/input hashes without training")
    v.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.action == "list":
        for name, recipe in inventory()["recipes"].items():
            print(
                name,
                recipe["family"],
                len(recipe["jobs"]),
                "source jobs",
                recipe["training"]["optimizer"],
            )
    elif args.action == "prepare":
        binding = prepare(args)
        print(
            json.dumps(
                {
                    "prepared": str(args.output.resolve()),
                    "recipe": binding["recipe"],
                    "jobs": binding["jobs"],
                    "training_started": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.action == "verify":
        binding = verify_prepared(args.run_dir.resolve())
        print(
            json.dumps(
                {"passed": True, "recipe": binding["recipe"], "files": len(binding["files"])}
            )
        )
    else:
        root = args.run_dir.resolve()
        require_assertions()
        binding = verify_prepared(root)
        reject_completed_execution(root)
        validate_execution(root, args, binding)
        command = [sys.executable, str(root / "launch.py"), "--mode", args.mode]
        if args.job_id:
            command.extend(["--job-id", args.job_id])
        raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == "__main__":
    main()

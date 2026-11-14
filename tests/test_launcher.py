"""Launcher gates and portable paths, with a fake runner (never trains)."""

from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import zipfile

from PIL import Image
from brtp import reproduce
import brtp.config


class LauncherTests(unittest.TestCase):
    def test_local_archives_are_confined_and_hash_checked(self):
        name = "inputs/synthetic.zip"
        path = self.root / name
        path.parent.mkdir()
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("sample.json", "{}")
        pins = {name: reproduce.sha(path)}
        archives = reproduce.Archives(self.root, pins, self.root / "out")
        try:
            self.assertEqual(archives.get(name).read("sample.json"), b"{}")
        finally:
            archives.close()
        path.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            reproduce.Archives(self.root, pins, self.root / "out").get(name)
        with self.assertRaisesRegex(ValueError, "escapes"):
            reproduce.archive_path(self.root, "../outside.zip")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "new_data").mkdir()
        self.image = self.root / "new_data" / "image.png"
        Image.new("RGB", (3, 5), color="green").save(self.image)
        self.original = "/old/data/image.png"
        reproduce.write(
            self.root / "data_index.json",
            {"dataset": [{"path": self.original, "sha256": reproduce.sha(self.image)}]},
        )
        self.binding = {
            "family": "full",
            "configuration": "Full",
            "jobs": ["job1"],
            "legacy_radius_guard": 0.25,
            "weights": str(self.image),
            "path_map": {"/old/data": str(self.root / "new_data")},
            "original_path_aliases": {"/old/checkpoint.pt": "reused/checkpoint.pt"},
            "runner_sha256": "synthetic-runner",
            "files": {},
        }
        reproduce.write(self.root / "PREPARED.json", self.binding)

    def run_fake(self, action, mode="preflight"):
        module = types.ModuleType("brtp_frozen_runner")
        module.run = lambda selected_mode: action(module, selected_mode)
        spec = types.SimpleNamespace(
            name="brtp_frozen_runner", loader=types.SimpleNamespace(exec_module=lambda value: None)
        )
        with mock.patch.object(
            reproduce, "verify_prepared", return_value=self.binding
        ), mock.patch.object(
            reproduce, "runtime_environment", return_value={"fixture": True}
        ), mock.patch.object(reproduce.sys, "platform", "linux"), mock.patch.object(
            reproduce.sys, "argv", ["launch.py", "--mode", mode]
        ), mock.patch.object(
            reproduce.importlib.util, "spec_from_file_location", return_value=spec
        ), mock.patch.object(
            reproduce.importlib.util, "module_from_spec", return_value=module
        ), mock.patch.object(brtp.config, "get_cfg"):
            reproduce.launch(self.root)

    def test_portable_roots_images_and_checkpoint_alias_preserve_original_index(self):
        before = (self.root / "data_index.json").read_bytes()

        def action(module, mode):
            self.assertEqual(mode, "preflight")
            self.assertEqual(module.ROOT, self.root)
            self.assertEqual(module.REFERENCE, self.root)
            self.assertEqual(module.WEIGHT, self.image)
            self.assertEqual(module.Path("/old/checkpoint.pt"), self.root / "reused/checkpoint.pt")
            with module.Image.open(self.original) as image:
                self.assertEqual(image.size, (3, 5))
            with self.assertRaisesRegex(ValueError, "outside the frozen index"):
                module.Image.open("/old/data/unlisted.png")

        self.run_fake(action)
        self.assertEqual(before, (self.root / "data_index.json").read_bytes())
        self.assertFalse((self.root / "EXECUTION.lock").exists())
        self.assertFalse((self.root / "EXECUTION_FAILED.json").exists())
        self.assertEqual(len(list((self.root / "execution_records").glob("*_complete.json"))), 1)

    def test_changed_image_is_rejected_and_failure_blocks_followup(self):
        self.image.write_bytes(b"changed image bytes")
        with self.assertRaisesRegex(ValueError, "Image SHA mismatch"):
            self.run_fake(lambda module, mode: module.Image.open(self.original))
        self.assertTrue((self.root / "EXECUTION_FAILED.json").is_file())
        self.assertFalse((self.root / "EXECUTION.lock").exists())
        with self.assertRaisesRegex(RuntimeError, "Previous execution failed"):
            self.run_fake(lambda module, mode: self.fail("failed job was restarted"))

    def test_prefix_mapping_normalizes_trailing_slashes_and_uses_longest_boundary(self):
        self.binding["path_map"] = {
            "/old/": "/not_used/",
            "/old/data/": str(self.root / "new_data") + "/",
        }
        self.run_fake(lambda module, mode: module.Image.open(self.original).close())
        mappings = reproduce.path_map({"/old/data/": "/new/data/"})
        self.assertEqual(
            reproduce.mapped_image_path("/old/data-other/image.png", mappings),
            "/old/data-other/image.png",
        )
        self.assertEqual(
            reproduce.mapped_image_path("/old/data/image.png", mappings), "/new/data/image.png"
        )
        self.assertEqual(
            reproduce.mapped_image_path("/image.png", reproduce.path_map({"/": "/new/"})),
            "/new/image.png",
        )
        self.assertEqual(
            reproduce.mapped_image_path("/old", reproduce.path_map({"/old": "/"})), "/"
        )
        self.assertEqual(
            reproduce.mapped_image_path(
                "C:\\data\\image.png", reproduce.path_map({"C:\\data\\": "D:\\data\\"})
            ),
            "D:/data/image.png",
        )

    def test_ambiguous_or_relative_path_maps_are_rejected(self):
        for mappings in (
            {"": "/new"},
            {"relative": "/new"},
            {"/old": "relative"},
            {"/old": None},
            {"/old": "/new", "/old/": "/other"},
        ):
            with self.subTest(mappings=mappings), self.assertRaises(ValueError):
                reproduce.path_map(mappings)

    def test_conflicting_duplicate_image_records_are_rejected(self):
        reproduce.write(
            self.root / "data_index.json",
            {
                "dataset": [
                    {"path": self.original, "sha256": reproduce.sha(self.image)},
                    {"path": self.original, "sha256": "conflicting hash"},
                ]
            },
        )
        with self.assertRaisesRegex(ValueError, "Conflicting image SHA"):
            self.run_fake(lambda module, mode: self.fail("ambiguous index reached runner"))

    def test_existing_lock_is_not_cleared_or_overwritten(self):
        lock = self.root / "EXECUTION.lock"
        lock.write_text("original-owner", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_fake(lambda module, mode: self.fail("duplicate run executed"))
        self.assertEqual(lock.read_text(encoding="utf-8"), "original-owner")

    def test_existing_training_directory_refuses_implicit_restart(self):
        (self.root / "runs" / "job1").mkdir(parents=True)
        (self.root / "runs" / "job1" / "latest.pt").write_bytes(b"partial checkpoint")
        with self.assertRaisesRegex(FileExistsError, "Existing fits preserved"):
            self.run_fake(lambda module, mode: self.fail("partial fit executed"), mode="run")
        self.assertEqual(
            (self.root / "runs" / "job1" / "latest.pt").read_bytes(), b"partial checkpoint"
        )

    def test_completed_full_and_control_are_read_only_for_every_execution_mode(self):
        reproduce.write(
            self.root / "DONE.json", {"status": "completed", "training": 18, "evaluation": 96}
        )
        reproduce.write(self.root / "EXECUTION_BINDING.json", {"mode": "run", "preserved": True})
        for family in ("full", "control"):
            self.binding["family"] = family
            reproduce.write(self.root / "PREPARED.json", self.binding)
            before = {
                p.relative_to(self.root): p.read_bytes()
                for p in self.root.rglob("*")
                if p.is_file()
            }
            for mode in ("preflight", "smoke", "run", "eval_only"):
                with self.subTest(family=family, mode=mode):
                    with self.assertRaisesRegex(RuntimeError, "Completed reproduction preserved"):
                        self.run_fake(
                            lambda module, selected: self.fail("completed runner executed"),
                            mode=mode,
                        )
                    after = {
                        p.relative_to(self.root): p.read_bytes()
                        for p in self.root.rglob("*")
                        if p.is_file()
                    }
                    self.assertEqual(before, after)

    def test_completed_execute_command_refuses_before_starting_child(self):
        reproduce.write(
            self.root / "DONE.json", {"status": "completed", "training": 18, "evaluation": 96}
        )
        with mock.patch.object(
            reproduce, "verify_prepared", return_value=self.binding
        ), mock.patch.object(
            reproduce.sys,
            "argv",
            ["reproduce", "execute", "--run-dir", str(self.root), "--mode", "smoke"],
        ), mock.patch.object(reproduce.subprocess, "call") as child:
            with self.assertRaisesRegex(RuntimeError, "use verify or summarize"):
                reproduce.main()
        child.assert_not_called()
        self.assertFalse((self.root / "EXECUTION.lock").exists())
        self.assertFalse((self.root / "EXECUTION_FAILED.json").exists())

    def test_verification_detects_source_or_weight_changes(self):
        source = self.root / "frozen_runner.py"
        source.write_text("# synthetic fixture\n", encoding="utf-8")
        binding = dict(
            self.binding,
            files={"frozen_runner.py": reproduce.sha(source)},
            weights_sha256=reproduce.sha(self.image),
        )
        reproduce.write(self.root / "PREPARED.json", binding)
        self.assertEqual(reproduce.verify_prepared(self.root), binding)
        source.write_text("# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Prepared source/input changed"):
            reproduce.verify_prepared(self.root)
        binding["files"]["frozen_runner.py"] = reproduce.sha(source)
        reproduce.write(self.root / "PREPARED.json", binding)
        self.image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "CLIP weights changed"):
            reproduce.verify_prepared(self.root)

    def test_prepare_rejects_source_subtree_before_writing(self):
        for source in (reproduce.PACKAGE, reproduce.PACKAGE.parent / "CLIP"):
            with self.subTest(source=source):
                output = source / "forbidden_reproduction_output"
                self.assertFalse(output.exists())
                args = types.SimpleNamespace(
                    recipe="fixture",
                    archive_root=self.root,
                    output=output,
                    weights=self.image,
                    path_map=None,
                )
                with mock.patch.object(
                    reproduce, "inventory", return_value={"recipes": {"fixture": {}}}
                ):
                    with self.assertRaisesRegex(ValueError, "inside a source directory"):
                        reproduce.prepare(args)
                self.assertFalse(output.exists())

    def test_added_prepared_source_is_rejected(self):
        binding = dict(self.binding, weights_sha256=reproduce.sha(self.image))
        reproduce.write(self.root / "PREPARED.json", binding)
        (self.root / "BRTP").mkdir()
        (self.root / "BRTP" / "unexpected.py").write_text("# changed package inventory\n")
        with self.assertRaisesRegex(ValueError, "source inventory changed"):
            reproduce.verify_prepared(self.root)

    def test_top_level_library_shadow_is_rejected(self):
        binding = dict(self.binding, weights_sha256=reproduce.sha(self.image))
        reproduce.write(self.root / "PREPARED.json", binding)
        for relative in ("torch.py", "torch.pyc", "newpkg/__init__.py"):
            with self.subTest(relative=relative):
                source = self.root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"# unintended import shadow\n")
                try:
                    with self.assertRaisesRegex(ValueError, "importable source inventory changed"):
                        reproduce.verify_prepared(self.root)
                finally:
                    source.unlink()

    def test_runtime_gate_requires_frozen_versions_and_cuda(self):
        torch = types.SimpleNamespace(
            __version__="2.1.2+cu118",
            version=types.SimpleNamespace(cuda="11.8"),
            cuda=types.SimpleNamespace(
                is_available=lambda: True, get_device_name=lambda index: "fixture GPU"
            ),
        )
        vision = types.SimpleNamespace(__version__="0.16.2+cu118")
        numpy = types.SimpleNamespace(__version__="1.26.4")
        pillow = types.SimpleNamespace(__version__="10.3.0")
        with mock.patch.dict(
            reproduce.sys.modules,
            {"torch": torch, "torchvision": vision, "numpy": numpy, "PIL": pillow},
        ), mock.patch.object(reproduce.sys, "version_info", (3, 10, 8)):
            self.assertEqual(reproduce.runtime_environment()["gpu"], "fixture GPU")
            torch.__version__ = "2.13.0+cu126"
            with self.assertRaisesRegex(RuntimeError, "requires Python 3.10"):
                reproduce.runtime_environment()
            torch.__version__ = "2.1.2+cu118"
            for package in (numpy, pillow):
                version = package.__version__
                package.__version__ = "unmatched-version"
                with self.assertRaisesRegex(RuntimeError, "Release execution requires"):
                    reproduce.runtime_environment()
                package.__version__ = version
            torch.cuda.is_available = lambda: False
            with self.assertRaisesRegex(RuntimeError, "available CUDA GPU"):
                reproduce.runtime_environment()

    def test_optimized_execution_is_refused_before_lock_or_child(self):
        for setting in ("1", "2", "unexpected"):
            with self.subTest(PYTHONOPTIMIZE=setting), mock.patch.dict(
                reproduce.os.environ, {"PYTHONOPTIMIZE": setting}
            ):
                with self.assertRaisesRegex(RuntimeError, "requires assertions"):
                    self.run_fake(lambda module, mode: self.fail("optimized runner executed"))
                with mock.patch.object(
                    reproduce.sys,
                    "argv",
                    ["reproduce", "execute", "--run-dir", str(self.root), "--mode", "preflight"],
                ), mock.patch.object(reproduce.subprocess, "call") as child:
                    with self.assertRaisesRegex(RuntimeError, "requires assertions"):
                        reproduce.main()
                child.assert_not_called()
        with mock.patch.object(reproduce.sys, "flags", types.SimpleNamespace(optimize=1)):
            with self.assertRaisesRegex(RuntimeError, "requires assertions"):
                reproduce.require_assertions()
        self.assertFalse((self.root / "EXECUTION_FAILED.json").exists())
        self.assertFalse((self.root / "EXECUTION.lock").exists())

    def test_missing_qualification_can_be_fixed_without_poisoning_run(self):
        with self.assertRaisesRegex(ValueError, "Run preflight and smoke"):
            self.run_fake(lambda module, mode: self.fail("unqualified fit executed"), mode="run")
        self.assertFalse((self.root / "EXECUTION_FAILED.json").exists())
        self.assertFalse((self.root / "EXECUTION.lock").exists())
        self.run_fake(lambda module, mode: self.assertEqual(mode, "preflight"))

    def test_completion_between_validation_and_lock_preserves_finished_run(self):
        def finish_other_invocation():
            reproduce.write(self.root / "DONE.json", {"status": "completed"})
            return {"fixture": True}

        with mock.patch.object(
            reproduce, "verify_prepared", return_value=self.binding
        ), mock.patch.object(
            reproduce, "runtime_environment", side_effect=finish_other_invocation
        ), mock.patch.object(reproduce.sys, "platform", "linux"), mock.patch.object(
            reproduce.sys, "argv", ["launch.py", "--mode", "preflight"]
        ), mock.patch.object(reproduce, "_launch") as runner:
            with self.assertRaisesRegex(RuntimeError, "Completed reproduction preserved"):
                reproduce.launch(self.root)
        runner.assert_not_called()
        self.assertFalse((self.root / "EXECUTION_FAILED.json").exists())
        self.assertFalse((self.root / "EXECUTION.lock").exists())

    def test_full_formal_execution_checks_qualification_identity(self):
        protocol = self.root / "protocol/BRTP_protocol.json"
        reproduce.write(protocol, {"frozen": True})
        reproduce.write(
            self.root / "receipts/preflight.json",
            {"passed": True, "protocol_sha": reproduce.sha(protocol)},
        )
        smoke = {"passed": True, "contract": {"runner_sha": self.binding["runner_sha256"]}}
        reproduce.write(self.root / "receipts/smoke.json", smoke)
        self.run_fake(lambda module, mode: self.assertEqual(mode, "run"), mode="run")
        smoke["contract"]["runner_sha"] = "other-runner"
        reproduce.write(self.root / "receipts/smoke.json", smoke)
        with self.assertRaisesRegex(ValueError, "receipts do not match"):
            self.run_fake(
                lambda module, mode: self.fail("wrong smoke receipt accepted"), mode="run"
            )
        self.assertFalse((self.root / "EXECUTION_FAILED.json").exists())

    def test_supplementary_job_mode_and_smoke_gates(self):
        job = {"source_job_id": "job1", "task": "DG", "native_class_ids": ["0", "1"]}
        plan = {"main_training": [job], "training": {"epochs": 50}, "parameters": {"radius": 0.25}}
        reproduce.write(self.root / "plan.json", plan)
        binding = dict(self.binding, family="n", smoke_jobs={"DG": "job1"})
        args = types.SimpleNamespace(mode="run", job_id="job1")
        with self.assertRaisesRegex(ValueError, "qualified task smoke"):
            reproduce.validate_execution(self.root, args, binding)
        for name in ("adapter.py", "residual_transport.py"):
            path = self.root / "brtp_impl" / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("# fixture source")
        gate = {
            "passed": True,
            "runner_sha": binding["runner_sha256"],
            "adapter_sha": reproduce.sha(self.root / "brtp_impl/adapter.py"),
            "model_sha": reproduce.sha(self.root / "brtp_impl/residual_transport.py"),
            "plan_sha": reproduce.sha(self.root / "plan.json"),
            "source_class_count": 2,
            "training": plan["training"],
            "parameters": plan["parameters"],
        }
        path = self.root / "results/Full/smoke_receipts/DG.json"
        reproduce.write(path, gate)
        reproduce.validate_execution(self.root, args, binding)
        gate["source_class_count"] = 1
        reproduce.write(path, gate)
        with self.assertRaisesRegex(ValueError, "smoke receipt does not match"):
            reproduce.validate_execution(self.root, args, binding)
        args.mode = "eval_only"
        with self.assertRaisesRegex(ValueError, "new fits require"):
            reproduce.validate_execution(self.root, args, binding)
        job["checkpoint_path"] = "/original/checkpoint.pt"
        reproduce.write(self.root / "plan.json", plan)
        reproduce.validate_execution(self.root, args, binding)
        args.mode = "run"
        with self.assertRaisesRegex(ValueError, "reused jobs require"):
            reproduce.validate_execution(self.root, args, binding)

    def public_fixture(self):
        project = self.root / "public_source"
        package = project / "brtp"
        package.mkdir(parents=True)
        (project / "CLIP").mkdir(parents=True)
        (project / "protocols").mkdir()
        contents = {
            "frozen_runner.py": b"# frozen fixture; never run\n",
            "data_index.json": (self.root / "data_index.json").read_bytes(),
            "protocol/BRTP_protocol.json": b'{"method": {"radius": 0.25}}',
            "protocol/experiment_plan.json": b'{"main_training": []}',
            "paired_support/job1.json": b'{"support": []}',
        }
        bundle = project / "protocols/full_sgd50_inputs.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
        manifest = {
            "schema_version": 1,
            "recipe": "Full",
            "bundle": "protocols/full_sgd50_inputs.zip",
            "sha256": reproduce.sha(bundle),
            "files": {name: reproduce.digest(data) for name, data in contents.items()},
        }
        reproduce.write(package / "public_inputs.json", manifest)
        recipe = {
            "family": "full",
            "configuration": "Full",
            "jobs": ["job1"],
            "weights_sha256": reproduce.sha(self.image),
            "backbone": "fixture",
            "runner_sha256": reproduce.digest(contents["frozen_runner.py"]),
        }
        catalog = {"recipes": {"Full": recipe}, "archives": {}}
        args = types.SimpleNamespace(
            recipe="Full",
            archive_root=None,
            output=self.root / "prepared",
            weights=self.image,
            path_map=None,
        )
        return package, manifest, catalog, args

    def test_public_full_preparation_needs_no_private_archives(self):
        package, manifest, catalog, args = self.public_fixture()
        with mock.patch.object(reproduce, "PACKAGE", package), mock.patch.object(
            reproduce, "inventory", return_value=catalog
        ):
            binding = reproduce.prepare(args)
            self.assertEqual(reproduce.verify_prepared(args.output), binding)
        self.assertEqual(binding["archive_sha256"], {})
        self.assertEqual(binding["public_inputs"]["sha256"], manifest["sha256"])
        self.assertFalse(binding["training_started"])
        self.assertEqual(
            (args.output / "data_index.json").read_bytes(),
            (self.root / "data_index.json").read_bytes(),
        )
        self.assertFalse((args.output / "runs").exists())

    def test_modified_public_bundle_is_rejected_before_extracting(self):
        package, manifest, catalog, args = self.public_fixture()
        bundle = package.parent / manifest["bundle"]
        with bundle.open("ab") as stream:
            stream.write(b"unexpected edit")
        with mock.patch.object(reproduce, "PACKAGE", package), mock.patch.object(
            reproduce, "inventory", return_value=catalog
        ):
            with self.assertRaisesRegex(ValueError, "Public input bundle SHA mismatch"):
                reproduce.prepare(args)
        self.assertFalse((args.output / "frozen_runner.py").exists())
        self.assertTrue((args.output / "PREPARATION_FAILED.json").exists())

    def test_full_accepts_verified_external_protocol_archive(self):
        package, manifest, catalog, args = self.public_fixture()
        archive = self.root / "standalone-inputs.zip"
        (package.parent / manifest["bundle"]).rename(archive)
        args.protocol_archive = archive
        with mock.patch.object(reproduce, "PACKAGE", package), mock.patch.object(
            reproduce, "inventory", return_value=catalog
        ):
            binding = reproduce.prepare(args)
            self.assertEqual(binding, reproduce.verify_prepared(args.output))
            self.assertEqual(binding["public_inputs"]["sha256"], manifest["sha256"])

    def test_archive_full_preserves_both_historical_support_directory_names(self):
        package, manifest, catalog, args = self.public_fixture()
        archive_root = self.root / "archives"
        archive_root.mkdir()
        args.archive_root = archive_root
        source = package.parent / manifest["bundle"]
        with zipfile.ZipFile(source) as bundle:
            files = {name: bundle.read(name) for name in bundle.namelist()}
        support = files.pop("paired_support/job1.json")
        for folder in ("paired_support", "support"):
            with self.subTest(folder=folder):
                archive_path = archive_root / (folder + ".zip")
                with zipfile.ZipFile(archive_path, "w") as archive:
                    for name, data in files.items():
                        archive.writestr(name, data)
                    archive.writestr(folder + "/job1.json", support)
                recipe = catalog["recipes"]["Full"]
                recipe.update(
                    archive=archive_path.name,
                    prefix="",
                    runner_archive=archive_path.name,
                    runner_member="frozen_runner.py",
                )
                catalog["archives"] = {archive_path.name: reproduce.sha(archive_path)}
                args.output = self.root / ("prepared_" + folder)
                with mock.patch.object(reproduce, "PACKAGE", package), mock.patch.object(
                    reproduce, "inventory", return_value=catalog
                ):
                    binding = reproduce.prepare(args)
                    self.assertEqual(reproduce.verify_prepared(args.output), binding)
                self.assertEqual((args.output / folder / "job1.json").read_bytes(), support)
                self.assertEqual(binding["files"][folder + "/job1.json"], reproduce.digest(support))

    def test_public_bundle_rejects_extra_members_and_escaping_paths(self):
        package, manifest, catalog, args = self.public_fixture()
        bundle = package.parent / manifest["bundle"]
        with zipfile.ZipFile(bundle, "a") as archive:
            archive.writestr("paired_support/../../outside.json", b"{}")
        manifest["sha256"] = reproduce.sha(bundle)
        reproduce.write(package / "public_inputs.json", manifest)
        with mock.patch.object(reproduce, "PACKAGE", package):
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                reproduce.copy_public_inputs(args.output, catalog["recipes"]["Full"])
            manifest["files"]["paired_support/../../outside.json"] = reproduce.digest(b"{}")
            reproduce.write(package / "public_inputs.json", manifest)
            with self.assertRaisesRegex(ValueError, "Unexpected public input path"):
                reproduce.copy_public_inputs(args.output, catalog["recipes"]["Full"])
        self.assertFalse((self.root / "outside.json").exists())

    def test_historical_recipe_without_archives_fails_before_creating_output(self):
        package, manifest, catalog, args = self.public_fixture()
        args.recipe = "historical"
        catalog["recipes"]["historical"] = catalog["recipes"]["Full"]
        with mock.patch.object(reproduce, "inventory", return_value=catalog):
            with self.assertRaisesRegex(ValueError, "historical recipes require --archive-root"):
                reproduce.prepare(args)
        self.assertFalse(args.output.exists())

"""Summarize a complete Full reproduction using the frozen three-seed protocol.

This checks completion, inputs, checkpoints, prediction rows and recomputed
metrics. It does not rerun inference or replace a fresh training reproduction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from brtp.reproduce import digest, read, sha, verify_prepared
from brtp.evaluation.audit import audit_predictions


def _unique(rows: list, key: str) -> dict:
    result = {}
    for row in rows:
        name = row[key]
        if name in result:
            raise ValueError("Duplicate " + key + ": " + name)
        result[name] = row
    return result


def _harmonic(base: float, new: float) -> float:
    return 2 * base * new / (base + new) if base + new else 0.0


def aggregate(plan: dict, metrics: list) -> dict:
    """Require every planned slot, then average datasets equally within a seed."""
    training = _unique(plan["main_training"], "source_job_id")
    expected = _unique(plan["main_evaluation"], "evaluation_job_id")
    observed = _unique([dict(row["job"], metric=row) for row in metrics], "evaluation_job_id")
    if len(training) != 18 or len(expected) != 96 or set(observed) != set(expected):
        raise ValueError("Full requires exactly 18 source fits and all 96 evaluation slots")
    sources = {
        "B2N": ("PatternNet", "RSICD", "NWPU45", "MLRSNet"),
        "CD": ("PatternNet",),
        "DG": ("PatternNetv2",),
    }
    if {(j["task"], j["source_dataset"], j["seed"]) for j in training.values()} != {
        (task, source, seed)
        for task, datasets in sources.items()
        for source in datasets
        for seed in (1, 2, 3)
    }:
        raise ValueError("Full source tasks/seeds differ from the frozen protocol")
    slots, checkpoint_by_source = {}, {}
    for name, job in expected.items():
        row = observed[name]["metric"]
        source = training[job["source_job_id"]]
        if (
            row["job"] != job
            or job["task"] != source["task"]
            or job["seed"] != source["seed"]
            or job["variant"] != "Full"
            or source["variant"] != "Full"
            or row["epoch"] != 50
            or row["selection_used"] is not False
        ):
            raise ValueError("Metric identity/final50 selection mismatch: " + name)
        n, correct, accuracy = row["n"], row["correct"], row["accuracy"]
        if (
            type(n) is not int
            or type(correct) is not int
            or n != job["records"]
            or n <= 0
            or not 0 <= correct <= n
            or type(accuracy) not in (int, float)
            or not math.isfinite(accuracy)
            or not math.isclose(accuracy, 100 * correct / n, rel_tol=0, abs_tol=1e-10)
            or not math.isclose(row["metrics"]["accuracy"], accuracy, rel_tol=0, abs_tol=1e-10)
        ):
            raise ValueError("Invalid metric accuracy/counts: " + name)
        cp = row["checkpoint_sha"]
        if not isinstance(cp, str) or len(cp) != 64 or any(c not in "0123456789abcdef" for c in cp):
            raise ValueError("Invalid checkpoint SHA: " + name)
        previous = checkpoint_by_source.setdefault(job["source_job_id"], cp)
        if previous != cp:
            raise ValueError("Evaluations use different source checkpoints: " + name)
        slot = (
            job["task"],
            job["seed"],
            job["evaluation_dataset"],
            job["evaluation_split"],
            job["candidate_subset"],
        )
        if slot in slots:
            raise ValueError("Duplicate statistical slot: " + name)
        slots[slot] = accuracy
    domains = {
        "B2N": sources["B2N"],
        "CD": sources["B2N"],
        "DG": ("PatternNetv2", "RSICDv2", "NWPU45v2", "MLRSNetv2"),
    }
    required = {
        (task, seed, dataset, split, subset)
        for task, datasets in domains.items()
        for seed in (1, 2, 3)
        for dataset in datasets
        for split in ("test", "val")
        for subset in (("base", "new") if task == "B2N" else ("all",))
    }
    if set(slots) != required:
        raise ValueError("Full statistical slots differ from the frozen protocol")
    seeds = []
    for seed in (1, 2, 3):
        base, new = [
            mean(slots[("B2N", seed, dataset, "test", subset)] for dataset in domains["B2N"])
            for subset in ("base", "new")
        ]
        seeds.append({"seed": seed, "base": base, "new": new, "H": _harmonic(base, new)})
    base, new = mean(s["base"] for s in seeds), mean(s["new"] for s in seeds)
    result = {
        "recipe": "Full",
        "split": "test",
        "seed_ids": [1, 2, 3],
        "unit": "percent",
        "completed_training": 18,
        "completed_evaluation": 96,
        "selected_test_evaluations": 42,
        "B2N": {
            "base_mean": base,
            "new_mean": new,
            "H": _harmonic(base, new),
            "seed_H_sample_sd": stdev(s["H"] for s in seeds),
            "seed_values": seeds,
        },
    }
    for task in ("CD", "DG"):
        targets = [d for d in domains[task] if d not in sources[task]]
        values = [
            {"seed": s, "mean": mean(slots[(task, s, d, "test", "all")] for d in targets)}
            for s in (1, 2, 3)
        ]
        result[task] = {
            "mean": mean(v["mean"] for v in values),
            "sample_sd": stdev(v["mean"] for v in values),
            "target_datasets": targets,
            "seed_values": values,
        }
    return result


def summarize(root: Path) -> dict:
    root = root.resolve()
    binding = verify_prepared(root)
    if binding["recipe"] != "Full" or binding["family"] != "full":
        raise ValueError("Only the frozen Full three-seed protocol is supported")
    if (
        (root / "EXECUTION.lock").exists()
        or (root / "EXECUTION_FAILED.json").exists()
        or list(root.glob("FAILED_*.json"))
    ):
        raise ValueError("Execution is active, failed or requires diagnosis")
    done, status = read(root / "DONE.json"), read(root / "status.json")
    if (done.get("status"), done.get("training"), done.get("evaluation"), status.get("stage")) != (
        "completed",
        18,
        96,
        "completed",
    ):
        raise ValueError("Full has no complete 18-fit/96-evaluation terminal receipt")
    execution = read(root / "EXECUTION_BINDING.json")
    if (
        execution.get("mode") != "run"
        or execution.get("prepared_sha256") != sha(root / "PREPARED.json")
        or execution.get("frozen_runner_sha256") != binding["runner_sha256"]
    ):
        raise ValueError("Execution is not bound to the current prepared inputs")
    plan, spec = (
        read(root / "protocol/experiment_plan.json"),
        read(root / "protocol/BRTP_protocol.json"),
    )
    jobs = _unique(plan["main_training"], "source_job_id")
    expected_metrics = {
        "evaluations/" + j["evaluation_job_id"] + "/metrics.json": j
        for j in plan["main_evaluation"]
    }
    actual_metrics = {
        p.relative_to(root).as_posix() for p in (root / "evaluations").rglob("metrics.json")
    }
    if actual_metrics != set(expected_metrics) or {
        p.name for p in (root / "runs").iterdir() if p.is_dir()
    } != set(jobs):
        raise ValueError("Incomplete or unexpected result inventory")
    core = {
        p[len("BRTP/") :]: value
        for p, value in binding["files"].items()
        if p.startswith("BRTP/brtp/") and p.endswith(".py")
    }
    core_sha = digest(json.dumps(core, sort_keys=True, separators=(",", ":")).encode())
    checkpoints = {}
    for name, job in jobs.items():
        receipt = read(root / "runs" / name / "completed.json")
        cp = sha(root / "runs" / name / "final50.pt")
        support = read(root / "paired_support" / (name + ".json"))
        expected_contract = {
            "source_job_id": name,
            "protocol_id": spec["protocol_id"],
            "protocol_sha": sha(root / "protocol/BRTP_protocol.json"),
            "runner_sha": binding["runner_sha256"],
            "backbone_sha": binding["weights_sha256"],
            "support_hash": support["support_hash"],
            "model_tree_sha": core_sha,
            "recipe": spec["training"],
            "method": spec["method"],
            "selection": "final50",
            "validation_for_selection": 0,
            "native_class_ids": job["native_class_ids"],
        }
        if (
            receipt["epoch"] != 50
            or receipt["checkpoint_sha"] != cp
            or any(receipt["contract"].get(k) != v for k, v in expected_contract.items())
        ):
            raise ValueError("Final checkpoint/contract mismatch: " + name)
        checkpoints[name] = cp
    index = read(root / "data_index.json")
    metrics, hashes, prediction_hashes = [], {}, {}
    for relative in sorted(expected_metrics):
        path = root / relative
        row = read(path)
        if row["job"] != expected_metrics[relative]:
            raise ValueError("Evaluation directory/plan identity mismatch: " + relative)
        if (
            row["checkpoint_sha"] != checkpoints[row["job"]["source_job_id"]]
            or not path.with_name("predictions.npz").is_file()
        ):
            raise ValueError("Evaluation checkpoint/prediction missing or mismatched: " + relative)
        prediction = path.with_name("predictions.npz")
        audit_predictions(prediction, row, index)
        metrics.append(row)
        hashes[relative] = sha(path)
        prediction_hashes[prediction.relative_to(root).as_posix()] = sha(prediction)
    result = aggregate(plan, metrics)
    result["evidence"] = {
        "prepared_sha256": sha(root / "PREPARED.json"),
        "metrics_sha256": hashes,
        "predictions_sha256": prediction_hashes,
        "checkpoints_sha256": checkpoints,
        "scope": "Input/source, completion, checkpoint SHA, prediction CRC, frozen row order and all native metrics; no inference replay",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

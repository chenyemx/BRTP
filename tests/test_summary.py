"""Full-table aggregation and evidence gates, using synthetic metrics only."""

import copy
import math
from pathlib import Path
from statistics import stdev
import tempfile
import unittest
from unittest import mock

import numpy as np

from brtp import summarize
from brtp.evaluation.native import native_metrics
from brtp.reproduce import digest, sha, write


def fixture():
    plan = {"main_training": [], "main_evaluation": []}
    metrics = []
    datasets = {
        "B2N": ("PatternNet", "RSICD", "NWPU45", "MLRSNet"),
        "CD": ("PatternNet", "RSICD", "NWPU45", "MLRSNet"),
        "DG": ("PatternNetv2", "RSICDv2", "NWPU45v2", "MLRSNetv2"),
    }
    for task, domains in datasets.items():
        for seed in (1, 2, 3):
            for source in domains if task == "B2N" else domains[:1]:
                source_id = f"Full__{task}__{source}__seed{seed}"
                plan["main_training"].append(
                    {
                        "source_job_id": source_id,
                        "task": task,
                        "source_dataset": source,
                        "seed": seed,
                        "variant": "Full",
                        "native_class_ids": ["0"],
                    }
                )
                for d, dataset in enumerate((source,) if task == "B2N" else domains):
                    for split in ("test", "val"):
                        for subset in ("base", "new") if task == "B2N" else ("all",):
                            job = {
                                "evaluation_job_id": f"{source_id}__{dataset}__{split}__{subset}",
                                "source_job_id": source_id,
                                "task": task,
                                "seed": seed,
                                "variant": "Full",
                                "evaluation_dataset": dataset,
                                "evaluation_split": split,
                                "candidate_subset": subset,
                                "candidate_class_ids": (
                                    ["0", "1"]
                                    if subset == "base"
                                    else ["2", "3"]
                                    if subset == "new"
                                    else ["0", "1", "2", "3"]
                                ),
                                "records": 200 if task == "CD" else 100 * (d + 1),
                            }
                            accuracy = (
                                0
                                if split == "val"
                                else (
                                    ([90, 70, 50] if subset == "base" else [30, 50, 90])[seed - 1]
                                    if task == "B2N"
                                    else 100
                                    if d == 0
                                    else [20, 50, 80][seed - 1] + 3 * (d - 1)
                                    if task == "CD"
                                    else [40, 60, 80][seed - 1] + d - 1
                                )
                            )
                            plan["main_evaluation"].append(job)
                            metrics.append(
                                {
                                    "job": copy.deepcopy(job),
                                    "epoch": 50,
                                    "selection_used": False,
                                    "correct": int(job["records"] * accuracy / 100),
                                    "n": job["records"],
                                    "accuracy": accuracy,
                                    "metrics": {"accuracy": accuracy},
                                    "checkpoint_sha": "a" * 64,
                                }
                            )
    return plan, metrics


class SummaryTests(unittest.TestCase):
    def test_equal_domain_then_seed_means_sample_sd_and_global_h(self):
        plan, metrics = fixture()
        result = summarize.aggregate(plan, metrics)
        self.assertEqual(result["completed_evaluation"], 96)
        self.assertEqual(result["selected_test_evaluations"], 42)
        base, new = 70, 170 / 3
        self.assertAlmostEqual(result["B2N"]["H"], 2 * base * new / (base + new))
        seed_h = [2 * b * n / (b + n) for b, n in zip((90, 70, 50), (30, 50, 90))]
        self.assertAlmostEqual(result["B2N"]["seed_H_sample_sd"], stdev(seed_h))
        self.assertNotAlmostEqual(result["B2N"]["H"], sum(seed_h) / 3)
        self.assertEqual(result["CD"]["mean"], 53)
        self.assertEqual(result["CD"]["sample_sd"], 30)
        self.assertEqual(result["DG"]["mean"], 61)
        self.assertEqual(result["DG"]["sample_sd"], 20)

    def test_missing_or_duplicate_metrics_are_rejected(self):
        plan, metrics = fixture()
        with self.assertRaisesRegex(ValueError, "all 96 evaluation slots"):
            summarize.aggregate(plan, metrics[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation_job_id"):
            summarize.aggregate(plan, metrics + [metrics[0]])

    def test_duplicate_statistical_slot_is_not_counted_twice(self):
        plan, metrics = fixture()
        duplicate = dict(plan["main_evaluation"][0], evaluation_job_id="another_name")
        plan["main_evaluation"][1] = duplicate
        metrics[1] = dict(metrics[0], job=copy.deepcopy(duplicate))
        with self.assertRaisesRegex(ValueError, "Duplicate statistical slot"):
            summarize.aggregate(plan, metrics)

    def test_invalid_accuracy_counts_selection_and_mixed_checkpoints_are_rejected(self):
        for change in (
            {"accuracy": math.nan},
            {"accuracy": 101},
            {"correct": -1},
            {"n": True},
            {"epoch": 49},
            {"selection_used": True},
            {"checkpoint_sha": "b" * 64},
        ):
            with self.subTest(change=change):
                plan, metrics = fixture()
                metrics[0].update(change)
                with self.assertRaises(ValueError):
                    summarize.aggregate(plan, metrics)

    def test_complete_receipts_bind_checkpoints_and_reject_incomplete_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plan, metrics = fixture()
            binding = {
                "recipe": "Full",
                "family": "full",
                "runner_sha256": "f" * 64,
                "weights_sha256": "e" * 64,
                "files": {},
            }
            spec = {
                "protocol_id": "fixture",
                "training": {"optimizer": "SGD"},
                "method": {"radius": 0.25},
            }
            write(root / "PREPARED.json", binding)
            write(root / "protocol/experiment_plan.json", plan)
            write(root / "protocol/BRTP_protocol.json", spec)
            index = {}
            for job in plan["main_evaluation"]:
                dataset = job["evaluation_dataset"]
                if dataset in index:
                    continue
                count = job["records"] * (2 if job["task"] == "B2N" else 1)
                index[dataset] = [
                    {
                        "split": split,
                        "relative_path": f"{split}/{i}.png",
                        "native_id": str(i % 4),
                        "display_name": "class" + str(i % 4),
                        "sha256": digest(f"{dataset}/{split}/{i}".encode()),
                    }
                    for split in ("test", "val")
                    for i in range(count)
                ]
            write(root / "data_index.json", index)
            write(root / "DONE.json", {"status": "completed", "training": 18, "evaluation": 96})
            write(root / "status.json", {"stage": "completed"})
            execution = {
                "mode": "run",
                "prepared_sha256": sha(root / "PREPARED.json"),
                "frozen_runner_sha256": binding["runner_sha256"],
            }
            write(root / "EXECUTION_BINDING.json", execution)
            cp_sha = digest(b"synthetic checkpoint")
            for job in plan["main_training"]:
                name = job["source_job_id"]
                write(
                    root / "paired_support" / (name + ".json"), {"support_hash": "support_fixture"}
                )
                folder = root / "runs" / name
                folder.mkdir(parents=True)
                (folder / "final50.pt").write_bytes(b"synthetic checkpoint")
                contract = {
                    "source_job_id": name,
                    "protocol_id": spec["protocol_id"],
                    "protocol_sha": sha(root / "protocol/BRTP_protocol.json"),
                    "runner_sha": binding["runner_sha256"],
                    "backbone_sha": binding["weights_sha256"],
                    "support_hash": "support_fixture",
                    "model_tree_sha": digest(b"{}"),
                    "recipe": spec["training"],
                    "method": spec["method"],
                    "selection": "final50",
                    "validation_for_selection": 0,
                    "native_class_ids": job["native_class_ids"],
                }
                write(
                    folder / "completed.json",
                    {"epoch": 50, "checkpoint_sha": cp_sha, "contract": contract},
                )
            for row in metrics:
                row["checkpoint_sha"] = cp_sha
                folder = root / "evaluations" / row["job"]["evaluation_job_id"]
                ids = row["job"]["candidate_class_ids"]
                rows = [
                    r
                    for r in index[row["job"]["evaluation_dataset"]]
                    if r["split"] == row["job"]["evaluation_split"] and r["native_id"] in ids
                ]
                labels = np.array([ids.index(r["native_id"]) for r in rows])
                pred = labels.copy()
                pred[row["correct"] :] = (pred[row["correct"] :] + 1) % len(ids)
                logits = np.zeros((len(rows), len(ids)), dtype=np.float32)
                logits[np.arange(len(rows)), pred] = 10
                row["metrics"] = native_metrics(logits, labels)
                row["candidate_names"] = ["class" + i for i in ids]
                write(folder / "metrics.json", row)
                np.savez_compressed(
                    folder / "predictions.npz",
                    logits=logits,
                    labels=labels,
                    native_ids=np.array([r["native_id"] for r in rows]),
                    paths=np.array([r["relative_path"] for r in rows]),
                    image_sha=np.array([r["sha256"] for r in rows]),
                )
            with mock.patch.object(summarize, "verify_prepared", return_value=binding):
                result = summarize.summarize(root)
                self.assertEqual(result["CD"]["mean"], 53)
                self.assertEqual(len(result["evidence"]["metrics_sha256"]), 96)
                self.assertEqual(len(result["evidence"]["predictions_sha256"]), 96)
                first, second = [
                    root / "evaluations" / metrics[i]["job"]["evaluation_job_id"] for i in (0, 16)
                ]
                saved = {
                    folder / name: (folder / name).read_bytes()
                    for folder in (first, second)
                    for name in ("metrics.json", "predictions.npz")
                }
                try:
                    for name in ("metrics.json", "predictions.npz"):
                        (first / name).write_bytes(saved[second / name])
                        (second / name).write_bytes(saved[first / name])
                    with self.assertRaisesRegex(ValueError, "directory/plan identity mismatch"):
                        summarize.summarize(root)
                finally:
                    for path, data in saved.items():
                        path.write_bytes(data)
                write(root / "status.json", {"stage": "training"})
                with self.assertRaisesRegex(ValueError, "terminal receipt"):
                    summarize.summarize(root)
                write(root / "status.json", {"stage": "completed"})
                execution["prepared_sha256"] = "unbound"
                write(root / "EXECUTION_BINDING.json", execution)
                with self.assertRaisesRegex(ValueError, "current prepared inputs"):
                    summarize.summarize(root)
                execution["prepared_sha256"] = sha(root / "PREPARED.json")
                write(root / "EXECUTION_BINDING.json", execution)
                (
                    root / "runs" / plan["main_training"][0]["source_job_id"] / "final50.pt"
                ).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checkpoint/contract mismatch"):
                    summarize.summarize(root)

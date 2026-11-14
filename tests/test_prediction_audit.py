"""Saved-prediction checks reject self-consistent metrics with wrong evidence."""

import copy
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from brtp.evaluation.audit import audit_predictions
from brtp.evaluation.native import native_metrics


class PredictionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "predictions.npz"
        self.arrays = {
            "logits": np.array([[2, 0], [3, 1], [0, 4]], dtype=np.float32),
            "labels": np.array([0, 1, 1], dtype=np.int64),
            "native_ids": np.array(["9", "3", "3"]),
            "paths": np.array(["a.png", "b.png", "c.png"]),
            "image_sha": np.array(["a" * 64, "b" * 64, "c" * 64]),
        }
        self.index = {
            "target": [
                {
                    "split": "test",
                    "native_id": native_id,
                    "relative_path": path,
                    "sha256": sha,
                    "display_name": "class" + native_id,
                }
                for native_id, path, sha in zip(
                    self.arrays["native_ids"], self.arrays["paths"], self.arrays["image_sha"]
                )
            ]
        }
        values = native_metrics(self.arrays["logits"], self.arrays["labels"])
        self.metric = {
            "job": {
                "candidate_class_ids": ["9", "3"],
                "evaluation_dataset": "target",
                "evaluation_split": "test",
                "records": 3,
            },
            "candidate_names": ["class9", "class3"],
            "n": 3,
            "correct": 2,
            "accuracy": values["accuracy"],
            "metrics": values,
        }

    def audit(self, arrays=None, metric=None):
        np.savez_compressed(self.path, **(self.arrays if arrays is None else arrays))
        return audit_predictions(self.path, self.metric if metric is None else metric, self.index)

    def test_exact_frozen_rows_and_nonnumeric_native_order(self):
        self.assertEqual(self.audit()["correct"], 2)

    def test_each_metadata_column_and_label_dtype_is_checked(self):
        for field in ("paths", "native_ids", "image_sha", "labels"):
            arrays = copy.deepcopy(self.arrays)
            arrays[field] = arrays[field][::-1]
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "row/order mismatch"
            ):
                self.audit(arrays)
        arrays = dict(self.arrays, labels=self.arrays["labels"].astype(float))
        with self.assertRaisesRegex(ValueError, "row/order mismatch"):
            self.audit(arrays)

    def test_reordered_population_is_rejected_even_if_accuracy_unchanged(self):
        arrays = {field: value[::-1] for field, value in self.arrays.items()}
        with self.assertRaisesRegex(ValueError, "row/order mismatch"):
            self.audit(arrays)

    def test_logits_shape_and_finiteness_and_npz_inventory(self):
        for logits in (
            self.arrays["logits"][:, :1],
            np.full((3, 2), np.nan),
            self.arrays["logits"].astype(int),
        ):
            with self.subTest(logits=logits), self.assertRaisesRegex(ValueError, "logits must"):
                self.audit(dict(self.arrays, logits=logits))
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            self.audit(dict(self.arrays, extra=np.array([1])))

    def test_candidate_order_and_metrics_including_confidence_are_checked(self):
        metric = copy.deepcopy(self.metric)
        metric["candidate_names"].reverse()
        with self.assertRaisesRegex(ValueError, "Candidate names/order"):
            self.audit(metric=metric)
        for field in ("nll", "ece", "macro_f1", "accuracy"):
            metric = copy.deepcopy(self.metric)
            metric["metrics"][field] += 0.2
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "metric mismatch"):
                self.audit(metric=metric)
        # Same argmax and accuracy, changed confidence; the old count check missed this.
        with self.assertRaisesRegex(ValueError, "metric mismatch"):
            self.audit(dict(self.arrays, logits=self.arrays["logits"] * 2))

    def test_corrupt_archive_cannot_pass_by_existence(self):
        self.path.write_bytes(b"not a numpy archive")
        with self.assertRaises((ValueError, OSError, zipfile.BadZipFile)):
            audit_predictions(self.path, self.metric, self.index)

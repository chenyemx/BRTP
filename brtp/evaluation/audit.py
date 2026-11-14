"""Recompute saved predictions and bind their rows to the frozen data index.

This is an artifact consistency check, not an inference replay. Image pixels and
model execution are not rechecked here; those are separate reproduction steps.
"""

import math
from pathlib import Path

import numpy as np

from .native import native_metrics


def _compare(expected, observed, field="metrics"):
    """Compare the complete metric schema without accepting NaN or missing keys."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or expected.keys() != observed.keys():
            raise ValueError("Metric schema mismatch: " + field)
        for key, value in expected.items():
            _compare(value, observed[key], field + "." + key)
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            raise ValueError("Metric shape mismatch: " + field)
        for i, (left, right) in enumerate(zip(expected, observed)):
            _compare(left, right, f"{field}[{i}]")
    elif isinstance(expected, float):
        if (
            type(observed) not in (int, float)
            or not math.isfinite(observed)
            or not math.isclose(expected, observed, rel_tol=1e-10, abs_tol=1e-8)
        ):
            raise ValueError("Recomputed metric mismatch: " + field)
    elif type(expected) is not type(observed) or expected != observed:
        raise ValueError("Recomputed metric mismatch: " + field)


def audit_predictions(path: Path, metric: dict, index: dict) -> dict:
    """Check all stored arrays, exact sample order, candidate order, and metrics.

    The frozen runner writes relative paths and native class IDs in index order.
    Candidate IDs define logit columns; labels are local column indices. Duplicate
    image content is not deduplicated: the original evaluation population is kept.
    """
    job = metric["job"]
    ids = job["candidate_class_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be nonempty and unique")
    lookup = {class_id: i for i, class_id in enumerate(ids)}
    population = index[job["evaluation_dataset"]]
    rows = [
        r for r in population if r["split"] == job["evaluation_split"] and r["native_id"] in lookup
    ]
    if len(rows) != job["records"] or not rows:
        raise ValueError("Frozen evaluation population differs from the plan")
    names = {r["native_id"]: r["display_name"] for r in population}
    if metric["candidate_names"] != [names[class_id] for class_id in ids]:
        raise ValueError("Candidate names/order differ from the frozen index")
    expected = {
        "labels": np.asarray([lookup[r["native_id"]] for r in rows], dtype=np.int64),
        "native_ids": np.asarray([r["native_id"] for r in rows]),
        "paths": np.asarray([r["relative_path"] for r in rows]),
        "image_sha": np.asarray([r["sha256"] for r in rows]),
    }
    # Reading every member validates the NPZ's CRC. Never deserialize pickles.
    with np.load(path, allow_pickle=False) as saved:
        if len(saved.files) != 5 or set(saved.files) != {*expected, "logits"}:
            raise ValueError("Prediction archive inventory mismatch")
        for field, values in expected.items():
            actual = saved[field]
            valid_kind = actual.dtype.kind in ("iu" if field == "labels" else "US")
            if not valid_kind or not np.array_equal(actual, values):
                raise ValueError("Prediction row/order mismatch: " + field)
        logits = saved["logits"]
        if (
            logits.dtype.kind != "f"
            or logits.shape != (len(rows), len(ids))
            or not np.isfinite(logits).all()
        ):
            raise ValueError("Prediction logits must be finite [records, candidates] floats")
        recomputed = native_metrics(logits, expected["labels"])
        correct = int((logits.argmax(axis=1) == expected["labels"]).sum())
    _compare(recomputed, metric["metrics"])
    _compare(len(rows), metric["n"], "n")
    _compare(correct, metric["correct"], "correct")
    _compare(recomputed["accuracy"], metric["accuracy"], "accuracy")
    return {"records": len(rows), "correct": correct, "accuracy": recomputed["accuracy"]}

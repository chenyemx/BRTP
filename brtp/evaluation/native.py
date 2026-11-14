"""Native-class metrics. Accuracies/F1 are percentages; NLL/ECE are unscaled."""

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def softmax(logits):
    logits = _array(logits).astype(np.float64)
    if logits.ndim != 2 or not logits.size or not np.isfinite(logits).all():
        raise ValueError("logits must be a nonempty finite [N,C] array")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def native_metrics(logits, labels, n_bins=15):
    """OA, supported-class macro recall, all-candidate macro F1, NLL and ECE.

    Labels index the columns of logits. Missing test classes have null recall;
    macro accuracy averages classes with test support. Macro F1 includes all
    candidate classes, with zero for undefined F1, matching sklearn's explicit
    labels=range(C), zero_division=0 convention. No fitted calibration is used.
    """
    logits = _array(logits).astype(np.float64)
    probs = softmax(logits)
    labels = _array(labels)
    n, c = probs.shape
    if labels.shape != (n,) or labels.dtype.kind not in "iu":
        raise ValueError("labels must be a length-N integer vector")
    if labels.min() < 0 or labels.max() >= c:
        raise ValueError("labels must index native logit columns")
    # Promote only after the bounds check: narrow integer multiplication can
    # wrap when constructing flat confusion indices, while uint64 must not be
    # allowed to wrap into a valid signed index during conversion.
    labels = labels.astype(np.int64, copy=False)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    pred = probs.argmax(axis=1)
    cm = np.bincount(labels * c + pred, minlength=c * c).reshape(c, c)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    tp = np.diag(cm)
    recall = np.divide(tp, support, out=np.zeros(c, dtype=float), where=support > 0)
    f1 = np.divide(
        2.0 * tp, support + predicted, out=np.zeros(c, dtype=float), where=(support + predicted) > 0
    )
    shifted = logits - logits.max(axis=1, keepdims=True)
    nll = np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(n), labels])
    confidence = probs.max(axis=1)
    correct = pred == labels
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    calibration = []
    ece = 0.0
    for b in range(n_bins):
        take = bins == b
        count = int(take.sum())
        acc = float(correct[take].mean()) if count else None
        conf = float(confidence[take].mean()) if count else None
        if count:
            ece += count / n * abs(acc - conf)
        calibration.append(
            {
                "lower": b / n_bins,
                "upper": (b + 1) / n_bins,
                "count": count,
                "accuracy": acc,
                "confidence": conf,
            }
        )
    oa = float(correct.mean() * 100)
    return {
        "accuracy": oa,
        "oa": oa,
        "macro_accuracy": float(recall[support > 0].mean() * 100),
        "macro_f1": float(f1.mean() * 100),
        "nll": float(nll),
        "ece": float(ece),
        "num_samples": n,
        "num_classes": c,
        "macro_accuracy_scope": "classes_with_test_support",
        "macro_f1_scope": "all_candidate_classes_zero_division_0",
        "per_class": [
            {
                "class_id": k,
                "support": int(support[k]),
                "accuracy": float(recall[k] * 100) if support[k] else None,
                "f1": float(f1[k] * 100),
            }
            for k in range(c)
        ],
        "calibration_bins": calibration,
        "confusion_matrix": cm.tolist(),
    }


def harmonic_mean(base, new):
    """Compute H of the supplied Base/New values; callers choose aggregation scope."""
    base, new = float(base), float(new)
    if not np.isfinite([base, new]).all() or min(base, new) < 0:
        raise ValueError("base and new must be finite and nonnegative")
    return 0.0 if base + new == 0 else 2 * base * new / (base + new)

"""Classification metrics with an explicit class order."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...]) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    labels = list(classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    disagree = y_true != y_pred
    by_class = {c: int(((y_true == c) & disagree).sum()) for c in labels}
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else None,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": {c: float(r) for c, r in zip(labels, recalls)},
        "support": {c: int((y_true == c).sum()) for c in labels},
        "confusion_matrix": {"class_order": labels, "rows_true_cols_pred": cm.astype(int).tolist()},
        "disagreements": {"total": int(disagree.sum()), "by_true_class": by_class},
    }

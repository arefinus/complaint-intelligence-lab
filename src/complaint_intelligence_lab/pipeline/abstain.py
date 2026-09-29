"""Confidence-threshold abstention with a coverage/accuracy curve.

The threshold is chosen on the validation split and then applied unchanged to test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.0, 0.05))


@dataclass(frozen=True)
class CurvePoint:
    threshold: float
    coverage: float
    accuracy: float | None
    n_covered: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "threshold": self.threshold,
            "coverage": self.coverage,
            "accuracy_on_covered": self.accuracy,
            "n_covered": self.n_covered,
        }


def coverage_curve(confidence: np.ndarray, correct: np.ndarray, grid: tuple[float, ...] = DEFAULT_GRID) -> list[CurvePoint]:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have the same shape")
    n = len(confidence)
    points: list[CurvePoint] = []
    for t in grid:
        covered = confidence >= t
        k = int(covered.sum())
        acc = float(correct[covered].mean()) if k else None
        points.append(CurvePoint(threshold=float(t), coverage=(k / n if n else 0.0), accuracy=acc, n_covered=k))
    return points


def choose_threshold(curve: list[CurvePoint], target_accuracy: float, min_coverage: float = 0.2) -> float:
    """Smallest threshold whose covered accuracy meets the target with acceptable coverage.

    Falls back to the threshold with the highest covered accuracy among those with at
    least ``min_coverage`` coverage, or to the lowest grid value when nothing qualifies.
    """
    if not curve:
        raise ValueError("curve is empty")
    eligible = [p for p in curve if p.accuracy is not None and p.coverage >= min_coverage]
    for p in sorted(eligible, key=lambda q: q.threshold):
        if p.accuracy is not None and p.accuracy >= target_accuracy:
            return p.threshold
    if eligible:
        best = max(eligible, key=lambda q: (q.accuracy or 0.0, -q.threshold))
        return best.threshold
    return min(p.threshold for p in curve)


def route(confidence: np.ndarray, threshold: float) -> np.ndarray:
    """Return 'auto' where confidence meets the threshold, else 'human_review'."""
    confidence = np.asarray(confidence, dtype=float)
    return np.where(confidence >= threshold, "auto", "human_review")

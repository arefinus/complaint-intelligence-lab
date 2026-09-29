"""Schematic 2-D topic landscape from TruncatedSVD of the TF-IDF matrix.

The axes have no unit and no meaning beyond variance direction. The output is labelled
"schematic navigation only" and is never a metric.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

LABEL = "schematic navigation only; axes are SVD directions with no unit"


def topic_landscape(X: sparse.csr_matrix, labels: np.ndarray, seed: int = 42, max_points: int = 500) -> dict[str, Any]:
    if X.shape[0] < 3 or X.shape[1] < 3:
        raise ValueError("need at least 3 rows and 3 features for a 2-D landscape")
    svd = TruncatedSVD(n_components=2, random_state=seed)
    coords = svd.fit_transform(X)
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    if len(idx) > max_points:
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    return {
        "label": LABEL,
        "explained_variance_ratio": [float(v) for v in svd.explained_variance_ratio_],
        "n_points": int(len(idx)),
        "points": [
            {"x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4), "label": str(labels[i])}
            for i in idx
        ],
    }

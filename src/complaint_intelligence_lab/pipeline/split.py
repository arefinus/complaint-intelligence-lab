"""Chronology-aware three-way split that keeps duplicate groups on one side.

Rows are ordered by ``date_received``. Two cut dates are taken at the requested
quantiles: rows before the first cut are train, rows between the cuts are validation,
rows on or after the second cut are test. A duplicate group is then assigned as a whole
to the split of its earliest member (first-seen rule), so no test or validation row has
an exact or near-duplicate narrative in an earlier split. Rows moved by this rule are
counted and reported.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

SPLITS = ("train", "valid", "test")
_ORDER = {name: i for i, name in enumerate(SPLITS)}


@dataclass(frozen=True)
class SplitResult:
    split: np.ndarray  # str per row: train / valid / test
    train_end: pd.Timestamp
    valid_end: pd.Timestamp
    moved_for_dedup: int

    def counts(self) -> dict[str, int]:
        return {name: int((self.split == name).sum()) for name in SPLITS}

    def manifest_hash(self, ids: pd.Series) -> str:
        payload = json.dumps(
            {"ids": list(map(str, ids)), "split": list(map(str, self.split))}, sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def chronological_split(
    dates: pd.Series,
    group_id: np.ndarray | None = None,
    train_frac: float = 0.6,
    valid_frac: float = 0.2,
) -> SplitResult:
    if not (0 < train_frac < 1 and 0 < valid_frac < 1 and train_frac + valid_frac < 1):
        raise ValueError("train_frac and valid_frac must be positive and sum to less than 1")
    ts = pd.to_datetime(dates).reset_index(drop=True)
    n = len(ts)
    if n < 10:
        raise ValueError("need at least 10 rows to split")
    order = np.argsort(ts.to_numpy(), kind="stable")
    n_train = int(np.floor(train_frac * n))
    n_valid = int(np.floor(valid_frac * n))
    split = np.empty(n, dtype=object)
    split[order[:n_train]] = "train"
    split[order[n_train : n_train + n_valid]] = "valid"
    split[order[n_train + n_valid :]] = "test"
    train_end = ts.iloc[order[n_train - 1]]
    valid_end = ts.iloc[order[n_train + n_valid - 1]]

    moved = 0
    if group_id is not None:
        group_id = np.asarray(group_id)
        if len(group_id) != n:
            raise ValueError("group_id length must match dates")
        frame = pd.DataFrame({"g": group_id, "rank": np.empty(n, dtype=int)})
        frame.loc[order, "rank"] = np.arange(n)
        first_rank = frame.groupby("g")["rank"].transform("min").to_numpy()
        first_split = split[order][first_rank]
        moved = int((first_split != split).sum())
        split = first_split.astype(object)
    return SplitResult(
        split=np.asarray(split, dtype=str),
        train_end=train_end,
        valid_end=valid_end,
        moved_for_dedup=moved,
    )


def assert_no_group_crosses(split: np.ndarray, group_id: np.ndarray) -> None:
    frame = pd.DataFrame({"g": group_id, "s": split})
    n_unique = frame.groupby("g")["s"].nunique()
    crossing = int((n_unique > 1).sum())
    if crossing:
        raise AssertionError(f"{crossing} duplicate groups span more than one split")

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from complaint_intelligence_lab.cfpb.fields import NARRATIVE
from complaint_intelligence_lab.pipeline.dedup import exact_hash, find_duplicates, jaccard, normalise, shingles
from complaint_intelligence_lab.pipeline.split import SPLITS, assert_no_group_crosses, chronological_split


def test_normalise_ignores_case_punctuation_and_spacing() -> None:
    assert normalise("Hello,  WORLD!") == normalise("hello world")
    assert exact_hash("A charge   on my statement.") == exact_hash("a charge on my statement")


def test_exact_duplicates_are_grouped() -> None:
    texts = ["I was charged a fee.", "i was charged a fee", "Something else entirely happened here."]
    result = find_duplicates(texts)
    assert result.group_id[0] == result.group_id[1]
    assert result.group_id[2] != result.group_id[0]
    assert result.exact_pairs == 1


def test_near_duplicates_are_grouped_by_shingle_overlap() -> None:
    base = "the servicer added an escrow shortage that was not explained and I asked for a review of the account"
    near = base + " please help"
    far = "my vehicle title was not released after the loan was paid off and nobody will answer"
    result = find_duplicates([base, near, far], k=5, threshold=0.8)
    assert result.group_id[0] == result.group_id[1]
    assert result.group_id[2] != result.group_id[0]
    assert result.near_pairs == 1


def test_distinct_texts_are_not_grouped() -> None:
    texts = [
        "a collector keeps contacting me about a debt that is not mine",
        "my mortgage payment was applied late even though it was sent before the due date",
        "i sent a transfer through the app and the recipient never received it",
    ]
    result = find_duplicates(texts)
    assert len(set(result.group_id)) == 3
    assert result.n_groups_with_duplicates == 0


def test_empty_narratives_are_never_grouped_together() -> None:
    result = find_duplicates(["", "   ", "", "real text about a fee"])
    assert result.n_groups_with_duplicates == 0


def test_jaccard_and_shingles_edge_cases() -> None:
    assert jaccard(frozenset(), frozenset()) == 0.0
    assert shingles("", k=5) == frozenset()
    assert shingles("two words", k=5) == frozenset({"two words"})
    with pytest.raises(ValueError):
        find_duplicates(["a"], threshold=0.0)


def test_chronological_split_train_before_valid_before_test() -> None:
    dates = pd.Series(pd.date_range("2024-01-01", periods=200, freq="D"))
    result = chronological_split(dates)
    s = result.split
    assert result.counts() == {"train": 120, "valid": 40, "test": 40}
    assert dates[s == "train"].max() < dates[s == "valid"].min()
    assert dates[s == "valid"].max() < dates[s == "test"].min()


def test_duplicate_groups_never_span_splits(small_frame: pd.DataFrame) -> None:
    dedup = find_duplicates(small_frame[NARRATIVE].tolist())
    result = chronological_split(small_frame["date_received"], dedup.group_id)
    assert_no_group_crosses(result.split, dedup.group_id)
    assert dedup.n_groups_with_duplicates > 0
    assert result.moved_for_dedup > 0


def test_singletons_keep_strict_chronology_after_dedup_moves(small_frame: pd.DataFrame) -> None:
    dedup = find_duplicates(small_frame[NARRATIVE].tolist())
    result = chronological_split(small_frame["date_received"], dedup.group_id)
    single = ~dedup.is_duplicate()
    dates = small_frame["date_received"].reset_index(drop=True)
    s = result.split
    assert dates[single & (s == "train")].max() <= dates[single & (s == "test")].min()
    # every row moved by the first-seen rule moved to an EARLIER split, never a later one
    order = {name: i for i, name in enumerate(SPLITS)}
    plain = chronological_split(small_frame["date_received"]).split
    moved = plain != s
    assert all(order[a] < order[b] for a, b in zip(s[moved], plain[moved]))


def test_assert_no_group_crosses_detects_a_leak() -> None:
    with pytest.raises(AssertionError, match="span"):
        assert_no_group_crosses(np.array(["train", "test"]), np.array([0, 0]))


def test_split_rejects_bad_fractions_and_tiny_inputs() -> None:
    dates = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))
    with pytest.raises(ValueError):
        chronological_split(dates, train_frac=0.8, valid_frac=0.3)
    with pytest.raises(ValueError):
        chronological_split(dates.head(5))


def test_split_manifest_hash_is_stable() -> None:
    dates = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    ids = pd.Series([f"SYN-{i}" for i in range(30)])
    a = chronological_split(dates).manifest_hash(ids)
    b = chronological_split(dates).manifest_hash(ids)
    assert a == b and len(a) == 64

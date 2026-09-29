from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from complaint_intelligence_lab.cfpb.fields import NARRATIVE
from complaint_intelligence_lab.pipeline.features import Featuriser, TfidfConfig
from complaint_intelligence_lab.pipeline.leakage import FeatureSpec, LeakageError, check_features
from complaint_intelligence_lab.pipeline.model import ComplaintClassifier, ModelConfig


def test_post_event_fields_are_refused() -> None:
    for field in ("company_response", "timely", "consumer_disputed"):
        with pytest.raises(LeakageError, match="post-event"):
            check_features("product", ["state", field])


def test_target_itself_is_refused() -> None:
    with pytest.raises(LeakageError, match="target itself"):
        check_features("product", ["product"])


def test_child_category_encoding_the_target_is_refused() -> None:
    with pytest.raises(LeakageError, match="encodes the target"):
        check_features("product", ["sub_product"])
    with pytest.raises(LeakageError, match="encodes the target"):
        check_features("issue", ["sub_issue"])


def test_allowed_intake_fields_pass_and_unknown_fields_fail() -> None:
    spec = check_features("product", ["state", "submitted_via", "company"])
    assert spec.structured == ("state", "submitted_via", "company")
    assert NARRATIVE in spec.all_columns()
    with pytest.raises(LeakageError, match="not an allowed intake field"):
        check_features("product", ["zip_code"])
    with pytest.raises(LeakageError):
        check_features("company", ["state"])


def _spec() -> FeatureSpec:
    return check_features("product", ["state", "submitted_via"])


def test_fitted_statistics_do_not_change_when_test_rows_are_perturbed(small_frame: pd.DataFrame) -> None:
    train = small_frame.iloc[:400]
    test = small_frame.iloc[400:].copy()
    feat = Featuriser(_spec(), TfidfConfig(max_features=800)).fit(train)
    vocab_before, idf_before = feat.vocabulary(), feat.idf().copy()
    test[NARRATIVE] = "completely new vocabulary zorblat quixotic " + test[NARRATIVE]
    test["state"] = "ZZ"
    X = feat.transform(test)
    assert X.shape[0] == len(test)
    assert feat.vocabulary() == vocab_before
    assert np.array_equal(feat.idf(), idf_before)
    assert "zorblat" not in feat.vocabulary()


def test_unknown_categories_are_ignored_not_errors(small_frame: pd.DataFrame) -> None:
    feat = Featuriser(_spec()).fit(small_frame.iloc[:300])
    row = small_frame.iloc[[300]].copy()
    row["state"] = "NOT-A-STATE"
    X = feat.transform(row)
    assert X.shape == (1, feat.n_features)


def test_classifier_predicts_known_classes_with_confidence(small_frame: pd.DataFrame) -> None:
    train, test = small_frame.iloc[:500], small_frame.iloc[500:]
    clf = ComplaintClassifier(_spec(), TfidfConfig(max_features=1500)).fit(train)
    pred = clf.predict(test)
    assert set(pred.labels).issubset(set(clf.classes))
    assert pred.proba.shape == (len(test), len(clf.classes))
    assert np.all((pred.confidence > 0) & (pred.confidence <= 1))
    assert np.allclose(pred.proba.sum(axis=1), 1.0)


def test_save_and_load_round_trip_gives_identical_predictions(small_frame: pd.DataFrame, tmp_path: Path) -> None:
    train, test = small_frame.iloc[:500], small_frame.iloc[500:]
    clf = ComplaintClassifier(_spec(), TfidfConfig(max_features=1500)).fit(train)
    clf.save(tmp_path / "model")
    assert not any(p.suffix in {".pkl", ".pickle", ".joblib"} for p in (tmp_path / "model").iterdir())
    loaded = ComplaintClassifier.load(tmp_path / "model")
    a, b = clf.predict(test), loaded.predict(test)
    assert np.array_equal(a.labels, b.labels)
    assert np.allclose(a.proba, b.proba, atol=1e-5)


def test_load_without_model_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train"):
        ComplaintClassifier.load(tmp_path)


def test_balanced_weighting_recovers_minority_class_recall(small_frame: pd.DataFrame) -> None:
    # Make one class very rare in training and check the balanced model still recalls it.
    rare = "Vehicle loan or lease"
    frame = small_frame.copy()
    rare_rows = frame[frame["product"] == rare]
    other = frame[frame["product"] != rare]
    train = pd.concat([other.iloc[:450], rare_rows.iloc[:6]])
    test = pd.concat([other.iloc[450:], rare_rows.iloc[6:]])
    balanced = ComplaintClassifier(_spec(), TfidfConfig(max_features=1500), ModelConfig(class_weight="balanced")).fit(train)
    plain = ComplaintClassifier(_spec(), TfidfConfig(max_features=1500), ModelConfig(class_weight=None)).fit(train)
    truth = test["product"].to_numpy()
    rec_b = (balanced.predict(test).labels[truth == rare] == rare).mean()
    rec_p = (plain.predict(test).labels[truth == rare] == rare).mean()
    assert rec_b > 0
    assert rec_b >= rec_p


def test_training_requires_two_classes(small_frame: pd.DataFrame) -> None:
    one = small_frame[small_frame["product"] == "Credit reporting"].iloc[:50]
    with pytest.raises(ValueError, match="two classes"):
        ComplaintClassifier(_spec()).fit(one)


def test_empty_narratives_still_get_a_prediction(small_frame: pd.DataFrame) -> None:
    clf = ComplaintClassifier(_spec(), TfidfConfig(max_features=1000)).fit(small_frame.iloc[:400])
    rows = small_frame.iloc[400:405].copy()
    rows[NARRATIVE] = ["", None, "   ", float("nan"), ""]
    pred = clf.predict(rows)
    assert len(pred.labels) == 5

"""Logistic-regression baseline with numpy/JSON persistence (no pickle)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from .features import Featuriser, TfidfConfig
from .leakage import FeatureSpec


@dataclass(frozen=True)
class ModelConfig:
    C: float = 1.0
    class_weight: str | None = "balanced"
    max_iter: int = 2000
    seed: int = 42


@dataclass(frozen=True)
class Prediction:
    labels: np.ndarray
    confidence: np.ndarray
    proba: np.ndarray
    classes: tuple[str, ...]


class ComplaintClassifier:
    def __init__(self, spec: FeatureSpec, tfidf: TfidfConfig = TfidfConfig(), model: ModelConfig = ModelConfig()) -> None:
        self.spec = spec
        self.featuriser = Featuriser(spec, tfidf)
        self.model_config = model
        self._clf: LogisticRegression | None = None

    def fit(self, train: pd.DataFrame) -> "ComplaintClassifier":
        y = train[self.spec.target].astype(str).to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError("training data must contain at least two classes")
        X = self.featuriser.fit(train).transform(train)
        self._clf = LogisticRegression(
            C=self.model_config.C,
            class_weight=self.model_config.class_weight,
            max_iter=self.model_config.max_iter,
            random_state=self.model_config.seed,
        )
        self._clf.fit(X, y)
        return self

    def _require(self) -> LogisticRegression:
        if self._clf is None:
            raise RuntimeError("classifier is not fitted; call fit() or load()")
        return self._clf

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self._require().classes_)

    def predict(self, frame: pd.DataFrame) -> Prediction:
        clf = self._require()
        X = self.featuriser.transform(frame)
        proba = clf.predict_proba(X)
        idx = proba.argmax(axis=1)
        return Prediction(
            labels=clf.classes_[idx].astype(str),
            confidence=proba.max(axis=1),
            proba=proba,
            classes=self.classes,
        )

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        return self.featuriser.transform(frame)

    # ---- persistence -------------------------------------------------------------
    def save(self, directory: Path) -> None:
        clf = self._require()
        directory.mkdir(parents=True, exist_ok=True)
        self.featuriser.save(directory)
        np.savez(
            directory / "classifier.npz",
            coef=clf.coef_.astype(np.float32),
            intercept=clf.intercept_.astype(np.float32),
        )
        meta = {"classes": self.classes, "model_config": self.model_config.__dict__}
        (directory / "classifier.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "ComplaintClassifier":
        if not (directory / "classifier.json").exists():
            raise FileNotFoundError(
                f"no saved model in {directory}; run `python -m complaint_intelligence_lab train` first"
            )
        featuriser = Featuriser.load(directory)
        meta = json.loads((directory / "classifier.json").read_text(encoding="utf-8"))
        obj = cls(featuriser.spec, featuriser.config, ModelConfig(**meta["model_config"]))
        obj.featuriser = featuriser
        arrays = np.load(directory / "classifier.npz")
        clf = LogisticRegression()
        clf.classes_ = np.asarray(meta["classes"], dtype=object)
        clf.coef_ = arrays["coef"].astype(np.float64)
        clf.intercept_ = arrays["intercept"].astype(np.float64)
        clf.n_features_in_ = clf.coef_.shape[1]
        obj._clf = clf
        return obj

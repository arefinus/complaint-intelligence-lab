"""TF-IDF narrative features plus one-hot structured features, fitted on train only.

The fitted vocabulary, idf weights and category lists are stored as plain JSON and numpy
arrays so that a saved model never requires pickle.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from ..cfpb.fields import NARRATIVE
from .leakage import FeatureSpec
from .sanitize import model_text


@dataclass(frozen=True)
class TfidfConfig:
    max_features: int = 5000
    ngram_max: int = 2
    min_df: int = 2
    max_narrative_chars: int = 5000


class Featuriser:
    """Fit on the training frame only; transform any frame with the same columns."""

    def __init__(self, spec: FeatureSpec, config: TfidfConfig = TfidfConfig()) -> None:
        self.spec = spec
        self.config = config
        self._tfidf: TfidfVectorizer | None = None
        self._onehot: OneHotEncoder | None = None

    def _texts(self, frame: pd.DataFrame) -> list[str]:
        if NARRATIVE not in frame.columns:
            return [""] * len(frame)
        return [model_text(v, self.config.max_narrative_chars) for v in frame[NARRATIVE]]

    def fit(self, frame: pd.DataFrame) -> "Featuriser":
        if self.spec.use_narrative:
            self._tfidf = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=(1, self.config.ngram_max),
                min_df=self.config.min_df,
                sublinear_tf=True,
                lowercase=True,
                dtype=np.float32,
            )
            self._tfidf.fit(self._texts(frame))
        if self.spec.structured:
            self._onehot = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
            self._onehot.fit(frame[list(self.spec.structured)].astype(str))
        return self

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        blocks: list[sparse.csr_matrix] = []
        if self._tfidf is not None:
            blocks.append(self._tfidf.transform(self._texts(frame)).tocsr())
        if self._onehot is not None:
            blocks.append(self._onehot.transform(frame[list(self.spec.structured)].astype(str)).tocsr())
        if not blocks:
            raise RuntimeError("featuriser is not fitted or has no feature blocks")
        return sparse.hstack(blocks, format="csr", dtype=np.float32)

    @property
    def n_features(self) -> int:
        n = 0
        if self._tfidf is not None:
            n += len(self._tfidf.vocabulary_)
        if self._onehot is not None:
            n += sum(len(c) for c in self._onehot.categories_)
        return n

    def vocabulary(self) -> dict[str, int]:
        if self._tfidf is None:
            return {}
        return {str(k): int(v) for k, v in self._tfidf.vocabulary_.items()}

    def idf(self) -> np.ndarray:
        if self._tfidf is None:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self._tfidf.idf_, dtype=np.float32)

    # ---- persistence -------------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "spec": {
                "target": self.spec.target,
                "structured": list(self.spec.structured),
                "use_narrative": self.spec.use_narrative,
            },
            "config": self.config.__dict__,
            "vocabulary": self.vocabulary(),
            "categories": [list(map(str, c)) for c in self._onehot.categories_] if self._onehot else [],
        }
        (directory / "featuriser.json").write_text(json.dumps(meta), encoding="utf-8")
        np.savez(directory / "featuriser.npz", idf=self.idf())

    @classmethod
    def load(cls, directory: Path) -> "Featuriser":
        meta = json.loads((directory / "featuriser.json").read_text(encoding="utf-8"))
        spec = FeatureSpec(
            target=meta["spec"]["target"],
            structured=tuple(meta["spec"]["structured"]),
            use_narrative=bool(meta["spec"]["use_narrative"]),
        )
        obj = cls(spec, TfidfConfig(**meta["config"]))
        idf = np.load(directory / "featuriser.npz")["idf"]
        if meta["vocabulary"]:
            vocab = {k: int(v) for k, v in meta["vocabulary"].items()}
            tfidf = TfidfVectorizer(
                vocabulary=vocab,
                ngram_range=(1, obj.config.ngram_max),
                sublinear_tf=True,
                lowercase=True,
                dtype=np.float32,
            )
            tfidf.fit(["placeholder"])  # builds the vectoriser around the fixed vocabulary
            tfidf.idf_ = idf.astype(np.float64)
            obj._tfidf = tfidf
        if meta["categories"]:
            onehot = OneHotEncoder(
                categories=[np.asarray(c, dtype=object) for c in meta["categories"]],
                handle_unknown="ignore",
                dtype=np.float32,
            )
            placeholder = pd.DataFrame(
                {col: [cats[0]] for col, cats in zip(spec.structured, meta["categories"])}
            )
            onehot.fit(placeholder)
            obj._onehot = onehot
        return obj

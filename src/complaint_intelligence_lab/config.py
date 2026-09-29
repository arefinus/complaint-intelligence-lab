"""Typed configuration loaded from YAML."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .pipeline.features import TfidfConfig
from .pipeline.model import ModelConfig
from .pipeline.trends import TrendConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class DedupConfig:
    shingle_k: int = 5
    jaccard_threshold: float = 0.8


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.6
    valid_frac: float = 0.2


@dataclass(frozen=True)
class AbstentionConfig:
    target_accuracy: float = 0.98
    min_coverage: float = 0.2


@dataclass(frozen=True)
class LabConfig:
    seed: int = 42
    target: str = "product"
    structured_features: tuple[str, ...] = ("state", "submitted_via", "company")
    use_narrative: bool = True
    fixture: str = "examples/fixtures/v1/complaints_synthetic.csv"
    output_dir: str = "examples/output"
    dedup: DedupConfig = field(default_factory=DedupConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    abstention: AbstentionConfig = field(default_factory=AbstentionConfig)
    trends: TrendConfig = field(default_factory=TrendConfig)
    landscape_max_points: int = 500

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def fixture_path(self) -> Path:
        p = Path(self.fixture)
        return p if p.is_absolute() else PACKAGE_ROOT / p

    def output_path(self) -> Path:
        p = Path(self.output_dir)
        return p if p.is_absolute() else PACKAGE_ROOT / p


def load_config(path: Path | None = None) -> LabConfig:
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model_raw = dict(raw.get("model", {}))
    model_raw.setdefault("seed", raw.get("seed", 42))
    return LabConfig(
        seed=int(raw.get("seed", 42)),
        target=str(raw.get("target", "product")),
        structured_features=tuple(raw.get("structured_features", ["state", "submitted_via", "company"])),
        use_narrative=bool(raw.get("use_narrative", True)),
        fixture=str(raw.get("fixture", LabConfig.fixture)),
        output_dir=str(raw.get("output_dir", LabConfig.output_dir)),
        dedup=DedupConfig(**raw.get("dedup", {})),
        split=SplitConfig(**raw.get("split", {})),
        tfidf=TfidfConfig(**raw.get("tfidf", {})),
        model=ModelConfig(**model_raw),
        abstention=AbstentionConfig(**raw.get("abstention", {})),
        trends=TrendConfig(**raw.get("trends", {})),
        landscape_max_points=int(raw.get("landscape", {}).get("max_points", 500)),
    )

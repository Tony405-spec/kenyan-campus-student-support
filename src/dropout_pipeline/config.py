"""Immutable, explicit run configuration."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class RunStage(StrEnum):
    """Named milestones reported by the CLI."""

    EXTRACT = "extract"
    VALIDATE = "validate"
    TRANSFORM = "transform"
    MODEL = "model"
    EVALUATE = "evaluate"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Paths and experiment settings; frozen to prevent accidental mutation."""

    data_path: Path = Path("data/raw/data.csv")
    output_dir: Path = Path("outputs")
    seed: int = 4201
    test_size: float = 0.20
    cv_folds: int = 5
    permutation_repeats: int = 5

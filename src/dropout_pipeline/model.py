"""Comparable model definitions and training-only cross-validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .config import PipelineConfig
from .transform import DatasetParts, make_preprocessor

LOGGER = logging.getLogger(__name__)


class Predictor(Protocol):
    """The fitted estimator interface used by the evaluation stage."""

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Predictor: ...


@dataclass(frozen=True)
class ModelResult:
    """Cross-validation summary and fitted final estimator."""

    name: str
    mean_macro_f1: float
    std_macro_f1: float
    mean_accuracy: float
    estimator: Pipeline


def model_factories(seed: int) -> dict[str, ClassifierMixin]:
    """Return modest, documented estimators on a common three-class task."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2_000, class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=2, random_state=seed
        ),
    }


def compare_models(
    parts: DatasetParts, config: PipelineConfig
) -> tuple[list[ModelResult], pd.DataFrame]:
    """Cross-validate on training rows only and fit candidates on those rows."""
    encoder = LabelEncoder().fit(parts.target)
    y = encoder.transform(parts.target)
    splitter = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    results: list[ModelResult] = []
    rows: list[dict[str, float | str]] = []
    for name, classifier in model_factories(config.seed).items():
        estimator = Pipeline([("preprocess", make_preprocessor(parts)), ("classifier", classifier)])
        scores = cross_validate(
            estimator,
            parts.features,
            y,
            cv=splitter,
            scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
            return_train_score=False,
            n_jobs=1,
        )
        final = Pipeline([("preprocess", make_preprocessor(parts)), ("classifier", classifier)])
        final.fit(parts.features, y)
        result = ModelResult(
            name,
            float(np.mean(scores["test_macro_f1"])),
            float(np.std(scores["test_macro_f1"], ddof=1)),
            float(np.mean(scores["test_accuracy"])),
            final,
        )
        results.append(result)
        rows.append(
            {
                "model": name,
                "cv_macro_f1_mean": result.mean_macro_f1,
                "cv_macro_f1_std": result.std_macro_f1,
                "cv_accuracy_mean": result.mean_accuracy,
            }
        )
        LOGGER.info("%s: CV macro-F1 %.3f ± %.3f", name, result.mean_macro_f1, result.std_macro_f1)
    return results, pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)


def encode_target(target: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode class labels in a stable, reversible order."""
    encoder = LabelEncoder()
    return encoder.fit_transform(target), encoder

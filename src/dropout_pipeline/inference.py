"""Validated model-artifact loading and transparent review-priority logic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

import joblib
import numpy as np
import pandas as pd

from .transform import split_features_target


class ProbabilisticClassifier(Protocol):
    """Minimum fitted-model interface required for an assessment."""

    classes_: np.ndarray

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class ReviewBand(StrEnum):
    """Demonstration-only prioritisation, not an institutional risk category."""

    ROUTINE = "Routine review"
    FOLLOW_UP = "Consider follow-up"
    PRIORITY = "Priority human review"


@dataclass(frozen=True, slots=True)
class Assessment:
    """Actual model output with provisional review thresholds."""

    predicted_outcome: str
    probabilities: dict[str, float]
    dropout_probability: float
    review_band: ReviewBand
    lower_threshold: float
    upper_threshold: float


def assign_review_band(
    dropout_probability: float, lower_threshold: float = 0.35, upper_threshold: float = 0.65
) -> ReviewBand:
    """Assign a provisional band from an estimated class probability."""
    if not 0 <= lower_threshold < upper_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= lower < upper <= 1")
    if not 0 <= dropout_probability <= 1:
        raise ValueError("Probability must be between zero and one")
    if dropout_probability >= upper_threshold:
        return ReviewBand.PRIORITY
    if dropout_probability >= lower_threshold:
        return ReviewBand.FOLLOW_UP
    return ReviewBand.ROUTINE


def load_model_artifact(path: Path) -> ProbabilisticClassifier:
    """Load a locally generated model artifact. Only load trusted project files."""
    if not path.is_file():
        raise FileNotFoundError(f"Trained model artifact not found: {path}")
    estimator = joblib.load(path)
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Saved estimator does not support probability estimates")
    return cast(ProbabilisticClassifier, estimator)


def assess_student(
    estimator: ProbabilisticClassifier,
    features: pd.DataFrame,
    feature_columns: list[str],
    class_names: list[str],
    lower_threshold: float = 0.35,
    upper_threshold: float = 0.65,
) -> Assessment:
    """Run the fitted sklearn pipeline after checking the exact feature contract."""
    if len(features) != 1:
        raise ValueError("Assessment expects exactly one profile row")
    if list(features.columns) != feature_columns:
        raise ValueError("Input feature names and order do not match the fitted pipeline")
    if features.isna().any().any():
        raise ValueError("Assessment inputs cannot contain missing values")
    if not all(pd.api.types.is_numeric_dtype(features[column]) for column in features.columns):
        raise ValueError("All model inputs must be numeric codes or measurements")
    probabilities = estimator.predict_proba(features)[0]
    predicted = int(estimator.predict(features)[0])
    model_classes = [int(value) for value in estimator.classes_]
    if len(probabilities) != len(model_classes) or len(class_names) != len(model_classes):
        raise ValueError("Model classes do not match the saved class-name metadata")
    name_by_class = dict(zip(model_classes, class_names, strict=True))
    probability_by_name = {
        name_by_class[class_id]: float(probability)
        for class_id, probability in zip(model_classes, probabilities, strict=True)
    }
    if "Dropout" not in probability_by_name:
        raise ValueError("Saved class metadata must include the Dropout class")
    dropout_probability = probability_by_name["Dropout"]
    band = assign_review_band(dropout_probability, lower_threshold, upper_threshold)
    return Assessment(
        predicted_outcome=name_by_class[predicted],
        probabilities=probability_by_name,
        dropout_probability=dropout_probability,
        review_band=band,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )


def make_synthetic_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a labelled synthetic form starting point from observed modes/medians."""
    parts = split_features_target(frame)
    values: dict[str, float | int] = {}
    for column in parts.features.columns:
        series = parts.features[column]
        if column in parts.categorical_columns:
            values[column] = int(series.mode(dropna=True).iloc[0])
        elif pd.api.types.is_integer_dtype(series):
            values[column] = int(round(float(series.median())))
        else:
            values[column] = float(series.median())
    return pd.DataFrame([values], columns=parts.features.columns)


def predict_dropout_probabilities(
    estimator: ProbabilisticClassifier,
    features: pd.DataFrame,
    feature_columns: list[str],
    class_names: list[str],
) -> np.ndarray:
    """Get actual model probabilities for rows respecting the saved feature contract."""
    if list(features.columns) != feature_columns:
        raise ValueError("Input feature names and order do not match the fitted pipeline")
    class_ids = [int(value) for value in estimator.classes_]
    try:
        dropout_id = class_names.index("Dropout")
        dropout_index = class_ids.index(dropout_id)
    except ValueError as exc:
        raise ValueError("Fitted estimator metadata does not contain the Dropout class") from exc
    return estimator.predict_proba(features)[:, dropout_index]

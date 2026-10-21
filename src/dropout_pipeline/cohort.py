"""Data-backed cohort summaries for the university overview and analysis pages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .inference import ProbabilisticClassifier, assign_review_band, predict_dropout_probabilities
from .transform import split_features_target


def outcome_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return observed target counts and within-filter proportions."""
    counts = frame["Target"].value_counts().rename_axis("Observed outcome").to_frame("Records")
    counts["Proportion"] = counts["Records"] / max(len(frame), 1)
    return counts.sort_index()


def make_age_band(frame: pd.DataFrame) -> pd.Series:
    """Group observed age-at-enrollment values for descriptive cohort analysis."""
    return pd.cut(
        frame["Age at enrollment"],
        bins=[0, 20, 25, 30, 40, float("inf")],
        labels=["20 or younger", "21–25", "26–30", "31–40", "Over 40"],
        right=True,
    )


def holdout_review_summary(
    frame: pd.DataFrame,
    estimator: ProbabilisticClassifier,
    feature_columns: list[str],
    class_names: list[str],
    seed: int,
    test_size: float,
    lower_threshold: float,
    upper_threshold: float,
) -> pd.DataFrame:
    """Recreate the held-out partition and score it using the trained artifact.

    The training CLI uses the same split seed and split proportion. These are
    de-identified Portuguese-cohort references, not Kenyan students.
    """
    parts = split_features_target(frame)
    encoded = LabelEncoder().fit_transform(parts.target)
    _, X_test, _, y_test = train_test_split(
        parts.features,
        encoded,
        test_size=test_size,
        stratify=encoded,
        random_state=seed,
    )
    probability = predict_dropout_probabilities(estimator, X_test, feature_columns, class_names)
    bands = [
        assign_review_band(float(item), lower_threshold, upper_threshold).value
        for item in probability
    ]
    return pd.DataFrame(
        {
            "observed_outcome": parts.target.loc[X_test.index].to_numpy(),
            "dropout_probability_estimate": probability,
            "provisional_review_band": bands,
        },
        index=X_test.index,
    )


def dataset_quality_summary(frame: pd.DataFrame, source_path: Path) -> dict[str, int | str]:
    """Compute structural facts from the currently loaded frame."""
    return {
        "source": source_path.name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "predictors": len(frame.columns) - 1,
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
    }

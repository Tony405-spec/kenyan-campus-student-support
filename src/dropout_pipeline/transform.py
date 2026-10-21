"""Feature splitting and scikit-learn preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class DatasetParts:
    """Predictors, observed outcome, and original feature names."""

    features: pd.DataFrame
    target: pd.Series
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


# Nominal codes include marital status, application mode, course, qualification,
# nationality, occupation/qualification codes, binary indicators, gender and debt.
CATEGORICAL_COLUMNS = (
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance\t",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
)


def split_features_target(frame: pd.DataFrame) -> DatasetParts:
    """Separate y and defensibly encode nominal integer codes as categories."""
    if "Target" not in frame.columns:
        raise ValueError("Target column is required")
    features = frame.drop(columns="Target").copy()
    categorical = tuple(column for column in CATEGORICAL_COLUMNS if column in features)
    numeric = tuple(str(column) for column in features if column not in categorical)
    return DatasetParts(features, frame["Target"].astype(str).copy(), numeric, categorical)


def make_preprocessor(parts: DatasetParts) -> ColumnTransformer:
    """Create fold-fitted scaling and one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(parts.numeric_columns)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(parts.categorical_columns),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

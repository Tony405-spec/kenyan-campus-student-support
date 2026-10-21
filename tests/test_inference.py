import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from dropout_pipeline.inference import (
    ReviewBand,
    assess_student,
    assign_review_band,
)


def fitted_classifier() -> tuple[LogisticRegression, pd.DataFrame]:
    features = pd.DataFrame({"academic_indicator": [0, 1, 2, 3, 4, 5]})
    target = np.array([0, 0, 1, 1, 2, 2])
    return LogisticRegression().fit(features, target), features.iloc[[2]].copy()


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.349, ReviewBand.ROUTINE),
        (0.35, ReviewBand.FOLLOW_UP),
        (0.649, ReviewBand.FOLLOW_UP),
        (0.65, ReviewBand.PRIORITY),
    ],
)
def test_provisional_review_band_boundaries(probability, expected) -> None:
    assert assign_review_band(probability) == expected


def test_review_band_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="Thresholds"):
        assign_review_band(0.5, 0.7, 0.4)


def test_assessment_uses_real_model_probabilities() -> None:
    estimator, row = fitted_classifier()
    result = assess_student(
        estimator, row, ["academic_indicator"], ["Dropout", "Enrolled", "Graduate"]
    )
    assert result.predicted_outcome in {"Dropout", "Enrolled", "Graduate"}
    assert result.dropout_probability == pytest.approx(float(estimator.predict_proba(row)[0, 0]))
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_assessment_checks_feature_contract() -> None:
    estimator, row = fitted_classifier()
    with pytest.raises(ValueError, match="feature names and order"):
        assess_student(estimator, row, ["other_feature"], ["Dropout", "Enrolled", "Graduate"])

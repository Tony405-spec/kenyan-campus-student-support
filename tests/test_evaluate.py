"""Metric report and outcome routing tests."""

from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from dropout_pipeline.config import PipelineConfig
from dropout_pipeline.evaluate import _interpret_outcome, evaluate_selected


def test_outcome_routes() -> None:
    assert _interpret_outcome("Enrolled") == "Still enrolled"
    assert _interpret_outcome("other") == "Unknown outcome"


def test_evaluation_writes_expected_outputs(tmp_path: Path) -> None:
    import pandas as pd

    X = pd.DataFrame({"signal": [0, 0, 1, 1, 2, 2]})
    y = np.array([0, 0, 1, 1, 2, 2])
    model = Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))]).fit(X, y)
    config = PipelineConfig(output_dir=tmp_path, permutation_repeats=2)
    metrics = evaluate_selected(model, X, pd.Series(y), ["Dropout", "Enrolled", "Graduate"], config)
    assert 0 <= metrics["test_macro_f1"] <= 1
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "confusion_matrix.png").is_file()

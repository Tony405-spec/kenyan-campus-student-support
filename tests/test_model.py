"""Model selection workflow tests."""

from pathlib import Path

from dropout_pipeline.config import PipelineConfig
from dropout_pipeline.model import compare_models
from dropout_pipeline.transform import split_features_target


def test_all_three_estimators_run_with_stratified_cv(sample_frame) -> None:
    repeated = sample_frame.loc[sample_frame.index.repeat(3)].reset_index(drop=True)
    parts = split_features_target(repeated)
    config = PipelineConfig(output_dir=Path("outputs"), cv_folds=3)
    results, comparison = compare_models(parts, config)
    assert {result.name for result in results} == {
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    }
    assert comparison["cv_macro_f1_mean"].between(0, 1).all()

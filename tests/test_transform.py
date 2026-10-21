"""Feature split and preprocessor tests."""

from dropout_pipeline.transform import make_preprocessor, split_features_target


def test_split_preserves_target_and_original_feature_names(sample_frame) -> None:
    parts = split_features_target(sample_frame)
    assert len(parts.features.columns) == 36
    assert len(parts.target) == 3
    assert "GDP" in parts.numeric_columns
    assert "Course" in parts.categorical_columns
    transformed = make_preprocessor(parts).fit_transform(parts.features)
    assert transformed.shape[0] == 3

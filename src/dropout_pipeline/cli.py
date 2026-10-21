"""Command-line entry point for the complete reproducible workflow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import PipelineConfig, RunStage
from .evaluate import evaluate_selected
from .extract import download_dataset, sha256_file
from .logging_setup import configure_logging
from .model import compare_models
from .transform import split_features_target
from .validate import load_and_validate

LOGGER = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig, download: bool = False) -> None:
    """Execute extraction, validation, model selection and final evaluation."""
    if download or not config.data_path.exists():
        data_path = download_dataset(config.data_path.parent)
    else:
        data_path = config.data_path
    LOGGER.info("Starting %s stage", RunStage.VALIDATE.value)
    frame = load_and_validate(data_path)
    LOGGER.info("Starting %s stage", RunStage.TRANSFORM.value)
    parts = split_features_target(frame)
    y_encoded, encoder = (
        LabelEncoder().fit_transform(parts.target),
        LabelEncoder().fit(parts.target),
    )
    X_train, X_test, y_train, y_test = train_test_split(
        parts.features,
        y_encoded,
        test_size=config.test_size,
        stratify=y_encoded,
        random_state=config.seed,
    )
    train_parts = type(parts)(
        X_train,
        pd.Series(y_train, index=X_train.index),
        parts.numeric_columns,
        parts.categorical_columns,
    )
    LOGGER.info("Starting %s stage", RunStage.MODEL.value)
    results, comparison = compare_models(train_parts, config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(config.output_dir / "model_comparison.csv", index=False)
    best = max(results, key=lambda result: result.mean_macro_f1)
    model_path = config.output_dir / "student_outcome_model.joblib"
    joblib.dump(best.estimator, model_path)
    LOGGER.info("Starting %s stage", RunStage.EVALUATE.value)
    metrics = evaluate_selected(
        best.estimator, X_test, pd.Series(y_test), list(encoder.classes_), config
    )
    cv = comparison.to_dict(orient="records")
    (config.output_dir / "cross_validation.json").write_text(
        json.dumps(cv, indent=2), encoding="utf-8"
    )
    (config.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "random_seed": config.seed,
                "test_size": config.test_size,
                "cv_folds": config.cv_folds,
                "permutation_repeats": config.permutation_repeats,
                "selected_model": best.name,
                "model_path": model_path.name,
                "feature_columns": parts.features.columns.tolist(),
                "class_names": encoder.classes_.tolist(),
                "source_path": str(data_path),
                "data_sha256": sha256_file(data_path),
                "rows": len(frame),
                "columns": len(frame.columns),
                "class_counts": frame["Target"].value_counts().to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = [
        f"Selected model: {best.name} (training CV macro-F1 "
        f"{best.mean_macro_f1:.3f} ± {best.std_macro_f1:.3f}).",
        f"Held-out test macro-F1: {metrics['test_macro_f1']:.3f}; "
        f"accuracy: {metrics['test_accuracy']:.3f}.",
        f"Held-out rows: {len(y_test)}; training rows: {len(y_train)}; seed: {config.seed}.",
        "These are predictions of recorded outcomes; they do not establish causes or "
        "future individual behaviour.",
    ]
    (config.output_dir / "results_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote verified outputs to %s", config.output_dir)


def main() -> None:
    """Parse CLI options and run the workflow."""
    parser = argparse.ArgumentParser(description="UCI 697 student outcome pipeline")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/data.csv"),
        help="Path to official semicolon CSV",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs"), help="Directory for results"
    )
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument(
        "--download", action="store_true", help="Download the official UCI archive first"
    )
    args = parser.parse_args()
    configure_logging()
    run_pipeline(
        PipelineConfig(data_path=args.data, output_dir=args.output, seed=args.seed), args.download
    )


if __name__ == "__main__":
    main()

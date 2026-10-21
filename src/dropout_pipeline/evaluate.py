"""Held-out metrics, permutation importance, and report-ready artifacts."""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


def _interpret_outcome(label: str) -> str:
    """Map observed outcome names to concise figure labels using pattern matching."""
    match label:
        case "Dropout":
            return "Dropout"
        case "Enrolled":
            return "Still enrolled"
        case "Graduate":
            return "Graduate"
        case _:
            return "Unknown outcome"


def evaluate_selected(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
    config: PipelineConfig,
) -> dict[str, object]:
    """Evaluate once on the held-out test set and write machine-readable outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = estimator.predict(X_test)
    truth = y_test.to_numpy()
    names = [_interpret_outcome(name) for name in class_names]
    report = classification_report(
        truth,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(truth, predictions, labels=list(range(len(class_names))))
    metrics: dict[str, object] = {
        "selected_model": estimator.named_steps["classifier"].__class__.__name__,
        "test_macro_f1": float(f1_score(truth, predictions, average="macro")),
        "test_accuracy": float(accuracy_score(truth, predictions)),
        "classification_report": report,
        "confusion_matrix_labels": class_names,
        "confusion_matrix": matrix.tolist(),
        "test_rows": int(len(y_test)),
        "random_seed": config.seed,
    }
    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(report).T.to_csv(
        config.output_dir / "classification_report.csv", index_label="class"
    )

    figure, axis = plt.subplots(figsize=(7.5, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Students")
    axis.set(
        xticks=range(len(names)),
        xticklabels=names,
        yticks=range(len(names)),
        yticklabels=names,
        xlabel="Predicted observed outcome",
        ylabel="Actual observed outcome",
        title="Held-out test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=20, ha="right")
    for row in range(len(names)):
        for column in range(len(names)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(config.output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    importance = permutation_importance(
        estimator,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=config.permutation_repeats,
        random_state=config.seed,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    table.to_csv(config.output_dir / "permutation_importance.csv", index=False)
    plot_data = table.head(12).sort_values("importance_mean")
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(
        plot_data["feature"],
        plot_data["importance_mean"],
        xerr=plot_data["importance_std"],
        color="#376795",
        capsize=2,
    )
    axis.set(
        xlabel="Decrease in macro-F1 after shuffling",
        title="Permutation importance (held-out test set)",
    )
    figure.tight_layout()
    figure.savefig(config.output_dir / "permutation_importance.png", dpi=180)
    plt.close(figure)
    LOGGER.info(
        "Held-out macro-F1 %.3f; accuracy %.3f", metrics["test_macro_f1"], metrics["test_accuracy"]
    )
    return metrics

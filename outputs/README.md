# Generated model artifacts

Run the data pipeline to create these local files. The artifacts are ignored by Git because they are reproducible outputs and may depend on a locally downloaded dataset.

```bash
python -m dropout_pipeline.cli --download
```

The command writes:

- `student_outcome_model.joblib`: fitted preprocessing and selected classifier used by the demo assessment. Load only artifacts generated locally from trusted project code.
- `run_config.json`: seed, split/fold settings, selected model, feature/class contract and dataset digest.
- `model_comparison.csv` and `cross_validation.json`: training-only stratified cross-validation summaries.
- `metrics.json` and `classification_report.csv`: held-out test metrics and per-class scores.
- `confusion_matrix.png` and `permutation_importance.png`: generated evaluation figures.
- `permutation_importance.csv` and `results_summary.txt`: feature permutation analysis and a short interpretation.
- `support_cases.json`: optional pseudonymous case-workflow demo data. It is not a secure student-record store; do not enter real personal information.

The report's plots and application screenshots are committed separately under `docs/report/screenshots/`. The original UCI source CSV remains under ignored `data/raw/` storage; see `data/raw/README.md` for provenance and setup.

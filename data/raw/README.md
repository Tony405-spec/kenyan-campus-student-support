# Dataset provenance

Source: UCI Machine Learning Repository, dataset ID 697, *Predict Students' Dropout and Academic Success*, https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success. Direct ZIP used: https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip. Licence: CC BY 4.0. Retrieved 2026-10-21.

Inspected archive member: `data.csv`. CSV encoding is UTF-8 with BOM; delimiter is semicolon. It contains 4,424 observations, 37 columns (36 predictors plus `Target`), all 37 columns unique. Target values are `Dropout` (1,421), `Enrolled` (794), and `Graduate` (2,209). Inspection found no missing/blank cells, no malformed row widths, no duplicate rows, and no duplicate columns. A source header has a trailing tab in `Daytime/evening attendance\t` and the source spelling `Nacionality`; both are preserved unchanged. All predictors were read as numeric values.

SHA-256 of extracted CSV: `3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae`. SHA-256 of downloaded ZIP: `e90e55fd65ec462ae283ebeb2cca409319e3460ed898d8754f62fb35cc83a65d`.

The available file is `data.csv`. It is ignored by Git to avoid committing the dataset; obtain it using `python -m dropout_pipeline.cli --download`.


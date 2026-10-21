# Project design

## Pipeline boundaries

`extract.py` fetches the official archive, uses a context manager for the HTTP response and ZIP handle, and writes the expected member. `validate.py` checks row width, parseability, unique schema, target values, missingness, and numeric input types. `transform.py` separates outcome from predictors and provides a scikit-learn `ColumnTransformer`. `model.py` owns model candidates and training-fold CV. `evaluate.py` owns held-out metrics and artifacts. `cli.py` connects these pieces and records the experiment configuration.

The implementation keeps raw column labels and the raw dataset. Validation reports duplicate observations rather than silently deleting them. The UCI file inspection found none.

## Assignment 3 extension ideas

1. Compare repeated stratified CV or nested cross-validation to describe model-selection uncertainty.
2. Add a calibrated probability review and class-specific decision thresholds, with explicit stakeholder costs.
3. Evaluate subgroup performance and fairness only when appropriate, ethically approved attributes and a clearly justified protocol are available.
4. Compare a gradient-boosted tree implementation or interpretable additive model without changing the established extraction/validation interfaces.
5. Establish a time-based evaluation if a documented chronological cohort becomes available.

Keep any extensions separate from the current held-out results; do not reuse this test set as a development set.


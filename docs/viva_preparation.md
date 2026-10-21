# Viva preparation

## Dataset and ETL

**Why UCI 697?** It is the official dataset named in the project brief, licensed CC BY 4.0, with 4,424 records and three observed outcomes.

**How did you inspect it?** Inspected ZIP membership, UTF-8 BOM, semicolon delimiter, 37 unique column names, row widths, dtypes, blanks, missingness, duplicates and label values. Preserved source header spellings.

**What is ETL here?** Extracting the UCI archive, validating schema and rows, then splitting features/target and producing fold-fitted sklearn transformations before modelling.

## Python and software engineering

**Why a context manager?** It guarantees a resource such as a response or file is closed when a block exits, including on errors.

**Why dataclasses and DataFrames?** Dataclasses make small configuration/result records explicit; DataFrames are still the right representation for the rectangular student table and sklearn workflow.

**Why a Protocol?** It defines the methods a replaceable validator/predictor must offer without forcing inheritance.

**What does the generator do?** It yields nonblank input lines and their positions one at a time, so validation need not store a second copy of the file.

**Why retry?** Network failures can be transient; bounded retries improve robustness without risking an endless loop. `wraps` preserves the wrapped function's metadata.

## Modelling and interpretation

**Why three classes?** The target records Dropout, Enrolled and Graduate; reducing it would discard a meaningful observed category.

**Why macro-F1?** It computes F1 per class and averages equally, so the smaller Enrolled class matters as much as Graduate. Accuracy alone is dominated more by common classes.

**How was leakage prevented?** The split happened before comparison; scaling and category discovery are inside the model pipeline so CV folds fit them only on training folds. The held-out test partition was used once for final evaluation and importance analysis.

**What is the concern with semester features?** They may only be observed near the time the final outcome is recorded. The result is retrospective outcome prediction, not validated early intervention forecasting.

**What does importance mean?** The decrease in held-out macro-F1 after permuting one input for the selected model. Correlation affects rankings; this is predictive reliance, not causality.

## Results and limits

**Which model won?** Logistic Regression by training CV macro-F1: 0.705 ± 0.030; held-out macro-F1 0.716 and accuracy 0.756.

**Which class was hardest?** Enrolled: precision 0.445, recall 0.642, F1 0.526. It is also the smallest class. Dropout recall was 0.704 and Graduate recall 0.830.

**Can a university act on it?** Not directly. This dataset and evaluation do not establish causal factors, fairness across groups, temporal generalisation, intervention benefit, or acceptable decision thresholds. Any real use would need governance, privacy controls, subgroup validation, human review and prospective evaluation.


# Methodology and verified data inspection

## Data and target

The source is UCI dataset 697, *Predict Students' Dropout and Academic Success*, retrieved 2026-10-21 under CC BY 4.0. The ZIP contains `data.csv`, UTF-8 with BOM, semicolon-delimited. Observed shape is 4,424 rows by 37 columns. There are 36 numeric predictors and the `Target` outcome. Original column names, including a trailing tab on `Daytime/evening attendance\t` and source spelling `Nacionality`, are retained. No missing values, duplicate rows/columns, empty records, or malformed row widths were found. Class counts: Graduate 2,209 (49.9%), Dropout 1,421 (32.1%), Enrolled 794 (17.9%).

## Feature treatment

Course, application, qualification, occupation, nationality, binary and other discrete codes are nominal categories and are one-hot encoded. Counts (curricular-unit counts), age, grades and macroeconomic measures are numeric and standardized for the linear baseline; tree learners also use that same transform for a consistent workflow (it does not affect tree ordering). Encoders and scalers are inside each sklearn Pipeline, so each CV fold learns preprocessing from only that fold's training rows. `handle_unknown="ignore"` accommodates categories not seen during fitting.

## Design and evaluation

Seed 4201 and stratified 80/20 train/test split: 3,539 training and 885 held-out test observations. Candidate models: balanced Logistic Regression (max_iter=2,000); balanced Random Forest (300 trees, minimum leaf 2); and Gradient Boosting (120 estimators, learning rate .08, max depth 2). Stratified five-fold CV compares candidates using macro-F1 as primary and accuracy as secondary. The candidate with highest mean training CV macro-F1 is fitted on all training rows and evaluated once on the held-out test set. The selected Logistic Regression achieved CV macro-F1 0.705 ± 0.030 and test macro-F1 0.716, accuracy 0.756. The CV mean accuracy was 0.746. Full values are in `outputs/`.

## Limitations

Semester academic records are powerful predictors but may encode progress observed close to the outcome. This is not an early-warning timing design. A single random split does not measure transfer to another institution, cohort, or time period. Class-specific metrics show weaker performance for Enrolled. Permutation importance measures this fitted model's score dependence under feature shuffling, not causal effect; correlated features can share or mask importance. No subgroup fairness analysis or deployment recommendation is made. Outputs should not be used to make consequential decisions about individuals.


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

## Application architecture and decision support

**What does the Streamlit app contain?** Eleven views: overview, individual assessment, cohort analysis, support cases, follow-up monitoring, model transparency, data administration, student support home, support chat, study reset and support directory. The wellbeing support and academic outcome model are separate; chat never receives a model prediction.

**Is the chatbot generative AI?** No. It applies ordered regular-expression patterns and returns prepared response text with optional resource links. It has specific Kenyan campus examples (CATs, HELB, upkeep, hostel, anxiety and loneliness) but only limited phrase coverage; it may miss paraphrases and context.

**How does individual assessment run?** The page starts from a synthetic row made from dataset medians and categorical modes. After feature validation it invokes the saved fitted preprocessing/classifier Pipeline and displays class probability estimates for the dataset's recorded labels. The starting row is not an actual person or Kenyan student.

**What is a review band?** It is a transparent rule over the model's estimated dropout probability: routine, consider follow-up or priority human review. The thresholds are adjustable in the interface, provisional, not selected through stakeholder analysis and not calibrated. A band is not a predicted fact, diagnosis or action; a person must review the context.

**How are cases handled?** A staff member can create or update a pseudonymous record with a category, next action, responsible role, status and due date. Atomic JSON replacement reduces the risk of a partially written file, but it does not protect confidentiality. There is no login, encryption, role-based access, audit trail or notification mechanism. The prototype must not store real student data.

**What does follow-up monitoring mean?** It counts open, due, overdue and closed records based on their dates and statuses. These are workflow indicators only; they do not measure whether a support intervention worked.

**What Kenyan resources are linked?** The app points to emergency numbers 112/999 and online/provider references for LVCT Health, Kenya Red Cross Chat Care and HELB. It does not assert an unverified KCA counselling number; students are directed to their own Student Affairs, Welfare, Dean of Students or department for the current local referral. Provider information should be checked against current official pages.

## Testing and safe interpretation

**What is covered by automated tests?** The 34-test suite covers the existing ETL/modelling modules, chatbot routing, threshold boundaries, saved-model inference contract, case JSON create/update/due behavior, and Streamlit AppTest routes. UI tests are not a substitute for actual student usability evaluation or clinical review.

**Why use a local joblib artifact?** The trained Pipeline preserves the fitted encoding/scaling and estimator together, reducing train/serve mismatch. Joblib deserialization can execute code, so the app only loads the locally generated trusted artifact; do not load arbitrary files.

**Can the threshold be treated as a clinical cutoff?** No. Neither the model probabilities nor bands are calibrated and no acceptable threshold has been set with students, advisers, counsellors or institutional leaders. Prospective evaluation and documented error costs are prerequisites even for a limited pilot.

**How should this become Kenyan-campus appropriate?** Co-design examples, language and referrals with Kenyan students and qualified support staff, then obtain appropriate approval and representative local data. Validate subgroup performance, calibration, privacy, accessibility, time ordering and intervention benefit before any operational use. Keep a human accountable for follow-up and crisis escalation.


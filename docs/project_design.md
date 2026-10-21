# Project design and boundaries

## Components

The Python package has two related but separate paths. The modelling path in `extract.py`, `validate.py`, `transform.py`, `model.py`, `evaluate.py` and `cli.py` downloads/reads the UCI file, validates its schema, preprocesses within a scikit-learn pipeline, compares three classifiers and writes fitted model and evaluation artifacts. The Streamlit app in `support_app.py` reads those artifacts for a transparent overview, cohort summaries and synthetic-profile assessment.

The campus support path keeps the student-facing wellbeing interaction independent of the outcome model. `support_chatbot.py` maps a limited set of phrases to prepared responses and human signposting. The study planner and check-in are session-only. The directory lists general referral guidance and online resources; campus referral information must be confirmed locally.

The case workflow is a separate, local demonstration. `case_management.py` validates pseudonymous references and dates, writes JSON atomically, and supports update and due-date calculations. The UI never creates a case from model output without a human action. This local JSON store has no security controls and is not suitable for actual student records.

## Data and validation

The model uses the public UCI student-outcome data from a Portuguese institution. Its metrics are not Kenyan campus performance estimates. Current results use one stratified random holdout and cross-validation for model selection. Some semester features may be recorded too late for intervention. The support examples are localized in wording and referral direction, but they are not evidence that a campus has adopted those specific services.

The synthetic assessment row combines observed numeric medians and categorical modes only to demonstrate the fitted inference contract. The review-band thresholds are adjustable examples; their probabilities are uncalibrated and their decision costs have not been established. All displayed predictions require human interpretation, and no eligibility, funding, disciplinary or clinical decision should be automated from them.

## Security, privacy and operations

The prototype has no authentication, authorization, encryption, secure database, audit events, retention policy, crisis queue or notifications. Chat state is session-only. Case files are local JSON and must contain no identifying or sensitive information. The emergency instructions direct users to live emergency or human services; the app itself is not monitored.

## Next engineering and research steps

1. Co-design language, examples, accessibility and escalation wording with Kenyan students, counsellors and university student-affairs staff.
2. Obtain institution-approved, representative Kenyan data and establish purpose limitation, consent, access control, retention and security review before any risk analytics.
3. Use time-aware validation and prospective evaluation; document class-specific error costs and calibration before considering operational thresholds.
4. Evaluate subgroup performance and potential harms using an ethically approved protocol; assess whether interventions improve outcomes rather than merely predicting records.
5. If case management is needed beyond a demo, replace local JSON with a secured, audited, role-restricted service and define human ownership and crisis escalation responsibilities.
6. Expand Kiswahili and Sheng coverage with user review and careful safety testing; regex matching alone cannot reliably interpret context.

These are future requirements, not features claimed to be implemented in this coursework prototype.

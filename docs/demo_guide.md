# Karibu Campus: Setup and Demonstration Guide

This guide covers the current Streamlit application and the separate academic-outcome modelling pipeline. The app combines a Kenyan campus student-support hub with a reproducible, retrospective model prototype. It is coursework software: chat replies are deterministic, risk bands are provisional, and no clinical, operational or Kenyan predictive validity has been established.

## 1. Prerequisites

- Python 3.11 or newer and Git, or a GitHub account with Codespaces access.
- Internet access for package installation and, if needed, downloading the UCI dataset.
- A modern web browser.
- For the full modelling demonstration, allow several minutes for cross-validation. The support pages work without model artifacts, but assessment and analytics require a local validated dataset and trained artifacts.

## 2. Start in GitHub Codespaces

1. Open [the project repository](https://github.com/Tony405-spec/kenyan-campus-student-support).
2. Select **Code → Codespaces → Create codespace on main** and wait for VS Code in the browser to finish loading.
3. Open **Terminal → New Terminal** and install the app and project dependencies:

   ```bash
   python -m pip install -e ".[dev,report,app]"
   ```

4. Start the application on the Codespaces network interface:

   ```bash
   python -m streamlit run src/dropout_pipeline/support_app.py --server.address 0.0.0.0 --server.port 8501
   ```

5. In the **Ports** panel, open the forwarded address for port `8501`. If it is absent, add port `8501`. Leave it private for a lecturer demonstration unless there is a reason to make it public.

6. To prepare model-backed pages, open a second terminal and run:

   ```bash
   python -m dropout_pipeline.cli --download
   ```

   Wait for the run to complete, then refresh the app. The command downloads the UCI dataset and creates local artifacts in `outputs/`. It does not create Kenyan student records.

## 3. Local setup

From the folder containing `pyproject.toml`:

```powershell
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,report,app]"
python -m dropout_pipeline.cli --download
python -m streamlit run src/dropout_pipeline/support_app.py
```

Open the URL shown in the terminal, usually `http://localhost:8501`. The virtual environment is recommended; do not commit `.venv`, downloaded data or generated artifacts.

If a local CSV already exists at `data/raw/data.csv`, replace the download step with `python -m dropout_pipeline.cli --data data/raw/data.csv`.

## 4. Application tour

The left navigation contains eleven views. Pages are designed to work without credentials; do not enter real student identifiers or sensitive disclosures. Assessment and outcome analysis display clear notices about dataset limitations.

### A. Home

Show the project overview, actual dataset/outcome counts and the actual saved model evaluation artifacts when they exist. Use the cohort filters to demonstrate the descriptive view. Point out the domain-shift warning: the source is a Portuguese dataset, not Kenyan university data. Any review bands on the dashboard are configurable prototypes, not validated interventions.

### B. Individual Assessment

The page creates an explicitly synthetic starting profile from the observed dataset's median numeric values and most common categorical values. Adjust the provided feature controls, then select **Run model assessment**. The persisted scikit-learn Pipeline produces its three-class outcome probabilities. The separate review band uses adjustable provisional probability thresholds and results in a human-review prompt. Explain that this is a demonstration of inference mechanics, not a Kenyan student's true risk, a clinical assessment or an automated decision. Do not enter real personal student data.

### C. Cohort Analysis

Show aggregate counts and outcome proportions for the selected groups. The page includes age-band and recorded academic/financial indicator summaries. Explain that these are descriptions of the supplied historical dataset. They do not establish that an indicator causes withdrawal or that a pattern transfers to Kenya.

### D. Support Cases

This local case register demonstrates a human follow-up workflow with pseudonymous references, category, owner role, next action, due date, status and notes. Create a fictional example such as `CASE-DEMO-01`, select an academic or fee-support category, and save it. Use the update view to change its status or follow-up fields. The records are stored as JSON under `outputs/support_cases.json` (or the path in `STUDENT_SUPPORT_CASES_PATH`). This prototype has no login, encryption, access controls or audit trail; never store real student cases in it.

### E. Follow-up Monitoring

Show the current counts of open, due and overdue local cases and the due-case table. The date logic uses the case follow-up date and excludes closed cases. Explain that this is a workflow demonstration, not a notification service; it does not send SMS, email or alerts.

### F. Model Transparency

Show the actual saved run configuration, cross-validation comparison, held-out metrics, per-class report, confusion matrix and permutation-importance output when present. If the model has not been trained, the page says which command creates the artifacts. Explain that a held-out score is not clinical validation, causal evidence, fairness certification or evidence of usefulness for Kenyan students.

### G. Data & Administration

Show the validator's schema and dataset-quality summary. The page can invoke the local pipeline to download/validate/train and regenerate artifacts. Describe the source dataset's class labels, data provenance and limitations. The model file is loaded as a local trusted artifact; do not use joblib files from untrusted sources.

### H. Student Support Home

This independent wellbeing hub offers a session-only optional check-in and links to chat, study planning and support resources. The check-in is not a clinical score and is not persisted to the case register.

### I. Support Chat

Use the quick-start buttons or type one example. The intent label and response are visible in the interface:

| Sample input | Intended route | Expected response direction |
|---|---|---|
| `I have two CATs next week and haven't started revision` | Academic/exam pressure | Break revision into a small topic and study block; seek lecturer, adviser or peer help if needed. |
| `HELB has not come through and I am worried about fees` | Financial pressure | Suggest Student Affairs/finance and official HELB routes; protect credentials. |
| `Nina wasiwasi kuhusu exams` | Anxiety/worry | Offer a brief grounding step and human support if worry keeps affecting daily life. |
| `Niko solo in the hostel` | Loneliness/isolation | Suggest one manageable connection, such as a classmate, society or study group. |
| `I feel hopeless and empty` | Low mood | Respond without diagnosing and encourage reaching a trusted person or qualified provider. |
| `I want to hurt myself tonight` | Immediate safety | Encourage immediate contact with emergency services (112/999), a nearby health facility and a trusted person who can stay. |

These routes are deterministic pattern matching over a limited set of phrases. A missed match is possible. The app does not monitor messages; in immediate danger do not wait for a reply from software.

### J. Study Reset

Enter a specific task (for example, “revise linked lists”), choose a 15, 25, 40 or 60 minute block and select **Create my small-step plan**. Show the focus, break and next step, plus the optional grounding routine. It is a planner and wellbeing prompt, not a running timer or saved calendar event.

### K. Support Directory

Show the emergency information, counselling/wellbeing, funding and online-support tabs. The app gives a generic route through Student Affairs, Student Welfare, Dean of Students or the department because institution-specific referral details have not been supplied. It also links to official HELB information and national providers; confirm provider details at their official sources before relying on them. The prototype is not monitored or an emergency service.

## 5. Five-minute demonstration script

**0:00–0:35 — Problem and scope.** “Karibu Campus combines a practical, Kenyan-context student-support interface with an independently evaluated academic-outcome pipeline. The source records are from Portugal, so the model is not a Kenyan risk tool.”

**0:35–1:10 — Home and cohort.** Show observed data counts and the domain-shift notice. Use one filter and explain the figures are descriptive. Avoid implying that group associations are causes.

**1:10–2:00 — Individual assessment.** Adjust one or two fields on the clearly synthetic profile, run inference, and show the three outcome probabilities and provisional human-review band. Explain the actual saved sklearn Pipeline produces the output; thresholds are not validated.

**2:00–2:40 — Cases and monitoring.** Create fictional case `CASE-DEMO-01`, show its local save, then show its due status in Follow-up Monitoring. Explain no real personal data should be stored and no alerts are sent.

**2:40–3:40 — Student support hub.** Show the chat using the CAT example and one financial or Kiswahili anxiety example. Identify the intent, response and privacy note. Show Study Reset and Support Directory. Clarify deterministic matching and human referral.

**3:40–4:25 — Model transparency and data quality.** Show actual held-out metrics and confusion matrix, then describe the schema checks and model comparison. Distinguish retrospective prediction from early intervention and wellbeing assessment.

**4:25–5:00 — Engineering and limitations.** Mention the modular ETL/ML package, real persisted model pipeline, eleven app views, JSON case-workflow prototype, and automated unit/UI tests. State the main next step: ethically governed Kenyan data, prospective validation, privacy controls and consultation with qualified campus services.

## 6. Testing and report

Run the checks from the repository root:

```bash
python -m pytest
ruff check .
ruff format --check .
mypy src/
```

The tests cover validators, pipeline behavior, threshold boundaries, inference contract, case persistence and Streamlit pages. Test success is software verification only; it does not validate the app's advice or model for Kenyan students. The 20-page report is `docs/report/Advanced_Python_Dropout_Prediction_Report.pdf`; its editable source and real UI screenshots are alongside it.

## 7. Troubleshooting

| Symptom | Resolution |
|---|---|
| `No module named streamlit` | Activate the project environment and run `python -m pip install -e ".[dev,report,app]"`. |
| Codespaces page does not open | Confirm the terminal still runs Streamlit, then open forwarded port `8501` from the **Ports** panel. Use the forwarded URL rather than your computer's localhost. |
| Port 8501 is busy | Stop the other app or use `--server.port 8502`, then forward `8502`. |
| Assessment/model pages report missing artifacts | Run `python -m dropout_pipeline.cli --download` and wait until it finishes. Refresh the page. |
| Data download fails | Check network access; if the official CSV is already present, run `python -m dropout_pipeline.cli --data data/raw/data.csv`. |
| Chat returns general support | Try the table's exact phrases. This small pattern matcher does not understand all paraphrases, spelling errors or context. |
| Charts or report fail on import | Reinstall dependencies with the documented extras in the active environment. Report generation requires `.[report]`; chart pages require `.[app]`. |
| Case write fails | Check that `outputs/` is writable and the configured `STUDENT_SUPPORT_CASES_PATH` points to a writable file location. |

## 8. Stop the application

In the terminal running Streamlit, press **Ctrl+C**. When using Codespaces, stop the Codespace from its menu or the GitHub Codespaces page when finished.

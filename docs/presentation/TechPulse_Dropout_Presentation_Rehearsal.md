# Presentation rehearsal and live-demo guide

**Project:** Predicting University Dropout Using an Advanced Python ETL and Machine Learning Pipeline  
**Course:** CPP 4201 — Advanced Python Programming, KCA University  
**Target delivery:** 8:45–9:15 minutes, including a short live demonstration

> The app is a prototype. The predictive model uses the UCI Portuguese cohort and is not validated on Kenyan university students. Its chatbot is deterministic intent routing with prepared replies, not a generative AI model. Never enter real student or sensitive personal information during a demonstration.

## 1. Prerequisites and setup

### Prerequisites

- Python 3.11 or newer.
- A terminal opened at the cloned repository root.
- Internet access for the first UCI dataset download, unless `data/raw/data.csv` is already present.
- A modern browser. In GitHub Codespaces, use the forwarded application port.
- Allow several minutes for the first dataset preparation and model evaluation.

### Local setup

From the project root:

```bash
python --version
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux / Codespaces
source .venv/bin/activate
```

Install the project and its app, report, and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,report,app]"
```

Prepare the official UCI dataset, validate it, compare models, fit the selected model, and generate evaluation artifacts:

```bash
python -m dropout_pipeline.cli --download
```

If the validated CSV already exists, use it without downloading:

```bash
python -m dropout_pipeline.cli --data data/raw/data.csv
```

Wait for the CLI to finish. The app’s modelling views need the saved files under `outputs/`. The wellbeing chat, study reset and support directory can run without model artifacts.

Start Streamlit:

```bash
python -m streamlit run src/dropout_pipeline/support_app.py
```

Open the local URL printed in the terminal, normally `http://localhost:8501`.

### GitHub Codespaces setup

Run the same installation and data-preparation commands in a Codespaces terminal. Start the app so it listens on the forwarded interface:

```bash
python -m streamlit run src/dropout_pipeline/support_app.py --server.address 0.0.0.0 --server.port 8501
```

Open the **Ports** panel in Codespaces and open forwarded port `8501` in the browser. If port forwarding is not visible, use the Ports panel to forward it. Keep the terminal running while presenting.

## 2. Feature navigation and demonstration checklist

1. **University Overview:** open the landing view. Point out 4,424 observed records, the three source outcomes, and the explicit notice that the model comes from Portugal and is not validated for Kenya.
2. **Model Transparency:** open from the left navigation. Show the selected Logistic Regression, held-out macro-F1 and accuracy, and the per-class table. Explain that the test partition has 885 Portuguese records and that `Enrolled` performance is weaker than the aggregate alone suggests.
3. **Support Chat:** open the intent-based chatbot. Use a prepared input from the table below; do not enter identifying details. Describe the matched label and prepared next step. The interface states that it does not use external generative AI and the conversation is session-only.
4. **Study Reset:** enter a fictional unit and one small task, keep the suggested 25-minute sprint, then create the plan. Explain that it stays in the session and is not saved to the model.
5. **Support Directory:** show the campus referral guidance and current provider links. Explain that a student should check availability and confidentiality with the provider; the app is not monitored.
6. **Support Cases:** use **Create case** with a pseudonymous fictional example only (for example, reference `CASE-DEMO-01`). Show the categories, agreed next action, responsible role and follow-up date. State that the register is local JSON without authentication or production security.
7. **Follow-up Monitoring:** open the follow-up view and show how the human-owned status/date filters help staff monitor a fictional case.
8. **Individual Assessment (optional):** if model artifacts are ready and time permits, run the synthetic profile. Point out its synthetic label and say clearly that the score is not a Kenyan student prediction.

Before the presentation, test the navigation and use a fictional case. The case demo is saved locally to `outputs/support_cases.json`; after the demonstration, remove that fictional record through the app or delete the local file if you do not need it. Do not commit local case or model outputs.

## 3. Chatbot demonstration inputs and expected behavior

The chatbot uses explicit regular-expression patterns and prepared replies. Phrase matching is not general conversation understanding. The listed examples are covered in the project’s tests where noted.

| Type this sample (fictional) | Expected matched intent | Expected response direction |
|---|---|---|
| `I have two CATs next week and haven't started revision` | Academic or exam pressure | Suggests one unit/topic, a 25-minute study sprint and a short break; recommends an academic adviser/department if deadlines or missed classes feel unmanageable. This is the captured slide example. |
| `HELB has not come through and I'm worried about fees` | Financial pressure | Suggests contacting campus finance/student affairs/department about current options; advises checking HELB/HEF only through the official portal and not sharing login or M-Pesa credentials. |
| `Nina wasiwasi about exams` | Anxiety or worry | Offers a brief grounding step and one small action; suggests a human support professional if worry continues disrupting sleep or study. |
| `Niko solo in the hostel` | Loneliness or isolation | Suggests a low-pressure connection such as a classmate, study group or student activity and offers campus referral signposting. |
| `I feel hopeless and empty` | Low mood or depression | Responds supportively, does not diagnose, and encourages speaking with campus welfare or a qualified professional; immediate safety options are available if needed. |
| `I want to hurt myself tonight` | Immediate safety | Shows urgent human-help signposting. Explain that the tool is not monitored and is not an emergency service. For a real crisis, use current local emergency/health services and a trusted person; do not wait for an app response. |

Try the CAT example first to show a predictable, useful reply. Crisis language is included in the test coverage, but it is usually better to explain the safety path rather than submit sensitive or alarming text in a classroom demonstration.

## 4. Timed speaking script (about 9 minutes)

| Time | Slide | Speaking cue |
|---|---:|---|
| 0:00–0:25 | 1 | Introduce the project, course, institution and team. Say it joins an ETL pipeline, machine learning and a campus-support prototype. |
| 0:25–1:10 | 2 | Explain student progression concerns and the proposed Kenyan adviser workflow. Emphasise that a risk score is not a decision, and the model is not locally validated. |
| 1:10–1:55 | 3 | Trace extraction → validation → transformation → training/evaluation → Streamlit → human support. State that the chat does not call the model. |
| 1:55–2:35 | 4 | Explain the semicolon CSV contract, 37 columns, target validation, typed findings and specific exceptions. |
| 2:35–3:15 | 5 | Explain feature/target separation, one-hot treatment of nominal codes, scaling numeric fields and preprocessing inside CV folds. Clarify that sklearn Pipeline is a library abstraction. |
| 3:15–3:55 | 6 | Explain the three target classes and the synthetic assessment. The model outputs class probabilities and a provisional review band; a person remains responsible for any follow-up. |
| 3:55–4:45 | 7 | Give the held-out 885-row metrics, then discuss class-level performance. Mention Portugal, model limits, possible leakage from late-semester features, and the need for Kenyan data and validation. |
| 4:45–5:30 | 8 | Use the CAT chatbot screenshot to show a localized intent and a prepared reply. Explain regex routing, explicit intent labels, no remote generative AI, session-only chat, and separate human-owned case tracking. |
| 5:30–6:25 | 9 | Select three Python examples to explain in detail: Protocol, decorator/ParamSpec and generator. Mention dataclasses/enums, context managers and match/case as additional evidence. |
| 6:25–6:55 | 10 | Summarise technical lessons and what would be needed for responsible Kenyan use. Transition to the live demo. |
| 6:55–8:45 | Live demo | Spend about 20 seconds on overview/transparency, 35 seconds on the CAT chat sample, 20 seconds on Study Reset/Directory, 25 seconds on a fictional case/follow-up, and use remaining time for a question or synthetic assessment if prepared. |

Speak to the slide rather than reading it. If the lecturer wants a short 7-minute version, skip the synthetic model demo and show only the CAT chat response plus one sentence about the case workflow.

## 5. Live-demo run sheet

### Before class

- [ ] Confirm Python is 3.11+ and the virtual environment is active.
- [ ] Install `.[dev,report,app]` and confirm the CLI can import the project.
- [ ] Run the CLI and confirm `outputs/` contains model and evaluation artifacts.
- [ ] Start the Streamlit app, open port 8501, and click the key navigation items.
- [ ] Check browser zoom and presentation screen resolution.
- [ ] Keep the CAT sample ready to paste; use no actual student details.
- [ ] If demonstrating case entry, use only `CASE-DEMO-01` and fictional values.
- [ ] Keep the deck and this guide available as a fallback if live app access fails.

### During class

1. Show **University Overview**, noting the cohort and domain caveat.
2. Show **Model Transparency**, including macro-F1, accuracy and per-class table.
3. Open **Support Chat** and submit the CAT sample. Point out its matched intent and suggested referral.
4. Show **Study Reset** and **Support Directory** briefly.
5. If time allows, create or browse the fictional support case, then show **Follow-up Monitoring**.
6. If model artifacts and time permit, open **Individual Assessment** and clearly identify the profile as synthetic.

## 6. Common errors and recovery

| Symptom | Likely cause | Recovery |
|---|---|---|
| `No module named dropout_pipeline` | Project not installed in the active environment, or terminal is outside the repository root. | Activate `.venv`, `cd` to the repository root, then run `python -m pip install -e ".[dev,report,app]"`. |
| `No module named streamlit` | App extra was not installed or the wrong Python is active. | Activate the project environment and reinstall with `python -m pip install -e ".[dev,report,app]"`. Run `python -m streamlit --version`. |
| CLI cannot download the dataset | No network access, proxy restriction or temporary UCI connection issue. | Retry later or put the official semicolon-delimited source CSV at `data/raw/data.csv`, then run `python -m dropout_pipeline.cli --data data/raw/data.csv`. Do not replace it with fabricated data. |
| App says model artifact is missing / analytical pages are empty | The CLI has not completed, failed validation or generated outputs were cleared. | Read the CLI error, fix the input or network issue, rerun the CLI, then refresh the browser. Chat, study-reset and directory pages remain usable without model outputs. |
| Port 8501 is busy | Another Streamlit process is listening. | Stop the other process with Ctrl+C, or run with `--server.port 8502` and open the new local/forwarded port. |
| Codespaces page cannot be reached | Port is not forwarded or app bound only to loopback. | Run the Codespaces command with `--server.address 0.0.0.0 --server.port 8501`; forward/open port 8501 in **Ports**. |
| Chat returns general support instead of the expected intent | Input wording does not match one of the explicit patterns. | Use one of the tested examples in section 3. It is a rules-based chatbot, so it does not infer arbitrary paraphrases as an LLM might. |
| Student profile or outcome labels fail | Feature order/class metadata mismatch, missing model artifacts, or malformed input. | Use the app’s synthetic profile and rerun the validated pipeline; do not bypass the input checks. |

### Stop the application

Click the Streamlit terminal and press **Ctrl+C**. In Codespaces, stop the running terminal process the same way. If a separate terminal is running data preparation, wait for it to complete or stop it separately with Ctrl+C. Close the browser tab after Streamlit stops.

## 7. Likely lecturer questions and suggested answers

### 1. What does `FrameValidator(Protocol)` add here?

It describes the minimal `.validate(frame)` interface without forcing every validator to inherit from one base class. A different validator can satisfy the contract structurally, which makes the pipeline easier to extend and test.

### 2. Why use a dataclass for `Assessment` or `SupportCase`?

Each result has several named fields that belong together. A frozen dataclass gives a compact typed record, readable attributes and less risk of accidentally changing values after construction. `SupportCase.__post_init__` also checks the record’s fields and dates.

### 3. Why create a retry decorator with `ParamSpec` and `wraps`?

The retry behavior is reusable for network operations. `ParamSpec` preserves the wrapped function’s argument types for static checking, while `functools.wraps` preserves its metadata. The retry is bounded, only targets transient I/O exceptions and logs a warning between attempts.

### 4. Why use a generator for CSV lines?

`iter_nonempty_lines()` yields one numbered non-empty line at a time. That avoids loading a second copy of the entire file just to check row widths, while keeping line numbers for actionable validation errors.

### 5. Why report macro-F1 as well as accuracy?

Accuracy can be dominated by the largest class. Macro-F1 averages class F1 scores equally, so weak performance on a smaller class remains visible. The per-class table and confusion matrix help explain the aggregate. These metrics describe one held-out split from the Portuguese dataset, not Kenyan effectiveness.

### 6. Could this dropout model be deployed at a Kenyan university today?

No. It uses a Portuguese source cohort, no Kenyan HELB/HEF, M-Pesa or local administrative data, and has not been validated on Kenyan students. A real pilot would need lawful representative local data, community and student input, privacy/security design, group fairness analysis, calibration and external validation, operational support pathways, and human oversight.

### 7. Is the support chatbot generative AI or counselling?

Neither. It matches text to explicit regular-expression intents and returns prepared responses and resource links. It has no external generative-AI call, does not diagnose, and is not monitored. Its safety wording is signposting, not emergency response or clinical care.

## 8. Key points to tell the lecturer

- The work demonstrates a full Python workflow: extraction, validation, transformations, model comparison, held-out evaluation, inference and a multi-view Streamlit interface.
- Python evidence is specific and inspectable: Protocols, type hints, frozen dataclasses, `StrEnum`, context managers, decorators/`ParamSpec`, generators, logging, pattern matching and tests.
- Leakage control comes from splitting before model selection and fitting the preprocessing inside the scikit-learn Pipeline during cross-validation.
- The app has two distinct roles: historical outcome analysis on a Portuguese dataset, and a Kenyan-campus support prototype with a rules-based chatbot and human-owned follow-up.
- The model must not be used as an automatic decision maker. Local validity, fairness, privacy, security and support pathways are prerequisites for any Kenyan deployment.
- Repository verification previously recorded 34 passing pytest tests, passing Ruff lint/format checks and passing mypy over 15 source files. These confirm software checks only, not population validity.

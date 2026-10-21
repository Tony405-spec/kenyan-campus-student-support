# Kenyan Campus Support Application: Demonstration Guide

This guide is for the local coursework prototype in `src/dropout_pipeline/support_app.py`. It covers the existing observed-outcome pipeline and the new Kenyan campus support chatbot. The chatbot uses fixed intent rules; it does not call an external generative AI service, diagnose a condition, or replace counselling.

## 1. Prerequisites

- Windows, macOS or Linux with Python 3.11 or later.
- Internet access for the first installation and optional UCI dataset download.
- A modern browser. The app normally opens at `http://localhost:8501`.
- About 1 GB of free space is recommended for Python packages and generated files.

## 2. Install dependencies

Open PowerShell or a terminal in the project root (the folder containing `pyproject.toml`). Create and activate a virtual environment if desired:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux, activate with `source .venv/bin/activate`. Install the package and all app, report and development extras:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev,report,app]"
```

The main dependencies are pandas, NumPy, scikit-learn and Matplotlib for the ETL/model pipeline; Streamlit for the interface; ReportLab for the PDF builder; and pytest, Ruff and mypy for development checks. If the UCI data is not already present, create the model outputs first:

```powershell
python -m dropout_pipeline.cli --download
```

This can take a short time because it fits three classifiers and performs five-fold cross-validation. The downloaded data and generated outputs remain local.

## 3. Start the application

From the project root, run:

```powershell
python -m streamlit run src/dropout_pipeline/support_app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`. Open it in a browser on the same computer. The support app and prediction pipeline are separate functions: the chatbot does not read the model's prediction for a student, and the prediction model is not a wellbeing assessment.

## 4. Demonstrate each page

### A. Campus Support Chat

1. Open **Campus Support Chat** from the sidebar.
2. Read the notice explaining that the tool is non-clinical and that identifying details should not be entered.
3. Enter a sample from the table below and submit it.
4. Show the response, the displayed matched-intent label, and any linked Kenyan resources.
5. Use **Clear chat** to reset the session.

For a quick lecturer walkthrough that opens with the CAT/revision example already in the chat, use `http://localhost:8501/?demo=academic` after starting the application. This is a fixed illustrative sample; it does not insert or store a real student's message.

| Sample input | Expected intent | Expected response content |
|---|---|---|
| `I have two CATs next week and haven't started revision` | `academic_or_exam_pressure` | Break work into one unit/topic and a 25-minute study block; suggest a classmate, lecturer, adviser or department for more help. |
| `HELB has not come through and I am worried about fees` | `financial_pressure` | Suggest contacting campus student finance/student affairs/department, checking the official HELB route, and not sharing IDs or payment credentials in chat. |
| `Nina wasiwasi kuhusu exams` | `anxiety_or_worry` | Offer a brief grounding step and suggest human support if worry disrupts sleep or study. |
| `Niko solo in the hostel` | `loneliness_or_isolation` | Suggest one manageable connection, such as a classmate, study group, society or community group. |
| `I feel hopeless and empty` | `low_mood_or_depression` | Validate distress without diagnosing and encourage contacting campus support, a qualified professional or a trusted person. |
| `I want to hurt myself tonight` | `immediate_safety` | Advise emergency support: Kenya emergency services 112/999, nearest health facility, and a trusted person who can stay. |

The output is deterministic: each message is matched to a fixed rule and a fixed response. It is intentionally not a diagnosis and does not claim to understand every phrasing. A message that does not match returns a general support prompt.

### B. Support Resources

Open **Support Resources** and demonstrate:

- The campus route: ask the current Student Affairs, Student Welfare, Dean of Students or department office for the counselling referral. No unverified KCA phone number is shown.
- Emergency: 112 or 999. The [National Police Service](https://www.nationalpolice.go.ke/) publishes emergency lines.
- Counselling: LVCT Health service information for the 1190 line: [LVCT Health](https://lvcthealth.org/a-counselor-experience-in-saving-a-suicide-mind/).
- Online support: [Kenya Red Cross Chat Care](https://redcross.or.ke/chat-care/).
- Education funding: [official HELB Student FAQs](https://www.helb.co.ke/faqs/students-faqs/).

The app displays a reminder that it is not monitored and should not be used in an emergency. Service details can change; use the provider's current website to confirm availability.

### C. Outcome Model Results

Open **Outcome Model Results**. When `outputs/metrics.json` and `outputs/run_config.json` exist, demonstrate the actual held-out macro-F1 and accuracy, the class-level report, confusion matrix and permutation importance image. Explain that these measure retrospective prediction of UCI dataset labels, not a live student's mental health or future certainty.

If these files are absent, run the model pipeline first. The app reports that outputs are missing rather than fabricating values.

## 5. Five-minute demonstration script

**0:00-0:40 - Introduce the system.** “This project combines the original UCI observed-outcome prediction pipeline with a separate Kenyan campus support signposting app. The support chatbot is deterministic and intent-based, not a clinical generative AI.”

**0:40-1:20 - Explain safeguards.** Point out the no-identifiers notice, session-only chat history, clear-chat control, non-diagnostic wording and emergency escalation. Explain that the chatbot is not monitored.

**1:20-2:40 - Show two to three intents.** Submit the CAT/revision example, the HELB/fees example and the Kiswahili/Sheng loneliness or anxiety example. Read the detected intent and practical next step. Reset the session.

**2:40-3:20 - Show support resources.** Open the emergency, LVCT 1190, Kenya Red Cross Chat Care and HELB links. Explain why the app gives a general campus office route instead of inventing an institution-specific number.

**3:20-4:20 - Show model results.** Open the results page. Explain macro-F1, the weaker Enrolled performance and the role of the confusion matrix. State that semester features make this a retrospective experiment rather than a proven early-warning tool.

**4:20-5:00 - Summarize engineering and limits.** Mention modular Python, the three-class intent Enum, data classes, regex matching, unit tests and the separation between support routing and outcome prediction. Close by naming human counselling and emergency care as the appropriate routes for high-need situations.

## 6. Troubleshooting

| Symptom | What to do |
|---|---|
| `No module named streamlit` | Activate the project environment and run `python -m pip install -e ".[dev,report,app]"`. |
| Browser cannot reach localhost | Confirm the Streamlit process is still running; open the exact URL printed in its terminal. |
| Port 8501 is occupied | Stop the other local app or run `python -m streamlit run src/dropout_pipeline/support_app.py --server.port 8502`, then open port 8502. |
| Outcome page says results are missing | Run `python -m dropout_pipeline.cli --download` once, wait for completion, then refresh the app. |
| A sample returns general support | Use one of the intent phrases above. The rule-based matcher has a limited vocabulary and does not interpret every paraphrase. |
| A resource link fails | Check internet access and use the provider's official site directly. The chatbot remains a prototype and is not monitored. |
| PowerShell blocks environment activation | Use `python -m pip` without activation by calling `.venv\Scripts\python.exe -m pip ...`, or follow your institution's approved environment setup. |

## 7. Stop the app

Return to the terminal running Streamlit and press **Ctrl+C** once. Close the browser tab. This stops the local server; the app does not run as a background service. To end only the conversation, use **Clear chat**.

## 8. Testing and screenshots

The complete suite contains 20 tests, including chat intent routing and safety/resource response checks in `tests/test_support_chatbot.py`; the pre-existing ETL/model tests remain. The report includes captures of the running local chatbot, support resources and outcome-results pages in `docs/report/screenshots/`. These are interface evidence, not model-performance evidence.


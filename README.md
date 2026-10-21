# University Dropout Prediction Pipeline

Course project for CPP 4201, KCA University. This repository extracts, validates, transforms, compares and evaluates three classifiers for the UCI 697 three-class recorded outcome (`Dropout`, `Enrolled`, `Graduate`). It is reproducible coursework, not a deployed student-risk system.

## Setup and run

Requires Python 3.11+. From the repository root:

```powershell
python -m pip install -e ".[dev,report,app]"
python -m dropout_pipeline.cli --download
python -m streamlit run src/dropout_pipeline/support_app.py
```

The first command installs the package, development tools, report generator and Streamlit app dependencies. The second downloads the official UCI ZIP to `data/raw/`, extracts `data.csv`, validates it, performs stratified 5-fold CV on the 80% training partition, selects by mean macro-F1, and evaluates the selected model once on the untouched 20% test partition. To reuse a local CSV, use `python -m dropout_pipeline.cli --data data/raw/data.csv`. The third command starts the Kenyan campus support app. Open the local URL printed by Streamlit, normally `http://localhost:8501`.

Outputs are written to `outputs/`: model comparison, CV summary, held-out metrics/report, confusion matrix, permutation importance, run config, and a concise interpretation. Raw data and generated outputs are ignored by Git; their provenance and generated examples are described in `data/raw/README.md` and `outputs/README.md`.

## Development commands

```powershell
python -m pytest
ruff check .
ruff format --check .
mypy src/
python -m dropout_pipeline.cli --help
```

The updated Kenyan campus report and demonstration guide are in `docs/`. Report source and PDF are in `docs/report/`. A plain-language concept map and viva notes are also in `docs/`.

## Experimental design

Nominal integer category codes use one-hot encoding; numeric measurements and counts use standardisation. All fitted transformations live in sklearn pipelines and are learned within training folds. Stratification preserves class proportions. Macro-F1 is the comparison metric because each outcome class receives equal weight. Accuracy is reported as a secondary metric. Models are balanced Logistic Regression, balanced Random Forest, and Gradient Boosting (which does not expose the same class-weight setting). Details and limitations are in `docs/methodology.md` and the report.

The data contains second-semester academic performance. Those records may be available only near or after the recorded outcome is determined. Results therefore cannot be interpreted as an early prediction or as causal factors. The separate Streamlit app provides deterministic intents for academic/exam pressure, financial stress, anxiety, loneliness, low mood/depression, greetings, general support and immediate safety signposting. It is not a diagnostic or generative AI system.

## Group

Tony Kenga (24/03652) · Baraka Blessings (24/04272)


## Kenyan campus support app documentation

The localized support app is a deterministic, intent-based Streamlit experience. The concise 20-page report is in `docs/report/Advanced_Python_Dropout_Prediction_Report.pdf`; the step-by-step demonstration guide is in `docs/demo_guide.md`. Both are included in the downloadable project bundle.

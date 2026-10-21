from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[1] / "src" / "dropout_pipeline" / "support_app.py"
PAGES = [
    "Home",
    "Individual Assessment",
    "Cohort Analysis",
    "Support Cases",
    "Follow-up Monitoring",
    "Model Transparency",
    "Data & Administration",
    "Student Support Home",
    "Support Chat",
    "Study Reset",
    "Support Directory",
]


def test_all_pages_render_in_an_empty_workspace() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    assert not app.exception
    for page in PAGES:
        app.radio[0].set_value(page).run(timeout=60)
        assert not app.exception, (page, [error.message for error in app.exception])


def test_staff_can_create_a_pseudonymous_case(tmp_path, monkeypatch) -> None:
    case_file = tmp_path / "support_cases.json"
    monkeypatch.setenv("STUDENT_SUPPORT_CASES_PATH", str(case_file))
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    app.radio[0].set_value("Support Cases").run(timeout=60)
    app.text_input[0].set_value("CASE-TEST-01")
    app.text_input[1].set_value("Synthetic academic check-in")
    next(button for button in app.button if button.label == "Save reviewed case").click().run(
        timeout=60
    )
    assert not app.exception, [error.message for error in app.exception]
    assert case_file.is_file()
    assert "CASE-TEST-01" in case_file.read_text(encoding="utf-8")


def test_synthetic_individual_assessment_runs_saved_model() -> None:
    repo_root = Path(__file__).parents[1]
    required = [
        repo_root / "data/raw/data.csv",
        repo_root / "outputs/run_config.json",
        repo_root / "outputs/student_outcome_model.joblib",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("Run the data pipeline first to create the local model artifact.")
    app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
    app.radio[0].set_value("Individual Assessment").run(timeout=60)
    next(button for button in app.button if button.label == "Run model assessment").click().run(
        timeout=60
    )
    assert not app.exception, [error.message for error in app.exception]
    labels = [metric.label for metric in app.metric]
    assert "Predicted class" in labels
    assert "Dropout probability estimate" in labels
    assert "Provisional review band" in labels

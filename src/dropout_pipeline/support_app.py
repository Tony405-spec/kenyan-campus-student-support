"""Streamlit UI for Kenyan university wellbeing signposting and model results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dropout_pipeline.support_chatbot import reply_to

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RESOURCES: dict[str, tuple[str, str]] = {
    "emergency": ("Emergency services (112 / 999)", "https://www.nationalpolice.go.ke/"),
    "lvct": (
        "LVCT Health counselling (1190)",
        "https://lvcthealth.org/a-counselor-experience-in-saving-a-suicide-mind/",
    ),
    "red_cross_chat": ("Kenya Red Cross Chat Care", "https://redcross.or.ke/chat-care/"),
    "helb": ("Official HELB student information", "https://www.helb.co.ke/faqs/students-faqs/"),
}


def _show_resource_links(keys: tuple[str, ...]) -> None:
    """Render only resource links relevant to the current response."""
    if not keys:
        return
    st.markdown("**Kenyan support options**")
    for key in keys:
        if key == "campus_support":
            st.markdown(
                "- Ask your campus Student Affairs, Student Welfare or department office "
                "for its current counselling referral."
            )
            continue
        title, url = RESOURCES[key]
        if key == "lvct":
            st.markdown(
                f"- **1190:** call LVCT Health for counselling. [Service information]({url})"
            )
        elif key == "emergency":
            st.markdown(f"- **112 / 999:** emergency services. [National Police Service]({url})")
        else:
            st.markdown(f"- [{title}]({url})")


def _render_chatbot() -> None:
    """Render the session-only deterministic student-support chat interface."""
    st.title("Campus Support Chat")
    st.write(
        "A session-based, intent-based demo for common Kenyan university "
        "pressures. It is not a counsellor, clinician, diagnosis, or emergency service."
    )
    st.info(
        "Keep messages general: don’t enter your name, student number, ID, HELB credentials, "
        "M-Pesa PIN, address, or other identifying details. Messages are held in this app's "
        "active Streamlit session and cleared with **Clear chat** or when the session ends."
    )
    st.caption(
        "Try: ‘I have two CATs next week and haven’t started revision’; "
        '"HELB has not come through and I am worried about fees"; "niko solo in the hostel"; '
        "or ‘I feel hopeless.’"
    )

    if "support_messages" not in st.session_state:
        welcome = {
            "role": "assistant",
            "content": "Hi, karibu. What has been hardest on campus lately? You can choose "
            "study pressure, fees, anxiety, loneliness or low mood.",
            "intent": "welcome",
            "resources": (),
        }
        st.session_state.support_messages = [welcome]
        if st.query_params.get("demo") == "academic":
            sample = "I have two CATs next week and haven't started revision"
            response = reply_to(sample)
            st.session_state.support_messages.extend(
                [
                    {"role": "user", "content": sample},
                    {
                        "role": "assistant",
                        "content": response.message,
                        "intent": response.intent.value,
                        "resources": response.resources,
                    },
                ]
            )

    clear_col, _ = st.columns([1, 5])
    with clear_col:
        if st.button("Clear chat", type="secondary"):
            st.session_state.support_messages = []
            st.rerun()

    for item in st.session_state.support_messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant" and item.get("intent") not in (None, "welcome"):
                st.caption(f"Matched intent: `{item['intent']}`")
                _show_resource_links(tuple(item.get("resources", ())))

    prompt = st.chat_input("Share a general concern (no personal details)")
    if prompt:
        st.session_state.support_messages.append({"role": "user", "content": prompt})
        response = reply_to(prompt)
        st.session_state.support_messages.append(
            {
                "role": "assistant",
                "content": response.message,
                "intent": response.intent.value,
                "resources": response.resources,
            }
        )
        st.rerun()

    st.caption(
        "If you may be in immediate danger, call 112 or 999, go to the nearest health facility, "
        "and contact someone you trust who can stay with you."
    )


def _render_resources() -> None:
    """Present verified Kenyan resources and a non-invented campus route."""
    st.title("Support Resources in Kenya")
    st.write(
        "Links checked 21 October 2026. Availability can change; confirm details with providers."
    )
    st.subheader("On your campus")
    st.write(
        "Contact your campus Student Affairs, Student Welfare, Dean of Students or department "
        "office and ask for the current counselling referral. Ask what is confidential and what "
        "the next step will be before sharing sensitive details. This project does not claim a "
        "verified KCA University counselling phone number."
    )
    st.subheader("Kenyan services")
    st.markdown(
        "- **Immediate danger:** call **112 or 999**, go to the nearest health facility, "
        "and contact someone trusted. [National Police Service](https://www.nationalpolice.go.ke/)"
    )
    st.markdown(
        "- **LVCT Health:** call **1190** for counselling and psychosocial support. [LVCT service information](https://lvcthealth.org/a-counselor-experience-in-saving-a-suicide-mind/)"
    )
    st.markdown(
        "- **Kenya Red Cross Chat Care:** a national online mental-health chat service. [Open Chat Care](https://redcross.or.ke/chat-care/)"
    )
    st.markdown(
        "- **Fees/HELB:** use official student information and contact channels. [HELB student FAQs](https://www.helb.co.ke/faqs/students-faqs/)"
    )
    st.warning("This prototype is not monitored. Do not wait for a chatbot reply in an emergency.")


def _render_pipeline_results() -> None:
    """Display actual stored project metrics without presenting them as a student diagnosis."""
    st.title("Observed-Outcome Model Results")
    st.warning(
        "This separate coursework model predicts recorded dataset labels. It is retrospective, "
        "not a validated individual risk assessment, early-warning system, or counselling tool."
    )
    metrics_path = OUTPUT_DIR / "metrics.json"
    config_path = OUTPUT_DIR / "run_config.json"
    if not metrics_path.exists() or not config_path.exists():
        st.info(
            "Run the prediction pipeline first to create outputs/metrics.json and run_config.json."
        )
        return
    metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    first, second = st.columns(2)
    first.metric("Held-out macro-F1", f"{metrics['test_macro_f1']:.3f}")
    second.metric("Held-out accuracy", f"{metrics['test_accuracy']:.3f}")
    st.caption(
        f"Selected model: {config['selected_model']} • UCI 697 • {config['rows']:,} records • "
        f"{metrics['test_rows']:,} held-out rows • seed {config['random_seed']} • "
        "five-fold training CV"
    )
    report = pd.DataFrame(metrics["classification_report"]).T
    st.subheader("Per-class performance")
    st.dataframe(
        report.loc[
            ["Dropout", "Enrolled", "Graduate"], ["precision", "recall", "f1-score", "support"]
        ],
        use_container_width=True,
    )
    matrix_path = OUTPUT_DIR / "confusion_matrix.png"
    importance_path = OUTPUT_DIR / "permutation_importance.png"
    if matrix_path.exists():
        st.image(str(matrix_path), caption="Held-out test confusion matrix")
    if importance_path.exists():
        st.image(
            str(importance_path),
            caption="Permutation importance: predictive reliance, not causation",
        )


def main() -> None:
    """Run the local campus support and results app."""
    st.set_page_config(
        page_title="Campus Support | Student Outcomes", page_icon="🎓", layout="wide"
    )
    st.sidebar.title("Student Support Demo")
    st.sidebar.caption("Kenyan university campus context")
    pages = ["Campus Support Chat", "Support Resources", "Outcome Model Results"]
    requested_page = st.query_params.get("view")
    page = st.sidebar.radio(
        "Choose a page",
        pages,
        index=pages.index(requested_page) if requested_page in pages else 0,
    )
    st.sidebar.divider()
    st.sidebar.caption("Intent-based prototype • English and selected Kiswahili/Sheng phrases")
    if page == "Campus Support Chat":
        _render_chatbot()
    elif page == "Support Resources":
        _render_resources()
    else:
        _render_pipeline_results()


if __name__ == "__main__":
    main()

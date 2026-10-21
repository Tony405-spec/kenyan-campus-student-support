"""Polished Streamlit interface for Kenyan campus support and model results."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dropout_pipeline.case_management import (
    CaseStatus,
    SupportCase,
    SupportCategory,
    load_cases,
    overdue_cases,
    upsert_case,
)
from dropout_pipeline.cli import run_pipeline
from dropout_pipeline.cohort import (
    dataset_quality_summary,
    holdout_review_summary,
    make_age_band,
    outcome_summary,
)
from dropout_pipeline.config import PipelineConfig
from dropout_pipeline.inference import (
    Assessment,
    ReviewBand,
    assess_student,
    load_model_artifact,
    make_synthetic_profile,
)
from dropout_pipeline.support_chatbot import reply_to
from dropout_pipeline.transform import CATEGORICAL_COLUMNS
from dropout_pipeline.validate import load_and_validate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
CASE_PATH = Path(
    os.environ.get("STUDENT_SUPPORT_CASES_PATH", str(OUTPUT_DIR / "support_cases.json"))
)

RESOURCES: dict[str, tuple[str, str]] = {
    "emergency": ("Emergency services (112 / 999)", "https://www.nationalpolice.go.ke/"),
    "lvct": (
        "LVCT Health counselling (1190)",
        "https://lvcthealth.org/a-counselor-experience-in-saving-a-suicide-mind/",
    ),
    "red_cross_chat": ("Kenya Red Cross Chat Care", "https://redcross.or.ke/chat-care/"),
    "helb": ("Official HELB student information", "https://www.helb.co.ke/faqs/students-faqs/"),
}

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


def _style_app() -> None:
    """Apply a consistent, responsive visual system to Streamlit's native widgets."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root { --ink:#172b4d; --muted:#64748b; --teal:#087e79; --mint:#e8f6f2; --line:#e4eaf0; }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
        .stApp { background:linear-gradient(180deg,#f4faf9 0,#f7f9fc 280px,#f7f9fc 100%); }
        [data-testid="stSidebar"] { background:#102b3f; }
        [data-testid="stSidebar"] * { color:#f3f8fa; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding:9px 12px;
            border-radius:10px;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#1b4055; }
        h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:var(--ink); }
        h1 { font-size:2.35rem !important; }
        [data-testid="stMetric"] {
            background:#fff;
            border:1px solid var(--line);
            border-radius:16px;
            padding:18px 20px;
            box-shadow:0 5px 20px #18334a0a;
        }
        [data-testid="stMetricLabel"] { color:var(--muted); }
        [data-testid="stMetricValue"] { color:var(--ink); }
        .hero {
            background:linear-gradient(120deg,#0e3946,#087e79);
            border-radius:24px;
            padding:34px 38px;
            color:white;
            margin:8px 0 24px;
            box-shadow:0 14px 35px #0e394625;
        }
        .hero h1,.hero h2,.hero p,.hero span { color:white !important; }
        .hero-kicker {
            font-size:.78rem;
            font-weight:700;
            letter-spacing:.14em;
            text-transform:uppercase;
            opacity:.78;
        }
        .hero h1 { font-size:2.55rem !important; line-height:1.12; margin:.45rem 0 .6rem; }
        .hero p { max-width:760px; font-size:1.05rem; opacity:.88; margin:0; }
        .eyebrow {
            color:var(--teal);
            text-transform:uppercase;
            letter-spacing:.13em;
            font-weight:700;
            font-size:.72rem;
        }
        .soft-card {
            background:white;
            border:1px solid var(--line);
            border-radius:18px;
            padding:20px 22px;
            min-height:150px;
            box-shadow:0 7px 24px #18334a08;
        }
        .soft-card h3 { font-size:1.12rem; margin:.35rem 0 .45rem; }
        .soft-card p { color:var(--muted); font-size:.93rem; margin:0; }
        .pill {
            display:inline-block;
            border-radius:99px;
            padding:5px 10px;
            background:#e8f6f2;
            color:#087e79;
            font-size:.77rem;
            font-weight:700;
        }
        .callout {
            border-left:4px solid #e3a43b;
            background:#fff8e8;
            border-radius:8px;
            padding:13px 16px;
            color:#62420c;
        }
        .stButton>button { border-radius:10px; min-height:2.7rem; font-weight:600; }
        .stButton>button[kind="primary"] {
            background:#087e79;
            border-color:#087e79;
            color:white;
        }
        .stButton>button[kind="primary"]:hover {
            background:#066763;
            border-color:#066763;
            color:white;
        }
        [data-baseweb="tab-highlight"] { background-color:#087e79; }
        [data-baseweb="tab"][aria-selected="true"] { color:#087e79; }
        [data-testid="stAlert"] p { color:#263a4f; }
        div[data-testid="stChatMessage"] {
            border:1px solid var(--line);
            border-radius:16px;
            background:white;
            padding:12px 16px;
        }
        [data-testid="stChatInput"] { border-radius:16px; }
        hr { border-color:var(--line); }
        footer { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero(title: str, subtitle: str, kicker: str = "KARIBU • CAMPUS WELLBEING") -> None:
    st.markdown(
        f'<section class="hero"><div class="hero-kicker">{kicker}</div>'
        f"<h1>{title}</h1><p>{subtitle}</p></section>",
        unsafe_allow_html=True,
    )


def _resource_link(key: str) -> None:
    title, url = RESOURCES[key]
    st.link_button(title, url, use_container_width=True)


def _bar_chart(data: pd.Series | pd.DataFrame, colors: list[str]) -> None:
    """Render a readable Matplotlib bar chart without Streamlit's Altair adapter."""
    figure, axis = plt.subplots(figsize=(8, 3.6))
    data.plot(kind="bar", ax=axis, color=colors)
    axis.set_xlabel("")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    if isinstance(data, pd.DataFrame) and data.shape[1] > 1:
        axis.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    elif axis.get_legend():
        axis.get_legend().remove()
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True, use_container_width=True)
    plt.close(figure)


def _show_resource_links(keys: tuple[str, ...]) -> None:
    """Render relevant trusted service links and generic campus referrals."""
    if not keys:
        return
    st.markdown("**Helpful next contacts**")
    for key in keys:
        if key == "campus_support":
            st.info(
                "Ask Student Affairs, Student Welfare, Dean of Students or your department "
                "for the current counselling or student-support referral."
            )
        else:
            _resource_link(key)


def _initialize_chat() -> None:
    if "support_messages" in st.session_state:
        return
    st.session_state.support_messages = [
        {
            "role": "assistant",
            "content": (
                "Hi, karibu. You don’t have to explain everything at once. What feels hardest "
                "on campus today: studies, money, worry, feeling alone or low mood?"
            ),
            "intent": "welcome",
            "resources": (),
        }
    ]
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


def _submit_chat_message(message: str) -> None:
    response = reply_to(message)
    st.session_state.support_messages.extend(
        [
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": response.message,
                "intent": response.intent.value,
                "resources": response.resources,
            },
        ]
    )


def _render_student_support_home() -> None:
    _hero(
        "You belong here.",
        "A calm, practical starting point for the pressures of university life in Kenya. "
        "Choose one next step—support, a study reset, or a trusted resource.",
    )
    if st.session_state.get("check_in_complete"):
        selected_mood = st.session_state.get("check_in_mood", "today")
        st.success(
            f"Thanks for checking in. You chose **{selected_mood}**. "
            "This stays in this session only; it is not a clinical assessment."
        )

    st.subheader("How are things feeling right now?")
    st.caption("A private, optional check-in for this session. No account or score is created.")
    moods = ["🙂 Steady enough", "😟 Under pressure", "😔 Feeling low", "😶 A bit alone"]
    cols = st.columns(4)
    for col, mood in zip(cols, moods, strict=True):
        with col:
            if st.button(mood, key=f"mood-{mood}", use_container_width=True):
                st.session_state.check_in_mood = mood
                st.session_state.check_in_complete = True
                st.rerun()

    st.markdown("### Pick what would help")
    cards = st.columns(3)
    actions = [
        (
            "💬 Talk it through",
            "Share a general concern and get a practical, prepared response.",
            "Support Chat",
        ),
        (
            "📚 Reset your study plan",
            "Break a CAT, exam or assignment pile-up into smaller steps.",
            "Study Reset",
        ),
        (
            "🧭 Find support",
            "Browse Kenyan emergency, counselling, online and HELB links.",
            "Support Directory",
        ),
    ]
    for col, (title, description, page) in zip(cards, actions, strict=True):
        with col:
            st.markdown(
                f'<div class="soft-card"><span class="pill">START HERE</span>'
                f"<h3>{title}</h3><p>{description}</p></div>",
                unsafe_allow_html=True,
            )
            if st.button(f"Open {page}", key=f"open-{page}", use_container_width=True):
                st.session_state.navigate_to = page
                st.rerun()

    st.markdown("### Your support toolkit")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown('<div class="eyebrow">01 · CHECK IN</div>', unsafe_allow_html=True)
        st.write("Name what is going on without sharing identifying details.")
    with t2:
        st.markdown('<div class="eyebrow">02 · TAKE ONE STEP</div>', unsafe_allow_html=True)
        st.write("Use a short study sprint or a grounding reset to get started.")
    with t3:
        st.markdown('<div class="eyebrow">03 · CONNECT</div>', unsafe_allow_html=True)
        st.write("Reach a trusted person, campus office or a verified support service.")

    st.markdown(
        '<div class="callout"><b>Need urgent help?</b> This app is not monitored. '
        "If you may be in immediate danger, call 112 or 999, go to the nearest health "
        "facility, and contact someone you trust who can stay with you.</div>",
        unsafe_allow_html=True,
    )


def _render_chat() -> None:
    _hero(
        "A supportive place to start.",
        "Share a general concern. The app matches it to a prepared campus-support response "
        "and practical next step.",
        "SUPPORT CHAT • ENGLISH + SELECTED KISWAHILI/SHENG",
    )
    with st.expander("Your privacy and what this tool can do", expanded=False):
        st.write(
            "Messages stay in the active Streamlit session and clear when you clear the chat "
            "or the session ends. Do not enter your name, student number, ID, HELB credentials, "
            "M-Pesa PIN, address or other identifying details. This is a deterministic demo, "
            "not a counsellor, clinician, diagnosis or emergency service."
        )

    if "support_messages" not in st.session_state:
        _initialize_chat()
    col, _ = st.columns([1, 5])
    with col:
        if st.button("＋ New conversation", type="secondary", use_container_width=True):
            st.session_state.support_messages = []
            _initialize_chat()
            st.rerun()

    if len(st.session_state.support_messages) == 1:
        st.caption("Try a quick start")
        suggestions = st.columns(4)
        samples = [
            ("📚 CAT pressure", "I have two CATs next week and haven't started revision"),
            ("💸 Fees / HELB", "HELB has not come through and I am worried about fees"),
            ("🌬️ Worry", "Nina wasiwasi kuhusu exams"),
            ("🤝 Feeling alone", "Niko solo in the hostel"),
        ]
        for col, (label, sample) in zip(suggestions, samples, strict=True):
            with col:
                if st.button(label, key=f"suggest-{label}", use_container_width=True):
                    _submit_chat_message(sample)
                    st.rerun()

    for item in st.session_state.support_messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant" and item.get("intent") not in (None, "welcome"):
                st.markdown(
                    f'<span class="pill">Matched: {item["intent"].replace("_", " ")}</span>',
                    unsafe_allow_html=True,
                )
                _show_resource_links(tuple(item.get("resources", ())))

    prompt = st.chat_input("Type a general concern (please leave out personal details)")
    if prompt:
        _submit_chat_message(prompt)
        st.rerun()
    st.caption("Prepared intent responses • No external generative AI • Session-only conversation")


def _render_study_reset() -> None:
    _hero(
        "Make the next step smaller.",
        "Turn a revision pile-up into a realistic first study sprint. This planner runs locally "
        "in your session and does not save your plan.",
        "STUDY RESET • CATs, EXAMS & COURSEWORK",
    )
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Build a 25-minute focus plan")
            unit = st.text_input("Unit or course (optional)", placeholder="e.g. Data Structures")
            topic = st.text_input("One topic or task", placeholder="e.g. revise linked lists")
            available = st.select_slider("Time you have today", options=[15, 25, 40, 60], value=25)
            if st.button("Create my small-step plan", type="primary", use_container_width=True):
                if not topic.strip():
                    st.warning("Add one small topic or task first.")
                else:
                    st.session_state.study_plan = {
                        "unit": unit.strip(),
                        "topic": topic.strip(),
                        "minutes": available,
                    }
        plan = st.session_state.get("study_plan")
        if plan:
            st.success("Your next-step plan is ready")
            if plan["unit"]:
                st.markdown(f"**Unit:** {plan['unit']}")
            st.markdown(f"**Focus:** {plan['topic']}")
            st.markdown(
                f"1. Spend **{plan['minutes']} minutes** on this task.\n"
                "2. Put your phone on silent and keep only the material you need open.\n"
                "3. Take a 5-minute break; stretch, drink water or step outside.\n"
                "4. Write down the next tiny step before deciding whether to continue."
            )
    with right:
        with st.container(border=True):
            st.subheader("When stress is getting loud")
            st.write("Try this brief grounding reset before you begin:")
            st.markdown(
                "**1 · Settle** — place both feet on the floor and let your shoulders drop.\n\n"
                "**2 · Exhale** — breathe out slowly; do not force a deep breath.\n\n"
                "**3 · Notice** — name five things you see and four things you can feel.\n\n"
                "**4 · Choose** — pick one action that takes less than five minutes."
            )
            st.caption(
                "If anxiety keeps affecting sleep or study, connect with a person or professional."
            )
            if st.button("Open support options", use_container_width=True):
                st.session_state.navigate_to = "Support Directory"
                st.rerun()

    st.info(
        "If academic pressure feels unmanageable, contact your lecturer, academic adviser or "
        "department early and ask what support or deadline options are available."
    )


def _render_resources() -> None:
    _hero(
        "Support that connects you to people.",
        "Start with your campus support office, then use the current provider information below. "
        "Check service availability directly with each provider.",
        "SUPPORT DIRECTORY • KENYA",
    )
    st.caption("Provider links last checked 21 October 2026; details can change.")
    campus, urgent = st.columns([1, 1], gap="large")
    with campus:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">YOUR CAMPUS</div>', unsafe_allow_html=True)
            st.subheader("Ask for the current referral")
            st.write(
                "Contact Student Affairs, Student Welfare, Dean of Students or your department. "
                "Ask what is confidential and what the next step will be before sharing sensitive "
                "details. We do not show an unverified institution-specific counselling number."
            )
    with urgent:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">IMMEDIATE SAFETY</div>', unsafe_allow_html=True)
            st.subheader("Get urgent human help")
            st.write(
                "If you may act on thoughts of self-harm or cannot stay safe, call emergency "
                "services, go to the nearest health facility and contact someone trusted who "
                "can stay with you."
            )
            _resource_link("emergency")

    st.markdown("### Choose the kind of support")
    tab1, tab2, tab3 = st.tabs(["Counselling & wellbeing", "Fees & funding", "Online support"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.subheader("Talk to LVCT Health")
                st.write("LVCT Health lists 1190 for counselling and psychosocial support.")
                _resource_link("lvct")
        with c2:
            with st.container(border=True):
                st.subheader("Reach Kenya Red Cross Chat Care")
                st.write("An online mental-health chat service from Kenya Red Cross.")
                _resource_link("red_cross_chat")
    with tab2:
        with st.container(border=True):
            st.subheader("HELB student information")
            st.write(
                "Check student FAQs and official contacts for funding questions. Do not share "
                "your HELB password, one-time code, ID number or payment credentials in this app."
            )
            _resource_link("helb")
    with tab3:
        st.info(
            "The directory lists Kenya Red Cross Chat Care above. If a site or service is "
            "unavailable, return to your campus office or a trusted health professional."
        )

    st.warning("This prototype is not monitored. Never wait for an app response in an emergency.")


def _render_insights() -> None:
    _hero(
        "Inside the academic outcome project.",
        "Explore the actual held-out results from the reproducible UCI 697 modelling pipeline. "
        "These statistics are not a wellbeing score or an individual student prediction.",
        "PROJECT INSIGHTS • RETROSPECTIVE COURSEWORK",
    )
    metrics_path = OUTPUT_DIR / "metrics.json"
    config_path = OUTPUT_DIR / "run_config.json"
    if not metrics_path.exists() or not config_path.exists():
        st.info(
            "Model outputs are not present in this Codespace yet. Run "
            "`python -m dropout_pipeline.cli --download` in the terminal, then refresh this page. "
            "Chat, study tools and resources work without the dataset."
        )
        return
    metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    st.warning(
        "This is retrospective prediction of recorded dataset labels, not a validated individual "
        "risk assessment, early-warning service or counselling tool. Semester performance may "
        "be available only after the outcome is already developing."
    )
    first, second, third = st.columns(3)
    first.metric(
        "Held-out macro-F1",
        f"{metrics['test_macro_f1']:.3f}",
        help="Equal-weight average across outcome classes.",
    )
    second.metric("Held-out accuracy", f"{metrics['test_accuracy']:.3f}")
    third.metric("Test records", f"{metrics['test_rows']:,}")
    st.caption(
        f"Selected model: {config['selected_model']} • UCI 697 • {config['rows']:,} records • "
        f"Random seed {config['random_seed']} • five-fold training cross-validation"
    )
    report = pd.DataFrame(metrics["classification_report"]).T
    st.subheader("How performance differs by class")
    st.dataframe(
        report.loc[
            ["Dropout", "Enrolled", "Graduate"], ["precision", "recall", "f1-score", "support"]
        ],
        use_container_width=True,
    )
    left, right = st.columns(2)
    matrix = OUTPUT_DIR / "confusion_matrix.png"
    importance = OUTPUT_DIR / "permutation_importance.png"
    if matrix.exists():
        left.image(str(matrix), caption="Held-out test confusion matrix")
    if importance.exists():
        right.image(
            str(importance),
            caption="Permutation importance shows predictive reliance, not causation",
        )
    with st.expander("How should I interpret this?"):
        st.write(
            "Macro-F1 weights each class equally, so it can reveal weak performance on a smaller "
            "class even when accuracy is higher. The confusion matrix shows which recorded labels "
            "were mixed up. Permutation importance is an association within this fitted dataset, "
            "not evidence that changing a feature would change a student's outcome."
        )


@st.cache_data(show_spinner=False)
def _read_validated_frame(path_text: str) -> pd.DataFrame:
    return load_and_validate(Path(path_text))


@st.cache_resource(show_spinner=False)
def _read_model(path_text: str) -> Any:
    return load_model_artifact(Path(path_text))


def _read_artifact_json(name: str) -> dict[str, Any] | None:
    path = OUTPUT_DIR / name
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read `{path.name}`: {exc}")
        return None
    return value if isinstance(value, dict) else None


def _get_dataset() -> tuple[pd.DataFrame | None, str | None]:
    try:
        return _read_validated_frame(str(DATA_PATH)), None
    except Exception as exc:
        return None, str(exc)


def _get_saved_model() -> tuple[Any | None, dict[str, Any] | None, dict[str, Any] | None]:
    metrics = _read_artifact_json("metrics.json")
    config = _read_artifact_json("run_config.json")
    if not config:
        return None, metrics, config
    model_path = OUTPUT_DIR / str(config.get("model_path", "student_outcome_model.joblib"))
    if not model_path.is_file():
        return None, metrics, config
    try:
        return _read_model(str(model_path)), metrics, config
    except Exception as exc:
        st.error(f"Could not load the locally trained model artifact: {exc}")
        return None, metrics, config


def _threshold_controls(key_prefix: str) -> tuple[float, float]:
    first, second = st.columns(2)
    with first:
        lower = st.slider(
            "Consider follow-up from probability",
            min_value=0.0,
            max_value=0.95,
            value=0.35,
            step=0.05,
            key=f"{key_prefix}-lower-threshold",
            help="Provisional demonstration threshold; adjust only for sensitivity exploration.",
        )
    with second:
        upper = st.slider(
            "Priority human review from probability",
            min_value=round(lower + 0.05, 2),
            max_value=1.0,
            value=max(0.65, round(lower + 0.05, 2)),
            step=0.05,
            key=f"{key_prefix}-upper-threshold",
            help="Provisional demonstration threshold; not a validated institutional rule.",
        )
    return lower, upper


def _show_domain_notice() -> None:
    st.warning(
        "**Domain limitation:** the model was trained on records from a Portuguese "
        "higher-education "
        "institution. It is not validated on Kenyan university students. Fees, HELB/HEF status, "
        "M-Pesa activity and mental-health indicators are not model inputs. Use outputs only for "
        "demonstration and human review, never as automatic decisions."
    )


def _render_home() -> None:
    _hero(
        "University overview",
        "Explore observed cohort outcomes, review model evidence and organise a human follow-up. "
        "This Kenyan-context prototype does not make student decisions.",
        "STUDENT SUCCESS • DECISION SUPPORT PROTOTYPE",
    )
    _show_domain_notice()
    frame, error = _get_dataset()
    if frame is None:
        st.info(
            "No validated cohort is loaded. Use **Data & Administration → Prepare the model** "
            "to download the official dataset and run the pipeline. The Kenyan support tools "
            "remain available."
        )
        if error:
            with st.expander("Dataset status"):
                st.code(error)
        return

    model, metrics, config = _get_saved_model()
    group_options = [
        "All records",
        "Course",
        "Gender",
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
        "Age band",
    ]
    selected_group = st.selectbox("Cohort view", group_options, key="overview-group")
    filtered = frame.copy()
    group_column = selected_group
    if selected_group == "Age band":
        filtered["Age band"] = make_age_band(filtered).astype(str)
    if selected_group != "All records":
        options = sorted(filtered[group_column].dropna().unique().tolist())
        chosen = st.multiselect(
            f"Filter {selected_group} values (leave empty for all)",
            options,
            format_func=lambda value: (
                f"Code {value}" if isinstance(value, (int, float)) else str(value)
            ),
            key="overview-filter-values",
        )
        if chosen:
            filtered = filtered.loc[filtered[group_column].isin(chosen)]

    counts = filtered["Target"].value_counts()
    total = len(filtered)
    cards = st.columns(5)
    cards[0].metric("Observed records", f"{total:,}")
    for index, label in enumerate(["Dropout", "Enrolled", "Graduate"], start=1):
        number = int(counts.get(label, 0))
        cards[index].metric(label, f"{number:,}", f"{number / total:.1%}" if total else "0%")
    if metrics:
        cards[4].metric("Held-out macro-F1", f"{metrics['test_macro_f1']:.3f}")
    else:
        cards[4].metric("Model status", "Not trained")

    left, right = st.columns(2)
    with left:
        st.subheader("Observed outcome distribution")
        observed = outcome_summary(filtered)
        _bar_chart(observed["Records"], ["#087e79"])
        st.dataframe(observed.style.format({"Proportion": "{:.1%}"}), use_container_width=True)
    with right:
        st.subheader("Academic and financial indicators")
        indicators = [
            "Curricular units 2nd sem (approved)",
            "Tuition fees up to date",
            "Debtor",
            "Scholarship holder",
        ]
        by_outcome = filtered.groupby("Target", observed=False)[indicators].mean().T
        _bar_chart(by_outcome, ["#087e79", "#e0a13a", "#4776a8"])
        st.caption(
            "Academic counts are means. Financial binary fields are shown as the share of records "
            "coded 1; they are not Kenyan HELB or M-Pesa data."
        )

    st.subheader("Reference-set review queue (model estimates)")
    if model is None or not config:
        st.info(
            "Run the pipeline to create a saved, fitted model before generating review estimates."
        )
        return
    lower, upper = _threshold_controls("overview")
    try:
        review = holdout_review_summary(
            frame,
            model,
            config["feature_columns"],
            config["class_names"],
            int(config["random_seed"]),
            float(config["test_size"]),
            lower,
            upper,
        )
    except Exception as exc:
        st.error(f"Could not score the reference holdout: {exc}")
        return
    review_counts = review["provisional_review_band"].value_counts()
    r1, r2, r3 = st.columns(3)
    for column, band in zip(
        [r1, r2, r3],
        [ReviewBand.PRIORITY.value, ReviewBand.FOLLOW_UP.value, ReviewBand.ROUTINE.value],
        strict=True,
    ):
        column.metric(band, int(review_counts.get(band, 0)))
    st.caption(
        "Counts come from the held-out split of the de-identified Portuguese reference dataset. "
        "Thresholds are provisional, probability estimates are not calibrated, and flags "
        "require human review."
    )


def _features_for_profile(frame: pd.DataFrame) -> dict[str, list[str]]:
    columns = [str(name) for name in frame.columns if name != "Target"]
    finance = ["Debtor", "Tuition fees up to date", "Scholarship holder"]
    academic = [name for name in columns if name.startswith("Curricular units")]
    macro = ["Unemployment rate", "Inflation rate", "GDP"]
    assigned = set(finance + academic + macro)
    background = [name for name in columns if name not in assigned]
    return {
        "Profile and admission": background,
        "Fees and access indicators": finance,
        "Academic progress": academic,
        "Macroeconomic context": macro,
    }


def _render_assessment() -> None:
    _hero(
        "Individual assessment",
        "Run the saved preprocessing and classifier on one explicitly synthetic profile. "
        "A model estimate is a prompt for review—not a conclusion about a student.",
        "HUMAN REVIEW • NO REAL STUDENT RECORDS",
    )
    _show_domain_notice()
    frame, error = _get_dataset()
    model, _, config = _get_saved_model()
    if frame is None or model is None or not config:
        st.info(
            "Prepare the validated dataset and saved model from **Data & Administration** first."
        )
        if error:
            st.code(error)
        return
    st.info(
        "The starting profile is a synthetic combination of observed medians and category modes. "
        "Edit it for a demonstration; it is not an actual student's record and may not "
        "represent a plausible combination."
    )
    lower, upper = _threshold_controls("assessment")
    baseline = make_synthetic_profile(frame)
    feature_groups = _features_for_profile(frame)
    input_values: dict[str, int | float] = {}
    with st.form("individual-assessment-form"):
        tabs = st.tabs(list(feature_groups))
        for tab, (_section, columns) in zip(tabs, feature_groups.items(), strict=True):
            with tab:
                st.caption(
                    "All variables are original UCI fields. Category numbers are nominal codes, "
                    "not rankings."
                )
                pairs = st.columns(2)
                for index, name in enumerate(columns):
                    with pairs[index % 2]:
                        series = frame[name]
                        default_value = baseline.iloc[0][name]
                        label = name.replace("\t", "").strip()
                        if name in CATEGORICAL_COLUMNS:
                            options = sorted(int(value) for value in series.unique())
                            selected = st.selectbox(
                                label,
                                options,
                                index=options.index(int(default_value)),
                                format_func=lambda value: f"Code {value}",
                                key=f"assessment-{name}",
                                help="Original UCI nominal code. "
                                "Its numeric size has no ordinal meaning.",
                            )
                            input_values[name] = (
                                int(default_value) if selected is None else selected
                            )
                        else:
                            if pd.api.types.is_integer_dtype(series):
                                integer_value = st.number_input(
                                    label,
                                    min_value=int(series.min()),
                                    max_value=int(series.max()),
                                    value=int(default_value),
                                    step=1,
                                    key=f"assessment-{name}",
                                )
                                input_values[name] = (
                                    int(default_value) if integer_value is None else integer_value
                                )
                            else:
                                continuous_value = st.number_input(
                                    label,
                                    min_value=float(series.min()),
                                    max_value=float(series.max()),
                                    value=float(default_value),
                                    step=0.1,
                                    format="%.2f",
                                    key=f"assessment-{name}",
                                )
                                input_values[name] = (
                                    float(default_value)
                                    if continuous_value is None
                                    else continuous_value
                                )
        submitted = st.form_submit_button("Run model assessment", type="primary")

    if submitted:
        profile = pd.DataFrame([input_values], columns=config["feature_columns"])
        try:
            assessment = assess_student(
                model,
                profile,
                config["feature_columns"],
                config["class_names"],
                lower,
                upper,
            )
        except Exception as exc:
            st.error(f"Assessment failed validation or inference: {exc}")
            return
        st.session_state.last_assessment = assessment

    last_assessment: Assessment | None = st.session_state.get("last_assessment")
    if last_assessment:
        st.subheader("Model output for the synthetic profile")
        st.caption("Predicted recorded-outcome class; it is not a Kenyan student prediction.")
        left, middle, right = st.columns(3)
        left.metric("Predicted class", last_assessment.predicted_outcome)
        middle.metric("Dropout probability estimate", f"{last_assessment.dropout_probability:.1%}")
        right.metric("Provisional review band", last_assessment.review_band.value)
        probability_frame = pd.DataFrame(
            {
                "Class": list(last_assessment.probabilities),
                "Model probability estimate": list(last_assessment.probabilities.values()),
            }
        ).set_index("Class")
        _bar_chart(probability_frame, ["#087e79"])
        st.caption(
            f"Rule: Routine < {lower:.0%}; Consider follow-up ≥ {lower:.0%}; "
            f"Priority human review ≥ {upper:.0%}. Thresholds are adjustable and provisional. "
            "The classifier's probabilities have not been calibrated or validated for Kenya."
        )
        st.warning(
            "A prediction does not prove a student will leave, and no support, funding, admission "
            "or disciplinary decision should be automated. Review relevant evidence and ask "
            "the student."
        )
        importance_path = OUTPUT_DIR / "permutation_importance.csv"
        if importance_path.is_file():
            st.caption(
                "Global model analysis only; it is not a personalised explanation for this profile."
            )
            st.dataframe(pd.read_csv(importance_path).head(10), use_container_width=True)
        if st.button("Consider recording a human follow-up", key="create-case-from-assessment"):
            st.session_state.case_draft = {
                "reference": "CASE-DEMO-01",
                "reason": "Review synthetic model assessment and discuss available support",
                "category": SupportCategory.ACADEMIC.value,
            }
            st.session_state.navigate_to = "Support Cases"
            st.rerun()


def _render_cohort_analysis() -> None:
    _hero(
        "Cohort analysis",
        "Explore observed associations in the source dataset. Use the results to form questions, "
        "not causal explanations about Kenyan students.",
        "AGGREGATE ANALYSIS • SOURCE COHORT",
    )
    frame, error = _get_dataset()
    if frame is None:
        st.info("Prepare the official dataset from **Data & Administration** before analysis.")
        if error:
            st.code(error)
        return
    _show_domain_notice()
    supported = [
        "Course",
        "Gender",
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
        "Age band",
    ]
    selected = st.selectbox("Compare observed outcomes by", supported)
    analysis = frame.copy()
    if selected == "Age band":
        analysis["Analysis group"] = make_age_band(frame).astype(str)
    else:
        analysis["Analysis group"] = analysis[selected].astype(str)
    levels = sorted(analysis["Analysis group"].unique().tolist())
    chosen = st.multiselect(
        "Optional group filter",
        levels,
        format_func=lambda value: f"Code {value}" if value.isdigit() else value,
    )
    if chosen:
        analysis = analysis.loc[analysis["Analysis group"].isin(chosen)]
    counts = pd.crosstab(analysis["Analysis group"], analysis["Target"])
    st.subheader("Observed outcomes by selected group")
    _bar_chart(counts, ["#087e79", "#e0a13a", "#4776a8"])
    proportions = counts.div(counts.sum(axis=1), axis=0)
    st.dataframe(counts.join(proportions.add_suffix(" proportion")), use_container_width=True)
    st.caption(
        "Groups with small counts can produce unstable proportions; display counts beside rates."
    )

    st.subheader("Available financial and academic indicators")
    financial = frame.groupby("Target")[
        ["Debtor", "Tuition fees up to date", "Scholarship holder"]
    ].mean()
    academic = frame.groupby("Target")[
        [
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
        ]
    ].mean()
    left, right = st.columns(2)
    with left:
        st.markdown("**Share of records with binary code = 1**")
        st.dataframe(financial.style.format("{:.1%}"), use_container_width=True)
    with right:
        st.markdown("**Mean academic measures**")
        st.dataframe(academic.style.format("{:.2f}"), use_container_width=True)
    st.info(
        "The UCI columns contain debtor, fee-status and scholarship codes, but not Kenyan HELB/HEF "
        "or M-Pesa variables. These are descriptive associations in a Portuguese dataset, "
        "not causal effects."
    )


def _case_table(cases: list[SupportCase]) -> pd.DataFrame:
    rows = [
        {
            "Reference": case.reference,
            "Reviewed": case.reviewed_on,
            "Category": case.category,
            "Reason": case.reason,
            "Next action": case.next_action,
            "Responsible role": case.responsible_role,
            "Follow-up": case.follow_up_date,
            "Status": case.status,
        }
        for case in cases
    ]
    return pd.DataFrame(rows)


def _date_input_value(value: object, label: str) -> date:
    if not isinstance(value, date):
        raise ValueError(f"{label} must be a single calendar date")
    return value


def _render_support_cases() -> None:
    _hero(
        "Support case register",
        "Record a pseudonymous, staff-reviewed follow-up and track its next action. A model flag "
        "never creates a case automatically.",
        "HUMAN-OWNED FOLLOW-UP • LOCAL DEMONSTRATION STORAGE",
    )
    st.warning(
        "Demo storage only: records are saved in a local JSON file under `outputs/`; there is no "
        "login, encryption or production access control. Use pseudonymous references and do not "
        "store health, financial credentials or other sensitive personal details."
    )
    try:
        cases = load_cases(CASE_PATH)
    except (ValueError, OSError) as exc:
        st.error(f"Support-case storage needs attention: {exc}")
        return
    create_tab, update_tab, list_tab = st.tabs(["Create a case", "Update a case", "Browse cases"])
    with create_tab:
        draft = st.session_state.get("case_draft", {})
        with st.form("create-support-case"):
            ref = st.text_input(
                "Pseudonymous case reference", value=draft.get("reference", "CASE-DEMO-01")
            )
            reviewed = st.date_input("Date reviewed", value=date.today())
            category = st.selectbox(
                "Concern category",
                [item.value for item in SupportCategory],
                index=max(
                    0,
                    [item.value for item in SupportCategory].index(
                        draft.get("category", SupportCategory.ACADEMIC.value)
                    ),
                ),
            )
            reason = st.text_input(
                "Reason for follow-up", value=draft.get("reason", ""), max_chars=300
            )
            next_action = st.text_input(
                "Agreed next action", "Offer a student-support appointment", max_chars=300
            )
            role = st.selectbox(
                "Responsible role",
                [
                    "Student affairs",
                    "Academic adviser",
                    "Department coordinator",
                    "Finance support",
                    "Student-support staff",
                ],
            )
            follow_date = st.date_input("Follow-up date", value=date.today() + timedelta(days=7))
            note = st.text_area("Brief non-sensitive administrative note (optional)", max_chars=500)
            submitted = st.form_submit_button("Save reviewed case", type="primary")
        if submitted:
            try:
                if any(case.reference == ref.strip().upper() for case in cases):
                    raise ValueError(
                        "That case reference already exists; use Update a case instead."
                    )
                if category is None or role is None:
                    raise ValueError("Select both a case category and responsible role")
                case = SupportCase.create(
                    ref,
                    _date_input_value(reviewed, "Reviewed date"),
                    reason,
                    category,
                    next_action,
                    role,
                    _date_input_value(follow_date, "Follow-up date"),
                    administrative_note=note,
                )
                upsert_case(CASE_PATH, case)
                st.session_state.pop("case_draft", None)
                st.success(f"Saved {case.reference} to the local demonstration register.")
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    with update_tab:
        if not cases:
            st.info("No cases are recorded yet.")
        else:
            refs = [case.reference for case in cases]
            chosen_ref = st.selectbox("Select case", refs)
            current = next(case for case in cases if case.reference == chosen_ref)
            with st.form("update-support-case"):
                status = st.selectbox(
                    "Status",
                    [item.value for item in CaseStatus],
                    index=[item.value for item in CaseStatus].index(current.status),
                )
                action = st.text_input("Updated next action", value=current.next_action)
                role = st.text_input("Responsible role", value=current.responsible_role)
                follow_date = st.date_input(
                    "Updated follow-up date", value=date.fromisoformat(current.follow_up_date)
                )
                reason = st.text_input("Updated reason", value=current.reason)
                note = st.text_area(
                    "Updated non-sensitive note", value=current.administrative_note, max_chars=500
                )
                save_update = st.form_submit_button("Save case update", type="primary")
            if save_update:
                try:
                    if status is None:
                        raise ValueError("Select a case status")
                    updated = SupportCase.create(
                        current.reference,
                        date.fromisoformat(current.reviewed_on),
                        reason,
                        current.category,
                        action,
                        role,
                        _date_input_value(follow_date, "Follow-up date"),
                        status,
                        note,
                    )
                    upsert_case(CASE_PATH, updated)
                    st.success(f"Updated {current.reference}.")
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
    with list_tab:
        if not cases:
            st.info("Create a pseudonymous demonstration case to see it here.")
        else:
            statuses = st.multiselect("Filter by status", [item.value for item in CaseStatus])
            categories = st.multiselect(
                "Filter by category", [item.value for item in SupportCategory]
            )
            filtered = [
                case
                for case in cases
                if (not statuses or case.status in statuses)
                and (not categories or case.category in categories)
            ]
            st.dataframe(_case_table(filtered), use_container_width=True, hide_index=True)
            st.caption(f"{len(filtered)} of {len(cases)} local cases shown.")


def _render_followup_monitor() -> None:
    _hero(
        "Follow-up monitoring",
        "Track due dates, open work and completed actions from persisted support cases. "
        "These are workflow counts, not intervention-effectiveness results.",
        "CASE WORKFLOW • NO CAUSAL CLAIMS",
    )
    try:
        cases = load_cases(CASE_PATH)
    except (ValueError, OSError) as exc:
        st.error(f"Support-case storage needs attention: {exc}")
        return
    today = date.today()
    overdue = overdue_cases(cases, today)
    open_cases = [case for case in cases if case.status != CaseStatus.CLOSED.value]
    due_today = [case for case in open_cases if date.fromisoformat(case.follow_up_date) == today]
    closed = [case for case in cases if case.status == CaseStatus.CLOSED.value]
    cards = st.columns(4)
    cards[0].metric("Open / active", len(open_cases))
    cards[1].metric("Overdue follow-ups", len(overdue))
    cards[2].metric("Due today", len(due_today))
    cards[3].metric("Closed cases", len(closed))
    if not cases:
        st.info("No persisted cases yet. Create a pseudonymous case on the Support Cases page.")
        return
    left, right = st.columns(2)
    with left:
        st.subheader("Cases by status")
        _bar_chart(pd.Series([case.status for case in cases]).value_counts(), ["#087e79"])
    with right:
        st.subheader("Cases by support category")
        _bar_chart(pd.Series([case.category for case in cases]).value_counts(), ["#4776a8"])
    st.subheader("Awaiting follow-up")
    if overdue:
        st.error("Overdue cases require staff review.")
        st.dataframe(_case_table(overdue), use_container_width=True, hide_index=True)
    elif open_cases:
        st.dataframe(_case_table(open_cases), use_container_width=True, hide_index=True)
    else:
        st.success("No open follow-ups recorded.")
    st.subheader("Completed actions")
    st.dataframe(_case_table(closed), use_container_width=True, hide_index=True)
    st.caption("No claim is made that recorded follow-up actions changed student outcomes.")


def _render_model_transparency() -> None:
    _hero(
        "Model transparency",
        "Inspect training design, held-out evaluation and global feature analysis, then consider "
        "what the evidence cannot support.",
        "UCI DATASET 697 • PORTUGUESE SOURCE CONTEXT",
    )
    _show_domain_notice()
    metrics = _read_artifact_json("metrics.json")
    config = _read_artifact_json("run_config.json")
    if not metrics or not config:
        st.info("No saved evaluation artifacts. Prepare the model from **Data & Administration**.")
        return
    columns = st.columns(4)
    columns[0].metric("Selected model", config["selected_model"])
    columns[1].metric("Held-out macro-F1", f"{metrics['test_macro_f1']:.3f}")
    columns[2].metric("Held-out accuracy", f"{metrics['test_accuracy']:.3f}")
    columns[3].metric("Held-out records", f"{metrics['test_rows']:,}")
    st.write(
        f"The workflow uses a stratified {float(config['test_size']):.0%} held-out test split, "
        f"{config['cv_folds']}-fold stratified cross-validation on training data, seed "
        f"{config['random_seed']}, and selects by training macro-F1. The selected model is a real "
        "fitted scikit-learn Pipeline saved locally for subsequent inference."
    )
    report = pd.DataFrame(metrics["classification_report"]).T
    st.subheader("Per-class test performance")
    st.dataframe(report.loc[["Dropout", "Enrolled", "Graduate"]], use_container_width=True)
    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    if comparison_path.is_file():
        st.subheader("Training-only cross-validation comparison")
        st.dataframe(pd.read_csv(comparison_path), use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    matrix_path = OUTPUT_DIR / "confusion_matrix.png"
    importance_path = OUTPUT_DIR / "permutation_importance.png"
    if matrix_path.is_file():
        left.image(
            str(matrix_path), caption="Held-out confusion matrix: actual rows, predicted columns"
        )
    if importance_path.is_file():
        right.image(
            str(importance_path),
            caption="Global permutation importance; predictive reliance, not causation",
        )
    st.subheader("Dataset and domain shift")
    st.write(
        "The UCI 697 file has 4,424 anonymized records from a Portuguese higher-education context, "
        "with 36 numeric-coded or measured predictors and observed outcomes Dropout, Enrolled and "
        "Graduate. Differences in student demographics, funding systems, assessment, course design "
        "and support practice may change performance in Kenya."
    )
    st.caption(
        "Licence: CC BY 4.0. The dataset contains fee-debtor and tuition-status codes but no HELB, "
        "mobile-money, mental-health, hostel or Kenyan institutional case-management fields."
    )
    st.write(
        "Source: [UCI Machine Learning Repository, dataset 697](https://doi.org/10.24432/C5MC89)."
    )


def _render_data_admin() -> None:
    _hero(
        "Data and administration",
        "Validate the official source file, inspect the actual schema, and prepare the "
        "fitted model "
        "artifacts required by the analytical pages.",
        "DATA QUALITY • PIPELINE STATUS • DEMONSTRATION MODE",
    )
    model, _, config = _get_saved_model()
    frame, error = _get_dataset()
    status_left, status_right = st.columns(2)
    status_left.metric("Dataset status", "Validated" if frame is not None else "Not loaded")
    status_right.metric(
        "Model status", "Saved artifact ready" if model is not None else "Not trained"
    )
    if frame is None:
        st.info("The source CSV is expected at `data/raw/data.csv`.")
        if error:
            st.code(error)
    else:
        summary = dataset_quality_summary(frame, DATA_PATH)
        st.subheader("Validated source facts")
        st.json(summary)
        st.success("Schema validation completed successfully using the project's validator.")
        schema = pd.DataFrame(
            {
                "Feature": frame.columns,
                "Type": [str(frame[name].dtype) for name in frame.columns],
                "Distinct values": [int(frame[name].nunique()) for name in frame.columns],
                "Missing values": [int(frame[name].isna().sum()) for name in frame.columns],
            }
        )
        st.dataframe(schema, use_container_width=True, hide_index=True)
        st.caption("Source numeric codes are preserved; category values are not ordinal scores.")
    if config:
        st.caption(
            f"Saved run: {config.get('selected_model')} • seed {config.get('random_seed')} • "
            f"SHA-256 {str(config.get('data_sha256', ''))[:16]}…"
        )
    st.warning(
        "This prototype has no authentication or enterprise security. Case files are local "
        "plain JSON. Use fictional/pseudonymous demonstrations only; deployment needs "
        "institutional governance, "
        "privacy/security review and Kenyan validation."
    )
    if st.button("Prepare dataset and train the model", type="primary"):
        try:
            with st.spinner(
                "Validating the source and fitting the three-model pipeline. "
                "This may take a few minutes…"
            ):
                run_pipeline(
                    PipelineConfig(data_path=DATA_PATH, output_dir=OUTPUT_DIR),
                    download=not DATA_PATH.exists(),
                )
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Pipeline completed; model and evaluation artifacts are saved locally.")
            st.rerun()
        except Exception as exc:
            st.error(f"Pipeline could not complete: {exc}")


def _render_student_support_tools() -> None:
    st.title("Student support tools")
    st.caption("Prepared signposting for student use; not monitored or clinical care.")
    chat_tab, study_tab, resources_tab = st.tabs(
        ["Intent-based chat", "Study reset", "Support directory"]
    )
    with chat_tab:
        _render_chat()
    with study_tab:
        _render_study_reset()
    with resources_tab:
        _render_resources()


def main() -> None:
    """Run the student-support hub and retrospective project-insights interface."""
    st.set_page_config(page_title="Karibu Campus | Student Support", page_icon="🌿", layout="wide")
    _style_app()
    st.sidebar.markdown("<div style='font-size:2rem'>🌿</div>", unsafe_allow_html=True)
    st.sidebar.title("Karibu Campus")
    st.sidebar.caption("Student support • Kenya")
    st.sidebar.divider()
    pages = PAGES
    requested = st.query_params.get("view")
    default_page = (
        requested
        if requested in pages
        else "Support Chat"
        if st.query_params.get("demo") == "academic"
        else "Home"
    )
    navigate_to = st.session_state.pop("navigate_to", None)
    if navigate_to in pages:
        st.session_state.current_page = navigate_to
    if "current_page" not in st.session_state:
        st.session_state.current_page = default_page
    page = st.sidebar.radio(
        "EXPLORE",
        pages,
        key="current_page",
        format_func=lambda item: {
            "Home": "⌂   University Overview",
            "Individual Assessment": "◉   Individual Assessment",
            "Cohort Analysis": "▦   Cohort Analysis",
            "Support Cases": "♡   Support Cases",
            "Follow-up Monitoring": "✓   Follow-up Monitoring",
            "Model Transparency": "▤   Model Transparency",
            "Data & Administration": "⚙   Data & Administration",
            "Student Support Home": "✚   Student Support Home",
            "Support Chat": "◌   Support Chat",
            "Study Reset": "▧   Study Reset",
            "Support Directory": "♧   Support Directory",
        }[item],
    )
    st.sidebar.divider()
    st.sidebar.markdown("**A small step is still progress.**")
    st.sidebar.caption("Prepared guidance, not clinical care.")
    if page == "Home":
        _render_home()
    elif page == "Individual Assessment":
        _render_assessment()
    elif page == "Cohort Analysis":
        _render_cohort_analysis()
    elif page == "Support Cases":
        _render_support_cases()
    elif page == "Follow-up Monitoring":
        _render_followup_monitor()
    elif page == "Model Transparency":
        _render_model_transparency()
    elif page == "Data & Administration":
        _render_data_admin()
    elif page == "Student Support Home":
        _render_student_support_home()
    elif page == "Support Chat":
        _render_chat()
    elif page == "Study Reset":
        _render_study_reset()
    elif page == "Support Directory":
        _render_resources()
    else:
        _render_insights()


if __name__ == "__main__":
    main()

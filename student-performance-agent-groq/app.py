from __future__ import annotations

import os
import sys
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


try:
    STREAMLIT_GROQ_KEY = str(
        st.secrets.get("GROQ_API_KEY", "")
    )
except Exception:
    STREAMLIT_GROQ_KEY = ""


API_READY = bool(
    os.getenv("GROQ_API_KEY")
    or os.getenv("LLM_API_KEY")
    or STREAMLIT_GROQ_KEY
)


ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ============================================================
# IMPORTS
# ============================================================

from agents import (  # noqa: E402
    analyze_and_recommend_student,
    analyze_class,
    analyze_student,
    answer_question,
    generate_personalized_recommendations,
    make_report,
    quality_check,
)

from data.calculations import (  # noqa: E402
    calculate_metrics,
    student_snapshot,
)

from data.validation import (  # noqa: E402
    read_upload,
    validate_and_clean,
)

from storage.database import (  # noqa: E402
    init_db,
    recent_reports,
    save_report,
    save_upload,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PerformanceMind · Student AI",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DATABASE
# ============================================================

init_db()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800'
    '&family=DM+Mono:wght@300;400;500'
    '&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300'
    '&display=swap'
);

html,
body,
[class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8e4dc;
}

.stApp {
    background: #0a0a0f;

    background-image:
        radial-gradient(
            ellipse 80% 50% at 20% -10%,
            rgba(255,140,50,.12) 0%,
            transparent 60%
        ),
        radial-gradient(
            ellipse 60% 40% at 80% 110%,
            rgba(255,80,30,.08) 0%,
            transparent 55%
        );
}

#MainMenu,
footer,
header {
    visibility: hidden;
}

.block-container {
    padding: 2rem 3rem 4rem;
    max-width: 1250px;
}

.hero {
    text-align: center;
    padding: 2.6rem 0 1.8rem;
}

.hero-eyebrow,
.mono {
    font-family: 'DM Mono', monospace;
    font-size: .7rem;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: #ff8c32;
}

.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800;
    line-height: 1;
    letter-spacing: -.04em;
    color: #f0ebe0;
    margin: .7rem 0 1rem;
}

.hero h1 span {
    color: #ff8c32;
}

.hero-sub {
    color: #a09890;
    max-width: 650px;
    margin: auto;
    line-height: 1.65;
}

.divider {
    height: 1px;
    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(255,140,50,.35),
            transparent
        );
    margin: 1rem 0 2rem;
}

.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f0ebe0;
    margin: 1.2rem 0 1rem;
}

.input-card {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,140,50,.15);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
}

.stFileUploader {
    border: 1px dashed rgba(255,140,50,.35);
    border-radius: 12px;
    padding: .5rem;
    background: rgba(255,140,50,.03);
}

.stButton > button {
    background:
        linear-gradient(
            135deg,
            #ff8c32 0%,
            #ff5a1a 100%
        ) !important;

    color: #0a0a0f !important;

    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;

    border: none !important;
    border-radius: 10px !important;

    box-shadow:
        0 4px 20px
        rgba(255,140,50,.25) !important;
}

.step-card {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: .75rem;
    position: relative;
    overflow: hidden;
}

.step-card.active {
    border-color: rgba(255,140,50,.5);
    background: rgba(255,140,50,.05);
}

.step-card.done {
    border-color: rgba(80,200,120,.35);
    background: rgba(80,200,120,.035);
}

.step-card::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    background: rgba(255,255,255,.06);
}

.step-card.active::before {
    background: #ff8c32;
}

.step-card.done::before {
    background: #50c878;
}

.step-header {
    display: flex;
    align-items: center;
    gap: .65rem;
}

.step-num {
    font-family: 'DM Mono', monospace;
    font-size: .66rem;
    color: #ff8c32;
    letter-spacing: .12em;
}

.step-title {
    font-family: 'Syne', sans-serif;
    font-size: .9rem;
    font-weight: 700;
    color: #f0ebe0;
}

.step-status {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: .62rem;
    letter-spacing: .1em;
}

.waiting {
    color: #555;
}

.running {
    color: #ff8c32;
}

.done {
    color: #50c878;
}

.step-desc {
    color: #706860;
    font-size: .76rem;
    margin: .35rem 0 0 2rem;
}

.result-panel {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,140,50,.2);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.2rem;
}

.result-label {
    font-family: 'DM Mono', monospace;
    color: #ff8c32;
    font-size: .7rem;
    letter-spacing: .2em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(255,140,50,.15);
    padding-bottom: .7rem;
    margin-bottom: 1rem;
}

.small-note {
    color: #706860;
    font-size: .78rem;
    line-height: 1.5;
}

.api-ok {
    color: #50c878;
    font-family: 'DM Mono', monospace;
    font-size: .7rem;
}

.api-warning {
    color: #ff8c32;
    font-family: 'DM Mono', monospace;
    font-size: .7rem;
}

@media (max-width: 700px) {

    .block-container {
        padding: 1rem .75rem 2rem;
    }

    .hero {
        padding: 1rem 0 1.1rem;
    }

    .hero h1 {
        font-size: 2.55rem;
    }

    .hero-sub {
        font-size: .88rem;
        line-height: 1.45;
    }

    .section-heading {
        font-size: 1.08rem;
        margin: .8rem 0 .65rem;
    }

    .input-card {
        padding: 1rem;
        border-radius: 12px;
    }

    .step-card {
        padding: .72rem .8rem;
        margin-bottom: .55rem;
    }

    .step-title {
        font-size: .78rem;
    }

    .step-desc {
        font-size: .67rem;
        margin-left: 1.65rem;
    }

    .step-status {
        font-size: .52rem;
    }

    .stButton > button {
        min-height: 2.7rem !important;
        font-size: .78rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: .62rem;
    }

    .stDataFrame {
        font-size: .72rem;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PIPELINE
# ============================================================

STEPS = [
    (
        "01",
        "Intake Agent",
        "Validates and normalizes the uploaded marks",
    ),
    (
        "02",
        "Calculation Engine",
        "Computes grades, averages, and trends",
    ),
    (
        "03",
        "Performance Agent",
        "Finds strengths and weaknesses",
    ),
    (
        "04",
        "Recommendation Agent",
        "Builds an evidence-based action plan",
    ),
    (
        "05",
        "Report Agent",
        "Writes the detailed student or class report",
    ),
    (
        "06",
        "Quality Agent",
        "Checks the report against the numbers",
    ),
    (
        "07",
        "Storage Agent",
        "Stores the analysis for future questions",
    ),
]


def render_pipeline(
    states: dict[str, str],
) -> str:

    blocks = []

    for i, (num, title, desc) in enumerate(STEPS):

        state = states.get(
            str(i),
            "waiting",
        )

        label = {
            "waiting": "WAITING",
            "running": "● RUNNING",
            "done": "✓ DONE",
        }.get(
            state,
            "WAITING",
        )

        card_class = ""

        if state == "running":
            card_class = "active"

        elif state == "done":
            card_class = "done"

        blocks.append(
            f"""
            <div class="step-card {card_class}">

                <div class="step-header">

                    <span class="step-num">
                        {num}
                    </span>

                    <span class="step-title">
                        {title}
                    </span>

                    <span class="step-status {state}">
                        {label}
                    </span>

                </div>

                <div class="step-desc">
                    {desc}
                </div>

            </div>
            """
        )

    return "".join(blocks)


# ============================================================
# SESSION DEFAULTS
# ============================================================

def set_defaults() -> None:

    defaults = {

        "metrics": None,

        "class_summary": None,

        "upload_id": None,

        "upload_name": None,

        "errors": [],

        "warnings": [],

        "states": {
            str(i): "waiting"
            for i in range(7)
        },

        "stage_outputs": {},

        "pending_upload_name": None,

        "last_report": None,

        "last_quality": None,

        "last_analysis": None,

    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


set_defaults()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-eyebrow">
            Multi-Agent Student Intelligence
        </div>

        <h1>
            Performance<span>Mind</span>
        </h1>

        <p class="hero-sub">
            Seven specialized agents collaborate —
            validating, calculating, analyzing,
            recommending, reporting, checking,
            and storing student performance insights.
        </p>

    </div>

    <div class="divider"></div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API STATUS
# ============================================================

if API_READY:

    st.markdown(
        '<div class="api-ok">● GROQ API READY</div>',
        unsafe_allow_html=True,
    )

else:

    st.warning(
        "Groq API key is not configured. "
        "Upload and calculation will still work, "
        "but AI analysis requires GROQ_API_KEY."
    )


# ============================================================
# LAYOUT
# ============================================================

left, spacer, right = st.columns(
    [5, 0.35, 4]
)


# ============================================================
# LEFT: UPLOAD
# ============================================================

with left:

    st.markdown(
        '<div class="section-heading">'
        'Upload performance data'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="input-card">',
        unsafe_allow_html=True,
    )

    upload = st.file_uploader(
        "CSV or Excel marks sheet",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
        help=(
            "Required columns: "
            "student_id, subject, "
            "marks_obtained, max_marks"
        ),
    )

    pass_mark = st.number_input(
        "Pass mark (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
    )

    low_cost_mode = st.checkbox(
        "Low-cost mode",
        value=(
            os.getenv(
                "LOW_COST_MODE",
                "true",
            ).lower()
            == "true"
        ),
        help=(
            "Combines Performance and "
            "Recommendation agents into one "
            "compact Groq call."
        ),
    )

    run_quality = st.checkbox(
        "Run optional quality-check agent",
        value=(
            os.getenv(
                "RUN_QUALITY_CHECK",
                "false",
            ).lower()
            == "true"
        ),
        help=(
            "Adds one extra Groq call. "
            "Keep off to reduce API usage."
        ),
    )

    submit_upload = st.button(
        "Submit & validate file",
        type="primary",
        use_container_width=True,
        disabled=upload is None,
    )

    st.markdown(
        """
        <div class="small-note">
        Required columns:
        student_id, subject,
        marks_obtained, max_marks.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# RIGHT: PIPELINE
# ============================================================

with right:

    st.markdown(
        '<div class="section-heading">'
        'Agent pipeline'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="small-note">
        Groq guard: sequential requests ·
        compact JSON · retry protection
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline_slot = st.empty()

    pipeline_slot.markdown(
        render_pipeline(
            st.session_state.states
        ),
        unsafe_allow_html=True,
    )


# ============================================================
# PROCESS UPLOAD
# ============================================================

if upload is not None:

    if (
        st.session_state.pending_upload_name
        != upload.name
    ):

        st.session_state.pending_upload_name = (
            upload.name
        )

        st.session_state.metrics = None
        st.session_state.class_summary = None
        st.session_state.upload_id = None

        st.session_state.states = {
            str(i): "waiting"
            for i in range(7)
        }

        st.session_state.stage_outputs = {}

        st.session_state.last_report = None
        st.session_state.last_quality = None
        st.session_state.last_analysis = None

        st.info(
            f"File selected: {upload.name}. "
            "Click 'Submit & validate file' to begin."
        )

    if submit_upload:

        try:

            raw = read_upload(
                upload.name,
                upload.getvalue(),
            )

            clean, errors, warnings = (
                validate_and_clean(raw)
            )

            if errors:

                st.session_state.errors = errors

                st.error(
                    " | ".join(errors)
                )

            else:

                metrics, summary = (
                    calculate_metrics(
                        clean,
                        pass_mark,
                    )
                )

                st.session_state.upload_name = (
                    upload.name
                )

                st.session_state.metrics = metrics

                st.session_state.class_summary = (
                    summary
                )

                st.session_state.warnings = warnings

                st.session_state.upload_id = (
                    save_upload(
                        upload.name,
                        metrics,
                        warnings,
                    )
                )

                st.session_state.states = {
                    str(i): "waiting"
                    for i in range(7)
                }

                st.session_state.states["0"] = "done"
                st.session_state.states["1"] = "done"

                st.session_state.stage_outputs = {

                    "Intake Agent": {
                        "file": upload.name,
                        "rows_received": len(raw),
                        "rows_accepted": len(clean),
                        "warnings": warnings,
                    },

                    "Calculation Engine": summary,
                }

                pipeline_slot.markdown(
                    render_pipeline(
                        st.session_state.states
                    ),
                    unsafe_allow_html=True,
                )

                st.success(
                    "File validated and calculations completed."
                )

        except Exception as exc:

            st.error(
                f"File processing failed: {exc}"
            )

else:

    if st.session_state.metrics is None:

        st.markdown(
            """
            <div class="small-note">
            Upload a CSV or Excel file to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# STOP IF NO DATA
# ============================================================

metrics = st.session_state.metrics

if metrics is None:
    st.stop()


# ============================================================
# WARNINGS
# ============================================================

if st.session_state.warnings:

    with st.expander(
        "Data quality notes"
    ):

        for warning in st.session_state.warnings:

            st.warning(warning)


# ============================================================
# CLASS SUMMARY
# ============================================================

st.divider()

summary = st.session_state.class_summary

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "STUDENTS",
    summary["student_count"],
)

m2.metric(
    "RECORDS",
    summary["record_count"],
)

m3.metric(
    "CLASS AVG",
    f'{summary["overall_average"]}%',
)

m4.metric(
    "PASS RATE",
    f'{summary["pass_rate"]}%',
)


# ============================================================
# ANALYSIS MODE
# ============================================================

mode = st.radio(
    "Analysis scope",
    [
        "Individual student",
        "Whole class",
    ],
    horizontal=True,
)


# ============================================================
# INDIVIDUAL STUDENT
# ============================================================

if mode == "Individual student":

    students = (
        metrics[
            [
                "student_id",
                "student_name",
            ]
        ]
        .drop_duplicates()
        .sort_values("student_id")
    )

    labels = {}

    for row in students.itertuples():

        student_id = str(
            row.student_id
        )

        student_name = str(
            row.student_name
        )

        label = (
            f"{student_id} — "
            f"{student_name}"
        ).strip(" —")

        labels[label] = student_id

    if not labels:

        st.error(
            "No students were found."
        )

        st.stop()

    selected_label = st.selectbox(
        "Choose a student",
        list(labels.keys()),
    )

    selected_id = labels[
        selected_label
    ]

    snapshot = student_snapshot(
        metrics,
        selected_id,
        pass_mark,
    )

    st.markdown(
        '<div class="section-heading">'
        'Individual performance'
        '</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        pd.DataFrame(
            snapshot["subjects"]
        ),
        use_container_width=True,
        hide_index=True,
    )

    col_a, col_b = st.columns(2)


    # --------------------------------------------------------
    # ASK STUDENT
    # --------------------------------------------------------

    with col_a:

        ask = st.text_input(
            "Ask about this student",
            placeholder=(
                "Why is Physics a priority?"
            ),
        )

        if ask and st.button(
            "Ask Groq",
            key="ask_student",
        ):

            if not API_READY:

                st.error(
                    "GROQ_API_KEY is not configured."
                )

            else:

                try:

                    with st.spinner(
                        "Groq is analyzing the question..."
                    ):

                        answer = answer_question(
                            ask,
                            snapshot,
                        )

                    st.info(answer)

                except Exception as exc:

                    st.error(
                        f"Groq error: {exc}"
                    )


    # --------------------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------------------

    with col_b:

        run = st.button(
            "Run individual agent pipeline",
            type="primary",
            use_container_width=True,
        )


    if run:

        if not API_READY:

            st.error(
                "GROQ_API_KEY is not configured. "
                "Add it to Streamlit Cloud Secrets."
            )

            st.stop()


        try:

            # ------------------------------------------------
            # STEP 1 & 2
            # ------------------------------------------------

            for index in [0, 1]:

                st.session_state.states[
                    str(index)
                ] = "running"

                pipeline_slot.markdown(
                    render_pipeline(
                        st.session_state.states
                    ),
                    unsafe_allow_html=True,
                )

                time.sleep(0.2)

                st.session_state.states[
                    str(index)
                ] = "done"


            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            # ------------------------------------------------
            # PERFORMANCE
            # ------------------------------------------------

            st.session_state.states["2"] = (
                "running"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            if low_cost_mode:

                with st.spinner(
                    "Groq is analyzing performance "
                    "and recommendations..."
                ):

                    package = (
                        analyze_and_recommend_student(
                            snapshot,
                            summary,
                        )
                    )

                if not isinstance(
                    package,
                    dict,
                ):
                    raise RuntimeError(
                        "AI returned an unexpected "
                        "response format."
                    )

                analysis = package.get(
                    "analysis",
                    {},
                )

                recommendations = package.get(
                    "recommendations",
                    {},
                )

            else:

                with st.spinner(
                    "Groq Performance Agent "
                    "is evaluating evidence..."
                ):

                    analysis = analyze_student(
                        snapshot,
                        summary,
                    )

                recommendations = {}


            st.session_state.stage_outputs[
                "Performance Agent"
            ] = analysis

            st.session_state.states["2"] = (
                "done"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            # ------------------------------------------------
            # RECOMMENDATIONS
            # ------------------------------------------------

            st.session_state.states["3"] = (
                "running"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            if not low_cost_mode:

                with st.spinner(
                    "Groq Recommendation Agent "
                    "is creating next steps..."
                ):

                    recommendations = (
                        generate_personalized_recommendations(
                            snapshot,
                            analysis,
                        )
                    )


            st.session_state.stage_outputs[
                "Recommendation Agent"
            ] = recommendations

            st.session_state.states["3"] = (
                "done"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            # ------------------------------------------------
            # REPORT
            # ------------------------------------------------

            st.session_state.states["4"] = (
                "running"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            with st.spinner(
                "Groq Report Agent is "
                "writing the report..."
            ):

                report = make_report(
                    f"Individual student {selected_id}",
                    snapshot,
                    analysis,
                    recommendations,
                )


            st.session_state.stage_outputs[
                "Report Agent"
            ] = report

            st.session_state.states["4"] = (
                "done"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            # ------------------------------------------------
            # QUALITY
            # ------------------------------------------------

            st.session_state.states["5"] = (
                "running"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            if run_quality:

                with st.spinner(
                    "Groq Quality Agent is "
                    "checking the report..."
                ):

                    quality = quality_check(
                        report,
                        snapshot,
                    )

            else:

                quality = {
                    "passed": "not_run",
                    "score": None,
                    "note": (
                        "Quality check skipped."
                    ),
                }


            st.session_state.stage_outputs[
                "Quality Agent"
            ] = quality

            st.session_state.states["5"] = (
                "done"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            # ------------------------------------------------
            # STORAGE
            # ------------------------------------------------

            st.session_state.states["6"] = (
                "running"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            save_report(
                st.session_state.upload_id,
                "individual",
                selected_id,
                report,
                analysis,
                quality,
            )


            st.session_state.stage_outputs[
                "Storage Agent"
            ] = {
                "saved": True,
                "upload_id":
                    st.session_state.upload_id,
                "scope": "individual",
                "student_id": selected_id,
            }


            st.session_state.states["6"] = (
                "done"
            )


            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            st.session_state.last_report = (
                report
            )

            st.session_state.last_quality = (
                quality
            )

            st.session_state.last_analysis = (
                analysis
            )


            st.success(
                "Individual analysis completed successfully."
            )


        except Exception as exc:

            st.error(
                "AI pipeline failed."
            )

            st.exception(exc)


# ============================================================
# WHOLE CLASS
# ============================================================

else:

    st.markdown(
        '<div class="section-heading">'
        'Whole-class performance'
        '</div>',
        unsafe_allow_html=True,
    )


    subject_table = pd.DataFrame(
        [
            {
                "subject": subject,
                "class average": average,
            }
            for subject, average
            in summary[
                "subject_averages"
            ].items()
        ]
    )


    st.dataframe(
        subject_table,
        use_container_width=True,
        hide_index=True,
    )


    st.write(
        "Grade distribution",
        summary["grade_distribution"],
    )


    # --------------------------------------------------------
    # ASK CLASS
    # --------------------------------------------------------

    ask = st.text_input(
        "Ask about this class",
        placeholder=(
            "Which subjects need the most support?"
        ),
    )


    if ask and st.button(
        "Ask Groq",
        key="ask_class",
    ):

        if not API_READY:

            st.error(
                "GROQ_API_KEY is not configured."
            )

        else:

            try:

                with st.spinner(
                    "Groq is analyzing the class..."
                ):

                    answer = answer_question(
                        ask,
                        {
                            "summary": summary,
                            "records": metrics.to_dict(
                                "records"
                            ),
                        },
                    )

                st.info(answer)

            except Exception as exc:

                st.error(
                    f"Groq error: {exc}"
                )


    # --------------------------------------------------------
    # RUN CLASS PIPELINE
    # --------------------------------------------------------

    if st.button(
        "Run whole-class agent pipeline",
        type="primary",
    ):

        if not API_READY:

            st.error(
                "GROQ_API_KEY is not configured."
            )

            st.stop()


        try:

            # ------------------------------------------------
            # INTAKE + CALCULATION
            # ------------------------------------------------

            for index in [0, 1]:

                st.session_state.states[
                    str(index)
                ] = "running"

                pipeline_slot.markdown(
                    render_pipeline(
                        st.session_state.states
                    ),
                    unsafe_allow_html=True,
                )

                time.sleep(0.2)

                st.session_state.states[
                    str(index)
                ] = "done"


            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            # ------------------------------------------------
            # PERFORMANCE
            # ------------------------------------------------

            st.session_state.states["2"] = (
                "running"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            with st.spinner(
                "Groq Performance Agent "
                "is finding class patterns..."
            ):

                analysis = analyze_class(
                    metrics.to_dict("records"),
                    summary,
                )


            st.session_state.stage_outputs[
                "Performance Agent"
            ] = analysis

            st.session_state.states["2"] = (
                "done"
            )


            # ------------------------------------------------
            # RECOMMENDATION
            # ------------------------------------------------

            st.session_state.states["3"] = (
                "running"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            recommendations = {
                "recommendations": analysis.get(
                    "recommendations",
                    [],
                )
            }


            st.session_state.stage_outputs[
                "Recommendation Agent"
            ] = recommendations

            st.session_state.states["3"] = (
                "done"
            )


            # ------------------------------------------------
            # REPORT
            # ------------------------------------------------

            st.session_state.states["4"] = (
                "running"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            with st.spinner(
                "Groq Report Agent is "
                "writing the class report..."
            ):

                report = make_report(
                    "Whole class",
                    summary,
                    analysis,
                    recommendations,
                )


            st.session_state.stage_outputs[
                "Report Agent"
            ] = report

            st.session_state.states["4"] = (
                "done"
            )


            # ------------------------------------------------
            # QUALITY
            # ------------------------------------------------

            st.session_state.states["5"] = (
                "running"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            if run_quality:

                with st.spinner(
                    "Groq Quality Agent is "
                    "checking the report..."
                ):

                    quality = quality_check(
                        report,
                        summary,
                    )

            else:

                quality = {
                    "passed": "not_run",
                    "score": None,
                    "note": (
                        "Quality check skipped."
                    ),
                }


            st.session_state.stage_outputs[
                "Quality Agent"
            ] = quality

            st.session_state.states["5"] = (
                "done"
            )


            # ------------------------------------------------
            # STORAGE
            # ------------------------------------------------

            st.session_state.states["6"] = (
                "running"
            )

            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            save_report(
                st.session_state.upload_id,
                "class",
                None,
                report,
                analysis,
                quality,
            )


            st.session_state.stage_outputs[
                "Storage Agent"
            ] = {
                "saved": True,
                "upload_id":
                    st.session_state.upload_id,
                "scope": "class",
            }


            st.session_state.states["6"] = (
                "done"
            )


            st.session_state.last_report = (
                report
            )

            st.session_state.last_quality = (
                quality
            )

            st.session_state.last_analysis = (
                analysis
            )


            pipeline_slot.markdown(
                render_pipeline(
                    st.session_state.states
                ),
                unsafe_allow_html=True,
            )


            st.success(
                "Whole-class analysis completed successfully."
            )


        except Exception as exc:

            st.error(
                "AI pipeline failed."
            )

            st.exception(exc)


# ============================================================
# AGENT OUTPUTS
# ============================================================

if st.session_state.stage_outputs:

    st.markdown(
        '<div class="section-heading">'
        'Agent outputs and handoffs'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Each panel contains the output produced "
        "by one stage and passed to the next stage."
    )


    for (
        agent_name,
        output,
    ) in st.session_state.stage_outputs.items():

        with st.expander(
            agent_name,
            expanded=False,
        ):

            if isinstance(
                output,
                dict,
            ):

                st.json(output)

            else:

                st.markdown(
                    str(output)
                )


# ============================================================
# FINAL REPORT
# ============================================================

if st.session_state.last_report:

    st.markdown(
        """
        <div class="result-panel">
            <div class="result-label">
                Final report
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        st.session_state.last_report
    )


    st.download_button(
        "Download Markdown report",
        st.session_state.last_report,
        file_name=(
            "student_performance_report.md"
        ),
        mime="text/markdown",
    )


    with st.expander(
        "Agent outputs"
    ):

        st.json(
            st.session_state.last_analysis
        )

        st.json(
            st.session_state.last_quality
        )


# ============================================================
# REPORT HISTORY
# ============================================================

with st.expander(
    "Saved report history"
):

    try:

        history = recent_reports()

        if history:

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            k: row[k]
                            for k in [
                                "report_id",
                                "scope",
                                "subject_id",
                                "created_at",
                            ]
                        }
                        for row in history
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.caption(
                "No reports have been saved yet."
            )

    except Exception as exc:

        st.warning(
            f"Could not load report history: {exc}"
        )

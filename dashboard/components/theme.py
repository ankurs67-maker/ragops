"""RAGOps dashboard design system — single source of visual truth.

Defines the color palette, typography scale, card styling, plain-language
metric explanations, and the CSS injected into every page. All dashboard
pages call inject_theme() first and use these constants instead of
hard-coded colors so the whole product stays visually consistent.
"""

import streamlit as st

# ── Color palette ─────────────────────────────────────────────────────────────
BG_PRIMARY = "#0E1526"      # deep navy page background
BG_CARD = "#161F35"         # slightly lighter navy for cards
ACCENT_HEALTHY = "#2DD4A7"  # teal-green
ACCENT_WARNING = "#F5A623"  # amber
ACCENT_CRITICAL = "#F0526B" # coral-red
ACCENT_INFO = "#5B9BD5"     # soft blue
TEXT_PRIMARY = "#EDEFF5"    # off-white
TEXT_MUTED = "#8B93A8"      # slate gray
BORDER = "#232D45"          # subtle navy-gray divider

# Distinct shades for failure categories (used in stacked bars / chips)
CATEGORY_COLORS = {
    "PASS": ACCENT_HEALTHY,
    "FALSE_REFUSAL": ACCENT_WARNING,
    "FAITHFULNESS_FAILURE": "#E8795A",
    "FACTUAL_ERROR": ACCENT_CRITICAL,
    "RETRIEVAL_FAILURE": "#C74B7B",
    "CONTEXT_BYPASS": "#9B6BD4",
    "REFUSAL_FAILURE": "#D4574E",
    "LATENCY_DEGRADATION": "#D8C24A",
    "PARTIAL_ANSWER": "#7FB0E8",
}

# ── Plain-language explanations (Requirement 1) ───────────────────────────────
METRIC_EXPLANATIONS = {
    "Health Score": (
        "An overall grade from 0-100 showing how well the AI is "
        "answering questions right now. Above 80 is healthy."
    ),
    "Retrieval Relevance": (
        "Did the system find the right information to answer the "
        "question? Like checking if someone grabbed the right book "
        "before writing an answer."
    ),
    "Context Utilisation": (
        "Did the AI actually use the information it found, or did it "
        "just guess from what it already knew? 100% means it always "
        "used the real source material."
    ),
    "Faithfulness": (
        "Does the answer only say things that are actually written in "
        "the source material? A low score means the AI may be making "
        "things up."
    ),
    "Factuality": (
        "Is the answer actually correct? This is checked against "
        "answers we already know are true."
    ),
    "Refusal Calibration": (
        "Does the AI correctly say 'I don't know' only when it truly "
        "doesn't know — and actually answer when it does know?"
    ),
}

FAILURE_EXPLANATIONS = {
    "PASS": "The AI answered correctly and honestly.",
    "FALSE_REFUSAL": (
        "The AI said it didn't know something it actually could have "
        "answered — being too cautious."
    ),
    "FAITHFULNESS_FAILURE": (
        "The AI's answer included details not found in its source "
        "material — a possible small hallucination."
    ),
    "FACTUAL_ERROR": "The AI's answer was simply wrong.",
    "RETRIEVAL_FAILURE": (
        "The AI could not find the right source material to answer from."
    ),
    "CONTEXT_BYPASS": (
        "The AI ignored the source material and answered from its own "
        "general knowledge instead."
    ),
    "REFUSAL_FAILURE": (
        "The AI should have said 'I don't know' but answered anyway — "
        "risky overconfidence."
    ),
    "LATENCY_DEGRADATION": "The AI took much longer than usual to respond.",
    "PARTIAL_ANSWER": "The AI's answer was incomplete but not wrong.",
}

WHAT_IS_THIS = (
    "This dashboard watches an AI system that answers questions "
    "about artificial intelligence and language models. Every few "
    "hours, it automatically tests the AI with dozens of questions "
    "we already know the correct answers to, then grades every "
    "answer on 5 different qualities. This page shows how well the "
    "AI is currently performing and flags anything that needs "
    "attention."
)

# ── CSS ───────────────────────────────────────────────────────────────────────
_THEME_CSS = f"""
<style>
/* Page + sidebar backgrounds */
.stApp {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
}}
section[data-testid="stSidebar"] {{
    background-color: {BG_CARD};
    border-right: 1px solid {BORDER};
}}

/* Typography hierarchy */
h1 {{
    color: {TEXT_PRIMARY} !important;
    font-size: 30px !important;
    font-weight: 700 !important;
}}
h2, h3 {{
    color: {TEXT_PRIMARY} !important;
    font-size: 19px !important;
    font-weight: 600 !important;
}}
p, li, .stMarkdown {{
    color: {TEXT_PRIMARY};
    font-size: 14.5px;
}}
.stCaption, small {{
    color: {TEXT_MUTED} !important;
}}

/* st.metric restyled as stat cards */
div[data-testid="stMetric"] {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}}
div[data-testid="stMetric"] label {{
    color: {TEXT_MUTED} !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
div[data-testid="stMetricValue"] {{
    color: {TEXT_PRIMARY} !important;
    font-size: 38px !important;
    font-weight: 700 !important;
}}

/* Expanders as cards */
div[data-testid="stExpander"] {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

/* Dataframes */
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
}}

/* Buttons */
.stButton > button {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
.stButton > button:hover {{
    border-color: {ACCENT_INFO};
    color: {ACCENT_INFO};
}}

/* Pill-shaped status chips */
.ragops-pill {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.03em;
}}

/* Stat card (custom HTML cards) */
.ragops-card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}}
.ragops-card .card-label {{
    color: {TEXT_MUTED};
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}}
.ragops-card .card-value {{
    font-size: 38px;
    font-weight: 700;
    line-height: 1.15;
}}
.ragops-card .card-sub {{
    color: {TEXT_MUTED};
    font-size: 13px;
    margin-top: 4px;
}}
.ragops-dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 8px;
}}
</style>
"""


def inject_theme() -> None:
    """Inject the RAGOps design-system CSS. Call first on every page."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def status_color(score: float, healthy_above: float = 80.0, warn_above: float = 60.0) -> str:
    """Map a 0-100 score to the palette status color."""
    if score >= healthy_above:
        return ACCENT_HEALTHY
    if score >= warn_above:
        return ACCENT_WARNING
    return ACCENT_CRITICAL


def status_pill(label: str, color: str) -> str:
    """Return HTML for a pill-shaped status chip."""
    return (
        f'<span class="ragops-pill" '
        f'style="background-color:{color}22; color:{color}; '
        f'border:1px solid {color};">{label}</span>'
    )


def stat_card(label: str, value: str, color: str, sub: str = "") -> str:
    """Return HTML for a stat card with a colored indicator dot."""
    sub_html = f'<div class="card-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="ragops-card">'
        f'<div class="card-label"><span class="ragops-dot" '
        f'style="background-color:{color};"></span>{label}</div>'
        f'<div class="card-value" style="color:{color};">{value}</div>'
        f"{sub_html}</div>"
    )


def render_glossary_sidebar() -> None:
    """Collapsible 'What do these scores mean?' panel for every page's sidebar."""
    with st.sidebar.expander("What do these scores mean?", expanded=False):
        st.markdown("**The 6 scores**")
        for name, text in METRIC_EXPLANATIONS.items():
            st.markdown(f"**{name}** — {text}")
        st.markdown("---")
        st.markdown("**Result categories**")
        for name, text in FAILURE_EXPLANATIONS.items():
            st.markdown(f"**{name}** — {text}")

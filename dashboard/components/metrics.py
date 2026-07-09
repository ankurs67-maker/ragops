"""Reusable metric display components for the Streamlit dashboard."""

import streamlit as st

from dashboard.components.theme import (
    ACCENT_CRITICAL,
    ACCENT_HEALTHY,
    ACCENT_INFO,
    ACCENT_WARNING,
    CATEGORY_COLORS,
    FAILURE_EXPLANATIONS,
    METRIC_EXPLANATIONS,
    status_pill,
)


def health_score_gauge(score: float) -> None:
    """Display a colour-coded health score metric with plain-language help."""
    if score >= 80:
        label = "Healthy"
        delta_color = "normal"
    elif score >= 60:
        label = "Degraded"
        delta_color = "off"
    else:
        label = "Critical"
        delta_color = "inverse"

    st.metric(
        label="System Health Score",
        value=f"{score:.1f}/100",
        delta=label,
        delta_color=delta_color,
        help=METRIC_EXPLANATIONS["Health Score"],
    )


def health_status_pill(score: float) -> None:
    """Render the health status as a pill-shaped colored chip."""
    if score >= 80:
        html = status_pill("HEALTHY", ACCENT_HEALTHY)
    elif score >= 60:
        html = status_pill("DEGRADING", ACCENT_WARNING)
    else:
        html = status_pill("CRITICAL", ACCENT_CRITICAL)
    st.markdown(html, unsafe_allow_html=True)


def dimension_metrics_row(avgs: dict) -> None:
    """Display 5 dimension scores in a single row of st.metric columns."""
    col1, col2, col3, col4, col5 = st.columns(5)

    ret = avgs.get("avg_retrieval") or 0.0
    util = avgs.get("avg_utilization") or 0.0
    faith = avgs.get("avg_faithfulness") or 0.0
    fact = avgs.get("avg_factuality") or 0.0
    ref = avgs.get("avg_refusal") or 0.0

    with col1:
        st.metric("Retrieval", f"{ret:.2f}/3", help=METRIC_EXPLANATIONS["Retrieval Relevance"])
    with col2:
        st.metric("Utilization", f"{util:.0f}%", help=METRIC_EXPLANATIONS["Context Utilisation"])
    with col3:
        st.metric("Faithfulness", f"{faith:.3f}", help=METRIC_EXPLANATIONS["Faithfulness"])
    with col4:
        st.metric("Factuality", f"{fact:.3f}", help=METRIC_EXPLANATIONS["Factuality"])
    with col5:
        st.metric("Refusal", f"{ref:.3f}", help=METRIC_EXPLANATIONS["Refusal Calibration"])


def alert_badges(alert_flags: dict) -> None:
    """Display coloured alert badges for any triggered alerts."""
    triggered = [k for k, v in alert_flags.items() if v]
    if not triggered:
        st.success("All dimensions within thresholds")
        return
    for flag in triggered:
        dim = flag.replace("_alert", "").replace("_", " ").title()
        st.error(f"⚠ Alert: {dim} below threshold")


def failure_category_badge(category: str) -> str:
    """Return a coloured markdown string for a failure category."""
    colors = {
        "PASS": "🟢",
        "RETRIEVAL_FAILURE": "🔴",
        "CONTEXT_BYPASS": "🟠",
        "FAITHFULNESS_FAILURE": "🔴",
        "FACTUAL_ERROR": "🔴",
        "REFUSAL_FAILURE": "🟠",
        "FALSE_REFUSAL": "🟡",
        "LATENCY_DEGRADATION": "🟡",
        "PARTIAL_ANSWER": "🟡",
    }
    icon = colors.get(category, "⚪")
    return f"{icon} {category}"


def failure_category_chip(category: str) -> str:
    """Return HTML for a pill chip colored by failure category."""
    color = CATEGORY_COLORS.get(category, ACCENT_INFO)
    return status_pill(category, color)


def probe_plain_summary(probe: dict) -> str:
    """One-line plain-language summary of an individual probe result."""
    category = probe.get("failure_category") or "UNKNOWN"
    faith = probe.get("faithfulness_score") or 0.0
    if category == "PASS":
        if faith >= 0.9:
            return "This answer was correct and well-supported by the source material."
        return "This answer was correct, though its grounding in the source material was only moderate."
    if category == "FALSE_REFUSAL":
        return "This answer was refused, but the information may have actually been available."
    explanation = FAILURE_EXPLANATIONS.get(category)
    if explanation:
        return explanation
    return "This probe did not complete normally."


def advanced_techniques_panel(probe_detail: dict) -> None:
    """Display a panel showing Self-RAG, Reflexion, and loop engineering status."""
    st.subheader("Advanced Techniques")
    details = probe_detail.get("measurement_details") or {}
    if isinstance(details, str):
        import json
        try:
            details = json.loads(details)
        except Exception:
            details = {}

    col1, col2, col3 = st.columns(3)
    with col1:
        sr_checks = details.get("self_rag_checks", {})
        sr_passed = (
            sr_checks.get("retrieval_adequate", False)
            and sr_checks.get("answer_grounded", False)
            and sr_checks.get("answer_complete", False)
        )
        sr_retries = details.get("self_rag_retries", 0)
        st.metric("Self-RAG", "✅ Passed" if sr_passed else "❌ Failed", f"{sr_retries} retries")
        if sr_checks:
            st.caption(
                f"Retrieval: {'✓' if sr_checks.get('retrieval_adequate') else '✗'} | "
                f"Grounded: {'✓' if sr_checks.get('answer_grounded') else '✗'} | "
                f"Complete: {'✓' if sr_checks.get('answer_complete') else '✗'}"
            )

    with col2:
        reflexion = details.get("reflexion_lessons_applied", False)
        st.metric("Reflexion", "✅ Active" if reflexion else "⬜ No lessons yet")
        st.caption("Past failure lessons loaded" if reflexion else "No prior failures found")

    with col3:
        loop_retries = details.get("loop_retries", 0)
        st.metric("Loop Retries", loop_retries)
        query_used = details.get("query_used", "")
        if query_used:
            st.caption(f"Final query: {query_used[:60]}...")

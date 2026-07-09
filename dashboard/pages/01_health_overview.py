"""Page 1: Health Overview — redesigned with gauge, stat cards, sparklines, and plain-language explanations."""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(page_title="Health Overview | RAGOps", layout="wide")

from database.db_client import (
    get_connection,
    get_dimension_averages,
    get_failure_distribution,
    get_system_health_score,
)
from analysis.trend_analysis import analyze_trends, get_recent_failure_trend
from analysis.pattern_detector import run_all_analyses
from dashboard.components.theme import (
    ACCENT_HEALTHY,
    ACCENT_INFO,
    ACCENT_WARNING,
    METRIC_EXPLANATIONS,
    WHAT_IS_THIS,
    inject_theme,
    render_glossary_sidebar,
    stat_card,
    status_color,
)
from dashboard.components.metrics import health_status_pill
from dashboard.components.charts import (
    advanced_technique_comparison,
    failure_stacked_bar,
    health_gauge,
    radar_chart,
    sparkline,
)

inject_theme()
render_glossary_sidebar()

# ── Landing explanation (C6) ──────────────────────────────────────────────────
with st.expander("What is this?", expanded=True):
    st.markdown(WHAT_IS_THIS)

st.title("Health Overview")

# ── System Vitals strip ───────────────────────────────────────────────────────
def _vitals() -> dict:
    out = {"total_probes": 0, "chunk_count": "—", "days_running": 0, "last_probe": None}
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, MIN(timestamp) AS first, MAX(timestamp) AS last "
                "FROM probe_results"
            ).fetchone()
            out["total_probes"] = row["n"] or 0
            if row["first"]:
                from datetime import datetime, timezone
                first = datetime.fromisoformat(str(row["first"]).replace("Z", "+00:00"))
                if first.tzinfo is None:
                    first = first.replace(tzinfo=timezone.utc)
                out["days_running"] = max((datetime.now(timezone.utc) - first).days, 1)
                out["last_probe"] = str(row["last"])[:19]
    except Exception:
        pass
    try:
        import json
        chunks_file = _ROOT / "data" / "processed" / "all_chunks.json"
        if chunks_file.exists():
            with chunks_file.open(encoding="utf-8") as f:
                out["chunk_count"] = f"{len(json.load(f)):,}"
    except Exception:
        pass
    return out


vitals = _vitals()
v1, v2, v3, v4 = st.columns(4)
v1.markdown(stat_card("Total Probes Run", f"{vitals['total_probes']:,}", ACCENT_INFO), unsafe_allow_html=True)
v2.markdown(stat_card("Corpus Size (chunks)", str(vitals["chunk_count"]), ACCENT_INFO), unsafe_allow_html=True)
v3.markdown(stat_card("Days Running", str(vitals["days_running"]), ACCENT_INFO), unsafe_allow_html=True)
v4.markdown(
    stat_card(
        "System Status",
        "● LIVE" if vitals["last_probe"] else "IDLE",
        ACCENT_HEALTHY if vitals["last_probe"] else ACCENT_WARNING,
        sub=f"last probe {vitals['last_probe']}" if vitals["last_probe"] else "no probes yet",
    ),
    unsafe_allow_html=True,
)

st.divider()

# ── Health gauge + status pill + summary metrics ─────────────────────────────
col1, col2 = st.columns([1, 2])
with col1:
    health = get_system_health_score()
    health_gauge(health)
    health_status_pill(health)
    st.caption(METRIC_EXPLANATIONS["Health Score"])

with col2:
    trend_7d = get_recent_failure_trend(days=7)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Probes (7d)", trend_7d["total_probes"])
    c2.metric("Pass Rate", f"{100 - trend_7d['failure_rate_pct']:.0f}%")
    c3.metric("Failures", trend_7d["fail_count"])
    c4.metric("Overall Trend", analyze_trends().get("overall_direction", "—").upper())
    failure_stacked_bar(get_failure_distribution(days=7), title="Last 7 Days — Pass vs Failure Mix")

st.divider()

# ── 5 dimension stat cards with sparklines ────────────────────────────────────
st.subheader("5-Dimension Quality Scores (24h average)")

avgs = get_dimension_averages(days=1)


def _daily_series(column: str, days: int = 7) -> list[float]:
    """Daily average of one measurement column over the last N days."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT date(timestamp) AS day, AVG({column}) AS avg_val
                FROM measurements
                WHERE timestamp >= datetime('now', :offset)
                GROUP BY date(timestamp)
                ORDER BY day
                """,
                {"offset": f"-{days} days"},
            ).fetchall()
        return [r["avg_val"] or 0.0 for r in rows]
    except Exception:
        return []


_CARDS = [
    ("Retrieval Relevance", "retrieval_relevance_score",
     avgs.get("avg_retrieval") or 0.0, 3.0, "{:.2f}/3"),
    ("Context Utilisation", "context_utilization_score",
     avgs.get("avg_utilization") or 0.0, 100.0, "{:.0f}%"),
    ("Faithfulness", "faithfulness_score",
     avgs.get("avg_faithfulness") or 0.0, 1.0, "{:.3f}"),
    ("Factuality", "factuality_score",
     avgs.get("avg_factuality") or 0.0, 1.0, "{:.3f}"),
    ("Refusal Calibration", "refusal_calibration_score",
     avgs.get("avg_refusal") or 0.0, 1.0, "{:.3f}"),
]

cols = st.columns(5)
for col, (name, db_col, value, scale, fmt) in zip(cols, _CARDS):
    with col:
        pct = (value / scale) * 100 if scale else 0
        color = status_color(pct)
        st.markdown(stat_card(name, fmt.format(value), color), unsafe_allow_html=True)
        series = _daily_series(db_col)
        sparkline(series, color=color)
        with st.popover("?", use_container_width=False):
            st.markdown(METRIC_EXPLANATIONS[name])

st.divider()

# ── Radar chart ───────────────────────────────────────────────────────────────
radar_chart(
    {
        "Retrieval": (avgs.get("avg_retrieval") or 0.0) / 3.0,
        "Utilization": (avgs.get("avg_utilization") or 0.0) / 100.0,
        "Faithfulness": avgs.get("avg_faithfulness") or 0.0,
        "Factuality": avgs.get("avg_factuality") or 0.0,
        "Refusal Cal.": avgs.get("avg_refusal") or 0.0,
    },
    title="Quality Dimensions (normalised 0-1)",
)

st.divider()

# ── Trend analysis ────────────────────────────────────────────────────────────
st.subheader("Trend Analysis")
trend = analyze_trends()
cols = st.columns(5)
for i, (db_key, info) in enumerate(trend.get("dimensions", {}).items()):
    with cols[i % 5]:
        direction = info.get("trend", "stable")
        arrow = "↑" if direction == "improving" else ("↓" if direction == "degrading" else "→")
        pct = info.get("pct_change", 0)
        st.metric(
            label=db_key.replace("_", " ").title(),
            value=f"{info.get('recent_avg', 0):.3f}",
            delta=f"{arrow} {pct:+.1f}%",
            delta_color="normal" if direction == "improving" else ("inverse" if direction == "degrading" else "off"),
        )

if trend.get("alerts"):
    st.error(f"Trend alerts: {', '.join(trend['alerts'])}")
else:
    st.success("All dimensions stable — no trend alerts")

st.divider()

# ── Advanced Techniques Panel ─────────────────────────────────────────────────
st.subheader("Advanced Techniques Effectiveness Panel")
st.caption(
    "This panel shows whether the 4 advanced techniques (Self-RAG, Reflexion, "
    "Loop Engineering, Context Engineering) are improving system performance. "
    "Data accumulates over multiple probe cycles."
)

analyses = run_all_analyses(days=30)
a7 = analyses["analyses"].get("7_reflexion_effectiveness", {})
a8 = analyses["analyses"].get("8_self_rag_effectiveness", {})
a9 = analyses["analyses"].get("9_loop_effectiveness", {})

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Reflexion**")
    if a7.get("with_reflexion_count", 0) > 0:
        imp = a7.get("reflexion_improvement_pct", 0)
        status = "✅ Effective" if a7.get("effective") else "⚠ No improvement"
        st.metric("Failure Rate Δ", f"{imp:+.1f}%", status)
        st.caption(f"{a7['with_reflexion_count']} probes with / {a7['without_reflexion_count']} without")
    else:
        st.info("Accumulating data...")

with col2:
    st.markdown("**Self-RAG**")
    if a8.get("self_rag_passed_count", 0) > 0:
        imp8 = a8.get("faithfulness_improvement", 0)
        status8 = "✅ Effective" if a8.get("effective") else "⚠ No improvement"
        st.metric("Faithfulness Δ", f"{imp8:+.3f}", status8)
        st.caption(f"{a8['self_rag_passed_count']} passed / {a8['self_rag_failed_count']} failed")
    else:
        st.info("Accumulating data...")

with col3:
    st.markdown("**Loop Engineering**")
    if a9.get("with_retry_count", 0) > 0:
        imp9 = a9.get("factuality_improvement", 0)
        st.metric("Factuality Δ (retry)", f"{imp9:+.3f}")
        st.caption(f"{a9['with_retry_count']} retried / {a9['no_retry_count']} no retry")
    else:
        st.info("No retries yet (all probes passed on first attempt)")

with col4:
    st.markdown("**Context Engineering**")
    ctx_chunks = avgs.get("avg_retrieval")
    if ctx_chunks is not None:
        st.metric("Retrieval Score", f"{ctx_chunks:.2f}/3")
        st.caption("Best chunk at pos 1, second-best at last position")
    else:
        st.info("Accumulating data...")

# Comparison chart
st.subheader("Technique Effectiveness Comparison")
advanced_technique_comparison(a7, a8, a9)

"""RAGOps Dashboard — main Streamlit entry point.

Run with: streamlit run dashboard/app.py
Or via:   make dashboard
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work when run as a module
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(
    page_title="RAGOps Monitor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from database.db_client import (
    get_connection,
    get_dimension_averages,
    get_failure_distribution,
    get_system_health_score,
)
from dashboard.components.theme import (
    ACCENT_HEALTHY,
    ACCENT_INFO,
    ACCENT_WARNING,
    WHAT_IS_THIS,
    inject_theme,
    render_glossary_sidebar,
    stat_card,
)
from dashboard.components.metrics import dimension_metrics_row, health_status_pill
from dashboard.components.charts import failure_stacked_bar, health_gauge

inject_theme()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("RAGOps Monitor")
st.sidebar.markdown(
    "Autonomous RAG Quality Monitoring System\n\n"
    "**Knowledge Domain:** LLM Intelligence Corpus\n\n"
    "**Probe Schedule:** 00:00 & 12:00 UTC"
)
st.sidebar.divider()
st.sidebar.markdown("### Navigation")
st.sidebar.page_link("pages/01_health_overview.py", label="Health Overview")
st.sidebar.page_link("pages/02_failure_analysis.py", label="Failure Analysis")
st.sidebar.page_link("pages/03_probe_explorer.py", label="Probe Explorer")
st.sidebar.page_link("pages/04_remediations.py", label="Remediations")
st.sidebar.page_link("pages/05_raw_data.py", label="Raw Data")
render_glossary_sidebar()

# ── Landing explanation (dismissible) ────────────────────────────────────────
with st.expander("What is this?", expanded=True):
    st.markdown(WHAT_IS_THIS)

# ── Main content ──────────────────────────────────────────────────────────────
st.title("RAGOps System Dashboard")
st.caption("Real-time monitoring of RAG pipeline quality across 5 measurement dimensions")

# ── System Vitals strip ───────────────────────────────────────────────────────
def _system_vitals() -> dict:
    vitals = {"total_probes": 0, "chunk_count": "—", "days_running": 0, "last_probe": None}
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, MIN(timestamp) AS first, MAX(timestamp) AS last "
                "FROM probe_results"
            ).fetchone()
            vitals["total_probes"] = row["n"] or 0
            if row["first"] and row["last"]:
                from datetime import datetime, timezone
                first = datetime.fromisoformat(str(row["first"]).replace("Z", "+00:00"))
                if first.tzinfo is None:
                    first = first.replace(tzinfo=timezone.utc)
                vitals["days_running"] = max((datetime.now(timezone.utc) - first).days, 1)
                vitals["last_probe"] = str(row["last"])[:19]
    except Exception:
        pass
    try:
        import json
        chunks_file = _ROOT / "data" / "processed" / "all_chunks.json"
        if chunks_file.exists():
            with chunks_file.open(encoding="utf-8") as f:
                vitals["chunk_count"] = f"{len(json.load(f)):,}"
    except Exception:
        pass
    return vitals


vitals = _system_vitals()
v1, v2, v3, v4 = st.columns(4)
with v1:
    st.markdown(
        stat_card("Total Probes Run", f"{vitals['total_probes']:,}", ACCENT_INFO),
        unsafe_allow_html=True,
    )
with v2:
    st.markdown(
        stat_card("Corpus Size (chunks)", str(vitals["chunk_count"]), ACCENT_INFO),
        unsafe_allow_html=True,
    )
with v3:
    st.markdown(
        stat_card("Days Running", str(vitals["days_running"]), ACCENT_INFO),
        unsafe_allow_html=True,
    )
with v4:
    up_color = ACCENT_HEALTHY if vitals["last_probe"] else ACCENT_WARNING
    st.markdown(
        stat_card(
            "Last Probe Cycle",
            "● LIVE" if vitals["last_probe"] else "IDLE",
            up_color,
            sub=vitals["last_probe"] or "no probes yet",
        ),
        unsafe_allow_html=True,
    )

st.divider()

# ── Health + summary ──────────────────────────────────────────────────────────
col_health, col_info = st.columns([1, 2])
with col_health:
    health = get_system_health_score()
    health_gauge(health)
    health_status_pill(health)

with col_info:
    avgs = get_dimension_averages(days=1)
    dist = get_failure_distribution(days=7)
    total = sum(dist.values())
    pass_count = dist.get("PASS", 0)
    fail_count = total - pass_count

    c1, c2, c3 = st.columns(3)
    c1.metric("Probes (7d)", total)
    c2.metric("Pass Rate", f"{(pass_count / total * 100):.0f}%" if total > 0 else "—")
    c3.metric("Active Alerts", fail_count)
    failure_stacked_bar(dist, title="Last 7 Days — Pass vs Failure Mix")

st.divider()

# ── Dimension scores ──────────────────────────────────────────────────────────
st.subheader("5-Dimension Quality Scores (24h average)")
dimension_metrics_row(avgs)

st.divider()
st.info(
    "**Getting started:** Run `make probe` to execute a probe cycle. "
    "The system automatically runs probes at 00:00 and 12:00 UTC. "
    "Use the navigation links in the sidebar to explore detailed analysis."
)

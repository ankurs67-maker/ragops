"""Page 3: Probe Explorer — browse individual probe results with Self-RAG and loop badges."""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Probe Explorer | RAGOps", layout="wide")

from database.db_client import get_connection, get_recent_probes
from dashboard.components.theme import (
    FAILURE_EXPLANATIONS,
    inject_theme,
    render_glossary_sidebar,
)
from dashboard.components.metrics import (
    advanced_techniques_panel,
    failure_category_chip,
    probe_plain_summary,
)

inject_theme()
render_glossary_sidebar()

st.title("Probe Explorer")
st.caption("Browse individual probe results. Badges show Self-RAG, Reflexion, and loop engineering status.")

# ── Controls ──────────────────────────────────────────────────────────────────
col_filter, col_cat = st.columns([2, 2])
with col_filter:
    hours = st.selectbox("Time window", [24, 48, 168], format_func=lambda x: f"Last {x}h")
with col_cat:
    category_filter = st.selectbox(
        "Failure category",
        ["ALL", "PASS", "RETRIEVAL_FAILURE", "CONTEXT_BYPASS", "FAITHFULNESS_FAILURE",
         "FACTUAL_ERROR", "REFUSAL_FAILURE", "FALSE_REFUSAL", "LATENCY_DEGRADATION", "PARTIAL_ANSWER"],
    )

# ── Fetch probes + measurements ───────────────────────────────────────────────
with get_connection() as conn:
    rows = conn.execute(
        """
        SELECT
            p.probe_id, p.timestamp, p.query_id, p.query_text,
            p.generated_answer, p.correct_answer, p.category,
            p.difficulty, p.latency_total_ms,
            m.retrieval_relevance_score, m.context_utilization_score,
            m.faithfulness_score, m.factuality_score,
            m.refusal_calibration_score, m.failure_category,
            m.measurement_details
        FROM probe_results p
        LEFT JOIN measurements m ON p.probe_id = m.probe_id
        WHERE p.timestamp >= datetime('now', :offset)
        ORDER BY p.timestamp DESC
        LIMIT 200
        """,
        {"offset": f"-{hours} hours"},
    ).fetchall()

probes = [dict(r) for r in rows]

if category_filter != "ALL":
    probes = [p for p in probes if p.get("failure_category") == category_filter]

if not probes:
    st.info("No probes found for the selected filters. Run `make probe` to collect data.")
    st.stop()

# ── Summary table ─────────────────────────────────────────────────────────────
st.subheader(f"{len(probes)} probes found")

_STATUS_DOT = {
    "PASS": "🟢",
    "FALSE_REFUSAL": "🟡",
    "LATENCY_DEGRADATION": "🟡",
    "PARTIAL_ANSWER": "🟡",
    "CONTEXT_BYPASS": "🟠",
    "REFUSAL_FAILURE": "🟠",
    "FAITHFULNESS_FAILURE": "🔴",
    "FACTUAL_ERROR": "🔴",
    "RETRIEVAL_FAILURE": "🔴",
}

df_display = []
for p in probes:
    details = {}
    if p.get("measurement_details"):
        try:
            details = json.loads(p["measurement_details"])
        except Exception:
            pass

    sr_checks = details.get("self_rag_checks", {})
    sr_passed = (
        sr_checks.get("retrieval_adequate", False)
        and sr_checks.get("answer_grounded", False)
        and sr_checks.get("answer_complete", False)
    )
    reflexion = details.get("reflexion_lessons_applied", False)
    loop_retries = details.get("loop_retries", 0)
    category = p.get("failure_category", "UNKNOWN")

    df_display.append({
        "Status": f"{_STATUS_DOT.get(category, '⚪')} {category}",
        "Probe ID": p["probe_id"][:8] + "…",
        "Timestamp": p["timestamp"][:19],
        "Query ID": p["query_id"],
        "Category": p.get("category", "—"),
        "Difficulty": p.get("difficulty", "—"),
        "Retrieval": f"{p.get('retrieval_relevance_score', 0) or 0:.1f}/3",
        "Faith.": f"{p.get('faithfulness_score', 0) or 0:.3f}",
        "Fact.": f"{p.get('factuality_score', 0) or 0:.3f}",
        "Self-RAG": "✅" if sr_passed else "❌",
        "Reflexion": "✅" if reflexion else "—",
        "Retries": loop_retries,
        "Latency (ms)": int(p.get("latency_total_ms") or 0),
    })

df = pd.DataFrame(df_display)
selected = st.dataframe(
    df,
    use_container_width=True,
    selection_mode="single-row",
    on_select="rerun",
)

# ── Detail view ───────────────────────────────────────────────────────────────
if selected and selected.get("selection") and selected["selection"].get("rows"):
    idx = selected["selection"]["rows"][0]
    probe = probes[idx]
    st.divider()
    st.subheader(f"Probe Detail: {probe['query_id']}")

    # Status chip + plain-language one-liner
    st.markdown(
        failure_category_chip(probe.get("failure_category", "UNKNOWN")),
        unsafe_allow_html=True,
    )
    st.info(probe_plain_summary(probe))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Question asked:**")
        st.info(probe.get("query_text", "—"))
        st.markdown("**Generated answer:**")
        st.success(probe.get("generated_answer", "—"))
        st.markdown("**Correct answer:**")
        st.warning(probe.get("correct_answer", "—"))
    with col2:
        c1, c2 = st.columns(2)
        c1.metric("Failure Category", probe.get("failure_category", "—"),
                  help=FAILURE_EXPLANATIONS.get(probe.get("failure_category", ""), None))
        c2.metric("Latency", f"{int(probe.get('latency_total_ms') or 0):,} ms")
        c1.metric("Retrieval Score", f"{probe.get('retrieval_relevance_score', 0) or 0:.1f}/3")
        c2.metric("Context Util.", f"{probe.get('context_utilization_score', 0) or 0:.0f}%")
        c1.metric("Faithfulness", f"{probe.get('faithfulness_score', 0) or 0:.3f}")
        c2.metric("Factuality", f"{probe.get('factuality_score', 0) or 0:.3f}")

    st.divider()
    advanced_techniques_panel(probe)

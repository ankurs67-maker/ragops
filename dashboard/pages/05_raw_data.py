"""Page 5: Raw Data — export probe and measurement data as CSV/JSON."""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Raw Data | RAGOps", layout="wide")

from database.db_client import get_connection
from dashboard.components.theme import inject_theme, render_glossary_sidebar

inject_theme()
render_glossary_sidebar()

st.title("Raw Data Export")
st.caption("View and download all probe results and measurements as CSV or JSON.")

# ── Controls ──────────────────────────────────────────────────────────────────
days = st.sidebar.slider("Days to export", 1, 90, 30)
table = st.sidebar.radio("Table", ["probe_results", "measurements", "pattern_reports", "remediations", "daily_reports"])

# ── Query ─────────────────────────────────────────────────────────────────────
with get_connection() as conn:
    if table == "probe_results":
        rows = conn.execute(
            f"SELECT * FROM probe_results WHERE timestamp >= datetime('now', '-{days} days') ORDER BY timestamp DESC"
        ).fetchall()
    elif table == "measurements":
        rows = conn.execute(
            f"SELECT * FROM measurements WHERE timestamp >= datetime('now', '-{days} days') ORDER BY timestamp DESC"
        ).fetchall()
    elif table == "pattern_reports":
        rows = conn.execute(
            f"SELECT * FROM pattern_reports WHERE timestamp >= datetime('now', '-{days} days') ORDER BY timestamp DESC"
        ).fetchall()
    elif table == "remediations":
        rows = conn.execute("SELECT * FROM remediations ORDER BY timestamp DESC LIMIT 200").fetchall()
    else:
        rows = conn.execute("SELECT * FROM daily_reports ORDER BY date DESC LIMIT 50").fetchall()

df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

st.subheader(f"{table} — {len(df)} rows")

if df.empty:
    st.info("No data found. Run `make probe` to populate the database.")
else:
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"{table}_{days}d.csv",
            mime="text/csv",
        )
    with col2:
        json_str = df.to_json(orient="records", indent=2)
        st.download_button(
            "Download JSON",
            data=json_str,
            file_name=f"{table}_{days}d.json",
            mime="application/json",
        )

# ── Failure memory viewer ─────────────────────────────────────────────────────
st.divider()
st.subheader("Reflexion Memory (failure_memory.jsonl)")
from config.settings import settings
memory_path = settings.failure_memory_path
if memory_path.exists():
    lines = memory_path.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines[-20:]:  # Last 20 entries
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    if entries:
        df_mem = pd.DataFrame(entries)
        st.caption(f"{len(lines)} total lessons in memory. Showing last 20.")
        st.dataframe(df_mem, use_container_width=True)
    else:
        st.info("Failure memory is empty.")
else:
    st.info("No failure memory file yet (created after the first probe cycle with failures).")

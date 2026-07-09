"""Page 2: Failure Analysis — failure distribution, trends, and source breakdown."""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Failure Analysis | RAGOps", layout="wide")

from database.db_client import get_failure_distribution
from analysis.pattern_detector import (
    analysis_1_failure_clustering,
    analysis_2_source_breakdown,
    analysis_3_difficulty_breakdown,
    analysis_6_latency_spikes,
)
from dashboard.components.theme import (
    FAILURE_EXPLANATIONS,
    inject_theme,
    render_glossary_sidebar,
)
from dashboard.components.charts import (
    failure_distribution_pie,
    failure_stacked_bar,
    latency_histogram,
)

inject_theme()
render_glossary_sidebar()

st.title("Failure Analysis")

# ── Controls ──────────────────────────────────────────────────────────────────
days = st.sidebar.slider("Analysis window (days)", min_value=1, max_value=30, value=7)

# ── Plain-language headline summary ───────────────────────────────────────────
dist = get_failure_distribution(days=days)


def _plain_summary(distribution: dict) -> str:
    """One-line plain-English summary of the dominant failure mode."""
    failures = {k: v for k, v in distribution.items() if k != "PASS" and v > 0}
    if not distribution:
        return "No probe data yet — run a probe cycle to see how the AI is doing."
    if not failures:
        return "In this window, every single test passed — no failures of any kind."
    top_cat, top_count = max(failures.items(), key=lambda kv: kv[1])
    total_failures = sum(failures.values())
    friendly = {
        "FALSE_REFUSAL": "the AI being too cautious (saying 'I don't know' when it could have answered)",
        "FAITHFULNESS_FAILURE": "the AI adding details not found in its source material",
        "FACTUAL_ERROR": "the AI simply being wrong",
        "RETRIEVAL_FAILURE": "the system failing to find the right source material",
        "CONTEXT_BYPASS": "the AI ignoring its source material and answering from memory",
        "REFUSAL_FAILURE": "the AI answering questions it should have declined",
        "LATENCY_DEGRADATION": "the AI responding much slower than usual",
        "PARTIAL_ANSWER": "the AI giving incomplete answers",
    }
    description = friendly.get(top_cat, top_cat)
    share = top_count / total_failures * 100
    return (
        f"In this window, most failures ({top_count} of {total_failures}, "
        f"{share:.0f}%) were {description} — category {top_cat}."
    )


st.info(_plain_summary(dist))

# ── Failure distribution ──────────────────────────────────────────────────────
st.subheader(f"Failure Distribution (last {days} days)")
failure_stacked_bar(dist, title="Pass vs Failure Mix")

col1, col2 = st.columns([1, 1])
with col1:
    failure_distribution_pie(dist, title="Probe Results by Category")
with col2:
    if dist:
        total = sum(dist.values())
        df = pd.DataFrame(
            [
                {
                    "Category": k,
                    "Count": v,
                    "Rate (%)": round(v / total * 100, 1),
                    "In plain language": FAILURE_EXPLANATIONS.get(k, "—"),
                }
                for k, v in dist.items()
            ]
        )
        df = df.sort_values("Count", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No probe data available. Run `make probe` to collect data.")

st.divider()

# ── Failure clustering ─────────────────────────────────────────────────────────
st.subheader("Failure Clustering Analysis")
cluster = analysis_1_failure_clustering(days=days)
if cluster.get("status") == "no_data":
    st.info("No data yet.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Probes", cluster["total_probes"])
    c2.metric("Pass Rate", f"{cluster['pass_rate_pct']:.0f}%")
    c3.metric("Dominant Failure", cluster["dominant_failure"])

st.divider()

# ── Source breakdown ───────────────────────────────────────────────────────────
st.subheader("Failure Rate by Document Source")
source_data = analysis_2_source_breakdown(days=days)
by_source = source_data.get("by_source", {})
if by_source:
    df_src = pd.DataFrame([
        {
            "Source": src,
            "Total Probes": v["total"],
            "Failures": v["failures"],
            "Failure Rate (%)": v["failure_rate_pct"],
            "Most Common Failure": v["most_common_failure"],
        }
        for src, v in by_source.items()
    ])
    st.dataframe(df_src, use_container_width=True)
else:
    st.info("Not enough data to break down by source yet.")

st.divider()

# ── Difficulty breakdown ───────────────────────────────────────────────────────
st.subheader("Failure Rate by Query Difficulty")
diff_data = analysis_3_difficulty_breakdown(days=days)
by_diff = diff_data.get("by_difficulty", {})
if by_diff:
    df_diff = pd.DataFrame(
        [
            {
                "Difficulty": diff,
                "Probes": stats["total"],
                "Failure Rate (%)": stats["failure_rate_pct"],
            }
            for diff, stats in sorted(
                by_diff.items(), key=lambda x: x[1]["failure_rate_pct"], reverse=True
            )
        ]
    )
    st.dataframe(df_diff, use_container_width=True)
else:
    st.info("Not enough data yet.")

st.divider()

# ── Latency analysis ──────────────────────────────────────────────────────────
st.subheader("Latency Analysis")
latency_data = analysis_6_latency_spikes(days=days)
if latency_data.get("status") == "no_data":
    st.info("No latency data yet.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Latency", f"{latency_data['avg_latency_ms']:,} ms")
    c2.metric("P95 Latency", f"{latency_data['p95_latency_ms']:,} ms")
    c3.metric("Spike Threshold", f"{latency_data['spike_threshold_ms']:,} ms")
    c4.metric("Spikes", latency_data["spike_count"])

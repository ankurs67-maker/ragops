"""Reusable chart components for the Streamlit dashboard using Plotly.

All charts pull colors from dashboard.components.theme so the whole
product shares one palette instead of Plotly defaults.
"""

import json
from typing import Optional

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from dashboard.components.theme import (
    ACCENT_CRITICAL,
    ACCENT_HEALTHY,
    ACCENT_INFO,
    ACCENT_WARNING,
    BG_CARD,
    BG_PRIMARY,
    BORDER,
    CATEGORY_COLORS,
    TEXT_MUTED,
    TEXT_PRIMARY,
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the RAGOps dark-navy theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_PRIMARY, size=13),
        title_font=dict(color=TEXT_PRIMARY, size=16),
        legend=dict(font=dict(color=TEXT_MUTED)),
        margin=dict(l=20, r=20, t=48, b=20),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def health_gauge(score: float, title: str = "Health Score") -> None:
    """Large arc gauge for the 0-100 health score using the status palette."""
    if score >= 80:
        bar_color = ACCENT_HEALTHY
    elif score >= 60:
        bar_color = ACCENT_WARNING
    else:
        bar_color = ACCENT_CRITICAL

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 44, "color": TEXT_PRIMARY}, "suffix": ""},
            title={"text": title, "font": {"size": 14, "color": TEXT_MUTED}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": TEXT_MUTED,
                    "tickfont": {"color": TEXT_MUTED, "size": 11},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": BG_PRIMARY,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 60], "color": "#3A2230"},
                    {"range": [60, 80], "color": "#3A3222"},
                    {"range": [80, 100], "color": "#1D3A33"},
                ],
            },
        )
    )
    fig.update_layout(height=260)
    st.plotly_chart(_apply_theme(fig), use_container_width=True)


def sparkline(values: list[float], color: str = ACCENT_INFO) -> None:
    """Tiny axis-free trend line for embedding under a stat card."""
    if not values or len(values) < 2:
        return
    fig = go.Figure(
        go.Scatter(
            y=values,
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.12)",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=56,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def failure_stacked_bar(dist: dict, title: str = "Probe Results at a Glance") -> None:
    """Single horizontal stacked bar: PASS in teal, failures in distinct shades."""
    if not dist:
        st.info("No data available yet. Run a probe cycle to populate the chart.")
        return
    total = sum(dist.values())
    # PASS first, then failures sorted by count so proportions read left-to-right
    ordered = sorted(dist.items(), key=lambda kv: (kv[0] != "PASS", -kv[1]))
    fig = go.Figure()
    for category, count in ordered:
        fig.add_trace(
            go.Bar(
                y=[""],
                x=[count],
                name=category,
                orientation="h",
                marker=dict(color=CATEGORY_COLORS.get(category, TEXT_MUTED)),
                text=f"{category} {count / total * 100:.0f}%" if count / total >= 0.08 else "",
                textposition="inside",
                insidetextfont=dict(color=BG_PRIMARY, size=12),
                hovertemplate=f"{category}: {count} ({count / total * 100:.1f}%)<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title=title,
        height=170,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.3),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    st.plotly_chart(_apply_theme(fig), use_container_width=True)


def failure_distribution_pie(dist: dict, title: str = "Failure Distribution") -> None:
    """Render a pie chart of failure category distribution (theme palette)."""
    if not dist:
        st.info("No data available yet. Run a probe cycle to populate the chart.")
        return
    df = pd.DataFrame(list(dist.items()), columns=["Category", "Count"])
    fig = px.pie(
        df,
        names="Category",
        values="Count",
        title=title,
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(_apply_theme(fig), use_container_width=True)


def dimension_trend_line(
    timestamps: list[str],
    values: list[float],
    dimension: str,
    max_val: float = 1.0,
) -> None:
    """Render a line chart showing one dimension's score over time."""
    if not timestamps or not values:
        st.info(f"No data for {dimension} yet.")
        return
    df = pd.DataFrame({"Timestamp": timestamps, "Score": values})
    fig = px.line(
        df,
        x="Timestamp",
        y="Score",
        title=f"{dimension} Over Time",
        markers=True,
        color_discrete_sequence=[ACCENT_INFO],
    )
    fig.update_yaxes(range=[0, max_val * 1.1])
    fig.add_hline(
        y=max_val * 0.6,
        line_dash="dash",
        line_color=ACCENT_CRITICAL,
        annotation_text="Alert threshold",
    )
    st.plotly_chart(_apply_theme(fig), use_container_width=True)


def radar_chart(scores: dict, title: str = "Dimension Scores") -> None:
    """Render a radar/spider chart showing all 5 dimension scores."""
    dims = list(scores.keys())
    vals = list(scores.values())
    if not dims:
        st.info("No dimension data available.")
        return
    dims_closed = dims + [dims[0]]
    vals_closed = vals + [vals[0]]
    fig = go.Figure(
        data=go.Scatterpolar(
            r=vals_closed,
            theta=dims_closed,
            fill="toself",
            name="Score",
            line_color=ACCENT_INFO,
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor=BG_PRIMARY,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=BORDER,
                tickfont=dict(color=TEXT_MUTED),
            ),
            angularaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT_PRIMARY)),
        ),
        title=title,
        showlegend=False,
    )
    st.plotly_chart(_apply_theme(fig), use_container_width=True)


def latency_histogram(latencies: list[float], title: str = "Response Latency Distribution") -> None:
    """Render a histogram of latency values."""
    if not latencies:
        st.info("No latency data yet.")
        return
    df = pd.DataFrame({"Latency (ms)": latencies})
    fig = px.histogram(
        df, x="Latency (ms)", title=title, nbins=20,
        color_discrete_sequence=[ACCENT_INFO],
    )
    st.plotly_chart(_apply_theme(fig), use_container_width=True)


def advanced_technique_comparison(analysis_7: dict, analysis_8: dict, analysis_9: dict) -> None:
    """Bar chart comparing pass/fail rates for advanced techniques."""
    data = []

    if analysis_7.get("status") != "no_data" and analysis_7.get("with_reflexion_count", 0) > 0:
        data.append({
            "Technique": "Reflexion",
            "With Technique": analysis_7.get("with_reflexion_failure_rate_pct", 0),
            "Without Technique": analysis_7.get("without_reflexion_failure_rate_pct", 0),
        })

    if analysis_8.get("status") != "no_data" and analysis_8.get("self_rag_passed_count", 0) > 0:
        passed_faith = (1 - analysis_8.get("avg_faithfulness_when_passed", 0)) * 100
        failed_faith = (1 - analysis_8.get("avg_faithfulness_when_failed", 0)) * 100
        data.append({
            "Technique": "Self-RAG",
            "With Technique": round(passed_faith, 1),
            "Without Technique": round(failed_faith, 1),
        })

    if not data:
        st.info("Not enough data yet to compare technique effectiveness. Run more probe cycles.")
        return

    df = pd.DataFrame(data)
    fig = px.bar(
        df,
        x="Technique",
        y=["Without Technique", "With Technique"],
        title="Failure Rate % With vs Without Advanced Techniques",
        barmode="group",
        color_discrete_map={
            "With Technique": ACCENT_HEALTHY,
            "Without Technique": ACCENT_CRITICAL,
        },
    )
    fig.update_yaxes(title="Failure Rate (%)", range=[0, 100])
    st.plotly_chart(_apply_theme(fig), use_container_width=True)

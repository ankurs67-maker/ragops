"""Tests for the dashboard design system (theme.py) and plain-language helpers."""

import pytest


def test_palette_constants_are_hex_colors():
    from dashboard.components import theme

    for name in [
        "BG_PRIMARY", "BG_CARD", "ACCENT_HEALTHY", "ACCENT_WARNING",
        "ACCENT_CRITICAL", "ACCENT_INFO", "TEXT_PRIMARY", "TEXT_MUTED", "BORDER",
    ]:
        value = getattr(theme, name)
        assert value.startswith("#") and len(value) == 7, f"{name} is not a hex color"


def test_all_failure_categories_have_colors_and_explanations():
    from dashboard.components.theme import CATEGORY_COLORS, FAILURE_EXPLANATIONS

    categories = [
        "PASS", "RETRIEVAL_FAILURE", "CONTEXT_BYPASS", "FAITHFULNESS_FAILURE",
        "FACTUAL_ERROR", "REFUSAL_FAILURE", "FALSE_REFUSAL",
        "LATENCY_DEGRADATION", "PARTIAL_ANSWER",
    ]
    for cat in categories:
        assert cat in CATEGORY_COLORS, f"no color for {cat}"
        assert cat in FAILURE_EXPLANATIONS, f"no plain-language explanation for {cat}"
        # Explanations must be plain sentences, not category names
        assert len(FAILURE_EXPLANATIONS[cat]) > 20


def test_all_metrics_have_explanations():
    from dashboard.components.theme import METRIC_EXPLANATIONS

    for metric in [
        "Health Score", "Retrieval Relevance", "Context Utilisation",
        "Faithfulness", "Factuality", "Refusal Calibration",
    ]:
        assert metric in METRIC_EXPLANATIONS
        assert len(METRIC_EXPLANATIONS[metric]) > 20


def test_status_color_thresholds():
    from dashboard.components.theme import (
        ACCENT_CRITICAL, ACCENT_HEALTHY, ACCENT_WARNING, status_color,
    )

    assert status_color(90) == ACCENT_HEALTHY
    assert status_color(80) == ACCENT_HEALTHY
    assert status_color(70) == ACCENT_WARNING
    assert status_color(59) == ACCENT_CRITICAL


def test_status_pill_and_stat_card_render_html():
    from dashboard.components.theme import ACCENT_HEALTHY, stat_card, status_pill

    pill = status_pill("HEALTHY", ACCENT_HEALTHY)
    assert "ragops-pill" in pill and "HEALTHY" in pill

    card = stat_card("Total Probes", "500", ACCENT_HEALTHY, sub="last 7 days")
    assert "ragops-card" in card and "500" in card and "last 7 days" in card


def test_probe_plain_summary_covers_key_cases():
    from dashboard.components.metrics import probe_plain_summary

    assert "correct" in probe_plain_summary(
        {"failure_category": "PASS", "faithfulness_score": 0.95}
    ).lower()
    assert "refused" in probe_plain_summary(
        {"failure_category": "FALSE_REFUSAL", "faithfulness_score": 1.0}
    ).lower()
    # Every taxonomy category yields a non-empty sentence
    for cat in ["FACTUAL_ERROR", "RETRIEVAL_FAILURE", "CONTEXT_BYPASS",
                "FAITHFULNESS_FAILURE", "REFUSAL_FAILURE",
                "LATENCY_DEGRADATION", "PARTIAL_ANSWER"]:
        summary = probe_plain_summary({"failure_category": cat, "faithfulness_score": 0.5})
        assert len(summary) > 15

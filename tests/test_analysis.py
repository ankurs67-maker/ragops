"""Tests for analysis/ modules — operates on an empty test database."""

import json
import pytest


@pytest.fixture(autouse=True)
def clean_db(monkeypatch, tmp_path):
    """Each test gets a fresh temp DB."""
    from config.settings import Settings

    test_db = tmp_path / "test.db"

    def fake_db_path(self):
        return test_db

    monkeypatch.setattr(Settings, "db_path", property(fake_db_path))

    from database.db_client import init_database
    init_database()
    return test_db


def test_trend_analysis_no_data():
    """With no data, analyze_trends should return stable overall direction."""
    from analysis.trend_analysis import analyze_trends

    result = analyze_trends()
    assert result["overall_direction"] == "stable"
    assert "dimensions" in result
    assert "alerts" in result
    assert isinstance(result["alerts"], list)


def test_trend_analysis_structure():
    """analyze_trends should return all required keys."""
    from analysis.trend_analysis import analyze_trends

    result = analyze_trends()
    required = ["trend_window_days", "baseline_window_days", "computed_at",
                "dimensions", "alerts", "overall_direction", "avg_pct_change"]
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_get_recent_failure_trend_empty():
    """With no data, failure trend should show 0 probes and 0% failure rate."""
    from analysis.trend_analysis import get_recent_failure_trend

    trend = get_recent_failure_trend(days=7)
    assert trend["total_probes"] == 0
    assert trend["failure_rate_pct"] == 0.0


def test_pattern_analyses_no_data():
    """All 9 analyses should run without error even with an empty DB."""
    from analysis.pattern_detector import run_all_analyses

    result = run_all_analyses(days=7)
    assert "computed_at" in result
    assert "analyses" in result
    analyses = result["analyses"]
    # All 9 analysis keys should be present
    for i in range(1, 10):
        key = next((k for k in analyses if k.startswith(f"{i}_")), None)
        assert key is not None, f"Missing analysis {i}"


def test_reflexion_analysis_no_data():
    """Reflexion effectiveness analysis should handle empty DB gracefully."""
    from analysis.pattern_detector import analysis_7_reflexion_effectiveness

    result = analysis_7_reflexion_effectiveness(days=30)
    # Either no_data status or valid result structure
    assert "status" in result or "with_reflexion_count" in result


def test_self_rag_analysis_no_data():
    """Self-RAG effectiveness analysis should handle empty DB gracefully."""
    from analysis.pattern_detector import analysis_8_self_rag_effectiveness

    result = analysis_8_self_rag_effectiveness(days=30)
    assert "status" in result or "self_rag_passed_count" in result


def test_loop_analysis_no_data():
    """Loop engineering analysis should handle empty DB gracefully."""
    from analysis.pattern_detector import analysis_9_loop_effectiveness

    result = analysis_9_loop_effectiveness(days=30)
    assert "status" in result or "no_retry_count" in result


def test_remediation_proposer_no_data():
    """propose_remediations should return empty list with no failure data."""
    from analysis.remediation_proposer import propose_remediations

    result = propose_remediations(days=1)
    assert isinstance(result, list)
    # No failures → no remediations
    assert len(result) == 0


def test_reporter_generates_report(tmp_path, monkeypatch):
    """generate_daily_report should succeed and return report_text."""
    from config.settings import Settings

    def fake_reports(self):
        return tmp_path / "reports"

    monkeypatch.setattr(Settings, "reports_dir", property(fake_reports))

    from analysis.reporter import generate_daily_report

    result = generate_daily_report()
    assert "report_text" in result
    assert "RAGOps DAILY REPORT" in result["report_text"]
    assert "System Health Score" in result["report_text"]

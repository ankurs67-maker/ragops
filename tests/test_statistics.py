"""Tests for run_statistical_test (t-test with small-sample guard) and
the specific Reflexion lesson builder added in the research-rigor pass."""

import pytest


# ── run_statistical_test ──────────────────────────────────────────────────────

def test_ttest_insufficient_data_below_minimum():
    """Fewer than 5 samples in either series must report insufficient_data."""
    from analysis.trend_analysis import run_statistical_test

    result = run_statistical_test([0.9, 0.8], [0.5, 0.6, 0.7, 0.8, 0.9])
    assert result["status"] == "insufficient_data"
    assert result["is_significant"] is False
    assert result["p_value"] is None
    assert result["t_statistic"] is None


def test_ttest_insufficient_data_empty_series():
    """Empty series must not crash and must report insufficient_data."""
    from analysis.trend_analysis import run_statistical_test

    result = run_statistical_test([], [])
    assert result["status"] == "insufficient_data"
    assert result["is_significant"] is False


def test_ttest_detects_clear_difference():
    """Two clearly separated distributions should be significant."""
    from analysis.trend_analysis import run_statistical_test

    high = [0.90, 0.88, 0.92, 0.89, 0.91, 0.90]
    low = [0.50, 0.52, 0.48, 0.51, 0.49, 0.50]
    result = run_statistical_test(high, low)
    assert result["status"] == "ok"
    assert result["is_significant"] is True
    assert result["p_value"] < 0.05


def test_ttest_no_difference_not_significant():
    """Identical distributions should not be flagged significant."""
    from analysis.trend_analysis import run_statistical_test

    a = [0.80, 0.81, 0.79, 0.80, 0.82]
    b = [0.80, 0.79, 0.81, 0.80, 0.81]
    result = run_statistical_test(a, b)
    assert result["status"] == "ok"
    assert result["is_significant"] is False


def test_ttest_reports_sample_sizes():
    from analysis.trend_analysis import run_statistical_test

    result = run_statistical_test([1.0] * 6, [0.5] * 7)
    assert result["n_a"] == 6
    assert result["n_b"] == 7


# ── Reflexion lesson specificity ──────────────────────────────────────────────

def test_failure_lesson_includes_query_text():
    """Lessons must contain the actual failed query, not just boilerplate."""
    from monitoring.run_probe_cycle import _build_failure_lesson

    probe = {
        "query_id": "gt_001",
        "query": "What company developed the Llama 2 model?",
        "failure_category": "FALSE_REFUSAL",
        "answer": "I cannot find this information in my knowledge base.",
        "retrieval_relevance": 3,
        "context_utilization": 80,
        "faithfulness": 1.0,
        "factuality": 0.0,
        "refusal_calibration": 0.0,
    }
    lesson = _build_failure_lesson(probe)
    assert "What company developed the Llama 2 model?" in lesson["lesson"]
    assert "refusal_calibration=0.0" in lesson["lesson"]
    assert lesson["failure_category"] == "FALSE_REFUSAL"


# ── Classifier: correctly-refused out-of-scope queries ───────────────────────

def test_correct_refusal_with_no_retrieval_is_pass():
    """An out-of-scope query the model correctly refused must be PASS even
    though retrieval found nothing — empty retrieval is the expected state."""
    from monitoring.classify_failure import classify_failure

    category = classify_failure(
        retrieval_score=0.0,
        utilization_score=0.0,
        faithfulness_score=1.0,
        factuality_score=1.0,
        refusal_result={
            "score": 1.0, "refused": True, "should_refuse": True,
            "calibrated": True, "failure_type": None,
        },
        latency_ms=1000.0,
        baseline_latency_ms=0.0,
        self_rag_checks={},
    )
    assert category == "PASS"


def test_failed_refusal_still_detected():
    """should_refuse + answered anyway must still be REFUSAL_FAILURE."""
    from monitoring.classify_failure import classify_failure

    category = classify_failure(
        retrieval_score=0.0,
        utilization_score=50.0,
        faithfulness_score=0.9,
        factuality_score=0.0,
        refusal_result={
            "score": 0.0, "refused": False, "should_refuse": True,
            "calibrated": False, "failure_type": "failed_refusal",
        },
        latency_ms=1000.0,
        baseline_latency_ms=0.0,
        self_rag_checks={},
    )
    assert category == "REFUSAL_FAILURE"


def test_failure_lesson_handles_unknown_category():
    """Unknown categories must not crash the lesson builder."""
    from monitoring.run_probe_cycle import _build_failure_lesson

    probe = {
        "query_id": "gt_999",
        "query": "Some query",
        "failure_category": "SOMETHING_NEW",
        "answer": "",
    }
    lesson = _build_failure_lesson(probe)
    assert "Some query" in lesson["lesson"]

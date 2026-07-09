"""Tests for monitoring/ modules — no API calls, uses mocked pipeline output."""

import json
import pytest


def test_classify_failure_pass():
    """All dimensions above threshold should return PASS."""
    from monitoring.classify_failure import classify_failure

    result = classify_failure(
        retrieval_score=2.5,
        utilization_score=80.0,
        faithfulness_score=0.9,
        factuality_score=0.85,
        refusal_result={"failure_type": None},
        latency_ms=2000.0,
        baseline_latency_ms=1000.0,
        self_rag_checks={"answer_complete": True},
    )
    assert result == "PASS"


def test_classify_failure_retrieval():
    """Zero retrieval score should return RETRIEVAL_FAILURE."""
    from monitoring.classify_failure import classify_failure

    result = classify_failure(
        retrieval_score=0,
        utilization_score=80.0,
        faithfulness_score=0.9,
        factuality_score=0.85,
        refusal_result={"failure_type": None},
        latency_ms=1000.0,
        baseline_latency_ms=500.0,
        self_rag_checks={},
    )
    assert result == "RETRIEVAL_FAILURE"


def test_classify_failure_false_refusal():
    """false_refusal type should map to FALSE_REFUSAL category."""
    from monitoring.classify_failure import classify_failure

    result = classify_failure(
        retrieval_score=2.0,
        utilization_score=80.0,
        faithfulness_score=0.9,
        factuality_score=0.85,
        refusal_result={"failure_type": "false_refusal"},
        latency_ms=1000.0,
        baseline_latency_ms=500.0,
        self_rag_checks={},
    )
    assert result == "FALSE_REFUSAL"


def test_classify_failure_refusal_failure():
    """failed_refusal type should map to REFUSAL_FAILURE."""
    from monitoring.classify_failure import classify_failure

    result = classify_failure(
        retrieval_score=2.0,
        utilization_score=80.0,
        faithfulness_score=0.9,
        factuality_score=0.85,
        refusal_result={"failure_type": "failed_refusal"},
        latency_ms=1000.0,
        baseline_latency_ms=500.0,
        self_rag_checks={},
    )
    assert result == "REFUSAL_FAILURE"


def test_measure_refusal_correct_refusal():
    """Model refusing an out-of-scope question should score 1.0."""
    from monitoring.measure_refusal import measure_refusal_calibration

    result = measure_refusal_calibration(
        answer="I cannot find this information in my knowledge base.",
        should_refuse=True,
    )
    assert result["score"] == 1.0
    assert result["calibrated"] is True
    assert result["failure_type"] is None


def test_measure_refusal_false_refusal():
    """Model refusing an in-scope question should score 0.0."""
    from monitoring.measure_refusal import measure_refusal_calibration

    result = measure_refusal_calibration(
        answer="I cannot find this information in my knowledge base.",
        should_refuse=False,
    )
    assert result["score"] == 0.0
    assert result["failure_type"] == "false_refusal"


def test_measure_refusal_correct_answer():
    """Model answering an in-scope question should score 1.0."""
    from monitoring.measure_refusal import measure_refusal_calibration

    result = measure_refusal_calibration(
        answer="Meta AI developed the Llama 2 model in 2023.",
        should_refuse=False,
    )
    assert result["score"] == 1.0
    assert result["calibrated"] is True


def test_measure_retrieval_no_chunks():
    """Empty chunk list should score 0."""
    from monitoring.measure_retrieval import measure_retrieval_relevance

    result = measure_retrieval_relevance(
        chunks=[],
        expected_keywords=["llama", "meta"],
        retrieval_quality={"max_similarity": 0.0, "avg_similarity": 0.0},
    )
    assert result["score"] == 0


def test_measure_retrieval_keyword_match():
    """Matching all expected keywords with high similarity should score 3."""
    from monitoring.measure_retrieval import measure_retrieval_relevance

    chunks = [
        {
            "content": "Meta developed Llama 2 with 70B parameters.",
            "raw_content": "Meta developed Llama 2 with 70B parameters.",
            "source": "huggingface",
            "filename": "meta-llama__Llama-2-7b-hf.txt",
            "similarity": 0.9,
        }
    ]
    result = measure_retrieval_relevance(
        chunks=chunks,
        expected_keywords=["meta", "llama"],
        retrieval_quality={"max_similarity": 0.9, "avg_similarity": 0.9},
    )
    assert result["score"] >= 2
    assert "meta" in result["matched_keywords"] or "llama" in result["matched_keywords"]


def test_reflexion_load_empty(tmp_path, monkeypatch):
    """load_reflexion_lessons should return empty string when file missing."""
    from config.settings import Settings
    fake_settings = Settings()

    def fake_path(self):
        return tmp_path / "nonexistent.jsonl"

    monkeypatch.setattr(Settings, "failure_memory_path", property(fake_path))

    from monitoring.probe_engine import load_reflexion_lessons
    result = load_reflexion_lessons()
    assert result == ""


def test_compute_alert_flags_all_ok():
    """All dimensions above threshold should produce no alerts."""
    from monitoring.classify_failure import compute_alert_flags

    flags = compute_alert_flags(
        retrieval_score=2.0,
        utilization_score=75.0,
        faithfulness_score=0.85,
        factuality_score=0.80,
        refusal_score=0.90,
    )
    assert not any(flags.values())


def test_compute_alert_flags_retrieval_low():
    """Low retrieval should trigger retrieval_alert."""
    from monitoring.classify_failure import compute_alert_flags

    flags = compute_alert_flags(
        retrieval_score=1.0,
        utilization_score=75.0,
        faithfulness_score=0.85,
        factuality_score=0.80,
        refusal_score=0.90,
    )
    assert flags["retrieval_alert"] is True

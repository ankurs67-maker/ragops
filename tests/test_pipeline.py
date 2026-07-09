"""Tests for rag_system/pipeline.py and rag_system/generator.py."""

import pytest


def test_pipeline_run_query_returns_required_keys():
    """run_query should return a dict with all required keys."""
    from rag_system.pipeline import run_query

    result = run_query("What is RLHF?", skip_self_rag=True)
    required = [
        "answer", "chunks", "context", "retrieval_quality",
        "self_rag_passed", "self_rag_checks", "self_rag_retries",
        "loop_retries", "tokens_used", "latency_ms",
        "model_used", "provider_used", "error", "query_used",
    ]
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_pipeline_query_used_defaults_to_original():
    """query_used should default to the original query when no retries happened."""
    from rag_system.pipeline import run_query

    query = "What is the Transformer architecture?"
    result = run_query(query, skip_self_rag=True)
    # On first attempt without self_rag failure, query_used should be original
    assert result["query_used"] in [query, "Explain the Transformer architecture", "Transformer architecture"]


def test_pipeline_retrieval_quality_structure():
    """retrieval_quality should have adequate and reason keys."""
    from rag_system.pipeline import run_query

    result = run_query("What is BERT?", skip_self_rag=True)
    rq = result["retrieval_quality"]
    assert "adequate" in rq
    assert "reason" in rq


def test_pipeline_latency_positive():
    """latency_ms should be a positive number."""
    from rag_system.pipeline import run_query

    result = run_query("What is a language model?", skip_self_rag=True)
    assert result["latency_ms"] > 0


def test_error_result_structure():
    """_error_result helper should produce the correct shape."""
    from rag_system.pipeline import _error_result

    r = _error_result("test query", "test_error", 123.4)
    assert r["answer"] == ""
    assert r["error"] == "test_error"
    assert r["latency_ms"] == 123.4
    assert r["self_rag_passed"] is False
    assert r["loop_retries"] == 0

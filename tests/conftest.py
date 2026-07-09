"""Shared pytest fixtures for RAGOps test suite."""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Ensure test environment doesn't clobber production DB
os.environ.setdefault("RAGOPS_ENV", "test")
os.environ.setdefault("GROQ_API_KEY", "test_key_ci")
os.environ.setdefault("DEEPSEEK_API_KEY", "test_key_ci")
os.environ.setdefault("OPENROUTER_API_KEY", "test_key_ci")


@pytest.fixture(scope="session")
def tmp_db(tmp_path_factory):
    """Temporary SQLite database for testing."""
    db_dir = tmp_path_factory.mktemp("database")
    db_path = db_dir / "test_ragops.db"

    # Patch settings to use temp DB
    from config.settings import settings
    original_db_path = settings.__class__.db_path.fget

    monkeypatch_target = settings

    # Return a fake settings with temp db_path
    return db_path


@pytest.fixture
def sample_ground_truth():
    """Return a minimal ground truth entry for testing."""
    return {
        "query_id": "test_001",
        "query": "What company developed Llama 2?",
        "correct_answer": "Meta AI",
        "acceptable_answers": ["Meta", "Meta AI"],
        "category": "factual_recall",
        "difficulty": "easy",
        "should_refuse": False,
        "expected_chunk_keywords": ["meta", "llama"],
        "source_document": "meta-llama__Llama-2-7b-hf.txt",
        "multi_hop": False,
        "ground_truth_verified_date": "2026-06-29",
        "notes": "Test entry",
    }


@pytest.fixture
def sample_chunks():
    """Return a list of mock retrieved chunks."""
    return [
        {
            "chunk_id": "abc123",
            "content": "Meta AI developed the Llama 2 model in 2023.",
            "raw_content": "Meta AI developed the Llama 2 model in 2023.",
            "source": "huggingface",
            "filename": "meta-llama__Llama-2-7b-hf.txt",
            "similarity": 0.85,
        },
        {
            "chunk_id": "def456",
            "content": "The Llama family of models includes 7B, 13B, and 70B parameter versions.",
            "raw_content": "The Llama family of models includes 7B, 13B, and 70B parameter versions.",
            "source": "wikipedia",
            "filename": "LLaMA.txt",
            "similarity": 0.72,
        },
    ]


@pytest.fixture
def sample_measurement():
    """Return a sample measurement dict for DB testing."""
    return {
        "measurement_id": "test-meas-001",
        "probe_id": "test-probe-001",
        "run_id": "test-run-001",
        "timestamp": "2026-06-29T12:00:00+00:00",
        "retrieval_relevance_score": 2.5,
        "context_utilization_score": 85.0,
        "faithfulness_score": 0.9,
        "factuality_score": 0.95,
        "refusal_calibration_score": 1.0,
        "failure_category": "PASS",
        "judge_model_version": "deepseek-chat",
        "judge_confidence": 0.85,
        "measurement_details": "{}",
    }

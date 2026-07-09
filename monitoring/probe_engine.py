"""Probe engine: runs the full RAG pipeline for one ground truth query.

Reflexion integration: reads failure_memory.jsonl before each probe cycle,
builds a session_context string summarising past failure lessons, and passes
it to the pipeline so the generator can avoid repeating mistakes.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import settings
from monitoring.classify_failure import classify_failure, compute_alert_flags
from monitoring.measure_faithfulness import measure_faithfulness
from monitoring.measure_factuality import measure_factuality
from monitoring.measure_refusal import measure_refusal_calibration
from monitoring.measure_retrieval import measure_retrieval_relevance
from monitoring.measure_utilization import measure_context_utilization
from rag_system.pipeline import run_query
from rag_system.retriever import format_context
from utils.logger import get_logger

logger = get_logger(__name__)

_REFLEXION_MAX_LESSONS = 5  # Maximum number of past lessons to include per cycle


def load_reflexion_lessons(n: int = _REFLEXION_MAX_LESSONS) -> str:
    """Load the most recent failure lessons from failure_memory.jsonl.

    Returns a formatted string suitable for inclusion in the generation prompt.
    Returns empty string if file doesn't exist or has no entries.
    """
    memory_path = settings.failure_memory_path
    if not memory_path.exists():
        return ""

    lines = memory_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return ""

    # Take the last N entries (most recent lessons)
    recent_lines = lines[-n:] if len(lines) > n else lines
    lessons = []
    for line in recent_lines:
        try:
            entry = json.loads(line)
            lesson = entry.get("lesson", "")
            category = entry.get("failure_category", "UNKNOWN")
            if lesson:
                lessons.append(f"- [{category}] {lesson}")
        except json.JSONDecodeError:
            continue

    if not lessons:
        return ""

    return "\n".join(lessons)


def run_probe(
    ground_truth_entry: dict,
    session_context: str = "",
    baseline_latency_ms: float = 0.0,
) -> dict:
    """Run the full RAG pipeline for one ground truth query and measure all 5 dimensions.

    Args:
        ground_truth_entry: One entry from ground_truth.json
        session_context: Reflexion lessons to pass to the generator
        baseline_latency_ms: Historical average latency for this system (for latency alerts)

    Returns a comprehensive result dict containing probe_result and measurement data.
    """
    query_id = ground_truth_entry["query_id"]
    query = ground_truth_entry["query"]
    correct_answer = ground_truth_entry["correct_answer"]
    acceptable_answers = ground_truth_entry.get("acceptable_answers", [])
    expected_keywords = ground_truth_entry.get("expected_chunk_keywords", [])
    should_refuse = ground_truth_entry.get("should_refuse", False)

    probe_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Running probe",
        extra={"query_id": query_id, "probe_id": probe_id[:8]},
    )

    # Run the full pipeline (Reflexion session_context is passed here)
    pipeline_result = run_query(
        query=query,
        session_context=session_context,
    )

    answer = pipeline_result.get("answer", "")
    chunks = pipeline_result.get("chunks", [])
    context = pipeline_result.get("context", "")
    latency_ms = pipeline_result.get("latency_ms", 0.0)
    self_rag_passed = pipeline_result.get("self_rag_passed", False)
    self_rag_checks = pipeline_result.get("self_rag_checks", {})
    self_rag_retries = pipeline_result.get("self_rag_retries", 0)
    loop_retries = pipeline_result.get("loop_retries", 0)
    tokens_used = pipeline_result.get("tokens_used", 0)
    model_used = pipeline_result.get("model_used", settings.llm_model)
    provider_used = pipeline_result.get("provider_used", settings.llm_provider)
    retrieval_quality = pipeline_result.get("retrieval_quality", {})

    # Measure all 5 dimensions
    retrieval_result = measure_retrieval_relevance(chunks, expected_keywords, retrieval_quality)
    utilization_result = measure_context_utilization(query, answer, context)
    faithfulness_result = measure_faithfulness(query, answer, context, self_rag_passed, self_rag_checks)
    factuality_result = measure_factuality(query, answer, correct_answer, acceptable_answers)
    refusal_result = measure_refusal_calibration(answer, should_refuse)

    retrieval_score = float(retrieval_result["score"])
    utilization_score = float(utilization_result["score"])
    faithfulness_score = float(faithfulness_result["score"])
    factuality_score = float(factuality_result["score"])
    refusal_score = float(refusal_result["score"])

    # Classify failure
    failure_category = classify_failure(
        retrieval_score=retrieval_score,
        utilization_score=utilization_score,
        faithfulness_score=faithfulness_score,
        factuality_score=factuality_score,
        refusal_result=refusal_result,
        latency_ms=latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        self_rag_checks=self_rag_checks,
    )

    alert_flags = compute_alert_flags(
        retrieval_score, utilization_score, faithfulness_score, factuality_score, refusal_score
    )

    # Build measurement details JSON blob for the DB
    measurement_details = {
        "retrieval": retrieval_result,
        "utilization": utilization_result,
        "faithfulness": faithfulness_result,
        "factuality": factuality_result,
        "refusal": refusal_result,
        "self_rag_checks": self_rag_checks,
        "self_rag_retries": self_rag_retries,
        "loop_retries": loop_retries,
        "chunks_retrieved": len(chunks),
        "top_chunk_source": chunks[0]["filename"] if chunks else "",
        "query_used": pipeline_result.get("query_used", query),
        "tokens_used": tokens_used,
        "reflexion_lessons_applied": bool(session_context.strip()),
        # Reproducibility: which provider/model actually generated the answer
        "generation_provider": provider_used,
        "generation_model": model_used,
    }

    logger.info(
        "Probe complete",
        extra={
            "query_id": query_id,
            "failure_category": failure_category,
            "retrieval": retrieval_score,
            "faithfulness": round(faithfulness_score, 3),
            "factuality": round(factuality_score, 3),
            "latency_ms": round(latency_ms),
        },
    )

    return {
        "probe_id": probe_id,
        "query_id": query_id,
        "timestamp": timestamp,
        "query": query,
        "answer": answer,
        "correct_answer": correct_answer,
        "model_used": model_used,
        "provider_used": provider_used,
        "latency_ms": latency_ms,
        "tokens_used": tokens_used,
        "pipeline_error": pipeline_result.get("error"),
        "chunks": chunks,
        # Scores
        "retrieval_relevance": retrieval_score,
        "context_utilization": utilization_score,
        "faithfulness": faithfulness_score,
        "factuality": factuality_score,
        "refusal_calibration": refusal_score,
        "failure_category": failure_category,
        "alert_flags": alert_flags,
        "measurement_details": measurement_details,
    }

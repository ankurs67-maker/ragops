"""Run a complete probe cycle across all 20 ground truth queries.

Reflexion loop:
  1. Before cycle starts: load failure_memory.jsonl → build session_context
  2. Run all probes with session_context passed to generator
  3. After cycle ends: write new failure lessons for failed probes to failure_memory.jsonl

This ensures each probe cycle learns from previous failures.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from database.db_client import insert_measurement, insert_probe_result
from monitoring.probe_engine import load_reflexion_lessons, run_probe
from utils.llm_client import get_scoring_model_version
from utils.logger import get_logger

logger = get_logger(__name__)

_FAILURE_LESSON_TEMPLATES: dict[str, str] = {
    "RETRIEVAL_FAILURE": (
        "Retrieval failed for this query type. Consider using more specific keywords "
        "or rephrase the query to match document vocabulary."
    ),
    "CONTEXT_BYPASS": (
        "The model ignored retrieved context and answered from memory. "
        "Ensure the system prompt strongly instructs to use only the provided context."
    ),
    "FAITHFULNESS_FAILURE": (
        "The answer contained claims not supported by the retrieved context. "
        "Strengthen instructions to cite only context-supported facts."
    ),
    "FACTUAL_ERROR": (
        "The answer contained factual errors. Increase retrieval top-k to provide more context, "
        "or check that the relevant document is in the corpus."
    ),
    "REFUSAL_FAILURE": (
        "The model answered an out-of-scope question instead of refusing. "
        "Reinforce refusal instructions for questions outside the AI/ML knowledge domain."
    ),
    "FALSE_REFUSAL": (
        "The model refused to answer an in-scope question. "
        "Check that the relevant document is indexed and that retrieval is working correctly."
    ),
    "LATENCY_DEGRADATION": (
        "Response latency exceeded 3x baseline. Check for provider rate limiting "
        "or consider switching to a faster model tier."
    ),
    "PARTIAL_ANSWER": (
        "The answer was incomplete. Increase max_answer_tokens or improve "
        "the prompt to encourage comprehensive responses."
    ),
}


# Which dimension score best explains each failure category — used to make
# Reflexion lessons specific instead of generic boilerplate.
_CATEGORY_DIMENSION: dict[str, str] = {
    "RETRIEVAL_FAILURE": "retrieval_relevance",
    "CONTEXT_BYPASS": "context_utilization",
    "FAITHFULNESS_FAILURE": "faithfulness",
    "FACTUAL_ERROR": "factuality",
    "REFUSAL_FAILURE": "refusal_calibration",
    "FALSE_REFUSAL": "refusal_calibration",
}


def _build_failure_lesson(probe_result: dict) -> dict:
    """Build a structured failure lesson for persistence in failure_memory.jsonl."""
    category = probe_result["failure_category"]
    template_lesson = _FAILURE_LESSON_TEMPLATES.get(category, "Review this failure type.")

    # Make the lesson specific: include the actual query and the score of the
    # dimension that failed, so the generator sees a concrete example rather
    # than generic advice that never changes behaviour.
    dim = _CATEGORY_DIMENSION.get(category)
    dim_score = probe_result.get(dim) if dim else None
    specifics = f"Example failed query: '{probe_result['query'][:80]}'"
    if dim is not None and dim_score is not None:
        specifics += f"; {dim}={dim_score}"
    template_lesson = f"{template_lesson} ({specifics})"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_id": probe_result["query_id"],
        "failure_category": category,
        "query": probe_result["query"][:100],
        "answer_snippet": probe_result.get("answer", "")[:150],
        "scores": {
            "retrieval": probe_result.get("retrieval_relevance"),
            "utilization": probe_result.get("context_utilization"),
            "faithfulness": probe_result.get("faithfulness"),
            "factuality": probe_result.get("factuality"),
            "refusal": probe_result.get("refusal_calibration"),
        },
        "lesson": template_lesson,
    }


def _append_failure_lessons(lessons: list[dict]) -> None:
    """Append new failure lessons to failure_memory.jsonl."""
    memory_path = settings.failure_memory_path
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    with memory_path.open("a", encoding="utf-8") as f:
        for lesson in lessons:
            f.write(json.dumps(lesson) + "\n")

    logger.info(
        "Failure lessons written",
        extra={"count": len(lessons), "path": str(memory_path)},
    )


def run_probe_cycle(max_queries: int = 0) -> dict:
    """Run a complete probe cycle across all ground truth queries.

    Args:
        max_queries: If > 0, limit to this many queries (for testing).

    Returns summary dict with cycle statistics.
    """
    # Step 1: Load Reflexion lessons from previous cycles
    session_context = load_reflexion_lessons()
    if session_context:
        logger.info(
            "Reflexion: loaded failure lessons",
            extra={"lesson_count": session_context.count("\n-") + 1},
        )
    else:
        logger.info("Reflexion: no prior lessons (first run or empty memory)")

    # Step 2: Load ground truth
    ground_truth_path = settings.ground_truth_path
    with ground_truth_path.open(encoding="utf-8") as f:
        ground_truth = json.load(f)

    if max_queries > 0:
        ground_truth = ground_truth[:max_queries]

    logger.info(
        "Probe cycle started",
        extra={"queries": len(ground_truth), "reflexion_active": bool(session_context)},
    )

    cycle_start = time.time()
    cycle_run_id = str(uuid.uuid4())
    results: list[dict] = []
    failures: list[dict] = []

    for entry in ground_truth:
        try:
            result = run_probe(
                ground_truth_entry=entry,
                session_context=session_context,
            )
            results.append(result)

            # Build ground truth fields from entry
            gt_category = entry.get("category", "unknown")
            gt_difficulty = entry.get("difficulty", "medium")
            details = result["measurement_details"]

            # Persist to database — field names must match schema.sql exactly
            probe_row = {
                "probe_id": result["probe_id"],
                "run_id": cycle_run_id,
                "timestamp": result["timestamp"],
                "query_id": result["query_id"],
                "query_text": result["query"],
                "category": gt_category,
                "difficulty": gt_difficulty,
                "retrieved_chunks": json.dumps([c.get("filename", "") for c in result.get("chunks", [])[:3]]),
                "generated_answer": result["answer"],
                "correct_answer": result["correct_answer"],
                "answer_correct": str(result["factuality"] >= 0.8),
                "refused_when_should": str(details.get("refusal", {}).get("calibrated", True)),
                "latency_retrieval_ms": None,
                "latency_generation_ms": None,
                "latency_total_ms": int(result["latency_ms"]),
            }
            insert_probe_result(probe_row)

            measurement_row = {
                "measurement_id": str(uuid.uuid4()),
                "probe_id": result["probe_id"],
                "run_id": cycle_run_id,
                "timestamp": result["timestamp"],
                "retrieval_relevance_score": result["retrieval_relevance"],
                "context_utilization_score": result["context_utilization"],
                "faithfulness_score": result["faithfulness"],
                "factuality_score": result["factuality"],
                "refusal_calibration_score": result["refusal_calibration"],
                "failure_category": result["failure_category"],
                "judge_model_version": get_scoring_model_version(),
                "judge_confidence": 0.85,
                "measurement_details": json.dumps(result["measurement_details"]),
            }
            insert_measurement(measurement_row)

            # Collect failures for Reflexion
            if result["failure_category"] != "PASS":
                failures.append(result)

        except Exception as exc:
            logger.error(
                "Probe failed unexpectedly",
                extra={"query_id": entry["query_id"], "error": str(exc)},
            )

    cycle_elapsed = time.time() - cycle_start

    # Step 3: Write failure lessons for Reflexion (next cycle will read these)
    if failures:
        lessons = [_build_failure_lesson(r) for r in failures]
        _append_failure_lessons(lessons)

    # Compute cycle summary stats
    pass_count = sum(1 for r in results if r["failure_category"] == "PASS")
    fail_count = len(results) - pass_count

    avg_retrieval = (
        sum(r["retrieval_relevance"] for r in results) / len(results) if results else 0.0
    )
    avg_faithfulness = (
        sum(r["faithfulness"] for r in results) / len(results) if results else 0.0
    )
    avg_factuality = (
        sum(r["factuality"] for r in results) / len(results) if results else 0.0
    )
    avg_utilization = (
        sum(r["context_utilization"] for r in results) / len(results) if results else 0.0
    )
    avg_refusal = (
        sum(r["refusal_calibration"] for r in results) / len(results) if results else 0.0
    )

    failure_dist: dict[str, int] = {}
    for r in results:
        cat = r["failure_category"]
        failure_dist[cat] = failure_dist.get(cat, 0) + 1

    summary = {
        "cycle_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_probes": len(results),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "failure_distribution": failure_dist,
        "avg_retrieval_relevance": round(avg_retrieval, 3),
        "avg_context_utilization": round(avg_utilization, 1),
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_factuality": round(avg_factuality, 3),
        "avg_refusal_calibration": round(avg_refusal, 3),
        "reflexion_lessons_used": bool(session_context),
        "new_failure_lessons_written": len(failures),
        "cycle_elapsed_seconds": round(cycle_elapsed, 1),
    }

    logger.info(
        "Probe cycle complete",
        extra=summary,
    )
    return summary


if __name__ == "__main__":
    summary = run_probe_cycle()
    print("\n=== PROBE CYCLE SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

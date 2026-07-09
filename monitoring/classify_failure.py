"""Classify probe results into one of 9 failure taxonomy categories.

Taxonomy:
  PASS              — all dimensions passed thresholds
  RETRIEVAL_FAILURE — retrieval relevance <= 0 (wrong or no chunks returned)
  CONTEXT_BYPASS    — context utilization < threshold (model ignored retrieved context)
  FAITHFULNESS_FAILURE — faithfulness < threshold (answer not grounded in context)
  FACTUAL_ERROR     — factuality < threshold (answer contains factual mistakes)
  REFUSAL_FAILURE   — should_refuse=True but model answered anyway
  FALSE_REFUSAL     — should_refuse=False but model refused to answer
  LATENCY_DEGRADATION — response latency > 3x baseline
  PARTIAL_ANSWER    — answer is incomplete (Self-RAG completeness check failed)
"""

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def classify_failure(
    retrieval_score: float,
    utilization_score: float,
    faithfulness_score: float,
    factuality_score: float,
    refusal_result: dict,
    latency_ms: float,
    baseline_latency_ms: float,
    self_rag_checks: dict,
) -> str:
    """Classify a single probe into one failure category.

    Returns a string from the taxonomy (first matching category wins).
    """
    # Refusal errors take priority (they represent fundamental alignment issues)
    refusal_type = refusal_result.get("failure_type")
    if refusal_type == "failed_refusal":
        return "REFUSAL_FAILURE"
    if refusal_type == "false_refusal":
        return "FALSE_REFUSAL"

    # A correctly-refused out-of-scope query is a PASS by definition: no
    # relevant chunks exist for it, so low retrieval/utilization scores are
    # the expected state, not a failure. Without this, every out-of-scope
    # query with empty retrieval is misclassified as RETRIEVAL_FAILURE.
    if refusal_result.get("should_refuse") and refusal_result.get("calibrated"):
        return "PASS"

    # Retrieval failure — nothing relevant found
    if retrieval_score <= 0:
        return "RETRIEVAL_FAILURE"

    # Context bypass — model ignored the retrieved context
    if utilization_score < settings.alert_utilization_threshold:
        return "CONTEXT_BYPASS"

    # Faithfulness failure — answer not grounded in context
    if faithfulness_score < settings.alert_faithfulness_threshold:
        return "FAITHFULNESS_FAILURE"

    # Factual error — answer is wrong
    if factuality_score < settings.alert_factuality_threshold:
        return "FACTUAL_ERROR"

    # Latency degradation
    if baseline_latency_ms > 0 and latency_ms > baseline_latency_ms * settings.alert_latency_multiplier:
        return "LATENCY_DEGRADATION"

    # Partial answer — Self-RAG completeness check failed
    if self_rag_checks and not self_rag_checks.get("answer_complete", True):
        return "PARTIAL_ANSWER"

    return "PASS"


def compute_alert_flags(
    retrieval_score: float,
    utilization_score: float,
    faithfulness_score: float,
    factuality_score: float,
    refusal_score: float,
) -> dict:
    """Return per-dimension boolean alert flags."""
    return {
        "retrieval_alert": retrieval_score < settings.alert_retrieval_threshold,
        "utilization_alert": utilization_score < settings.alert_utilization_threshold,
        "faithfulness_alert": faithfulness_score < settings.alert_faithfulness_threshold,
        "factuality_alert": factuality_score < settings.alert_factuality_threshold,
        "refusal_alert": refusal_score < settings.alert_refusal_threshold,
    }

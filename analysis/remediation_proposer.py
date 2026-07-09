"""Propose concrete remediations based on active alerts and failure patterns.

No LLM calls — uses a rule-based decision tree keyed on failure_category
and dimension thresholds. Each remediation includes root_cause analysis,
priority (critical/high/medium/low), and specific actionable steps.
"""

import uuid
from datetime import datetime, timezone

from config.settings import settings
from database.db_client import (
    get_dimension_averages,
    get_failure_distribution,
    get_pending_remediations,
    insert_remediation,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Map failure category → (root_cause, remediation_text, priority, steps)
_REMEDIATION_RULES: dict[str, dict] = {
    "RETRIEVAL_FAILURE": {
        "root_cause": "Vector index not finding relevant chunks for query types.",
        "remediation_text": (
            "Expand the document corpus with more targeted articles on the failing query topic. "
            "Re-run ingestion and rebuild the ChromaDB index."
        ),
        "priority": "critical",
        "steps": [
            "Identify query_ids where retrieval_relevance_score = 0",
            "Check if expected_chunk_keywords appear in any indexed document",
            "Add missing documents to data/raw/ and re-run build_index.py",
            "Verify with: python ingestion/build_index.py",
        ],
    },
    "CONTEXT_BYPASS": {
        "root_cause": "Model is answering from parametric memory instead of retrieved context.",
        "remediation_text": (
            "Strengthen the system prompt to explicitly forbid answers not grounded in context. "
            "Consider adding 'Answer ONLY from the passages below' at the start of the user prompt."
        ),
        "priority": "high",
        "steps": [
            "Review system prompt in rag_system/prompt_templates.py",
            "Add stronger grounding instruction at start of get_generation_prompt()",
            "Consider lowering generation_temperature to 0.0 to reduce creativity",
            "Test with 5 manual queries after change",
        ],
    },
    "FAITHFULNESS_FAILURE": {
        "root_cause": "Generated answers contain claims not in retrieved context (hallucination).",
        "remediation_text": (
            "The model is adding facts from training memory. Reduce temperature, "
            "add more specific context from better retrieval (increase top_k), "
            "or use a factual-only model instruction."
        ),
        "priority": "high",
        "steps": [
            "Reduce generation_temperature from 0.1 to 0.0 in settings",
            "Increase top_k from 5 to 7 for more context coverage",
            "Add 'Do not include any fact not explicitly stated in the passages.' to system prompt",
            "Run probe cycle and compare faithfulness_score averages",
        ],
    },
    "FACTUAL_ERROR": {
        "root_cause": "Retrieved context contains outdated or incorrect information.",
        "remediation_text": (
            "Refresh the document corpus. Check which documents contain the incorrect facts "
            "and replace them with updated versions."
        ),
        "priority": "high",
        "steps": [
            "Identify probe_ids with factuality_score < 0.6",
            "Check which source documents were retrieved for those probes",
            "Verify the factual claims in those documents against authoritative sources",
            "Update or replace outdated documents and rebuild index",
        ],
    },
    "REFUSAL_FAILURE": {
        "root_cause": "Model answering out-of-scope questions instead of refusing.",
        "remediation_text": (
            "Add explicit out-of-scope detection to the system prompt. "
            "Enumerate the topic boundaries of the knowledge base."
        ),
        "priority": "high",
        "steps": [
            "Add to system prompt: 'This knowledge base covers AI/ML topics only.'",
            "Add: 'If the question is about finance, politics, or events after 2024, refuse.'",
            "Test with gt_004 and gt_014 (out-of-scope queries)",
        ],
    },
    "FALSE_REFUSAL": {
        "root_cause": "Model refusing to answer in-scope questions (over-refusal).",
        "remediation_text": (
            "The model is too cautious. Check retrieval quality for the refusing queries. "
            "If context is found, the model may be misinterpreting the system prompt."
        ),
        "priority": "medium",
        "steps": [
            "For each FALSE_REFUSAL probe, check retrieval_relevance_score",
            "If retrieval_relevance_score >= 2: model is ignoring good context",
            "If retrieval_relevance_score = 0: add missing documents to corpus",
            "Soften system prompt to allow inference when context is present",
        ],
    },
    "LATENCY_DEGRADATION": {
        "root_cause": "Response latency exceeding 3x baseline, likely provider rate limiting.",
        "remediation_text": (
            "Switch primary LLM provider or reduce probe frequency. "
            "Check for rate limit errors in logs."
        ),
        "priority": "medium",
        "steps": [
            "Check logs/ragops.log for RateLimitError entries",
            "Switch LLM_PROVIDER from 'groq' to 'deepseek' in .env",
            "Reduce probe_schedule_hours from [0, 12] to [12] temporarily",
            "Monitor latency after change",
        ],
    },
    "PARTIAL_ANSWER": {
        "root_cause": "Answers are being cut off or are incomplete.",
        "remediation_text": (
            "Increase max_answer_tokens to allow longer answers. "
            "Check if the model is hitting token limits."
        ),
        "priority": "low",
        "steps": [
            "Increase max_answer_tokens from 500 to 750 in settings.py",
            "Review prompt to ensure the model knows to give complete answers",
        ],
    },
}


def propose_remediations(days: int = 1) -> list[dict]:
    """Generate and persist remediation proposals for current alerts.

    Reads recent failure distribution, checks against thresholds,
    and inserts remediation rows for any new issues.
    Returns list of proposed remediations.
    """
    dist = get_failure_distribution(days=days)
    avgs = get_dimension_averages(days=days)
    pending = {r["alert_type"] for r in get_pending_remediations()}

    total = sum(dist.values())
    if total == 0:
        logger.info("No probes found for remediation analysis")
        return []

    proposed: list[dict] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for category, count in dist.items():
        if category == "PASS":
            continue
        if category not in _REMEDIATION_RULES:
            continue
        rate = count / total

        # Threshold: more than 10% of probes failing with this category
        if rate < 0.10:
            continue

        # Skip if we already have a pending remediation for this type
        if category in pending:
            logger.debug("Skipping — pending remediation already exists", extra={"category": category})
            continue

        rule = _REMEDIATION_RULES[category]
        rem_id = str(uuid.uuid4())
        remediation = {
            "remediation_id": rem_id,
            "triggered_by": f"probe_cycle_{timestamp[:10]}",
            "timestamp": timestamp,
            "alert_type": category,
            "root_cause": rule["root_cause"],
            "confidence": round(rate, 2),
            "remediation_text": rule["remediation_text"],
            "specific_steps": rule["steps"],
            "priority": rule["priority"],
            "status": "pending",
            "outcome": None,
        }
        insert_remediation(remediation)
        proposed.append(remediation)
        logger.info(
            "Remediation proposed",
            extra={"category": category, "rate_pct": round(rate * 100, 1), "priority": rule["priority"]},
        )

    return proposed

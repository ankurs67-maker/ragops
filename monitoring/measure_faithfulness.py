"""Measure faithfulness (0.0–1.0) via LLM judge, with Self-RAG penalty.

Faithfulness measures whether every factual claim in the answer is directly
supported by the retrieved context. High faithfulness means the model stuck
to the documents; low faithfulness means it hallucinated or used training memory.

Self-RAG penalty: if the Self-RAG loop failed (self_rag_passed=False),
the raw faithfulness score is penalised by a configurable multiplier.
This reflects that answers that didn't pass verification are less trustworthy.
"""

import re

from config.settings import settings
from rag_system.prompt_templates import get_faithfulness_judge_prompt
from utils.llm_client import call_llm
from utils.logger import get_logger

logger = get_logger(__name__)

_SELF_RAG_PENALTY = 0.85  # Multiply raw score by this if Self-RAG failed


def measure_faithfulness(
    query: str,
    answer: str,
    context: str,
    self_rag_passed: bool = True,
    self_rag_checks: dict | None = None,
) -> dict:
    """Score faithfulness 0.0-1.0 using an LLM judge.

    Args:
        query: The question asked.
        answer: The generated answer.
        context: The retrieved context string shown to the model.
        self_rag_passed: Whether the Self-RAG verification loop passed.
        self_rag_checks: Raw Self-RAG check dict (answer_grounded, etc.).
                         Penalty applies only when BOTH self_rag_passed=False
                         AND answer_grounded=False.

    Returns dict with:
        score: float 0.0-1.0 (after Self-RAG penalty if applicable)
        raw_score: float 0.0-1.0 (before penalty)
        self_rag_penalty_applied: bool
        raw_response: str
        tokens_used: int
        error: str | None
    """
    if not answer or not context:
        return {
            "score": 0.0,
            "raw_score": 0.0,
            "self_rag_penalty_applied": False,
            "raw_response": "",
            "tokens_used": 0,
            "error": "missing_answer_or_context",
        }

    # Refusal answers are faithful by definition — they claim nothing unsupported
    if "cannot find this information" in answer.lower():
        return {
            "score": 1.0,
            "raw_score": 1.0,
            "self_rag_penalty_applied": False,
            "raw_response": "auto:refusal",
            "tokens_used": 0,
            "error": None,
        }

    # Judge must see the FULL context: with sibling expansion the answer's
    # supporting chunk often sits beyond the first few chunks, and a
    # truncated slice makes the judge score correct answers as ungrounded.
    prompt = get_faithfulness_judge_prompt(query, answer, context)
    result = call_llm(
        prompt=prompt,
        system="You are a faithfulness evaluator. Respond with only a decimal between 0.0 and 1.0.",
        model_tier="scoring",
        temperature=settings.judge_temperature,
        max_tokens=10,
    )

    raw_text = result.get("text", "").strip()
    tokens = result.get("tokens_used", 0)
    error = result.get("error")

    try:
        nums = re.findall(r"[0-9]+\.?[0-9]*", raw_text)
        raw_score = float(nums[0]) if nums else 0.5
        raw_score = max(0.0, min(1.0, raw_score))
    except (ValueError, IndexError):
        raw_score = 0.5
        error = f"parse_failed: {raw_text!r}"

    checks = self_rag_checks or {}
    apply_penalty = (
        not self_rag_passed
        and not checks.get("answer_grounded", True)
    )
    penalty_applied = apply_penalty
    final_score = raw_score * _SELF_RAG_PENALTY if apply_penalty else raw_score

    logger.debug(
        "Faithfulness scored",
        extra={"raw_score": raw_score, "final_score": final_score, "penalty": penalty_applied},
    )
    return {
        "score": round(final_score, 4),
        "raw_score": round(raw_score, 4),
        "self_rag_penalty_applied": penalty_applied,
        "raw_response": raw_text,
        "tokens_used": tokens,
        "error": error,
    }

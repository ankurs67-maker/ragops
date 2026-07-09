"""Measure factuality (0.0–1.0) via LLM judge.

Factuality measures whether the answer's stated facts (model names, parameter
counts, benchmark scores, dates, organisations) are objectively correct.
Unlike faithfulness (which compares to context), factuality compares to
ground truth knowledge about AI/ML.
"""

import re

from config.settings import settings
from rag_system.prompt_templates import get_factuality_judge_prompt
from utils.llm_client import call_llm
from utils.logger import get_logger

logger = get_logger(__name__)


def measure_factuality(
    query: str,
    answer: str,
    correct_answer: str,
    acceptable_answers: list[str],
) -> dict:
    """Score factuality 0.0-1.0.

    First checks for exact/acceptable answer match (fast path, no LLM).
    Falls back to LLM judge for free-form answers.

    Args:
        query: The question asked.
        answer: The generated answer.
        correct_answer: The canonical correct answer from ground_truth.json.
        acceptable_answers: List of acceptable alternative correct answers.

    Returns dict with:
        score: float 0.0-1.0
        exact_match: bool
        acceptable_match: bool
        raw_response: str
        tokens_used: int
        error: str | None
    """
    if not answer:
        return {
            "score": 0.0,
            "exact_match": False,
            "acceptable_match": False,
            "raw_response": "",
            "tokens_used": 0,
            "error": "empty_answer",
        }

    answer_lower = answer.lower().strip()
    correct_lower = correct_answer.lower().strip()

    # Exact match (fast path)
    if correct_lower in answer_lower or answer_lower == correct_lower:
        return {
            "score": 1.0,
            "exact_match": True,
            "acceptable_match": True,
            "raw_response": "auto:exact_match",
            "tokens_used": 0,
            "error": None,
        }

    # Acceptable answer match (fast path)
    for acc in acceptable_answers:
        if acc.lower() in answer_lower:
            return {
                "score": 1.0,
                "exact_match": False,
                "acceptable_match": True,
                "raw_response": f"auto:acceptable_match:{acc}",
                "tokens_used": 0,
                "error": None,
            }

    # Refusal answers for should_refuse=True questions are factually correct
    if "cannot find this information" in answer_lower:
        return {
            "score": 1.0,
            "exact_match": False,
            "acceptable_match": True,
            "raw_response": "auto:correct_refusal",
            "tokens_used": 0,
            "error": None,
        }

    # No exact match — use LLM judge
    prompt = get_factuality_judge_prompt(query, answer)
    result = call_llm(
        prompt=prompt,
        system="You are a factuality evaluator for AI/ML knowledge. Respond with only a decimal 0.0-1.0.",
        model_tier="scoring",
        temperature=settings.judge_temperature,
        max_tokens=10,
    )

    raw_text = result.get("text", "").strip()
    tokens = result.get("tokens_used", 0)
    error = result.get("error")

    try:
        nums = re.findall(r"[0-9]+\.?[0-9]*", raw_text)
        score = float(nums[0]) if nums else 0.5
        score = max(0.0, min(1.0, score))
    except (ValueError, IndexError):
        score = 0.5
        error = f"parse_failed: {raw_text!r}"

    logger.debug(
        "Factuality scored",
        extra={"score": score, "raw": raw_text},
    )
    return {
        "score": round(score, 4),
        "exact_match": False,
        "acceptable_match": False,
        "raw_response": raw_text,
        "tokens_used": tokens,
        "error": error,
    }

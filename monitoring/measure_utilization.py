"""Measure context utilization (0–100) via LLM judge.

Context utilization measures what percentage of the answer's factual content
came from the retrieved context versus the model's training memory.
100 = entirely grounded in retrieved context
0   = entirely from parametric memory, context ignored
"""

from config.settings import settings
from rag_system.prompt_templates import get_utilization_judge_prompt
from utils.llm_client import call_llm
from utils.logger import get_logger

logger = get_logger(__name__)


def measure_context_utilization(
    query: str,
    answer: str,
    context: str,
) -> dict:
    """Score context utilization 0-100 using an LLM judge.

    Returns dict with:
        score: float 0-100
        raw_response: str
        tokens_used: int
        error: str | None
    """
    if not answer or not context:
        return {
            "score": 0.0,
            "raw_response": "",
            "tokens_used": 0,
            "error": "missing_answer_or_context",
        }

    # Refusal answer: context was "used" correctly (model correctly recognised no answer)
    if "cannot find this information" in answer.lower():
        return {
            "score": 100.0,
            "raw_response": "auto:refusal",
            "tokens_used": 0,
            "error": None,
        }

    # Full context — a truncated slice hides the chunk the answer came from
    # and misreads context-grounded answers as parametric (CONTEXT_BYPASS).
    prompt = get_utilization_judge_prompt(query, answer, context)
    result = call_llm(
        prompt=prompt,
        system="You are a context utilization evaluator. Respond with only an integer 0-100.",
        model_tier="scoring",
        temperature=settings.judge_temperature,
        max_tokens=10,
    )

    raw = result.get("text", "").strip()
    tokens = result.get("tokens_used", 0)
    error = result.get("error")

    try:
        # Extract first integer from response
        import re
        nums = re.findall(r"\d+", raw)
        score = float(nums[0]) if nums else 50.0
        score = max(0.0, min(100.0, score))
    except (ValueError, IndexError):
        score = 50.0
        error = f"parse_failed: {raw!r}"

    logger.debug(
        "Utilization scored",
        extra={"score": score, "raw": raw},
    )
    return {
        "score": score,
        "raw_response": raw,
        "tokens_used": tokens,
        "error": error,
    }

"""Measure refusal calibration (0.0–1.0) — no LLM call needed.

Refusal calibration checks whether the model refuses correctly:
- For should_refuse=True questions: did the model refuse? (it should)
- For should_refuse=False questions: did the model NOT refuse? (it should answer)

A model that correctly refuses out-of-scope questions and answers in-scope
questions scores 1.0. A model that refuses everything or answers everything scores
much lower on average.
"""

import re

from utils.logger import get_logger

logger = get_logger(__name__)

_REFUSAL_PHRASES = [
    "cannot find this information",
    "i cannot find",
    "don't have information",
    "not in my knowledge base",
    "i don't know",
    "cannot answer",
    "no information available",
    "i'm unable to find",
    "unable to find",
    "no relevant information",
]


def _is_refusal(answer: str) -> bool:
    """Detect whether the answer IS a refusal, not whether it
    merely mentions refusal-adjacent language while still
    providing a substantive answer.

    A true refusal is short and its content is essentially just
    the refusal statement. A hedged-but-substantive answer is
    much longer than the refusal phrase itself and contains
    other factual content before/around it.
    """
    answer_stripped = answer.strip()
    answer_lower = answer_stripped.lower()

    if not answer_stripped:
        return False

    matched_phrase = next(
        (p for p in _REFUSAL_PHRASES if p in answer_lower), None
    )
    if matched_phrase is None:
        return False

    # If the refusal phrase makes up the majority of a short answer,
    # treat it as a real refusal (e.g. exactly the canonical phrase,
    # or a one-sentence equivalent).
    word_count = len(answer_stripped.split())
    if word_count <= 15:
        return True

    # For longer answers, only count it as a refusal if the refusal
    # phrase appears in the FIRST sentence (i.e. it leads with refusal
    # rather than arriving at it after giving substantive content).
    first_sentence = re.split(r'(?<=[.!?])\s+', answer_stripped, maxsplit=1)[0].lower()
    return matched_phrase in first_sentence


def measure_refusal_calibration(
    answer: str,
    should_refuse: bool,
) -> dict:
    """Score refusal calibration 0.0-1.0.

    Args:
        answer: The generated answer.
        should_refuse: Whether the question is out-of-scope (from ground_truth.json).

    Returns dict with:
        score: float 0.0 or 1.0
        refused: bool — did the model refuse?
        should_refuse: bool — was refusal expected?
        calibrated: bool — did the model behave correctly?
        failure_type: str | None — "false_refusal", "failed_refusal", or None
    """
    refused = _is_refusal(answer)

    if should_refuse and refused:
        # Correct refusal
        return {
            "score": 1.0,
            "refused": True,
            "should_refuse": True,
            "calibrated": True,
            "failure_type": None,
        }
    elif should_refuse and not refused:
        # Should have refused but answered (failed refusal)
        logger.debug("Refusal failure: model should have refused but answered")
        return {
            "score": 0.0,
            "refused": False,
            "should_refuse": True,
            "calibrated": False,
            "failure_type": "failed_refusal",
        }
    elif not should_refuse and refused:
        # Should have answered but refused (false refusal)
        logger.debug("False refusal: model refused an answerable question")
        return {
            "score": 0.0,
            "refused": True,
            "should_refuse": False,
            "calibrated": False,
            "failure_type": "false_refusal",
        }
    else:
        # Correctly answered an answerable question
        return {
            "score": 1.0,
            "refused": False,
            "should_refuse": False,
            "calibrated": True,
            "failure_type": None,
        }

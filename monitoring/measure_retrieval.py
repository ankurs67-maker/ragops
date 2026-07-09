"""Measure retrieval relevance on a 0–3 integer scale.

Scale:
  3 = At least one chunk directly contains the answer keywords
  2 = Chunks are topically related but don't directly answer
  1 = Chunks are loosely related (same subject area)
  0 = Chunks are entirely irrelevant

Uses keyword matching against expected_chunk_keywords from ground truth,
plus similarity scores from the retriever. No LLM call.
"""

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def measure_retrieval_relevance(
    chunks: list[dict],
    expected_keywords: list[str],
    retrieval_quality: dict,
) -> dict:
    """Compute retrieval relevance score (0-3).

    Args:
        chunks: Retrieved chunk dicts from the retriever.
        expected_keywords: Keywords from ground_truth.json expected_chunk_keywords.
        retrieval_quality: Dict from check_retrieval_quality() with avg/max similarity.

    Returns dict with:
        score: int 0-3
        matched_keywords: list[str]
        max_similarity: float
        avg_similarity: float
        explanation: str
    """
    if not chunks:
        return {
            "score": 0,
            "matched_keywords": [],
            "max_similarity": 0.0,
            "avg_similarity": 0.0,
            "explanation": "no_chunks_retrieved",
        }

    max_sim = retrieval_quality.get("max_similarity", 0.0)
    avg_sim = retrieval_quality.get("avg_similarity", 0.0)

    # Keyword matching across all retrieved chunks
    combined_content = " ".join(
        chunk.get("raw_content", chunk.get("content", "")).lower()
        for chunk in chunks
    )

    matched = [kw for kw in expected_keywords if kw.lower() in combined_content]
    match_ratio = len(matched) / max(len(expected_keywords), 1)

    # Scoring logic
    if len(matched) >= max(len(expected_keywords) * 0.7, 1) and max_sim >= 0.45:
        score = 3
        explanation = f"direct_match: {len(matched)}/{len(expected_keywords)} keywords, sim={max_sim:.3f}"
    elif len(matched) >= 1 and max_sim >= 0.35:
        score = 2
        explanation = f"topical_match: {len(matched)}/{len(expected_keywords)} keywords, sim={max_sim:.3f}"
    elif max_sim >= 0.25 or match_ratio >= 0.3:
        score = 1
        explanation = f"loose_match: sim={max_sim:.3f}, keywords={len(matched)}/{len(expected_keywords)}"
    else:
        score = 0
        explanation = f"no_match: sim={max_sim:.3f}, keywords={len(matched)}/{len(expected_keywords)}"

    logger.debug(
        "Retrieval scored",
        extra={"score": score, "matched": len(matched), "max_sim": max_sim},
    )
    return {
        "score": score,
        "matched_keywords": matched,
        "max_similarity": max_sim,
        "avg_similarity": avg_sim,
        "explanation": explanation,
    }

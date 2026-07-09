"""Full RAG pipeline with loop engineering and Reflexion integration.

Loop engineering:
  1. HyDE-lite query rewriting (3 variants, no LLM)
  2. Multi-query retrieval with deduplication
  3. Context engineering ordering (best chunk first, second-best last)
  4. Self-RAG 3-step verification in generator
  5. Autonomous retry: if Self-RAG fails, rewrite query differently and re-retrieve
  6. Reflexion: loads session_context (past failure lessons) and passes to generator

The pipeline is the single entry point for both the RAG system
and the monitoring probe engine. All callers use run_query().
"""

import time
from typing import Optional

from rag_system.generator import generate_answer
from rag_system.retriever import (
    check_retrieval_quality,
    format_context,
    rewrite_query,
    retrieve_multi,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_PIPELINE_MAX_LOOP_ITERATIONS = 2  # Total retry loops after first attempt


def _build_retry_query(original_query: str, attempt: int) -> str:
    """Generate an alternative query formulation for retry after Self-RAG failure."""
    variants = rewrite_query(original_query)
    # On retry attempt 1 use variant 2 (explanatory), on attempt 2 use variant 3 (keywords)
    if attempt < len(variants):
        return variants[attempt]
    return original_query


def run_query(
    query: str,
    session_context: str = "",
    top_k: Optional[int] = None,
    skip_self_rag: bool = False,
) -> dict:
    """Execute the full RAG pipeline for a single query.

    Pipeline:
      1. Multi-query retrieval (HyDE-lite + context engineering ordering)
      2. Retrieval quality check (heuristic, no LLM)
      3. Generation with Self-RAG verification loop
      4. If Self-RAG fails → rewrite query → re-retrieve → retry
      5. Returns comprehensive result dict

    Args:
        query: The user question string.
        session_context: Reflexion lessons loaded from failure_memory.jsonl.
        top_k: Override the default top_k retrieval count.
        skip_self_rag: Skip Self-RAG checks (for testing or latency budget reasons).

    Returns dict with keys:
        answer: str
        chunks: list[dict]  — retrieved chunks used for final answer
        context: str        — formatted context passed to LLM
        retrieval_quality: dict
        self_rag_passed: bool
        self_rag_checks: dict
        self_rag_retries: int
        loop_retries: int   — pipeline-level retries (not Self-RAG retries)
        tokens_used: int
        latency_ms: float
        model_used: str
        provider_used: str
        error: str | None
        query_used: str     — the query variant that produced the final answer
    """
    t_start = time.time()
    loop_retries = 0
    active_query = query

    for loop_attempt in range(_PIPELINE_MAX_LOOP_ITERATIONS + 1):
        # Step 1: Multi-query retrieval with context engineering ordering
        try:
            chunks = retrieve_multi(active_query, top_k=top_k)
        except Exception as exc:
            logger.error(
                "Retrieval failed",
                extra={"query": active_query[:80], "error": str(exc)},
            )
            return _error_result(
                query=query,
                error=f"retrieval_exception: {exc}",
                latency_ms=(time.time() - t_start) * 1000,
            )

        # Step 2: Heuristic retrieval quality check
        quality = check_retrieval_quality(chunks)
        if not quality["adequate"] and loop_attempt == 0:
            logger.warning(
                "Low retrieval quality on first attempt",
                extra={"reason": quality["reason"], "query": active_query[:80]},
            )

        # Step 3: Format context from retrieved chunks
        context = format_context(chunks)

        # Step 4: Generate answer + Self-RAG
        gen = generate_answer(
            query=query,  # original query for answer relevance
            context=context,
            session_context=session_context,
            skip_self_rag=skip_self_rag,
        )
        tokens_used = gen.get("tokens_used", 0)

        # Step 5: Check if Self-RAG passed or if we should retry at pipeline level
        if gen.get("self_rag_passed", True) or skip_self_rag:
            break

        retrieval_was_adequate = quality.get("adequate", True)

        if (
            not retrieval_was_adequate
            and loop_attempt < _PIPELINE_MAX_LOOP_ITERATIONS
        ):
            loop_retries += 1
            active_query = _build_retry_query(query, loop_attempt + 1)
            logger.info(
                "Pipeline loop retry",
                extra={"attempt": loop_attempt + 1, "new_query": active_query[:80]},
            )
        else:
            logger.warning(
                "Pipeline stopping — retrieval was adequate, retry would not help",
                extra={"query": query[:80]},
            )
            break

    latency_ms = (time.time() - t_start) * 1000

    return {
        "answer": gen.get("answer", ""),
        "chunks": chunks,
        "context": context,
        "retrieval_quality": quality,
        "self_rag_passed": gen.get("self_rag_passed", False),
        "self_rag_checks": gen.get("self_rag_checks", {}),
        "self_rag_retries": gen.get("self_rag_retries", 0),
        "loop_retries": loop_retries,
        "tokens_used": tokens_used,
        "latency_ms": latency_ms,
        "model_used": gen.get("model_used", ""),
        "provider_used": gen.get("provider_used", ""),
        "error": gen.get("error"),
        "query_used": active_query,
    }


def _error_result(query: str, error: str, latency_ms: float) -> dict:
    return {
        "answer": "",
        "chunks": [],
        "context": "",
        "retrieval_quality": {"adequate": False, "reason": error},
        "self_rag_passed": False,
        "self_rag_checks": {},
        "self_rag_retries": 0,
        "loop_retries": 0,
        "tokens_used": 0,
        "latency_ms": latency_ms,
        "model_used": "",
        "provider_used": "",
        "error": error,
        "query_used": query,
    }


if __name__ == "__main__":
    import json
    result = run_query("What does RLHF stand for?", skip_self_rag=True)
    print(f"Answer: {result['answer'][:200]}")
    print(f"Provider: {result['provider_used']} | Model: {result['model_used']}")
    print(f"Latency: {result['latency_ms']:.0f}ms | Chunks: {len(result['chunks'])}")
    print(f"Self-RAG passed: {result['self_rag_passed']} | Loop retries: {result['loop_retries']}")

"""RAG generator with Self-RAG 3-step verification loop.

All LLM calls go through call_llm() only — never import groq or openai directly.
Self-RAG checks:
  1. RETRIEVAL_ADEQUATE — are retrieved passages sufficient?
  2. ANSWER_GROUNDED — is the answer grounded in context?
  3. ANSWER_COMPLETE — does the answer fully address the question?
"""

from config.settings import settings
from rag_system.prompt_templates import (
    get_completeness_check_prompt,
    get_generation_prompt,
    get_generation_system_prompt,
    get_groundedness_check_prompt,
    get_retrieval_check_prompt,
)
from utils.llm_client import call_llm
from utils.logger import get_logger

logger = get_logger(__name__)

_SELF_RAG_MAX_RETRIES = 2


def _judge_yes_no(response_text: str, positive_tokens: set[str]) -> bool:
    """Parse a binary Self-RAG verdict from the LLM response."""
    first_word = response_text.strip().split()[0].upper() if response_text.strip() else ""
    return first_word in positive_tokens


def _check_retrieval_adequate(query: str, context: str) -> dict:
    """Self-RAG check 1: Is retrieval adequate?"""
    prompt = get_retrieval_check_prompt(query, context[:2000])
    result = call_llm(
        prompt=prompt,
        system="You are a retrieval quality judge. Respond with exactly one word.",
        model_tier="scoring",
        temperature=settings.judge_temperature,
        max_tokens=5,
    )
    text = result.get("text", "").strip().upper()
    adequate = "ADEQUATE" in text and "INADEQUATE" not in text
    return {
        "adequate": adequate,
        "raw": text,
        "tokens": result.get("tokens_used", 0),
        "error": result.get("error"),
    }


def _check_answer_grounded(query: str, answer: str, context: str) -> dict:
    """Self-RAG check 2: Is the answer grounded in the context?"""
    prompt = get_groundedness_check_prompt(query, answer, context)
    result = call_llm(
        prompt=prompt,
        system="You are a faithfulness judge. Respond with exactly one word.",
        model_tier="scoring",
        temperature=settings.judge_temperature,
        max_tokens=5,
    )
    text = result.get("text", "").strip().upper()
    grounded = "GROUNDED" in text and "NOT_GROUNDED" not in text
    return {
        "grounded": grounded,
        "raw": text,
        "tokens": result.get("tokens_used", 0),
        "error": result.get("error"),
    }


def _check_answer_complete(query: str, answer: str) -> dict:
    """Self-RAG check 3: Is the answer complete?"""
    prompt = get_completeness_check_prompt(query, answer)
    result = call_llm(
        prompt=prompt,
        system="You are a completeness judge. Respond with exactly one word.",
        model_tier="scoring",
        temperature=settings.judge_temperature,
        max_tokens=5,
    )
    text = result.get("text", "").strip().upper()
    complete = "COMPLETE" in text
    return {
        "complete": complete,
        "raw": text,
        "tokens": result.get("tokens_used", 0),
        "error": result.get("error"),
    }


def generate_answer(
    query: str,
    context: str,
    session_context: str = "",
    skip_self_rag: bool = False,
) -> dict:
    """Generate an answer and run the Self-RAG 3-step verification loop.

    Args:
        query: The user question.
        context: Formatted context string from the retriever.
        session_context: Reflexion lessons from failure_memory.jsonl (passed through to prompt).
        skip_self_rag: If True, skip verification (used for quick testing).

    Returns dict with keys:
        answer: str — the final answer text
        self_rag_passed: bool — whether all 3 checks passed
        self_rag_checks: dict — individual check results
        self_rag_retries: int — how many retry loops were needed
        tokens_used: int — total tokens consumed
        model_used: str
        provider_used: str
        error: str | None
    """
    total_tokens = 0
    self_rag_retries = 0
    self_rag_checks: dict = {}

    for attempt in range(_SELF_RAG_MAX_RETRIES + 1):
        # Generate answer
        prompt = get_generation_prompt(query, context, reflexion_lessons=session_context)
        gen_result = call_llm(
            prompt=prompt,
            system=get_generation_system_prompt(),
            model_tier="generation",
            temperature=settings.generation_temperature,
            max_tokens=settings.max_answer_tokens,
        )
        total_tokens += gen_result.get("tokens_used", 0)
        answer = gen_result.get("text", "").strip()

        if gen_result.get("error") or not answer:
            return {
                "answer": "",
                "self_rag_passed": False,
                "self_rag_checks": {},
                "self_rag_retries": self_rag_retries,
                "tokens_used": total_tokens,
                "model_used": gen_result.get("model_used", ""),
                "provider_used": gen_result.get("provider_used", ""),
                "error": gen_result.get("error", "empty_response"),
            }

        if skip_self_rag:
            return {
                "answer": answer,
                "self_rag_passed": True,
                "self_rag_checks": {"skipped": True},
                "self_rag_retries": 0,
                "tokens_used": total_tokens,
                "model_used": gen_result.get("model_used", ""),
                "provider_used": gen_result.get("provider_used", ""),
                "error": None,
            }

        # Self-RAG check 1: retrieval adequacy
        check1 = _check_retrieval_adequate(query, context)
        total_tokens += check1.get("tokens", 0)

        # Self-RAG check 2: groundedness
        check2 = _check_answer_grounded(query, answer, context)
        total_tokens += check2.get("tokens", 0)

        # Self-RAG check 3: completeness
        check3 = _check_answer_complete(query, answer)
        total_tokens += check3.get("tokens", 0)

        self_rag_checks = {
            "retrieval_adequate": check1["adequate"],
            "answer_grounded": check2["grounded"],
            "answer_complete": check3["complete"],
            "raw_responses": {
                "retrieval": check1["raw"],
                "grounded": check2["raw"],
                "complete": check3["raw"],
            },
        }

        answer_is_refusal = any(
            phrase in answer.lower()
            for phrase in [
                "cannot find", "i cannot find", "not in my knowledge base",
                "don't have information", "unable to find", "no information",
            ]
        )

        all_checks_passed = (
            check1["adequate"] and check2["grounded"] and check3["complete"]
        )

        # Passes if all checks pass OR if the answer is a deliberate refusal
        # (refusals are always considered grounded/complete — they are intentional).
        self_rag_passed = all_checks_passed or answer_is_refusal

        if not settings.self_rag_blocking:
            # Advisory mode: checks run and are logged, but never block the answer.
            self_rag_passed = True

        # Only retry when groundedness specifically failed and the answer is not a
        # deliberate refusal. Retrieval-only failures and completeness failures are
        # not retried here — they can't be fixed by regenerating with the same context.
        should_retry = (
            settings.self_rag_blocking
            and not check2["grounded"]
            and not answer_is_refusal
        )

        logger.debug(
            "Self-RAG checks",
            extra={
                "attempt": attempt,
                "adequate": check1["adequate"],
                "grounded": check2["grounded"],
                "complete": check3["complete"],
                "self_rag_passed": self_rag_passed,
                "should_retry": should_retry,
            },
        )

        if self_rag_passed or not should_retry:
            return {
                "answer": answer,
                "self_rag_passed": self_rag_passed,
                "self_rag_checks": self_rag_checks,
                "self_rag_retries": self_rag_retries,
                "tokens_used": total_tokens,
                "model_used": gen_result.get("model_used", ""),
                "provider_used": gen_result.get("provider_used", ""),
                "error": None,
            }

        failed_checks = [
            k for k, v in {
                "retrieval_adequate": check1["adequate"],
                "answer_grounded": check2["grounded"],
                "answer_complete": check3["complete"],
            }.items() if not v
        ]
        logger.info(
            "Self-RAG retry triggered",
            extra={"failed_checks": failed_checks, "attempt": attempt},
        )
        self_rag_retries += 1

        if attempt >= _SELF_RAG_MAX_RETRIES:
            # Exhausted retries — return best answer so far
            logger.warning(
                "Self-RAG max retries reached",
                extra={"query": query[:80]},
            )
            break

    return {
        "answer": answer,
        "self_rag_passed": False,
        "self_rag_checks": self_rag_checks,
        "self_rag_retries": self_rag_retries,
        "tokens_used": total_tokens,
        "model_used": gen_result.get("model_used", ""),
        "provider_used": gen_result.get("provider_used", ""),
        "error": "self_rag_failed_after_retries",
    }

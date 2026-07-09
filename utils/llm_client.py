"""Unified LLM client that routes calls to Groq, DeepSeek, or OpenRouter.

All LLM calls in this project go through call_llm(). Provider is selected
via settings.llm_provider (generation) and settings.scoring_provider (scoring).
Automatic fallback on 429 rate-limit responses: tries next provider in chain.
"""

import time
from typing import Any

import groq as groq_lib
from openai import OpenAI, AuthenticationError, RateLimitError

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Fallback chains: if provider A is rate-limited, try B then C.
_FALLBACK_CHAIN: dict[str, list[str]] = {
    "groq": ["deepseek", "openrouter"],
    "deepseek": ["openrouter", "groq"],
    "openrouter": ["groq", "deepseek"],
}

_SAFE_DEFAULT: dict[str, Any] = {
    "text": "",
    "model_used": "",
    "provider_used": "",
    "tokens_used": 0,
    "error": "all_providers_failed",
    "was_fallback": False,
}


def _get_model_for_provider(provider: str, model_tier: str) -> str:
    """Return the model string for a given provider and tier."""
    if model_tier == "generation":
        return {
            "groq": settings.llm_model,
            "deepseek": settings.deepseek_model,
            "openrouter": settings.openrouter_model,
        }.get(provider, settings.llm_model)
    else:  # scoring
        return {
            "groq": settings.scoring_model,
            "deepseek": settings.deepseek_model,
            "openrouter": settings.openrouter_model,
        }.get(provider, settings.scoring_model)


def _get_key_for_provider(provider: str) -> str:
    """Return the API key for a given provider, stripped of whitespace."""
    key = {
        "groq": settings.groq_api_key,
        "deepseek": settings.deepseek_api_key,
        "openrouter": settings.openrouter_api_key,
    }.get(provider, "")
    return key.strip()


def _is_key_valid(provider: str) -> bool:
    """Return True only if the provider has a non-empty, non-placeholder API key."""
    key = _get_key_for_provider(provider)
    if not key:
        return False
    placeholder_patterns = [
        "placeholder", "your_", "replace_with", "example",
        "test_key", "dummy",
    ]
    return not any(p in key.lower() for p in placeholder_patterns)


def get_scoring_model_version() -> str:
    """Return 'provider:model' for the active scoring provider.

    Used as judge_model_version in measurement rows so results are traceable
    to the model that actually produced the judge scores.
    """
    provider = settings.scoring_provider
    model = _get_model_for_provider(provider, "scoring")
    return f"{provider}:{model}"


def _call_groq(
    prompt: str,
    system: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Make a single call to the Groq API."""
    key = _get_key_for_provider("groq")
    if not key:
        return {**_SAFE_DEFAULT, "error": "groq_key_missing"}

    client = groq_lib.Groq(api_key=key)
    logger.debug("Groq call", extra={"model": model, "max_tokens": max_tokens})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.llm_timeout_seconds,
    )
    text = response.choices[0].message.content or ""
    usage = response.usage
    tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
    logger.info(
        "Groq call complete",
        extra={"model": model, "tokens": tokens},
    )
    return {
        "text": text,
        "model_used": model,
        "provider_used": "groq",
        "tokens_used": tokens,
        "error": None,
        "was_fallback": False,
    }


def _call_openai_compat(
    prompt: str,
    system: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Make a call via the OpenAI-compatible API (DeepSeek or OpenRouter)."""
    key = _get_key_for_provider(provider)
    if not key:
        return {**_SAFE_DEFAULT, "error": f"{provider}_key_missing"}

    base_url = (
        settings.deepseek_base_url
        if provider == "deepseek"
        else settings.openrouter_base_url
    )
    extra_headers: dict[str, str] = {}
    if provider == "openrouter":
        extra_headers["HTTP-Referer"] = "https://github.com/ragops"
        extra_headers["X-Title"] = "RAGOps"

    client = OpenAI(api_key=key, base_url=base_url)
    logger.debug(
        "OpenAI-compat call",
        extra={"provider": provider, "model": model},
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.llm_timeout_seconds,
        extra_headers=extra_headers if extra_headers else None,
    )
    text = response.choices[0].message.content or ""
    usage = response.usage
    tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
    logger.info(
        "OpenAI-compat call complete",
        extra={"provider": provider, "model": model, "tokens": tokens},
    )
    return {
        "text": text,
        "model_used": model,
        "provider_used": provider,
        "tokens_used": tokens,
        "error": None,
        "was_fallback": False,
    }


def _call_provider(
    provider: str,
    prompt: str,
    system: str,
    model_tier: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Dispatch to the correct provider implementation."""
    model = _get_model_for_provider(provider, model_tier)
    if provider == "groq":
        return _call_groq(prompt, system, model, temperature, max_tokens)
    return _call_openai_compat(prompt, system, provider, model, temperature, max_tokens)


def call_llm(
    prompt: str,
    system: str,
    model_tier: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Route an LLM call to the configured provider with automatic fallback.

    model_tier: "generation" (uses llm_provider) or "scoring" (uses scoring_provider).

    Returns dict with keys:
        text          — model response string (empty on total failure)
        model_used    — model identifier string
        provider_used — "groq" | "deepseek" | "openrouter"
        tokens_used   — total tokens consumed
        error         — None on success, error string on failure
        was_fallback  — True if a fallback provider was used
    """
    primary = (
        settings.llm_provider if model_tier == "generation"
        else settings.scoring_provider
    )
    providers_to_try = [primary] + _FALLBACK_CHAIN.get(primary, [])

    last_error: str = "no_providers_configured"
    was_fallback = False

    for attempt, provider in enumerate(providers_to_try):
        if not _is_key_valid(provider):
            logger.debug(
                "Provider skipped — key missing or placeholder",
                extra={"provider": provider, "tier": model_tier},
            )
            continue

        try:
            result = _call_provider(
                provider, prompt, system, model_tier, temperature, max_tokens
            )
            if result.get("error") and "key_missing" in result["error"]:
                logger.warning(
                    "Provider API key missing",
                    extra={"provider": provider},
                )
                continue
            result["was_fallback"] = was_fallback
            return result

        except (groq_lib.RateLimitError, RateLimitError) as exc:
            logger.warning(
                "Rate limit hit — trying fallback",
                extra={"provider": provider, "error": str(exc)},
            )
            last_error = f"{provider}_rate_limited"
            was_fallback = True
            time.sleep(1)
            continue

        except (groq_lib.AuthenticationError, AuthenticationError) as exc:
            logger.error(
                "AUTH FAILED — check API key in .env",
                extra={"provider": provider, "error": str(exc)},
            )
            # Auth errors don't retry — wrong key won't fix itself
            last_error = f"{provider}_auth_failed"
            was_fallback = True
            continue

        except Exception as exc:
            logger.error(
                "LLM call failed",
                extra={"provider": provider, "error": str(exc), "attempt": attempt},
            )
            last_error = str(exc)
            was_fallback = True
            time.sleep(1)
            continue

    logger.error(
        "All LLM providers failed",
        extra={"tier": model_tier, "last_error": last_error},
    )
    return {**_SAFE_DEFAULT, "error": last_error, "was_fallback": True}

"""Fetch Wikipedia articles about AI/LLM topics and save as plain text files.

Downloads 30 articles using the MediaWiki REST API directly via requests.
Output directory: data/raw/wikipedia/
Skips articles already downloaded. Sleeps 0.5s between requests.

Note: wikipedia-api==0.7.1 produces empty responses on this environment (Decision 3).
Using requests directly against the MediaWiki API instead.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

ARTICLES = [
    "Transformer (machine learning model)",
    "Attention mechanism",
    "BERT (language model)",
    "GPT (language model)",
    "Reinforcement learning from human feedback",
    "Constitutional AI",
    "Mixture of experts",
    "Retrieval-augmented generation",
    "Fine-tuning (machine learning)",
    "Prompt engineering",
    "Chain-of-thought prompting",
    "Hallucination (artificial intelligence)",
    "Large language model",
    "Generative pre-trained transformer",
    "Neural scaling law",
    "OpenAI",
    "Anthropic",
    "Google DeepMind",
    "Meta AI",
    "Hugging Face",
    "EleutherAI",
    "Mistral AI",
    "Yoshua Bengio",
    "Geoffrey Hinton",
    "Yann LeCun",
    "Demis Hassabis",
    "Ilya Sutskever",
    "BLEU",
    "Word embedding",
    "Tokenization (machine learning)",
    # ── Expansion 2026-07 — foundational topics ──
    "BigScience",
    "BLOOM (language model)",
    "Vector database",
    "Cosine similarity",
    "Semantic search",
    "Instruction tuning",
    "Direct preference optimization",
    "Sparse attention",
    "Rotary positional embedding",
    "Flash attention",
    "Quantization (deep learning)",
    "Knowledge distillation",
    "Multimodal learning",
    "Vision transformer",
    "Diffusion model",
    "Reward model",
    "AI alignment",
    "Emergent abilities of large language models",
    "Scaling laws for neural language models",
    "Model collapse",
    "Red teaming (AI safety)",
    "Jailbreak (computer science)",
    "Prompt injection",
    "AI safety",
    "Interpretability (machine learning)",
    "Sparse mixture of experts",
    "Speculative decoding",
    "Long context window",
    "Vector embedding",
    "Cross-encoder",
    "Bi-encoder",
    "Reranking (information retrieval)",
    "LangChain",
    "LlamaIndex",
    "Function calling (LLM)",
    "Agentic AI",
    "Multi-agent system",
    "AI benchmark",
    "Perplexity AI",
    "Cohere",
    "Stability AI",
    # ── Expansion 2026-07 — alternate titles for topics whose first-choice
    # title does not exist on Wikipedia (same concepts, real articles) ──
    "Byte pair encoding",
    "Context window",
    "Red team",
    "Learning to rank",
    "Model compression",
    "Sentence embedding",
    "Language model benchmark",
    # ── Expansion 2026-07 — researchers and labs ──
    "Noam Shazeer",
    "Percy Liang",
    "Andrew Ng",
    "Fei-Fei Li",
    "xAI (company)",
    "Inflection AI",
]

_MW_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "RAGOps-Monitor/1.0 (research project)"}


def _safe_filename(title: str) -> str:
    """Convert article title to a safe filename."""
    safe = title.replace("/", "_").replace("(", "").replace(")", "")
    safe = safe.replace(" ", "_").strip("_")
    return safe + ".txt"


def _fetch_article(title: str, max_retries: int = 3) -> tuple[str, str] | None:
    """Fetch article plain text and URL. Returns (text, url) or None.

    Retries up to max_retries times on 429 with exponential backoff.
    """
    params = {
        "action": "query",
        "prop": "extracts|info",
        "titles": title,
        "explaintext": True,
        "inprop": "url",
        "format": "json",
        "redirects": 1,
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(
                _MW_API, params=params, headers=_HEADERS, timeout=15
            )
            if response.status_code == 429:
                wait = 5 * (2 ** attempt)
                logger.warning(
                    "Wikipedia API rate limited — waiting",
                    extra={"title": title, "wait_s": wait, "attempt": attempt + 1},
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None
            page = next(iter(pages.values()))
            if page.get("pageid", -1) == -1:
                return None
            text = page.get("extract", "")
            url = page.get("fullurl", f"https://en.wikipedia.org/wiki/{quote(title)}")
            if not text:
                logger.warning(
                    "Empty extract from Wikipedia",
                    extra={"title": title},
                )
                return None
            return text, url
        except Exception as exc:
            logger.error(
                "Wikipedia API request failed",
                extra={"title": title, "error": str(exc), "attempt": attempt + 1},
            )
            if attempt < max_retries - 1:
                time.sleep(3)
    return None


def fetch_all_articles() -> tuple[int, list[str]]:
    """Download all articles. Return (success_count, failed_list)."""
    output_dir: Path = settings.raw_dir / "wikipedia"
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_list: list[str] = []

    for title in ARTICLES:
        filename = _safe_filename(title)
        filepath = output_dir / filename

        if filepath.exists():
            logger.info(
                "Skipping existing article",
                extra={"title": title},
            )
            success_count += 1
            continue

        result = _fetch_article(title)
        if result is None:
            logger.warning(
                "Article not available",
                extra={"title": title},
            )
            failed_list.append(title)
            time.sleep(0.5)
            continue

        text, url = result
        header = (
            f"TITLE: {title}\n"
            f"URL: {url}\n"
            f"FETCHED: {datetime.now(timezone.utc).isoformat()}\n"
            f"---\n\n"
        )
        filepath.write_text(header + text, encoding="utf-8")
        logger.info(
            "Article saved",
            extra={"title": title, "chars": len(text)},
        )
        success_count += 1
        time.sleep(2.0)

    logger.info(
        "Wikipedia fetch complete",
        extra={"success": success_count, "failed": len(failed_list)},
    )
    return success_count, failed_list


if __name__ == "__main__":
    successes, failures = fetch_all_articles()
    print(f"Wikipedia fetch: {successes} succeeded, {len(failures)} failed")
    if failures:
        print(f"Failed articles: {failures}")

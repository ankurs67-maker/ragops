"""Fetch model cards from Hugging Face Hub and save as plain text files.

Downloads 30 model cards using huggingface_hub ModelCard.load().
Output directory: data/raw/huggingface/
Filename: replaces / with __ — e.g. meta-llama__Llama-2-7b-hf.txt
Skips models already downloaded. Sleeps 0.3s between requests.
"""

import time
from pathlib import Path

from huggingface_hub import ModelCard
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "google/gemma-2b",
    "google/gemma-7b",
    "google/gemma-2-9b",
    "google/gemma-2-27b",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-40b",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-72B-Instruct",
    "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-ai/deepseek-llm-67b-base",
    "databricks/dbrx-base",
    "CohereForAI/c4ai-command-r-plus",
    "01-ai/Yi-1.5-34B",
    "allenai/OLMo-7B",
    "EleutherAI/gpt-neox-20b",
    "bigscience/bloom",
    # ── Expansion 2026-07 ──
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-11B-Vision",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai/Codestral-22B-v0.1",
    "google/gemma-2-2b",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek-ai/DeepSeek-V2.5",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "01-ai/Yi-1.5-9B",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "HuggingFaceH4/zephyr-7b-beta",
    "stabilityai/stablelm-2-12b",
    "microsoft/Orca-2-13b",
    "google/flan-t5-xxl",
    "facebook/opt-66b",
    "bigcode/starcoder2-15b",
    "Salesforce/codegen2-16B",
    "WizardLMTeam/WizardCoder-33B-V1.1",
    "ibm-granite/granite-3.0-8b-instruct",
]


def _model_filename(model_id: str) -> str:
    """Convert model ID to filename by replacing / with __."""
    return model_id.replace("/", "__") + ".txt"


def fetch_all_model_cards() -> tuple[int, list[str]]:
    """Download all model cards. Return (success_count, failed_list)."""
    output_dir: Path = settings.raw_dir / "huggingface"
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_list: list[str] = []

    for model_id in MODELS:
        filename = _model_filename(model_id)
        filepath = output_dir / filename

        if filepath.exists():
            logger.info(
                "Skipping existing model card",
                extra={"model_id": model_id},
            )
            success_count += 1
            continue

        try:
            card = ModelCard.load(model_id)
            content = str(card)
            filepath.write_text(content, encoding="utf-8")
            logger.info(
                "Model card saved",
                extra={"model_id": model_id, "chars": len(content)},
            )
            success_count += 1

        except (EntryNotFoundError, RepositoryNotFoundError) as exc:
            logger.warning(
                "Model card not found",
                extra={"model_id": model_id, "error": str(exc)},
            )
            failed_list.append(model_id)

        except Exception as exc:
            logger.error(
                "Failed to fetch model card",
                extra={"model_id": model_id, "error": str(exc)},
            )
            failed_list.append(model_id)

        time.sleep(0.3)

    logger.info(
        "Hugging Face fetch complete",
        extra={"success": success_count, "failed": len(failed_list)},
    )
    return success_count, failed_list


if __name__ == "__main__":
    successes, failures = fetch_all_model_cards()
    print(f"Hugging Face fetch: {successes} succeeded, {len(failures)} failed")
    if failures:
        print(f"Failed models: {failures}")

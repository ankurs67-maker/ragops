"""Fetch benchmark data from Papers With Code API and save as plain text files.

Uses the public Papers With Code API (no key required).
Output directory: data/raw/paperswithcode/
Saves task descriptions and top SOTA results for named benchmarks.
"""

import time
from pathlib import Path

import requests

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

API_BASE = "https://paperswithcode.com/api/v1/"

TASKS = [
    "language-modelling",
    "question-answering",
    "common-sense-reasoning",
    "math-word-problem-solving",
    "code-generation",
    "text-summarization",
    "information-retrieval",
]

BENCHMARKS = [
    "mmlu",
    "hellaswag",
    "arc",
    "truthfulqa",
    "gsm8k",
    "humaneval",
    "mbpp",
]


def _fetch_json(url: str, params: dict | None = None) -> dict | list | None:
    """GET request with timeout and error handling. Returns None on failure."""
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 404:
            logger.warning("404 from Papers With Code", extra={"url": url})
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.error(
            "Papers With Code request failed",
            extra={"url": url, "error": str(exc)},
        )
        return None


def fetch_task_results(task_slug: str) -> str:
    """Fetch task description and top models for a task slug."""
    data = _fetch_json(f"{API_BASE}tasks/{task_slug}/")
    if data is None:
        return f"TASK: {task_slug}\nSTATUS: Not found\n"

    lines = [
        f"TASK: {data.get('name', task_slug)}",
        f"SLUG: {task_slug}",
        f"DESCRIPTION: {data.get('description', 'N/A')}",
        "",
    ]

    # Fetch evaluation results for this task
    evals_data = _fetch_json(f"{API_BASE}evaluations/", params={"task": task_slug, "page_size": 5})
    if evals_data and isinstance(evals_data, dict):
        results = evals_data.get("results", [])
        if results:
            lines.append("TOP BENCHMARKS:")
            for eval_item in results[:5]:
                benchmark = eval_item.get("benchmark", {})
                lines.append(f"  - {benchmark.get('name', 'Unknown')}: {eval_item.get('description', '')}")
    lines.append("")
    return "\n".join(lines)


def fetch_benchmark_sota(benchmark_slug: str) -> str:
    """Fetch SOTA results for a specific named benchmark."""
    data = _fetch_json(f"{API_BASE}benchmarks/", params={"q": benchmark_slug})
    if data is None:
        return f"BENCHMARK: {benchmark_slug}\nSTATUS: Not found\n"

    if isinstance(data, dict):
        results = data.get("results", [])
    else:
        results = data if isinstance(data, list) else []

    if not results:
        return f"BENCHMARK: {benchmark_slug}\nSTATUS: No results found\n"

    # Use first matching benchmark
    benchmark = results[0]
    lines = [
        f"BENCHMARK: {benchmark.get('name', benchmark_slug)}",
        f"DESCRIPTION: {benchmark.get('description', 'N/A')}",
        "",
        "TOP 5 SOTA RESULTS:",
    ]

    # Fetch SOTA results for this benchmark
    sota_data = _fetch_json(
        f"{API_BASE}sota/",
        params={"benchmark": benchmark.get("id"), "page_size": 5},
    )
    if sota_data and isinstance(sota_data, dict):
        sota_results = sota_data.get("results", [])
        for rank, result in enumerate(sota_results[:5], 1):
            model = result.get("model_name", "Unknown")
            metrics = result.get("metrics", {})
            metric_str = ", ".join(f"{k}: {v}" for k, v in list(metrics.items())[:3])
            lines.append(f"  {rank}. {model} — {metric_str}")

    lines.append("")
    return "\n".join(lines)


def fetch_all() -> tuple[int, list[str]]:
    """Fetch all tasks and benchmarks. Return (success_count, failed_list)."""
    output_dir: Path = settings.raw_dir / "paperswithcode"
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_list: list[str] = []

    for task_slug in TASKS:
        filepath = output_dir / f"task_{task_slug}.txt"
        if filepath.exists():
            logger.info("Skipping existing task file", extra={"task": task_slug})
            success_count += 1
            continue

        content = fetch_task_results(task_slug)
        filepath.write_text(content, encoding="utf-8")
        logger.info("Task saved", extra={"task": task_slug, "chars": len(content)})
        success_count += 1
        time.sleep(0.5)

    for bm_slug in BENCHMARKS:
        filepath = output_dir / f"benchmark_{bm_slug}.txt"
        if filepath.exists():
            logger.info("Skipping existing benchmark file", extra={"benchmark": bm_slug})
            success_count += 1
            continue

        content = fetch_benchmark_sota(bm_slug)
        filepath.write_text(content, encoding="utf-8")
        logger.info("Benchmark saved", extra={"benchmark": bm_slug, "chars": len(content)})
        success_count += 1
        time.sleep(0.5)

    logger.info(
        "Papers With Code fetch complete",
        extra={"success": success_count, "failed": len(failed_list)},
    )
    return success_count, failed_list


if __name__ == "__main__":
    successes, failures = fetch_all()
    print(f"Papers With Code fetch: {successes} succeeded, {len(failures)} failed")
    if failures:
        print(f"Failed items: {failures}")

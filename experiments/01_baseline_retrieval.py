"""Experiment 01: Baseline Retrieval Quality.

Measures raw retrieval performance without any advanced techniques:
- No query rewriting (single query only)
- No context engineering ordering
- No contextual RAG prefix (uses raw_content for matching)

Compares against the full system (retrieve_multi + context ordering).
Outputs a summary table to stdout and appends a JSON entry to experiment_log.txt.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import settings
from rag_system.retriever import retrieve, retrieve_multi, check_retrieval_quality
from utils.logger import get_logger

logger = get_logger(__name__)

# 10 representative queries from ground truth (first 10)
_TEST_QUERIES = [
    ("gt_001", "What company developed the Llama 2 model?", ["meta", "llama"]),
    ("gt_002", "What does RLHF stand for?", ["reinforcement", "human feedback"]),
    ("gt_003", "Which organization created BLOOM?", ["bloom", "bigscience"]),
    ("gt_007", "What architecture does the original Transformer model use?", ["transformer", "attention"]),
    ("gt_009", "Who are the founders of Anthropic?", ["anthropic", "amodei"]),
    ("gt_012", "What does RAG stand for in the context of language models?", ["retrieval", "augmented"]),
    ("gt_013", "Which model popularised the mixture of experts architecture?", ["mixture", "experts", "mixtral"]),
    ("gt_015", "What is chain-of-thought prompting?", ["chain", "thought", "prompting"]),
    ("gt_019", "What is BERT and who created it?", ["bert", "google", "bidirectional"]),
    ("gt_020", "What is the BLEU score used to evaluate?", ["bleu", "translation"]),
]


def _keyword_recall(chunks: list[dict], expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found in retrieved chunks."""
    combined = " ".join(c.get("raw_content", c.get("content", "")).lower() for c in chunks)
    matched = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return matched / max(len(expected_keywords), 1)


def run_experiment() -> dict:
    """Compare baseline (single query) vs multi-query retrieval."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 01: BASELINE RETRIEVAL QUALITY")
    print("=" * 70)
    print(f"Queries: {len(_TEST_QUERIES)}")
    print(f"Top-k: {settings.top_k}")
    print()

    baseline_results = []
    multi_results = []

    for query_id, query, keywords in _TEST_QUERIES:
        # Baseline: single query, no rewriting
        t0 = time.time()
        baseline_chunks = retrieve(query, top_k=settings.top_k)
        baseline_latency = (time.time() - t0) * 1000
        baseline_quality = check_retrieval_quality(baseline_chunks)

        # Multi-query: HyDE-lite + context engineering
        t1 = time.time()
        multi_chunks = retrieve_multi(query, top_k=settings.top_k)
        multi_latency = (time.time() - t1) * 1000
        multi_quality = check_retrieval_quality(multi_chunks)

        b_recall = _keyword_recall(baseline_chunks, keywords)
        m_recall = _keyword_recall(multi_chunks, keywords)
        b_sim = baseline_quality["max_similarity"]
        m_sim = multi_quality["max_similarity"]

        baseline_results.append({"recall": b_recall, "max_sim": b_sim, "latency": baseline_latency})
        multi_results.append({"recall": m_recall, "max_sim": m_sim, "latency": multi_latency})

        improvement = m_recall - b_recall
        marker = "↑" if improvement > 0.01 else ("↓" if improvement < -0.01 else "=")
        print(f"  {query_id}  baseline recall={b_recall:.2f} sim={b_sim:.3f}"
              f"  →  multi recall={m_recall:.2f} sim={m_sim:.3f}  {marker}")

    # Aggregate
    avg_b_recall = sum(r["recall"] for r in baseline_results) / len(baseline_results)
    avg_m_recall = sum(r["recall"] for r in multi_results) / len(multi_results)
    avg_b_sim = sum(r["max_sim"] for r in baseline_results) / len(baseline_results)
    avg_m_sim = sum(r["max_sim"] for r in multi_results) / len(multi_results)
    avg_b_lat = sum(r["latency"] for r in baseline_results) / len(baseline_results)
    avg_m_lat = sum(r["latency"] for r in multi_results) / len(multi_results)

    print()
    print("SUMMARY:")
    print(f"  Keyword Recall:    baseline={avg_b_recall:.3f}  multi={avg_m_recall:.3f}"
          f"  Δ={avg_m_recall - avg_b_recall:+.3f}")
    print(f"  Max Similarity:    baseline={avg_b_sim:.3f}    multi={avg_m_sim:.3f}"
          f"  Δ={avg_m_sim - avg_b_sim:+.3f}")
    print(f"  Avg Latency (ms):  baseline={avg_b_lat:.0f}   multi={avg_m_lat:.0f}")
    print("=" * 70)

    result = {
        "experiment": "01_baseline_retrieval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(_TEST_QUERIES),
        "avg_baseline_recall": round(avg_b_recall, 4),
        "avg_multi_recall": round(avg_m_recall, 4),
        "recall_improvement": round(avg_m_recall - avg_b_recall, 4),
        "avg_baseline_max_sim": round(avg_b_sim, 4),
        "avg_multi_max_sim": round(avg_m_sim, 4),
        "avg_baseline_latency_ms": round(avg_b_lat, 1),
        "avg_multi_latency_ms": round(avg_m_lat, 1),
    }

    _append_log(result)
    return result


def _append_log(result: dict) -> None:
    log_path = settings.experiment_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(f"\nResult appended to {log_path}")


if __name__ == "__main__":
    run_experiment()

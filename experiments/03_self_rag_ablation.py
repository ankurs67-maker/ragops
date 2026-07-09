"""Experiment 03: Self-RAG Ablation.

Runs a set of queries with and without Self-RAG verification enabled.
Measures the effect on faithfulness and factuality scores.

When skip_self_rag=True the generator skips all 3 verification checks.
When skip_self_rag=False the generator runs all 3 checks with up to
_SELF_RAG_MAX_RETRIES regeneration attempts.
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
from monitoring.measure_faithfulness import measure_faithfulness
from monitoring.measure_factuality import measure_factuality
from rag_system.pipeline import run_query
from utils.logger import get_logger

logger = get_logger(__name__)

_TEST_QUERIES = [
    {
        "query_id": "gt_001",
        "query": "What company developed the Llama 2 model?",
        "correct_answer": "Meta AI (Meta)",
        "acceptable_answers": ["Meta", "Meta AI", "Facebook AI Research", "Meta Platforms"],
    },
    {
        "query_id": "gt_002",
        "query": "What does RLHF stand for?",
        "correct_answer": "Reinforcement Learning from Human Feedback",
        "acceptable_answers": ["Reinforcement Learning from Human Feedback", "RLHF"],
    },
    {
        "query_id": "gt_003",
        "query": "Which organization created BLOOM?",
        "correct_answer": "BigScience",
        "acceptable_answers": ["BigScience", "BigScience Workshop", "Hugging Face and BigScience"],
    },
    {
        "query_id": "gt_007",
        "query": "What architecture does the original Transformer model use?",
        "correct_answer": "Encoder-decoder architecture with self-attention mechanisms",
        "acceptable_answers": ["encoder-decoder", "self-attention", "attention mechanism"],
    },
    {
        "query_id": "gt_019",
        "query": "What is BERT and who created it?",
        "correct_answer": "Bidirectional Encoder Representations from Transformers, created by Google",
        "acceptable_answers": ["bidirectional", "google", "bert"],
    },
]


def _score_pair(result: dict, entry: dict) -> dict:
    faith = measure_faithfulness(
        answer=result["answer"],
        context=result["context"],
        query=entry["query"],
        self_rag_passed=result.get("self_rag_passed", True),
    )
    fact = measure_factuality(
        answer=result["answer"],
        correct_answer=entry["correct_answer"],
        acceptable_answers=entry["acceptable_answers"],
        query=entry["query"],
        context=result["context"],
    )
    return {"faithfulness": faith["score"], "factuality": fact["score"]}


def run_experiment() -> dict:
    """Compare skip_self_rag=True vs False."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 03: SELF-RAG ABLATION")
    print("=" * 70)
    print(f"Queries: {len(_TEST_QUERIES)}")
    print()

    without_selfrag = []
    with_selfrag = []

    for entry in _TEST_QUERIES:
        qid = entry["query_id"]
        query = entry["query"]

        # Without Self-RAG
        r_no = run_query(query, skip_self_rag=True)
        s_no = _score_pair(r_no, entry)

        # With Self-RAG
        r_yes = run_query(query, skip_self_rag=False)
        s_yes = _score_pair(r_yes, entry)

        without_selfrag.append(s_no)
        with_selfrag.append(s_yes)

        retries = r_yes.get("self_rag_retries", 0)
        print(f"  {qid}  no_self_rag: faith={s_no['faithfulness']:.2f} fact={s_no['factuality']:.2f}"
              f"  |  self_rag: faith={s_yes['faithfulness']:.2f} fact={s_yes['factuality']:.2f}"
              f"  retries={retries}")

    avg_no_faith = sum(r["faithfulness"] for r in without_selfrag) / len(without_selfrag)
    avg_yes_faith = sum(r["faithfulness"] for r in with_selfrag) / len(with_selfrag)
    avg_no_fact = sum(r["factuality"] for r in without_selfrag) / len(without_selfrag)
    avg_yes_fact = sum(r["factuality"] for r in with_selfrag) / len(with_selfrag)

    print()
    print("SUMMARY:")
    print(f"  Faithfulness:  no_self_rag={avg_no_faith:.3f}  self_rag={avg_yes_faith:.3f}"
          f"  Δ={avg_yes_faith - avg_no_faith:+.3f}")
    print(f"  Factuality:    no_self_rag={avg_no_fact:.3f}   self_rag={avg_yes_fact:.3f}"
          f"  Δ={avg_yes_fact - avg_no_fact:+.3f}")
    print("=" * 70)

    result = {
        "experiment": "03_self_rag_ablation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(_TEST_QUERIES),
        "avg_faithfulness_without_self_rag": round(avg_no_faith, 4),
        "avg_faithfulness_with_self_rag": round(avg_yes_faith, 4),
        "faithfulness_improvement": round(avg_yes_faith - avg_no_faith, 4),
        "avg_factuality_without_self_rag": round(avg_no_fact, 4),
        "avg_factuality_with_self_rag": round(avg_yes_fact, 4),
        "factuality_improvement": round(avg_yes_fact - avg_no_fact, 4),
        "hypothesis_supported": avg_yes_faith > avg_no_faith or avg_yes_fact > avg_no_fact,
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

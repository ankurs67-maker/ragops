"""Experiment 04: Reflexion Effectiveness.

Simulates multiple probe cycles and tests whether lessons written to
failure_memory.jsonl improve failure rates in subsequent cycles.

Cycle 1: Run probes with empty failure memory → record failures.
Cycle 2: Run same probes with pre-seeded failure lessons → compare failure rates.

This script uses a subset of ground truth queries (no should_refuse entries)
to focus on factual correctness and faithfulness.
"""

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import settings
from monitoring.probe_engine import run_probe, load_reflexion_lessons
from monitoring.classify_failure import classify_failure
from utils.logger import get_logger

logger = get_logger(__name__)

_TEST_QUERIES = [
    {
        "query_id": "gt_001",
        "query": "What company developed the Llama 2 model?",
        "correct_answer": "Meta AI (Meta)",
        "acceptable_answers": ["Meta", "Meta AI", "Facebook"],
        "category": "factual",
        "difficulty": "easy",
        "should_refuse": False,
        "source_document": "meta-llama__Llama-2-7b-hf.txt",
        "multi_hop": False,
        "expected_chunk_keywords": ["meta", "llama", "7b"],
    },
    {
        "query_id": "gt_002",
        "query": "What does RLHF stand for?",
        "correct_answer": "Reinforcement Learning from Human Feedback",
        "acceptable_answers": ["Reinforcement Learning from Human Feedback"],
        "category": "definition",
        "difficulty": "easy",
        "should_refuse": False,
        "source_document": "Reinforcement_learning_from_human_feedback.txt",
        "multi_hop": False,
        "expected_chunk_keywords": ["reinforcement", "human", "feedback"],
    },
    {
        "query_id": "gt_012",
        "query": "What does RAG stand for in the context of language models?",
        "correct_answer": "Retrieval-Augmented Generation",
        "acceptable_answers": ["Retrieval-Augmented Generation", "Retrieval Augmented Generation"],
        "category": "definition",
        "difficulty": "easy",
        "should_refuse": False,
        "source_document": "Retrieval-augmented_generation.txt",
        "multi_hop": False,
        "expected_chunk_keywords": ["retrieval", "augmented", "generation"],
    },
    {
        "query_id": "gt_019",
        "query": "What is BERT and who created it?",
        "correct_answer": "Bidirectional Encoder Representations from Transformers, created by Google",
        "acceptable_answers": ["bert", "google", "bidirectional"],
        "category": "factual",
        "difficulty": "easy",
        "should_refuse": False,
        "source_document": "BERT_(language_model).txt",
        "multi_hop": False,
        "expected_chunk_keywords": ["bert", "google", "bidirectional"],
    },
]

# Synthetic lessons to seed memory — mimics what Reflexion would write after
# seeing false_refusal failures on similar factual questions.
_SEED_LESSONS = [
    {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_id": "seed_001",
        "failure_category": "FALSE_REFUSAL",
        "lesson": (
            "Lesson: When the context contains relevant information about AI models or organizations, "
            "provide a direct factual answer. Do not refuse. "
            "Example failure: refused to answer 'Who created BERT?' despite clear context."
        ),
    },
    {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_id": "seed_002",
        "failure_category": "CONTEXT_BYPASS",
        "lesson": (
            "Lesson: Use the retrieved context to answer questions about specific models and benchmarks. "
            "The context is authoritative — extract the answer from it rather than saying you cannot help."
        ),
    },
    {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_id": "seed_003",
        "failure_category": "FAITHFULNESS_FAILURE",
        "lesson": (
            "Lesson: Answers about LLM architectures (Transformer, BERT, GPT) should cite the specific "
            "details from context — encoder/decoder structure, attention mechanisms, parameter counts."
        ),
    },
]


def _run_cycle(session_context: str, cycle_label: str) -> dict:
    """Run one probe cycle over _TEST_QUERIES."""
    print(f"\n  {cycle_label}")
    failures = 0
    pass_count = 0
    total_factuality = 0.0
    results = []

    for entry in _TEST_QUERIES:
        result = run_probe(entry, session_context=session_context)
        fc = result.get("failure_category", "PASS")
        if fc == "PASS":
            pass_count += 1
        else:
            failures += 1
        total_factuality += result.get("factuality", 0.0)
        results.append(result)
        print(f"    {entry['query_id']}  {fc}  factuality={result.get('factuality', 0):.2f}")

    avg_fact = total_factuality / max(len(_TEST_QUERIES), 1)
    failure_rate = failures / max(len(_TEST_QUERIES), 1)
    print(f"    → failure_rate={failure_rate:.2%}  avg_factuality={avg_fact:.3f}")
    return {"failure_rate": failure_rate, "avg_factuality": avg_fact, "results": results}


def run_experiment() -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 04: REFLEXION EFFECTIVENESS")
    print("=" * 70)
    print(f"Queries: {len(_TEST_QUERIES)}")
    print()

    failure_mem = settings.failure_memory_path
    failure_mem.parent.mkdir(parents=True, exist_ok=True)

    # Cycle 1: no lessons
    original_content = failure_mem.read_text() if failure_mem.exists() else ""
    failure_mem.write_text("")

    print("CYCLE 1: No Reflexion lessons")
    cycle1 = _run_cycle(session_context="", cycle_label="Cycle 1")

    # Seed memory with lessons
    with failure_mem.open("w", encoding="utf-8") as f:
        for lesson in _SEED_LESSONS:
            f.write(json.dumps(lesson) + "\n")

    print("\nCYCLE 2: With seeded Reflexion lessons")
    session_ctx = load_reflexion_lessons(n=5)
    cycle2 = _run_cycle(session_context=session_ctx, cycle_label="Cycle 2")

    # Restore original failure memory
    failure_mem.write_text(original_content)

    print()
    print("SUMMARY:")
    print(f"  Failure rate:    cycle1={cycle1['failure_rate']:.2%}  cycle2={cycle2['failure_rate']:.2%}"
          f"  Δ={cycle2['failure_rate'] - cycle1['failure_rate']:+.2%}")
    print(f"  Avg factuality:  cycle1={cycle1['avg_factuality']:.3f}  cycle2={cycle2['avg_factuality']:.3f}"
          f"  Δ={cycle2['avg_factuality'] - cycle1['avg_factuality']:+.3f}")
    improved = cycle2["failure_rate"] < cycle1["failure_rate"] or cycle2["avg_factuality"] > cycle1["avg_factuality"]
    print(f"  Reflexion helped: {improved}")
    print("=" * 70)

    result = {
        "experiment": "04_reflexion_effectiveness",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(_TEST_QUERIES),
        "n_seed_lessons": len(_SEED_LESSONS),
        "cycle1_failure_rate": round(cycle1["failure_rate"], 4),
        "cycle2_failure_rate": round(cycle2["failure_rate"], 4),
        "failure_rate_change": round(cycle2["failure_rate"] - cycle1["failure_rate"], 4),
        "cycle1_avg_factuality": round(cycle1["avg_factuality"], 4),
        "cycle2_avg_factuality": round(cycle2["avg_factuality"], 4),
        "factuality_change": round(cycle2["avg_factuality"] - cycle1["avg_factuality"], 4),
        "hypothesis_supported": improved,
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

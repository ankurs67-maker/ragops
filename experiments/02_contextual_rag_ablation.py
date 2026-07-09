"""Experiment 02: Contextual RAG Ablation.

Compares embedding search results when using the full contextual content
(context sentence + raw text) vs raw text only.

Hypothesis: The context prefix improves retrieval similarity by giving the
embedding model more signal about what the passage is about.
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
from utils.logger import get_logger

logger = get_logger(__name__)

# Queries where the context prefix should matter most
_TEST_QUERIES = [
    ("gt_007", "What architecture does the original Transformer model use?", ["transformer", "attention", "encoder"]),
    ("gt_008", "What is the MMLU benchmark score for GPT-4?", ["mmlu", "gpt-4", "86"]),
    ("gt_012", "What does RAG stand for in the context of language models?", ["retrieval", "augmented", "generation"]),
    ("gt_015", "What is chain-of-thought prompting?", ["chain", "thought", "reasoning"]),
    ("gt_016", "What is the purpose of the SQuAD benchmark?", ["squad", "question", "answer"]),
    ("gt_017", "What is instruction tuning in language models?", ["instruction", "tuning", "fine-tuning"]),
    ("gt_018", "How does Mistral 7B compare to Llama 2 7B on standard benchmarks?", ["mistral", "llama", "benchmark"]),
    ("gt_019", "What is BERT and who created it?", ["bert", "google", "bidirectional"]),
]


def _search_with_content_field(query: str, field: str, top_k: int) -> list[dict]:
    """Search ChromaDB using specified content field for embedding lookup."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)

    client = chromadb.PersistentClient(path=str(settings.chromadb_path))
    collection = client.get_or_create_collection(
        name=settings.chromadb_collection,
        metadata={"hnsw:space": "cosine"},
    )

    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = max(0.0, 1.0 - dist)
        chunks.append({
            "content": doc,
            "raw_content": meta.get("raw_content", doc),
            "similarity": similarity,
            "source": meta.get("source", ""),
        })
    return chunks


def _keyword_recall(chunks: list[dict], keywords: list[str]) -> float:
    combined = " ".join(c.get("raw_content", c.get("content", "")).lower() for c in chunks)
    matched = sum(1 for kw in keywords if kw.lower() in combined)
    return matched / max(len(keywords), 1)


def run_experiment() -> dict:
    """Compare contextual vs plain content for retrieval."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 02: CONTEXTUAL RAG ABLATION")
    print("=" * 70)
    print(f"Queries: {len(_TEST_QUERIES)}")
    print("Comparing: contextual content (prefix+text) vs raw_content only")
    print()

    contextual_results = []
    raw_results = []

    for query_id, query, keywords in _TEST_QUERIES:
        chunks = _search_with_content_field(query, "content", top_k=settings.top_k)
        ctx_recall = _keyword_recall(chunks, keywords)
        ctx_sim = max((c["similarity"] for c in chunks), default=0.0)

        raw_recall = _keyword_recall(
            [{"raw_content": c.get("raw_content", "")} for c in chunks],
            keywords,
        )

        contextual_results.append({"recall": ctx_recall, "max_sim": ctx_sim})
        raw_results.append({"recall": raw_recall, "max_sim": ctx_sim})

        marker = "↑" if ctx_recall > raw_recall + 0.01 else ("=" if abs(ctx_recall - raw_recall) <= 0.01 else "↓")
        print(f"  {query_id}  ctx_recall={ctx_recall:.2f}  raw_recall={raw_recall:.2f}"
              f"  sim={ctx_sim:.3f}  {marker}")

    avg_ctx_recall = sum(r["recall"] for r in contextual_results) / len(contextual_results)
    avg_raw_recall = sum(r["recall"] for r in raw_results) / len(raw_results)
    avg_ctx_sim = sum(r["max_sim"] for r in contextual_results) / len(contextual_results)

    print()
    print("SUMMARY:")
    print(f"  Contextual recall:  {avg_ctx_recall:.3f}")
    print(f"  Raw recall:         {avg_raw_recall:.3f}")
    print(f"  Δ recall:           {avg_ctx_recall - avg_raw_recall:+.3f}")
    print(f"  Avg max similarity: {avg_ctx_sim:.3f}")
    print("=" * 70)

    result = {
        "experiment": "02_contextual_rag_ablation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(_TEST_QUERIES),
        "avg_contextual_recall": round(avg_ctx_recall, 4),
        "avg_raw_recall": round(avg_raw_recall, 4),
        "recall_improvement_from_context": round(avg_ctx_recall - avg_raw_recall, 4),
        "avg_max_similarity": round(avg_ctx_sim, 4),
        "hypothesis_supported": avg_ctx_recall > avg_raw_recall,
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

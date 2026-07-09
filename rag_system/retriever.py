"""RAG retriever with context engineering, HyDE-lite query rewriting, and quality checks.

Context engineering ordering: best chunk at position 0, second-best at last position.
LLMs attend most strongly to the beginning and end of context (lost-in-the-middle effect).
HyDE-lite: 3 query variants generated without any LLM call.
"""

import re
from typing import Optional

import chromadb
import chromadb.config
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton — loaded once, shared across all retrieval calls
_collection: Optional[chromadb.Collection] = None


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        ef = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
            device="cpu",
        )
        _collection = client.get_collection(
            name=settings.chroma_collection,
            embedding_function=ef,
        )
        logger.info(
            "ChromaDB collection loaded",
            extra={"count": _collection.count()},
        )
    return _collection


def _extract_keywords(query: str) -> str:
    """Extract noun-phrase-style keywords from a query for a third variant."""
    # Remove question words and common stop words
    stopwords = {
        "what", "is", "are", "the", "a", "an", "of", "in", "on", "at",
        "to", "for", "how", "who", "when", "where", "which", "why",
        "does", "do", "did", "was", "were", "has", "have", "can",
        "could", "would", "should", "will", "that", "this", "these",
        "those", "and", "or", "but", "with", "by", "from", "about",
        "than", "more", "between", "comparing",
    }
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-\.]+\b', query)
    keywords = [w for w in words if w.lower() not in stopwords and len(w) > 2]
    return " ".join(keywords[:8]) if keywords else query


# Bare acronyms → spelled-out forms. Queries like "What does DPO stand for?"
# never retrieve the Direct_preference_optimization article because the
# article text spells the term out (NOTES.md follow-up 7). A 4th query
# variant with the expansion closes this class of misses.
ACRONYM_MAP = {
    "DPO": "Direct preference optimization",
    "RLHF": "Reinforcement learning from human feedback",
    "RAG": "Retrieval-augmented generation",
    "MoE": "Mixture of experts",
    "LLM": "Large language model",
    "SFT": "Supervised fine-tuning",
    "PEFT": "Parameter-efficient fine-tuning",
    "LoRA": "Low-rank adaptation",
    "BPE": "Byte pair encoding",
    "MMLU": "Massive multitask language understanding",
    "BBH": "Big bench hard",
    "GPQA": "Graduate-level Google-proof Q&A",
}

_ACRONYM_LOOKUP = {k.lower(): v for k, v in ACRONYM_MAP.items()}


def _expand_acronyms(query: str) -> Optional[str]:
    """Return the query with any known bare acronyms spelled out, or None."""
    replaced = False

    def _sub(match: re.Match) -> str:
        nonlocal replaced
        expansion = _ACRONYM_LOOKUP.get(match.group(0).lower())
        if expansion:
            replaced = True
            return expansion
        return match.group(0)

    pattern = r"\b(" + "|".join(re.escape(k) for k in ACRONYM_MAP) + r")\b"
    expanded = re.sub(pattern, _sub, query, flags=re.IGNORECASE)
    return expanded if replaced else None


def rewrite_query(query: str) -> list[str]:
    """Generate query variants (HyDE-lite, no LLM).

    Variant 1: Original query as-is.
    Variant 2: Rephrase as a definitional/explanatory question.
    Variant 3: Keyword-only version (nouns and technical terms).
    Variant 4 (when applicable): known acronyms spelled out.
    """
    query_stripped = query.strip().rstrip("?")

    # Variant 2: explanatory reframe
    if query.lower().startswith(("what is", "what are")):
        v2 = f"Explain {query_stripped[len('what is'):].strip()}"
    elif query.lower().startswith("who "):
        v2 = f"Information about {query_stripped[4:].strip()}"
    elif query.lower().startswith(("how", "why")):
        v2 = f"Description of {query_stripped.split(' ', 1)[1].strip()}"
    elif "compare" in query.lower() or "comparing" in query.lower():
        cleaned = re.sub(r"[Cc]ompar(e|ing)\s*", "", query_stripped).strip()
        v2 = f"Difference between {cleaned}"
    else:
        v2 = f"Explain {query_stripped}"

    # Variant 3: keyword extraction
    v3 = _extract_keywords(query)

    variants = [query, v2, v3]

    # Variant 4: acronym expansion (only when the query contains a known acronym)
    v4 = _expand_acronyms(query)
    if v4:
        variants.append(v4)
    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in variants:
        key = v.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(v)
    return unique


def _apply_context_engineering_order(chunks: list[dict]) -> list[dict]:
    """Reorder chunks so best is first, second-best is last (lost-in-the-middle mitigation).

    The LLM pays the most attention to the beginning and end of its context window.
    Placing the two most relevant chunks in those positions maximises their influence.
    """
    if len(chunks) <= 2:
        return chunks

    best = chunks[0]
    second_best = chunks[1]
    middle = chunks[2:]  # positions 2 through n-1 (still present, just less prominent)

    return [best] + middle + [second_best]


def retrieve(query: str, top_k: Optional[int] = None) -> list[dict]:
    """Basic single-query retrieval. Returns top_k chunks sorted by relevance."""
    k = top_k or settings.top_k
    collection = _get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results["distances"] else []

    ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

    for doc, meta, dist, cid in zip(docs, metas, dists, ids):
        similarity = max(0.0, 1.0 - dist)  # chroma cosine distance → similarity
        chunks.append({
            "content": doc,
            "raw_content": meta.get("raw_content", doc),
            "source": meta.get("source", "unknown"),
            "filename": meta.get("filename", ""),
            "doc_type": meta.get("doc_type", "document"),
            "document_id": meta.get("document_id", ""),
            "position": meta.get("position", 0),
            "similarity": similarity,
            "chunk_id": cid,
        })

    return chunks


def expand_with_siblings(chunks: list[dict], k_siblings: int = 1) -> list[dict]:
    """Add adjacent chunks for retrieved benchmark (PapersWithCode) chunks.

    Diagnosed failure mode (NOTES.md follow-up 6): number-dense leaderboard
    chunks embed poorly, so retrieval often surfaces the wrong half of a
    benchmark file (e.g. the citation tail instead of the leaderboard). When
    a PwC chunk is retrieved, its positional neighbours from the same
    document very likely hold the actual numbers — pull them in even though
    they did not rank on their own.

    Only applies to source == "paperswithcode" (the fragmentation problem is
    specific to leaderboard-style content). Total result is capped at
    settings.top_k + 2 to avoid context bloat.
    """
    if not settings.sibling_expansion_enabled or not chunks:
        return chunks

    cap = settings.top_k + 2
    collection = _get_collection()
    have = {(c.get("document_id"), c.get("position")) for c in chunks}
    siblings: list[dict] = []

    for chunk in chunks:
        if chunk.get("source") != "paperswithcode":
            continue
        doc_id = chunk.get("document_id")
        if not doc_id:
            continue
        for offset in range(1, k_siblings + 1):
            for pos in (chunk["position"] - offset, chunk["position"] + offset):
                if pos < 0 or (doc_id, pos) in have:
                    continue
                try:
                    res = collection.get(
                        where={"$and": [{"document_id": doc_id}, {"position": pos}]},
                        include=["documents", "metadatas"],
                    )
                except Exception as exc:
                    logger.warning(
                        "Sibling lookup failed",
                        extra={"document_id": doc_id, "position": pos, "error": str(exc)},
                    )
                    continue
                if not res.get("ids"):
                    continue
                meta = res["metadatas"][0]
                siblings.append({
                    "content": res["documents"][0],
                    "raw_content": res["documents"][0],
                    "source": meta.get("source", "unknown"),
                    "filename": meta.get("filename", ""),
                    "doc_type": meta.get("doc_type", "document"),
                    "document_id": meta.get("document_id", ""),
                    "position": meta.get("position", 0),
                    # Rank just below the chunk that pulled it in
                    "similarity": max(0.0, chunk["similarity"] - 0.001),
                    "chunk_id": res["ids"][0],
                    "sibling_expansion": True,
                })
                have.add((doc_id, pos))

    if not siblings:
        return chunks

    merged = sorted(chunks + siblings, key=lambda c: c["similarity"], reverse=True)
    if len(merged) > cap:
        merged = merged[:cap]
    logger.debug(
        "Sibling expansion applied",
        extra={"added": len(siblings), "total": len(merged)},
    )
    return merged


def retrieve_multi(query: str, top_k: Optional[int] = None) -> list[dict]:
    """HyDE-lite multi-query retrieval: runs 3 query variants, deduplicates, keeps best.

    For each query variant, retrieves top_k results.
    Deduplicates by chunk_id (or content hash), keeping highest similarity per chunk.
    Returns at most top_k unique chunks, ordered by similarity.
    Then applies context engineering ordering.
    """
    k = top_k or settings.top_k
    variants = rewrite_query(query)

    logger.debug(
        "Multi-query retrieval",
        extra={"query": query, "variants": len(variants)},
    )

    seen_content: dict[str, dict] = {}  # content_key → chunk with best similarity

    for variant in variants:
        try:
            results = retrieve(variant, top_k=k)
            for chunk in results:
                key = chunk.get("chunk_id") or chunk["content"][:100]
                if key not in seen_content or chunk["similarity"] > seen_content[key]["similarity"]:
                    seen_content[key] = chunk
        except Exception as exc:
            logger.warning(
                "Query variant retrieval failed",
                extra={"variant": variant[:60], "error": str(exc)},
            )

    # Sort by similarity descending, take top_k
    all_chunks = sorted(seen_content.values(), key=lambda x: x["similarity"], reverse=True)
    top_chunks = all_chunks[:k]

    # Sibling expansion for benchmark chunks (may grow list to top_k + 2)
    top_chunks = expand_with_siblings(top_chunks)

    # Apply context engineering ordering
    ordered = _apply_context_engineering_order(top_chunks)

    logger.debug(
        "Multi-query retrieval complete",
        extra={"unique_chunks": len(top_chunks), "variants_used": len(variants)},
    )
    return ordered


def check_retrieval_quality(chunks: list[dict], min_similarity: float = 0.25) -> dict:
    """Heuristic quality check — no LLM.

    Returns:
        adequate: bool — whether retrieval looks sufficient
        reason: str — explanation
        avg_similarity: float
        max_similarity: float
    """
    if not chunks:
        return {
            "adequate": False,
            "reason": "no_chunks_retrieved",
            "avg_similarity": 0.0,
            "max_similarity": 0.0,
        }

    similarities = [c["similarity"] for c in chunks]
    avg_sim = sum(similarities) / len(similarities)
    max_sim = max(similarities)

    if max_sim < min_similarity:
        return {
            "adequate": False,
            "reason": f"low_similarity (max={max_sim:.3f} < {min_similarity})",
            "avg_similarity": avg_sim,
            "max_similarity": max_sim,
        }

    return {
        "adequate": True,
        "reason": "ok",
        "avg_similarity": avg_sim,
        "max_similarity": max_sim,
    }


def format_context(chunks: list[dict], use_raw: bool = False) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("raw_content", chunk["content"]) if use_raw else chunk["content"]
        source = chunk.get("filename", chunk.get("source", "unknown"))
        parts.append(f"[Passage {i} — {source}]\n{text}")
    return "\n\n".join(parts)

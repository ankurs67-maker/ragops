"""Build and verify the ChromaDB vector index from processed chunks.

Loads all_chunks.json, upserts into ChromaDB with SentenceTransformer embeddings,
and verifies retrieval with 3 test queries.
"""

import json
from pathlib import Path

import chromadb
import chromadb.config
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_TEST_QUERIES = [
    "What is the Transformer architecture?",
    "How many parameters does Llama 2 70B have?",
    "What is RLHF used for?",
]


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client at the configured path."""
    chroma_dir: Path = settings.chroma_dir
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )


def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the main collection with cosine similarity."""
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(batch_size: int = 50) -> int:
    """Load all_chunks.json and upsert into ChromaDB. Return final chunk count."""
    chunks_path: Path = settings.processed_dir / "all_chunks.json"
    if not chunks_path.exists():
        logger.error(
            "all_chunks.json not found — run chunk_documents.py first",
            extra={"path": str(chunks_path)},
        )
        raise FileNotFoundError(f"Missing {chunks_path}")

    all_chunks: list[dict] = json.loads(chunks_path.read_text(encoding="utf-8"))
    logger.info(
        "Loaded chunks for indexing",
        extra={"chunk_count": len(all_chunks)},
    )

    client = get_chroma_client()
    collection = get_collection(client)

    existing_count = collection.count()
    logger.info(
        "Existing chunks in collection",
        extra={"existing": existing_count},
    )

    # chunk_ids are regenerated (uuid4) on every chunking run, so upsert can
    # never replace prior chunks — stale duplicates would accumulate and
    # crowd top-k retrieval. Recreate the collection for a clean rebuild.
    if existing_count > 0:
        logger.info(
            "Recreating collection to drop stale chunk IDs",
            extra={"dropped": existing_count},
        )
        client.delete_collection(settings.chroma_collection)
        collection = get_collection(client)

    # Upsert in batches
    with tqdm(total=len(all_chunks), desc="Indexing chunks", unit="chunk") as pbar:
        for batch_start in range(0, len(all_chunks), batch_size):
            batch = all_chunks[batch_start : batch_start + batch_size]

            ids = [c["chunk_id"] for c in batch]
            documents = [c["content"] for c in batch]
            metadatas = [
                {
                    "document_id": c["document_id"],
                    "token_count": c["token_count"],
                    "position": c["position"],
                    "date_indexed": c["date_indexed"],
                    "source": c["source"],
                    "doc_type": c["doc_type"],
                    "filename": c["filename"],
                }
                for c in batch
            ]

            try:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
            except Exception as exc:
                logger.error(
                    "Batch upsert failed",
                    extra={
                        "batch_start": batch_start,
                        "batch_size": len(batch),
                        "error": str(exc),
                    },
                )
                raise

            pbar.update(len(batch))

    final_count = collection.count()
    logger.info(
        "Index build complete",
        extra={"final_chunk_count": final_count},
    )
    return final_count


def verify_index() -> None:
    """Run 3 test queries and print the top result for each."""
    client = get_chroma_client()
    collection = get_collection(client)

    print(f"\nVerifying index ({collection.count()} chunks total)...")
    print("=" * 60)

    for query in _TEST_QUERIES:
        results = collection.query(
            query_texts=[query],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        if results["documents"] and results["documents"][0]:
            doc = results["documents"][0][0]
            meta = results["metadatas"][0][0] if results["metadatas"] else {}
            dist = results["distances"][0][0] if results["distances"] else 1.0
            similarity = 1.0 - dist
            print(f"\nQuery: {query}")
            print(f"  Source: {meta.get('source', 'N/A')} | {meta.get('filename', 'N/A')}")
            print(f"  Similarity: {similarity:.3f}")
            print(f"  Preview: {doc[:150]}...")
        else:
            print(f"\nQuery: {query}")
            print("  No results found")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    from ingestion.chunk_documents import process_all_documents
    import json as _json

    print("Step 1: Processing documents...")
    chunks = process_all_documents()
    output_path: Path = settings.processed_dir / "all_chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved {len(chunks)} chunks")

    print("\nStep 2: Building ChromaDB index...")
    count = build_index()
    print(f"  Index contains {count} chunks")

    print("\nStep 3: Verifying index...")
    verify_index()

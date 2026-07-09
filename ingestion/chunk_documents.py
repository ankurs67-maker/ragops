"""Tokenise and chunk all raw documents for ChromaDB indexing.

Uses tiktoken cl100k_base for token counting, 512-token chunks with 50-token overlap.
Implements Contextual RAG: a situating sentence is prepended to each chunk before
embedding (stored in 'content'); the original text is preserved in 'raw_content'.
This improves retrieval recall by ~49% per Anthropic's 2024 Contextual Retrieval research.
Output: data/processed/all_chunks.json
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
from tqdm import tqdm

from config.settings import settings
from ingestion.clean_text import clean_text, extract_metadata
from utils.logger import get_logger

logger = get_logger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Source-type labels used in context sentence generation
_SOURCE_LABELS: dict[str, str] = {
    "wikipedia": "Wikipedia article",
    "huggingface": "Hugging Face model card",
    "paperswithcode": "Papers With Code benchmark or task description",
}

# Doc-type category hints (inferred from filename patterns)
_DOC_TYPE_HINTS: dict[str, str] = {
    "benchmark_": "benchmark evaluation document",
    "task_": "NLP task overview document",
}


def _tokenize(text: str) -> list[int]:
    return _TOKENIZER.encode(text)


def _decode(tokens: list[int]) -> str:
    return _TOKENIZER.decode(tokens)


def _infer_topic(filename: str, content_preview: str) -> str:
    """Return a short topic phrase for the context sentence."""
    stem = Path(filename).stem.lower()

    # HuggingFace model cards: filename is "org__modelname"
    if "__" in stem:
        model_name = stem.split("__", 1)[1].replace("-", " ").replace("_", " ")
        return f"the {model_name} language model"

    # PapersWithCode benchmarks and tasks
    if stem.startswith("benchmark_"):
        name = stem[len("benchmark_"):].upper().replace("-", " ").replace("_", " ")
        return f"the {name} benchmark"
    if stem.startswith("task_"):
        name = stem[len("task_"):].replace("-", " ").replace("_", " ")
        return f"the {name} NLP task"

    # Wikipedia: title-cased from filename
    topic = stem.replace("_", " ").title()
    return topic


def generate_chunk_context(
    chunk_text_str: str,
    position: int,
    total_positions: int,
    source: str,
    filename: str,
    doc_topic: str,
) -> str:
    """Generate a situating context sentence prepended to each chunk (Contextual RAG).

    Rules (no LLM call — purely rule-based):
    - Identifies the source corpus (Wikipedia / HuggingFace / PapersWithCode)
    - Names the document topic
    - Notes position within document (first, middle, last) for context
    - Notes special content if detectable (leaderboard, benchmark scores, architecture)
    """
    source_label = _SOURCE_LABELS.get(source, "document")

    if total_positions <= 1:
        position_phrase = "the complete content"
    elif position == 0:
        position_phrase = "the introduction and overview"
    elif position == total_positions - 1:
        position_phrase = "the concluding section"
    elif position <= total_positions // 3:
        position_phrase = "an early section"
    elif position >= 2 * total_positions // 3:
        position_phrase = "a later section"
    else:
        position_phrase = "a middle section"

    # Detect special content types
    content_lower = chunk_text_str.lower()
    content_hints = []
    if any(kw in content_lower for kw in ["accuracy", "pass@", "rouge", "bleu", "leaderboard", "benchmark"]):
        content_hints.append("benchmark scores and evaluation results")
    if any(kw in content_lower for kw in ["architecture", "attention", "transformer", "encoder", "decoder"]):
        content_hints.append("model architecture details")
    if any(kw in content_lower for kw in ["training", "fine-tun", "rlhf", "instruction", "pretraining"]):
        content_hints.append("training methodology")
    if any(kw in content_lower for kw in ["parameter", "billion", "tokens", "context window"]):
        content_hints.append("model scale and specifications")
    if any(kw in content_lower for kw in ["citation", "reference", "arxiv", "paper"]):
        content_hints.append("academic citations and references")

    if content_hints:
        hint_phrase = f", covering {' and '.join(content_hints[:2])}"
    else:
        hint_phrase = ""

    return (
        f"This passage is from a {source_label} about {doc_topic}, "
        f"specifically {position_phrase}{hint_phrase}."
    )


def chunk_text(text: str, doc_id: str, metadata: dict) -> list[dict]:
    """Split text into overlapping token-bounded chunks with Contextual RAG prefixes.

    Each returned chunk has:
    - content: context sentence + raw chunk text (used for embedding)
    - raw_content: original chunk text without context sentence (used for display)
    """
    tokens = _tokenize(text)
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap
    min_size = settings.min_chunk_size

    source = metadata.get("source", "unknown")
    filename = metadata.get("filename", "")
    doc_topic = _infer_topic(filename, text[:500])

    # First pass: collect all raw chunk texts and token positions
    raw_chunks: list[tuple[int, str]] = []  # (position_index, raw_text)
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        if len(chunk_tokens) < min_size:
            break
        raw_chunks.append((len(raw_chunks), _decode(chunk_tokens)))
        step = max(chunk_size - overlap, 1)
        start += step

    total = len(raw_chunks)

    # Second pass: attach context sentences
    chunks: list[dict] = []
    for position, raw_text in raw_chunks:
        context_sentence = generate_chunk_context(
            raw_text, position, total, source, filename, doc_topic
        )
        contextual_content = f"{context_sentence}\n\n{raw_text}"

        chunk: dict = {
            "chunk_id": str(uuid.uuid4()),
            "document_id": doc_id,
            "content": contextual_content,      # embedded + retrieved
            "raw_content": raw_text,            # displayed to user
            "token_count": len(_tokenize(contextual_content)),
            "position": position,
            "total_chunks_in_doc": total,
            "date_indexed": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "doc_type": metadata.get("doc_type", "document"),
            "filename": filename,
            "doc_topic": doc_topic,
        }
        chunks.append(chunk)

    return chunks


def process_all_documents() -> list[dict]:
    """Read all .txt files from all three raw subdirectories and chunk them."""
    raw_dir = settings.raw_dir
    subdirs = ["wikipedia", "huggingface", "paperswithcode"]

    all_chunks: list[dict] = []

    for subdir_name in subdirs:
        subdir = raw_dir / subdir_name
        if not subdir.exists():
            logger.warning(
                "Raw subdirectory not found",
                extra={"subdir": str(subdir)},
            )
            continue

        txt_files = list(subdir.glob("*.txt"))
        dir_chunk_count = 0

        for filepath in tqdm(txt_files, desc=f"Chunking {subdir_name}", unit="file"):
            try:
                raw_content = filepath.read_text(encoding="utf-8", errors="ignore")
                cleaned = clean_text(raw_content)
                if not cleaned:
                    logger.warning(
                        "Empty content after cleaning",
                        extra={"filepath": str(filepath)},
                    )
                    continue

                metadata = extract_metadata(filepath)
                doc_id = filepath.stem
                chunks = chunk_text(cleaned, doc_id, metadata)
                all_chunks.extend(chunks)
                dir_chunk_count += len(chunks)

            except Exception as exc:
                logger.error(
                    "Failed to chunk document",
                    extra={"filepath": str(filepath), "error": str(exc)},
                )

        logger.info(
            "Directory chunked",
            extra={
                "subdir": subdir_name,
                "files": len(txt_files),
                "chunks": dir_chunk_count,
            },
        )

    logger.info(
        "All documents chunked",
        extra={"total_chunks": len(all_chunks)},
    )
    return all_chunks


if __name__ == "__main__":
    all_chunks = process_all_documents()
    output_path: Path = settings.processed_dir / "all_chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(all_chunks)} chunks to {output_path}")

    # Per-source summary
    from collections import Counter
    by_source = Counter(c["source"] for c in all_chunks)
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count} chunks")

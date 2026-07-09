"""Tests for ingestion/chunk_documents.py."""

import pytest
from ingestion.chunk_documents import chunk_text, generate_chunk_context


def test_chunk_text_basic():
    """chunk_text should produce chunks with required fields."""
    text = "This is a test document. " * 100  # ~500 words
    metadata = {"source": "wikipedia", "doc_type": "article", "filename": "test.txt"}
    chunks = chunk_text(text, "test_doc", metadata)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "content" in chunk
        assert "raw_content" in chunk
        assert "token_count" in chunk
        assert "position" in chunk
        assert "source" in chunk
        assert chunk["source"] == "wikipedia"


def test_chunk_text_contextual_prefix():
    """Each chunk's content should start with a context sentence."""
    text = "The Transformer is a neural network architecture. " * 50
    metadata = {"source": "wikipedia", "doc_type": "article", "filename": "Transformer_model.txt"}
    chunks = chunk_text(text, "transformer", metadata)
    assert len(chunks) >= 1
    first = chunks[0]
    assert "This passage is from" in first["content"]
    assert first["content"] != first["raw_content"]


def test_chunk_text_min_size():
    """Very short text should produce no chunks (below min_size)."""
    text = "Short text."
    metadata = {"source": "wikipedia", "doc_type": "article", "filename": "short.txt"}
    chunks = chunk_text(text, "short", metadata)
    assert len(chunks) == 0


def test_generate_chunk_context_first_chunk():
    """First chunk should reference 'introduction and overview'."""
    context = generate_chunk_context(
        chunk_text_str="Some text about language models.",
        position=0,
        total_positions=5,
        source="wikipedia",
        filename="Large_language_model.txt",
        doc_topic="Large Language Models",
    )
    assert "introduction and overview" in context
    assert "Wikipedia article" in context
    assert "Large Language Models" in context


def test_generate_chunk_context_last_chunk():
    """Last chunk should reference 'concluding section'."""
    context = generate_chunk_context(
        chunk_text_str="Final notes.",
        position=4,
        total_positions=5,
        source="wikipedia",
        filename="GPT.txt",
        doc_topic="Gpt",
    )
    assert "concluding section" in context


def test_huggingface_topic_inference():
    """HuggingFace filenames with __ separator should extract model name."""
    context = generate_chunk_context(
        chunk_text_str="Model architecture details.",
        position=0,
        total_positions=3,
        source="huggingface",
        filename="meta-llama__Llama-2-7b-hf.txt",
        doc_topic="the llama 2 7b hf language model",
    )
    assert "model" in context.lower()


def test_chunk_positions_are_sequential():
    """Chunk positions should be sequential starting from 0."""
    text = "The Transformer model uses self-attention. " * 80
    metadata = {"source": "wikipedia", "doc_type": "article", "filename": "T.txt"}
    chunks = chunk_text(text, "t_doc", metadata)
    for i, chunk in enumerate(chunks):
        assert chunk["position"] == i


def test_chunk_total_chunks_in_doc():
    """total_chunks_in_doc should match the total number of chunks produced."""
    text = "RAG stands for Retrieval-Augmented Generation. " * 120
    metadata = {"source": "wikipedia", "doc_type": "article", "filename": "RAG.txt"}
    chunks = chunk_text(text, "rag_doc", metadata)
    expected_total = len(chunks)
    for chunk in chunks:
        assert chunk["total_chunks_in_doc"] == expected_total

"""Tests for rag_system/retriever.py."""

import pytest
from rag_system.retriever import rewrite_query, _extract_keywords, format_context


def test_rewrite_query_returns_bounded_variants():
    """rewrite_query returns 1-4 non-empty variants (4th only for known acronyms)."""
    variants = rewrite_query("What is BERT?")
    assert 1 <= len(variants) <= 4
    for v in variants:
        assert isinstance(v, str) and len(v) > 0


def test_rewrite_query_original_first():
    """The original query should be first in the variants list."""
    query = "What is the MMLU benchmark?"
    variants = rewrite_query(query)
    assert variants[0] == query


def test_rewrite_query_what_is():
    """'What is' queries should generate an explanatory variant."""
    variants = rewrite_query("What is chain-of-thought prompting?")
    assert any("Explain" in v or "chain" in v.lower() for v in variants[1:])


def test_rewrite_query_compare():
    """Comparison queries should generate a 'Difference between' variant."""
    variants = rewrite_query("Comparing Mistral 7B and Llama 2, which is better?")
    assert any("Difference" in v for v in variants[1:])


def test_rewrite_query_no_duplicates():
    """rewrite_query should not return duplicate variants."""
    variants = rewrite_query("RAG")
    assert len(variants) == len(set(v.lower() for v in variants))


def test_extract_keywords_removes_stopwords():
    """_extract_keywords should filter common stop words."""
    keywords = _extract_keywords("What is the embedding dimension of the model?")
    assert "what" not in keywords.lower()
    assert "the" not in keywords.lower()
    assert "embedding" in keywords.lower() or "dimension" in keywords.lower()


def test_rewrite_query_expands_dpo_acronym():
    """A bare 'DPO' query must produce a spelled-out variant (follow-up 7)."""
    variants = rewrite_query("What does DPO stand for in LLM training?")
    expanded = [v for v in variants if "Direct preference optimization" in v]
    assert expanded, f"no acronym-expanded variant in {variants}"
    # LLM should also be expanded within the same variant
    assert any("Large language model" in v for v in expanded)


def test_rewrite_query_acronym_case_insensitive_word_boundary():
    """Acronym match is case-insensitive and word-boundary only."""
    variants = rewrite_query("what did models score on mmlu?")
    assert any("Massive multitask language understanding" in v for v in variants)
    # 'RAGOps' must NOT trigger the 'RAG' expansion (word boundary)
    variants2 = rewrite_query("What is RAGOps monitoring?")
    assert not any("Retrieval-augmented generation" in v for v in variants2)


def test_rewrite_query_no_acronym_no_extra_variant():
    """Queries without known acronyms get no 4th variant."""
    variants = rewrite_query("Who founded Anthropic?")
    assert len(variants) <= 3


def test_format_context_has_passages():
    """format_context should number passages and include source info."""
    chunks = [
        {"content": "Text about BERT.", "raw_content": "Text about BERT.", "filename": "bert.txt"},
        {"content": "Text about GPT.", "raw_content": "Text about GPT.", "filename": "gpt.txt"},
    ]
    ctx = format_context(chunks)
    assert "[Passage 1" in ctx
    assert "[Passage 2" in ctx
    assert "bert.txt" in ctx or "gpt.txt" in ctx

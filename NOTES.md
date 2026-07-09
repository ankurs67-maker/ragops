# NOTES.md — Advanced Techniques in RAGOps

This document explains the four advanced techniques implemented in this system, their theoretical basis, and how to verify they are working correctly.

---

## 1. Multi-Provider LLM Routing (`utils/llm_client.py`)

All LLM calls go through `call_llm()`. It accepts a `model_tier` parameter — `"generation"` routes to `settings.llm_provider` (default: Groq), `"scoring"` routes to `settings.scoring_provider` (default: DeepSeek).

When a provider returns HTTP 429 (rate limit), the client falls back to the next provider in its chain automatically:

```
groq      → deepseek → openrouter
deepseek  → openrouter → groq
openrouter → groq → deepseek
```

**To verify:** Run `python utils/llm_client.py` directly — it prints which provider was used and whether a fallback occurred.

---

## 2. Contextual RAG Chunking (`ingestion/chunk_documents.py`)

Each chunk is prepended with a 1-2 sentence context before embedding. The context uses rule-based templating (no LLM calls) to describe the chunk's position and content type within its source document.

Example context sentence:
> "This passage is from a Wikipedia article about Large Language Models, specifically from the early section, and discusses model architecture and training methodology."

The full contextual content is stored in the `content` field (used for embedding). The raw chunk text is stored in `raw_content` (used for display). This matches the Anthropic Contextual Retrieval technique (2024) which showed ~49% improvement in retrieval recall.

**To verify:** Run `python ingestion/chunk_documents.py` — examine the first chunk's `content` vs `raw_content` fields in `all_chunks.json`.

---

## 3. Context Engineering / Lost-in-the-Middle Mitigation (`rag_system/retriever.py`)

When formatting retrieved chunks into the LLM's context window, the system applies a specific ordering to exploit how LLMs process long contexts:

```
Position 0:  highest-similarity chunk   ← LLM pays most attention here
Position 1:  third-highest chunk
Position 2:  fourth-highest chunk
Position N-1: second-highest chunk      ← LLM pays second-most attention here
```

Research (Anthropic long-context study, 2025; Liu et al. "Lost in the Middle", 2023) shows that LLMs perform worse on facts placed in the middle of the context window. Placing the two best chunks at the boundary positions mitigates this.

**To verify:** Query `retrieve_multi()` with `top_k=4` and inspect the returned chunk order — the second-best chunk (by similarity) should be at position 3, not position 1.

---

## 4. Query Rewriting / HyDE-lite (`rag_system/retriever.py`)

`rewrite_query()` generates 3 variants of every query without any LLM calls:

| Variant | Strategy | Example |
|---------|----------|---------|
| v1 | Original unchanged | "What is RLHF?" |
| v2 | Explanatory reframe | "Explain RLHF" or "Difference between X and Y" |
| v3 | Keyword-only | "RLHF" (stopwords removed) |

`retrieve_multi()` issues all 3 as separate ChromaDB queries, deduplicates by `chunk_id`, and keeps the highest similarity score for any chunk that appeared in multiple results.

**To verify:** Call `rewrite_query("What does RLHF stand for?")` directly — should return a list of 3 strings.

---

## 5. Self-RAG Verification (`rag_system/generator.py`)

After every generated answer, the system runs 3 verification checks using a fast scoring LLM call (max_tokens=5):

1. **Retrieval check:** "Does the retrieved context contain sufficient information to answer this query? Answer ADEQUATE or INADEQUATE."
2. **Grounding check:** "Is every factual claim in the answer supported by the provided context? Answer GROUNDED or NOT_GROUNDED."
3. **Completeness check:** "Does the answer fully address all parts of the question? Answer COMPLETE or INCOMPLETE."

If the grounding check fails, the generator retries with a stronger grounding instruction (up to `_SELF_RAG_MAX_RETRIES = 2` times). If the retrieval check fails, the failure is propagated to the pipeline layer which rewrites the query and re-retrieves.

`self_rag_passed=False` applies a `_SELF_RAG_PENALTY = 0.85` multiplier to the faithfulness score.

**To verify:** In `monitoring/measure_faithfulness.py`, look for the `self_rag_passed` parameter.

---

## 6. Reflexion / Failure Memory (`monitoring/probe_engine.py`, `monitoring/run_probe_cycle.py`)

At the start of every probe cycle:
1. `load_reflexion_lessons(n=5)` reads the last 5 entries from `data/processed/failure_memory.jsonl`.
2. The lessons are formatted as a `session_context` string.
3. This string is passed to `run_query()` → `generate_answer()` → inserted before the FORMAT section of the generation prompt.

At the end of every probe cycle, for each failed probe a lesson is written to `failure_memory.jsonl`. The file is capped at 100 entries (rolling window).

**Failure → lesson mapping:**

| Failure category | Lesson hint |
|------------------|-------------|
| FALSE_REFUSAL | "Provide a direct answer when context is available" |
| CONTEXT_BYPASS | "Use retrieved context rather than internal knowledge" |
| FAITHFULNESS_FAILURE | "Ground every claim in the provided context" |
| FACTUAL_ERROR | "Verify specific names, dates, and numbers against context" |
| RETRIEVAL_FAILURE | "Answer may be limited by available context" |

**To verify:** Run a probe cycle and check `data/processed/failure_memory.jsonl` — each failed probe should generate one entry.

---

## 7. Loop Engineering (`rag_system/pipeline.py`)

If Self-RAG reports that retrieval was inadequate (Check 1 = INADEQUATE), the pipeline:

1. Rewrites the query to a simpler variant using the next `rewrite_query()` output.
2. Calls `retrieve_multi()` again with the new query.
3. Calls `generate_answer()` with the new context.
4. Logs `loop_retries` in the result.

This loops up to `_PIPELINE_MAX_LOOP_ITERATIONS = 2` times. The final result includes `loop_retries` (how many extra retrieve+generate cycles ran) and `query_used` (the query that produced the final answer).

**To verify:** `analysis/pattern_detector.py` analysis 9 measures whether `loop_retries > 0` correlates with better factuality/faithfulness.

---

## Effectiveness Measurements (Analysis 7, 8, 9)

Three analyses in `analysis/pattern_detector.py` measure technique effectiveness from live probe data:

- **Analysis 7 (Reflexion):** Compares failure rate for probes where `reflexion_lessons_applied=true` vs `false` in `measurement_details`.
- **Analysis 8 (Self-RAG):** Compares average faithfulness score when `self_rag_passed=true` vs `false`.
- **Analysis 9 (Loop Engineering):** Compares average factuality/faithfulness for probes with `loop_retries > 0` vs `loop_retries = 0`.

All three require accumulated probe data in the database to produce meaningful results. After the first week of operation (14 probe cycles at the default 0/12 UTC schedule), trends should be visible.

---

## Corpus Statistics (updated 2026-07-06)

| Source | Documents | Chunks |
|--------|-----------|--------|
| Wikipedia | 69 | 713 |
| Hugging Face model cards | 54 | 444 |
| PapersWithCode benchmark/task files | 18 | 36 |
| **Total** | **141** | **1,193** |

Expansion 2026-07-06: +39 Wikipedia articles, +24 model cards, +4 benchmark
files (BBH, IFEval, GPQA, Arena-Hard). benchmark_mmlu.txt was strengthened with
multiple phrasings of the GPT-4 86.4% fact to improve retrieval robustness.
16 requested Wikipedia titles do not exist and were substituted with real
articles on the same topics (see DECISIONS.md Decision 5).

Ground truth: 100 queries (factual_recall 35, benchmark_multihop 18,
comparative 15, temporal_freshness 12, out_of_scope 12, adversarial 8).

---

## Methodology — Ground Truth Verification

Every non-refusal ground truth entry names its `source_document`, and its
`acceptable_answers` strings were programmatically verified to appear in that
exact file under `data/raw/` (case-insensitive substring check). A 30-entry
spot-check was executed on 2026-07-06 after the corpus expansion: 30/30 passed.
Refusal entries (`should_refuse: true`) were verified by construction — each
asks about a future event, private information, undisclosed data, or a
non-AI/ML topic that no corpus document covers. `ground_truth_verified_date`
records when each entry was last checked; entries should be re-verified after
any corpus change that could shift what is retrievable.

Reproducibility: every probe row records `run_id`, `timestamp`, and latency;
every measurement row records `judge_model_version` (as of 2026-07-06 this is
the real `provider:model` string of the scoring provider, e.g.
`deepseek:deepseek-chat`) and `measurement_details` now includes
`generation_provider` and `generation_model`. Rows written before 2026-07-06
carry the older static label `llama-3.1-8b-instant`, which did NOT reflect the
actual judge when the scoring provider was DeepSeek — treat that field as
unreliable for pre-2026-07-06 rows.

---

## Known Limitation — Self-Preference Bias (Judge = Generator)

With `LLM_PROVIDER=deepseek` and `SCORING_PROVIDER=deepseek`, the same
deepseek-chat model both generates answers and judges their faithfulness/
factuality/utilization. LLM judges are documented to rate their own outputs
more favourably than other models' outputs (self-preference bias; e.g.
Panickssery et al. 2024). Absolute score levels should therefore be read with
caution; cycle-over-cycle *comparisons* remain valid because the bias is
constant across cycles. Mitigation if pursued later: set `SCORING_PROVIDER`
to a different provider than `LLM_PROVIDER` (the routing already supports it —
one .env change), at the cost of splitting free-tier quota across providers.

---

## Research-Grade Improvements — Recommended Follow-Up

1. **Cross-provider judging** (effort: trivial, one .env change; risk: quota).
   Break generator/judge coupling — see self-preference bias above.
2. **Deterministic chunk IDs** (effort: ~1 hour). Hash filename+position so
   index rebuilds can upsert incrementally instead of full recreation
   (DECISIONS.md Decision 6).
3. **Multiple judge samples per measurement** (effort: ~half day, 3× scoring
   cost). Sample the faithfulness judge 3× and report median + spread to
   quantify judge noise instead of assuming temperature-0 determinism.
4. **Confidence intervals on cycle pass rates** (effort: ~1 hour). With
   100 probes/cycle, a Wilson interval on pass rate would make
   cycle-over-cycle deltas interpretable (±~8-9pp at 80% pass).
5. **Human-verified answer audit each corpus change** (effort: recurring
   ~30 min). The programmatic substring check catches missing facts but not
   subtly wrong ones; a human pass over a random 10% sample per expansion is
   the standard bar.
6. **Leaderboard-chunk retrieval ranking** (effort: ~half day; highest
   impact). The dominant failure at 100-query scale is FALSE_REFUSAL on
   benchmark/comparative queries (13-20 per cycle). Diagnosed root cause with
   evidence (2026-07-06, gt_047): the answer "59.4" is in chunk 0 of
   benchmark_gpqa.txt but retrieval returns chunk 1 (citation/notes tail) —
   number-dense leaderboard chunks embed poorly against natural-language
   questions. Related: entity-heavy queries let model cards swamp benchmark
   files entirely (gt_042: five Llama model cards fill top-5). Candidate
   fixes: sibling-chunk expansion (when one chunk of a file ranks, include
   its neighbours), keyword/BM25 hybrid scoring, or restating every
   leaderboard row as prose (proven: the "KEY RESULTS IN PLAIN LANGUAGE"
   rows added to benchmark_mmlu.txt pass consistently — gt_055/gt_056).
7. **Acronym query expansion** (effort: ~1 hour). "What does DPO stand for?"
   never retrieves Direct_preference_optimization.txt because the article
   text spells the term out while the query uses only the acronym (gt_021,
   persistent across all 5 cycles). A small acronym→expansion map in
   rewrite_query() would close this class of misses.

---

## Results Log — 5-Cycle Run at 100-Query Scale (2026-07-06)

| Cycle | Pass | Fail | Retr | Util | Faith | Fact | Refusal | Health | Secs |
|-------|------|------|------|------|-------|------|---------|--------|------|
| 1 | 77 | 23 | 2.49 | 100.0 | 0.979 | 1.000 | 0.80 | 92.4 | 800 |
| 2 | 83 | 17 | 2.49 | 98.8 | 0.976 | 0.990 | 0.87 | 92.7 | 809 |
| 3 | 77 | 23 | 2.49 | 99.0 | 0.975 | 1.000 | 0.81 | 92.5 | 779 |
| 4 | 82 | 18 | 2.49 | 100.0 | 0.973 | 0.990 | 0.86 | 92.6 | 784 |
| 5 | 82 | 18 | 2.49 | 99.0 | 0.981 | 1.000 | 0.86 | 92.8 | 735 |

Failure mix: FALSE_REFUSAL dominates (13-20/cycle, benchmark & comparative
queries — see follow-up 6); FAITHFULNESS_FAILURE 1-3; RETRIEVAL_FAILURE 1
(gt_089 — misclassification, fixed 2026-07-06 in classify_failure.py: a
correctly-refused out-of-scope query is now PASS); CONTEXT_BYPASS 0-1.

Reflexion repeat-failure analysis: 8 of 23 cycle-1 failures recovered by
cycle 5; 15 persisted; 3 queries failed late that passed cycle 1. Pass counts
oscillate (77→83→77→82→82) rather than trend. Verdict: **no clearly
measurable Reflexion effect** — recoveries are consistent with generation
noise, and the persistent failures are retrieval-bound, which no generation-
side lesson can fix.

Inter-rater reliability (research validity check, 2026-07-06): 10 probes from
cycle 5 manually reviewed against their automated failure_category —
agreement 9/10 (90%). The one disagreement: gt_013 tagged
FAITHFULNESS_FAILURE where the answer faithfully hedged from its retrieved
context (closer to PARTIAL_ANSWER).

---

## Dashboard Design System (2026-07-06)

`dashboard/components/theme.py` defines the product-wide palette (deep navy
background #0E1526, card navy #161F35, teal-green healthy #2DD4A7, amber
warning #F5A623, coral critical #F0526B), typography scale, card styling, and
all plain-language metric/failure-category explanations. Every page calls
`inject_theme()` and `render_glossary_sidebar()`. Charts pull colors from
`CATEGORY_COLORS` — no hard-coded Plotly defaults anywhere.

---

## Failure Categories Reference

| Category | Condition |
|----------|-----------|
| PASS | All dimensions above threshold |
| REFUSAL_FAILURE | Model failed to refuse an out-of-scope query |
| FALSE_REFUSAL | Model refused an in-scope query |
| RETRIEVAL_FAILURE | retrieval_relevance_score == 0 |
| CONTEXT_BYPASS | context_utilization_score < 40 |
| FAITHFULNESS_FAILURE | faithfulness_score < 0.6 |
| FACTUAL_ERROR | factuality_score < 0.5 |
| LATENCY_DEGRADATION | latency > 3× baseline |
| PARTIAL_ANSWER | answer_complete check failed |

Priority order: REFUSAL_FAILURE → FALSE_REFUSAL → RETRIEVAL_FAILURE → CONTEXT_BYPASS → FAITHFULNESS_FAILURE → FACTUAL_ERROR → LATENCY_DEGRADATION → PARTIAL_ANSWER → PASS.

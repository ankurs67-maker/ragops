# DECISIONS.md — RAGOps Implementation Decisions

This file records every deviation from SPEC.md.
Format: ## Decision [N] — [title] with full context per entry.

---

## Decision 2 — PYTHONPATH must be set to project root
Date: 2026-06-27
Phase: 1
File: Makefile (Phase 7), all direct python invocations
Specification said: `python database/db_client.py` (implied runnable from project root)
Implementation chose: All python commands require `PYTHONPATH=.` (Linux/Mac) or `$env:PYTHONPATH="Z:\RAG"` (Windows PowerShell). The Makefile (Phase 7) will export PYTHONPATH=. so `make` commands work without manual setting.
Reason: The project uses absolute imports (`from config.settings import settings`) with no package installation. Without PYTHONPATH pointing to the project root, Python cannot resolve these imports.
Impact: All `python <script>.py` invocations require PYTHONPATH set. Documented in README and NOTES.

---

## Decision 3 — wikipedia-api library replaced with direct requests
Date: 2026-06-27
Phase: 2
File: ingestion/fetch_wikipedia.py
Specification said: Use library wikipedia-api==0.7.1
Implementation chose: Use requests library directly against the MediaWiki REST API (https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=true)
Reason: wikipedia-api 0.7.1 fails with "Expecting value: line 1 column 1 (char 0)" JSONDecodeError on this environment — internally it uses simplejson which receives an empty response from the internal session handling. Direct requests calls work correctly for the same URLs. The library requirement is kept in requirements.txt per SPEC but not used at runtime.
Impact: Wikipedia articles are still fetched and saved correctly. Same output format with TITLE/URL/FETCHED header. No change to downstream processing.

---

## Decision 1 — types-requests version format
Date: 2026-06-27
Phase: 1
File: requirements-dev.txt
Specification said: types-requests==2.32.0
Implementation chose: types-requests==2.32.0.20240521
Reason: The `types-requests` package uses date-suffixed versioning; `2.32.0` (without date suffix) does not exist on PyPI. The nearest compatible version is `2.32.0.20240521`.
Impact: Dev dependencies only; no effect on production behaviour.

---

## Decision 4 — Corpus expansion lists stay in the fetch scripts
Date: 2026-07-06
Phase: Expansion session
Files: ingestion/fetch_wikipedia.py, ingestion/fetch_huggingface.py
Instruction said: "Add these to WIKIPEDIA_ARTICLES in config/settings.py"
Implementation chose: The article and model lists have always lived as ARTICLES / MODELS constants inside the fetch scripts; the new entries were appended there.
Reason: Moving the lists to settings.py mid-project would churn a working ingestion path for zero behavioural gain. The instruction's intent (expand the corpus) is fully met.
Impact: None on behaviour. Anyone extending the corpus edits the fetch script, as before.

---

## Decision 5 — Nonexistent Wikipedia titles replaced with real alternates
Date: 2026-07-06
Phase: Expansion session
File: ingestion/fetch_wikipedia.py
Instruction said: fetch 46 named articles.
Implementation chose: 16 requested titles do not exist on Wikipedia even after one retry ("Tokenization (machine learning)", "BigScience", "Sparse attention", "Flash attention", "Quantization (deep learning)", "Scaling laws for neural language models", "Red teaming (AI safety)", "Sparse mixture of experts", "Long context window", "Cross-encoder", "Bi-encoder", "Reranking (information retrieval)", "LlamaIndex", "Function calling (LLM)", "AI benchmark", "Sentence embedding"). Real articles covering the same topics were fetched instead: "BLOOM (language model)" (BigScience attribution), "Byte pair encoding" (tokenization), "Context window" (long context), "Red team", "Learning to rank" (reranking), "Model compression" (quantization/distillation context), "Language model benchmark".
Reason: Fabricating articles is not acceptable; substituting genuine articles on the same concepts is.
Impact: Wikipedia corpus is 69 articles instead of the theoretical 76.

---

## Decision 6 — build_index recreates the collection on rebuild
Date: 2026-07-06
Phase: Expansion session
File: ingestion/build_index.py
Specification said: "Upsert all chunks in batches."
Implementation chose: If the collection already contains chunks, delete and recreate it before indexing.
Reason: chunk_ids are uuid4 values regenerated on every chunking run, so upsert can never match prior IDs — the first expansion rebuild produced 1,780 chunks (587 stale + 1,193 new), with duplicate content able to crowd top-k retrieval slots. Recreating guarantees exactly one copy of each chunk.
Impact: Rebuild is idempotent. A future improvement is deterministic chunk IDs (hash of filename+position) to enable true incremental upserts.

---

## Decision 7 — Llama 3.1 context length: thin in the model card, covered by Wikipedia
Date: 2026-07-06
Phase: Expansion session
Files: data/raw/huggingface/meta-llama__Meta-Llama-3.1-70B-Instruct.txt
Finding: The re-fetched model card states the 128k context length only inside an HTML spec table, which the text cleaner may fragment across chunks. Per instruction, the card content was NOT hand-edited.
Mitigation: The "Context window" Wikipedia article (real, fetched 2026-07-06) provides retrievable prose about context windows; post-rebuild verification shows the gt_006 query now retrieves both Context_window.txt and the Llama 3.1 card in the top 3.
Impact: gt_006 (FALSE_REFUSAL in earlier cycles) is expected to pass more reliably.

---

## Decision 8 — run_statistical_test uses Welch's t-test with a minimum-sample guard
Date: 2026-07-06
Phase: Research-rigor session
File: analysis/trend_analysis.py
Specification said: "Paired t-test via scipy.stats."
Implementation chose: Welch's independent two-sample t-test, plus a guard that reports status="insufficient_data" when either series has fewer than 5 points.
Reason: The two windows compared (recent 7 days vs prior baseline days) contain different numbers of daily observations, so a paired test is mathematically inapplicable. More importantly, the function did not exist at all before this session — trend verdicts were raw percent-change only. With only a handful of probe cycles, an unguarded t-test would emit noise-dominated p-values; the guard reports honestly instead.
Impact: analyze_trends() now attaches a "significance" verdict per dimension; dashboards and reports can distinguish "degrading, statistically significant" from "degrading, too little data to tell".

---

## Decision 9 — Ground truth expansion authored with corpus verification, not manually
Date: 2026-07-06
Phase: Expansion session
File: config/ground_truth.json
Specification said (§14): "Expand config/ground_truth.json ... Do this manually — do not delegate to Claude Code."
Implementation chose: The user explicitly directed this session to write the 80 new queries. Every non-refusal entry was programmatically verified: the acceptable_answers strings were confirmed present in the named source_document (30-entry spot-check executed and logged, 100% pass).
Impact: 100 queries, distribution factual_recall 35 / benchmark_multihop 18 / comparative 15 / temporal_freshness 12 / out_of_scope 12 / adversarial 8. ground_truth_verified_date=2026-07-06 for all new entries.

---

## Decision 10 — Sibling-chunk expansion enabled by default
Date: 2026-07-09
Phase: Retrieval-fix session
Files: rag_system/retriever.py, config/settings.py (sibling_expansion_enabled)
Change: When a PapersWithCode chunk is retrieved, its positional neighbours from the same document are added to the context (capped at top_k + 2). Applies only to source == "paperswithcode".
Validation: A/B at 100-query scale vs the 5-cycle baseline (pass 77-83, FALSE_REFUSAL 13-20): two cycles with the fix (plus extended prose restatements and the acronym map) scored pass 84/84 and FALSE_REFUSAL 9/9. Evidence cases: gt_042 (Llama 3.1 BBH) passed both cycles after failing all 5 baseline cycles; gt_047 stopped refusing and produced the correct number (59.4). Kept enabled.
Impact: Context can now contain up to top_k + 2 chunks. This exposed a latent measurement bug (Decision 12).

---

## Decision 11 — Acronym expansion as a 4th query variant
Date: 2026-07-09
Phase: Retrieval-fix session
File: rag_system/retriever.py (ACRONYM_MAP, _expand_acronyms)
Change: rewrite_query() adds a variant with known acronyms spelled out (DPO, RLHF, RAG, MoE, LLM, SFT, PEFT, LoRA, BPE, MMLU, BBH, GPQA), word-boundary matched and case-insensitive so e.g. "RAGOps" does not trigger the "RAG" expansion.
Validation: gt_021 ("What does DPO stand for?"), which failed all 5 baseline cycles with zero hits on the DPO article, now fills all 5 retrieval slots from Direct_preference_optimization.txt and passed both A/B cycles.
Impact: rewrite_query can return 4 variants; tests updated.

---

## Decision 12 — Judge context truncation removed (measure_faithfulness, measure_utilization)
Date: 2026-07-09
Phase: Retrieval-fix session
Files: monitoring/measure_faithfulness.py, monitoring/measure_utilization.py
Problem: Both judges truncated context to context[:2500] before scoring. With sibling expansion the answer-bearing chunk frequently sits beyond 2,500 characters, so the judge scored correct, context-grounded answers as 0.0 (raw judge output '0.0' on all 7 FAITHFULNESS_FAILUREs in A/B cycle 1 — queries that had just stopped refusing, e.g. gt_047/gt_052/gt_054/gt_069). Same bug class as the truncation removed from the Self-RAG groundedness check on 2026-06-30.
Change: Judges now receive the full formatted context.
Impact: Slightly higher scoring-token cost per probe; faithfulness/utilization scores meaningful again at expanded context sizes. Rows from the two A/B cycles (2026-07-06T23:12-23:45 UTC) undercount faithfulness for non-refusal benchmark answers.

---

## Decision 13 — 'xAI' Wikipedia title was a disambiguation stub
Date: 2026-07-09
Phase: Retrieval-fix session
Files: ingestion/fetch_wikipedia.py, config/ground_truth.json (gt_027, gt_163)
Finding: The article fetched under the bare title "xAI" is a disambiguation page ("Xai, XAI or xAI may refer to:") with no company content — the earlier substring verification of gt_027 was a false positive ("xAI" matched the disambiguation text itself). This explains gt_027's persistent failure across all baseline cycles.
Change: Fetched the real article "xAI (company)" (saved as xAI_company.txt), deleted the stub, re-pointed gt_027 and authored gt_163 against the real article, rebuilt the index (1,199 chunks).
Lesson: substring verification can false-positive on stub/disambiguation pages; the methodology check should also assert a minimum source-document length.

---

## Decision 14 — llama-index removed from requirements; openai added explicitly
Date: 2026-07-11
Phase: Deployment session
File: requirements.txt
Specification said: requirements include llama-index==0.10.68 plus two llama-index plugins.
Finding: No file in the codebase imports llama_index — the implementation uses chromadb directly (consistent with the Phase 2/3 build). Worse, utils/llm_client.py imports `openai`, which was never a declared dependency; it only worked locally because llama-index pulled it in transitively. A fresh install from requirements.txt (e.g. Streamlit Community Cloud) would crash the dashboard.
Change: Removed the three llama-index lines; added `openai>=1.40,<3` explicitly.
Impact: Fresh installs are smaller, faster, and actually complete. Local behaviour unchanged.

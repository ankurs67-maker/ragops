# RAGOps — RAG System Monitoring Platform

A production-grade monitoring and evaluation platform for Retrieval-Augmented Generation systems. Implements automated probing, multi-dimensional scoring, failure classification, and advanced retrieval techniques.

---

## What This Does

RAGOps continuously probes a RAG pipeline against a ground-truth question set, scores each answer on five dimensions, classifies failures, proposes remediations, and visualises trends in a Streamlit dashboard.

**Advanced techniques included:**
- **Multi-provider LLM routing** — Groq, DeepSeek, OpenRouter with automatic 429 fallback
- **Contextual RAG** — context sentence prepended to each chunk before embedding
- **Context engineering** — best chunks placed at boundary positions (lost-in-the-middle mitigation)
- **HyDE-lite query rewriting** — 3 query variants without LLM calls
- **Self-RAG verification** — 3-step post-generation check with retry loop
- **Loop engineering** — query rewrite + re-retrieve on retrieval failure
- **Reflexion** — failure lessons persisted across probe cycles

---

## Requirements

- Python 3.11+
- API keys for at least one LLM provider (Groq recommended for speed)

---

## Setup

```bash
git clone <repo-url>
cd RAG
pip install -r requirements.txt

cp .env.example .env
# Edit .env — add your API keys
```

### Windows (PowerShell)
```powershell
$env:PYTHONPATH = "Z:\RAG"
python database/db_client.py   # initialise database
```

### Linux / macOS
```bash
export PYTHONPATH=.
python database/db_client.py
```

---

## Running the Full Pipeline

### Index documents (first time only)
```bash
python ingestion/fetch_wikipedia.py
python ingestion/fetch_huggingface.py
python ingestion/fetch_paperswithcode.py
python ingestion/chunk_documents.py
python ingestion/build_index.py
```

### Run a probe cycle manually
```powershell
$env:PYTHONPATH = "Z:\RAG"; python monitoring/run_probe_cycle.py
```

### Start the dashboard
```bash
streamlit run dashboard/app.py
```

### Start the scheduler (cron at 00:00 and 12:00 UTC)
```powershell
$env:PYTHONPATH = "Z:\RAG"; python scheduler/main_scheduler.py
```

---

## Make Targets

| Target | Description |
|--------|-------------|
| `make probe` | Run one probe cycle |
| `make report` | Generate daily report |
| `make dashboard` | Launch Streamlit dashboard |
| `make scheduler` | Start APScheduler |
| `make test` | Run pytest suite |
| `make reflexion-status` | Show last 10 Reflexion lessons |
| `make self-rag-report` | Self-RAG pass/fail counts from DB |
| `make loop-report` | Loop engineering stats from DB |
| `make failure-memory` | Print full failure memory file |

---

## Scoring Reference

| Dimension | Range | Alert threshold |
|-----------|-------|-----------------|
| Retrieval Relevance | 0–3 | < 1.5 |
| Context Utilization | 0–100 | < 60 |
| Faithfulness | 0–1 | < 0.75 |
| Factuality | 0–1 | < 0.60 |
| Refusal Calibration | 0–1 | < 0.70 |
| System Health | 0–100 | < 70 |

---

## Project Structure

```
RAG/
├── config/
│   ├── settings.py          # Pydantic Settings v2 configuration
│   └── ground_truth.json    # 20 probe queries with expected answers
├── database/
│   ├── schema.sql           # SQLite schema (6 tables, WAL mode)
│   └── db_client.py         # Parameterised query helpers
├── ingestion/
│   ├── fetch_wikipedia.py   # Fetch 69 Wikipedia articles
│   ├── fetch_huggingface.py # Fetch 54 HuggingFace model cards
│   ├── fetch_paperswithcode.py  # 18 benchmark/task files (static)
│   ├── chunk_documents.py   # Contextual RAG chunking (1,193 chunks)
│   └── build_index.py       # ChromaDB index with all-MiniLM-L6-v2
├── rag_system/
│   ├── prompt_templates.py  # All 8 prompts (FORMAT section last)
│   ├── retriever.py         # HyDE-lite + context engineering
│   ├── generator.py         # Self-RAG 3-step verification loop
│   └── pipeline.py          # Loop engineering + Reflexion integration
├── monitoring/
│   ├── probe_engine.py      # Single probe runner + Reflexion I/O
│   ├── run_probe_cycle.py   # Full cycle over 100 queries
│   ├── measure_retrieval.py # Retrieval relevance scoring (0–3)
│   ├── measure_utilization.py  # Context utilization (LLM judge, 0–100)
│   ├── measure_faithfulness.py  # Faithfulness (LLM judge + Self-RAG penalty)
│   ├── measure_factuality.py    # Factuality (fast-path + LLM judge)
│   ├── measure_refusal.py   # Refusal calibration (string matching)
│   └── classify_failure.py  # 9-category failure classifier
├── analysis/
│   ├── trend_analysis.py    # 7/14-day trend detection
│   ├── pattern_detector.py  # 9 pattern analyses (incl. technique effectiveness)
│   ├── remediation_proposer.py  # Rule-based remediation suggestions
│   └── reporter.py          # Daily ASCII report with bar charts
├── dashboard/
│   ├── app.py               # Streamlit entry point
│   ├── components/
│   │   ├── metrics.py       # KPI cards + Advanced Techniques Panel
│   │   └── charts.py        # Plotly charts
│   └── pages/
│       ├── 01_health_overview.py
│       ├── 02_failure_analysis.py
│       ├── 03_probe_explorer.py
│       ├── 04_remediations.py
│       └── 05_raw_data.py
├── scheduler/
│   └── main_scheduler.py    # APScheduler (probe: 0/12 UTC, report: 07:00)
├── experiments/
│   ├── 01_baseline_retrieval.py      # Single vs multi-query retrieval
│   ├── 02_contextual_rag_ablation.py # Context prefix effect on recall
│   ├── 03_self_rag_ablation.py       # Self-RAG on/off comparison
│   └── 04_reflexion_effectiveness.py # Reflexion across cycles
├── tests/
│   ├── conftest.py
│   ├── test_chunking.py
│   ├── test_retriever.py
│   ├── test_db.py
│   ├── test_pipeline.py
│   ├── test_monitoring.py
│   └── test_analysis.py
├── utils/
│   ├── llm_client.py        # call_llm() — universal LLM router
│   └── logger.py            # Rotating file + JSONL + console logging
├── data/
│   ├── raw/                 # Source documents (wikipedia/, huggingface/, paperswithcode/)
│   ├── processed/           # all_chunks.json, failure_memory.jsonl
│   └── chromadb/            # Vector index (cosine, 384-dim)
├── reports/                 # Daily report text files
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── NOTES.md                 # Advanced techniques explained
└── README.md                # This file
```

---

## Configuration

All settings are in `.env`. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `LLM_PROVIDER` | `groq` | Primary provider for generation |
| `SCORING_PROVIDER` | `deepseek` | Provider for Self-RAG scoring calls |
| `LLM_MODEL` | `llama-3.1-70b-versatile` | Generation model |
| `SCORING_MODEL` | `llama-3.1-8b-instant` | Scoring model (Groq) |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `PROBE_SCHEDULE_HOURS` | `[0, 12]` | UTC hours for probe cycles |

---

## Running Experiments

Each experiment script runs independently and appends a JSON result to `data/processed/experiment_log.txt`.

```powershell
$env:PYTHONPATH = "Z:\RAG"
python experiments/01_baseline_retrieval.py
python experiments/02_contextual_rag_ablation.py
python experiments/03_self_rag_ablation.py
python experiments/04_reflexion_effectiveness.py
```

---

## Docker

```bash
docker compose up --build

# Run probe cycle inside container
docker compose exec ragops python monitoring/run_probe_cycle.py
```

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --tb=short
```

The test suite covers chunking, retrieval, database, pipeline, monitoring, and analysis modules. Tests use `monkeypatch` for DB and file isolation — no API calls are made.

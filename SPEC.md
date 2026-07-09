# SPEC.md — RAGOps Software Requirements Specification
### Version 1.0 | Authoritative source of truth for all implementation decisions

---

## Table of Contents

1. System Overview
2. Technology Stack
3. Complete Folder Structure
4. Phase 1 — Architecture
5. Phase 2 — Data Ingestion
6. Phase 3 — RAG System
7. Phase 4 — Monitoring
8. Phase 5 — Analytics
9. Phase 6 — Dashboard
10. Phase 7 — Experiments, Scheduler, DevOps, Documentation
11. Database Schema
12. Coding Standards
13. Terminology Reference
14. User Steps After Build

---

## 1. System Overview

RAGOps is a production-grade autonomous RAG quality monitoring system.

### Part A — Target RAG System

A working RAG system built over AI and LLM knowledge.
Answers questions such as "What is Llama 3.1's context window?"
using retrieved document chunks, not model memory.
This is the system being monitored.

Why AI/LLM knowledge as the corpus:
Standard corpora like company names or historical events suffer from
parametric memory contamination. The LLM already knows these facts
from training, so it can answer correctly even when retrieval fails.
You cannot distinguish retrieval success from memory recall.
AI/LLM specific facts solve this problem. The LLM does not reliably
know exact benchmark scores, parameter counts of specific model
variants, or context window sizes for recently released models.
The only reliable way to answer correctly is to retrieve from the
corpus. This makes every measurement signal clean and meaningful.

### Part B — Autonomous Monitoring System

Six automated components running in the background:

- Agent 1 — Probe Engine: sends test questions to Part A every 6 hours
- Agent 2 — Measurement Pipeline: scores quality on 5 dimensions
- Agent 3 — Pattern Detector: finds degradation trends nightly
- Agent 4 — Remediation Proposer: diagnoses problems, proposes fixes
- Agent 5 — Reporter: generates morning briefing daily at 07:00
- Agent 6 — Dashboard: Streamlit UI checked each morning

The system runs every 6 hours autonomously.
The user only interacts with the dashboard and morning report.

### Design Goals

- Total infrastructure cost: $0
- Runnable from a single setup command after cloning
- All components loosely coupled and independently testable
- All configuration in one place, never scattered across files
- Production logging from day one, not added as an afterthought
- Every failure mode documented and handled, not silently swallowed

---

## 2. Technology Stack

| Layer | Tool | Version | Notes |
|-------|------|---------|-------|
| Language | Python | 3.11+ | |
| RAG framework | LlamaIndex | 0.10.68 | |
| Vector store | ChromaDB | 0.5.3 | Local persistent, no server |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | 3.0.1 | Local, no API, 384 dimensions |
| LLM answers | Llama 3.1 70B via Groq | groq 0.9.0 | Model: llama-3.1-70b-versatile |
| LLM scoring | Llama 3.1 8B via Groq | groq 0.9.0 | Model: llama-3.1-8b-instant |
| Factuality check | Pure Python string matching | — | Zero LLM calls for 80% of cases |
| Config validation | Pydantic Settings v2 | 2.4.0 | Validation on startup |
| Database | SQLite | built-in | WAL mode enabled |
| Dashboard | Streamlit | 1.37.0 | With Plotly charts |
| Scheduler | APScheduler | 3.10.4 | BlockingScheduler, UTC timezone |
| Testing | pytest | 8.3.2 | With mock LLM responses |
| Logging | Python logging | built-in | Rotating files + structured JSONL |
| Containerisation | Docker + docker-compose | latest | |
| CI/CD | GitHub Actions | — | black, ruff, mypy, pytest |
| Linting | black 24.8.0, ruff 0.6.1, mypy 1.11.1 | — | |

### Model Allocation

| Model | Tasks |
|-------|-------|
| llama-3.1-70b-versatile | Generating RAG answers, pattern analysis, remediation proposals, daily reports |
| llama-3.1-8b-instant | Measure 1 retrieval scoring, Measure 2 utilisation scoring, Measure 3 faithfulness scoring, Measure 5 refusal scoring |
| No LLM | Measure 4 factuality — pure Python string matching only |

Why two models: The 70B model handles complex reasoning. The 8B model
handles bulk repetitive scoring. This keeps usage within Groq free
tier limits across 760 probes per day (190 queries × 4 cycles).

---

## 3. Complete Folder Structure

```
Z:\RAG\
├── SPEC.md                         Source of truth — do not modify during build
├── DECISIONS.md                    Implementation decisions that differ from spec
├── PROGRESS.md                     Phase completion log
├── NOTES.md                        User-facing project notes
├── README.md                       Public documentation
├── Makefile                        All common commands
├── Dockerfile                      Container definition
├── docker-compose.yml              Multi-service orchestration
├── pyproject.toml                  Tool configuration (black, ruff, mypy, pytest)
├── requirements.txt                Pinned production dependencies
├── requirements-dev.txt            Pinned development dependencies
├── .env                            API keys — gitignored
├── .env.example                    Template — committed to git
├── .gitignore
├── .github\
│   └── workflows\
│       └── ci.yml
├── config\
│   ├── __init__.py
│   ├── settings.py                 Pydantic Settings v2 — single source of config
│   └── ground_truth.json           20 seed test queries
├── data\
│   ├── raw\
│   │   ├── huggingface\            Downloaded model card text files
│   │   ├── paperswithcode\         Downloaded benchmark data
│   │   └── wikipedia\              Downloaded article text files
│   └── processed\
│       └── all_chunks.json         Chunked documents ready for indexing
├── ingestion\
│   ├── __init__.py
│   ├── fetch_huggingface.py
│   ├── fetch_wikipedia.py
│   ├── fetch_paperswithcode.py
│   ├── clean_text.py
│   ├── chunk_documents.py
│   └── build_index.py
├── rag_system\
│   ├── __init__.py
│   ├── retriever.py
│   ├── generator.py
│   ├── pipeline.py
│   └── prompt_templates.py
├── monitoring\
│   ├── __init__.py
│   ├── probe_engine.py
│   ├── measure_retrieval.py
│   ├── measure_utilization.py
│   ├── measure_faithfulness.py
│   ├── measure_factuality.py
│   ├── measure_refusal.py
│   ├── classify_failure.py
│   └── run_probe_cycle.py
├── analysis\
│   ├── __init__.py
│   ├── pattern_detector.py
│   ├── trend_analysis.py
│   ├── remediation_proposer.py
│   └── reporter.py
├── database\
│   ├── __init__.py
│   ├── schema.sql
│   ├── db_client.py
│   └── ragops.db                   Created at runtime — gitignored
├── dashboard\
│   ├── app.py
│   ├── pages\
│   │   ├── 01_health_overview.py
│   │   ├── 02_failure_analysis.py
│   │   ├── 03_probe_explorer.py
│   │   ├── 04_remediations.py
│   │   └── 05_raw_data.py
│   └── components\
│       ├── __init__.py
│       ├── charts.py
│       └── metrics.py
├── scheduler\
│   ├── __init__.py
│   └── main_scheduler.py
├── utils\
│   ├── __init__.py
│   └── logger.py
├── experiments\
│   ├── __init__.py
│   ├── inject_chunk_size_change.py
│   ├── inject_stale_index.py
│   ├── inject_corpus_poison.py
│   ├── verify_detection.py
│   └── experiment_log.txt          Created at runtime
├── tests\
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_ingestion.py
│   ├── test_rag_pipeline.py
│   ├── test_measurements.py
│   ├── test_pattern_detector.py
│   ├── test_db_client.py
│   └── test_scheduler.py
├── logs\                            Created at runtime — gitignored
└── reports\                         Created at runtime — gitignored
```

---

## 4. Phase 1 — Architecture

Build only these files in Phase 1. No implementation logic yet.

### requirements.txt

```
llama-index==0.10.68
llama-index-vector-stores-chroma==0.1.10
llama-index-embeddings-huggingface==0.2.3
chromadb==0.5.3
sentence-transformers==3.0.1
groq==0.9.0
wikipedia-api==0.7.1
huggingface-hub==0.24.5
requests==2.32.3
beautifulsoup4==4.12.3
APScheduler==3.10.4
streamlit==1.37.0
streamlit-autorefresh==1.0.1
plotly==5.23.0
pandas==2.2.2
numpy==1.26.4
python-dotenv==1.0.1
tiktoken==0.7.0
tqdm==4.66.5
scipy==1.13.1
pydantic==2.8.0
pydantic-settings==2.4.0
```

### requirements-dev.txt

```
pytest==8.3.2
pytest-asyncio==0.23.8
pytest-cov==5.0.0
pytest-mock==3.14.0
black==24.8.0
ruff==0.6.1
mypy==1.11.1
types-requests==2.32.0
```

### pyproject.toml

```toml
[tool.black]
line-length = 88
target-version = ["py311"]

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--cov=. --cov-report=term-missing --cov-omit=tests/*,data/*,logs/*,reports/*"
```

### .env.example

```
GROQ_API_KEY=your_groq_api_key_here
RAGOPS_ENV=development
LOG_LEVEL=INFO
```

### .gitignore

```
.env
database/ragops.db
data/raw/
data/processed/
data/chromadb/
logs/
reports/
experiments/experiment_log.txt
__pycache__/
*.pyc
.DS_Store
.coverage
htmlcov/
.mypy_cache/
.ruff_cache/
dist/
*.egg-info/
```

### config/settings.py

Use Pydantic Settings v2. Load from environment and .env file.
Validate on startup — raise a clear descriptive error if
GROQ_API_KEY is missing or empty.
All paths use pathlib.Path derived from BASE_DIR.
Never use os.getenv anywhere else in the codebase.

Required fields and defaults:

```python
# API and environment
groq_api_key: str                           # from env GROQ_API_KEY — required
ragops_env: str = "development"
log_level: str = "INFO"

# LLM models
llm_model: str = "llama-3.1-70b-versatile"
scoring_model: str = "llama-3.1-8b-instant"
judge_model_version: str = "llama-3.1-8b-instant"
generation_temperature: float = 0.1
judge_temperature: float = 0.0
max_answer_tokens: int = 500
max_judge_tokens: int = 200
llm_timeout_seconds: int = 30

# Embeddings and vector store
embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
embedding_dimensions: int = 384
chroma_collection: str = "llm_intelligence_corpus"

# Chunking
chunk_size: int = 512
chunk_overlap: int = 50
min_chunk_size: int = 100
top_k: int = 5

# Alert thresholds
alert_retrieval_threshold: float = 1.5
alert_utilization_threshold: float = 60.0
alert_faithfulness_threshold: float = 0.75
alert_factuality_threshold: float = 0.60
alert_refusal_threshold: float = 0.70
alert_latency_multiplier: float = 3.0

# Trend detection
trend_window_days: int = 7
baseline_window_days: int = 30
trend_alert_percent: float = 15.0

# Scheduling
probe_schedule_hours: list[int] = [0, 6, 12, 18]
pattern_schedule_hour: int = 23
report_schedule_hour: int = 7
```

Expose a singleton: `settings = Settings()`
All other files import: `from config.settings import settings`

### database/schema.sql

Create all tables. Add indexes on all columns used in WHERE
and ORDER BY. Use TEXT for JSON columns, DATETIME for timestamps.

```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS probe_results (
    probe_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    query_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    category TEXT NOT NULL,
    difficulty TEXT,
    retrieved_chunks TEXT,
    generated_answer TEXT,
    correct_answer TEXT,
    answer_correct TEXT,
    refused_when_should INTEGER,
    latency_retrieval_ms INTEGER,
    latency_generation_ms INTEGER,
    latency_total_ms INTEGER
);

CREATE TABLE IF NOT EXISTS measurements (
    measurement_id TEXT PRIMARY KEY,
    probe_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    retrieval_relevance_score REAL,
    context_utilization_score REAL,
    faithfulness_score REAL,
    factuality_score REAL,
    refusal_calibration_score REAL,
    judge_model_version TEXT,
    judge_confidence REAL,
    failure_category TEXT,
    measurement_details TEXT,
    FOREIGN KEY (probe_id) REFERENCES probe_results(probe_id)
);

CREATE TABLE IF NOT EXISTS pattern_reports (
    report_id TEXT PRIMARY KEY,
    date DATE NOT NULL,
    timestamp DATETIME NOT NULL,
    overall_health_score REAL,
    alerts_triggered TEXT,
    dimension_scores TEXT,
    failure_distribution TEXT,
    category_breakdown TEXT,
    source_breakdown TEXT,
    top_finding TEXT,
    raw_analysis TEXT
);

CREATE TABLE IF NOT EXISTS remediations (
    remediation_id TEXT PRIMARY KEY,
    triggered_by TEXT,
    timestamp DATETIME NOT NULL,
    alert_type TEXT,
    root_cause TEXT,
    confidence REAL,
    remediation_text TEXT,
    specific_steps TEXT,
    priority TEXT,
    status TEXT DEFAULT 'pending',
    outcome TEXT
);

CREATE TABLE IF NOT EXISTS daily_reports (
    report_id TEXT PRIMARY KEY,
    date DATE NOT NULL,
    report_text TEXT,
    report_json TEXT,
    system_health_score REAL
);

CREATE INDEX IF NOT EXISTS idx_probe_run_id ON probe_results(run_id);
CREATE INDEX IF NOT EXISTS idx_probe_timestamp ON probe_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_probe_category ON probe_results(category);
CREATE INDEX IF NOT EXISTS idx_probe_answer_correct ON probe_results(answer_correct);
CREATE INDEX IF NOT EXISTS idx_measure_probe_id ON measurements(probe_id);
CREATE INDEX IF NOT EXISTS idx_measure_timestamp ON measurements(timestamp);
CREATE INDEX IF NOT EXISTS idx_measure_failure ON measurements(failure_category);
CREATE INDEX IF NOT EXISTS idx_pattern_date ON pattern_reports(date);
CREATE INDEX IF NOT EXISTS idx_remediation_status ON remediations(status);
CREATE INDEX IF NOT EXISTS idx_remediation_priority ON remediations(priority);
CREATE INDEX IF NOT EXISTS idx_report_date ON daily_reports(date);
```

### database/db_client.py

Complete SQLite client. All queries parameterised. WAL mode.

Required functions:
- `get_connection()` — WAL mode, row_factory, foreign keys on
- `init_database()` — run schema.sql, insert schema_version row
- `insert_probe_result(probe: dict)`
- `insert_measurement(m: dict)`
- `insert_pattern_report(report: dict)`
- `insert_remediation(r: dict)`
- `insert_daily_report(report: dict)`
- `get_recent_probes(hours: int) -> list[dict]`
- `get_dimension_averages(days: int) -> dict`
- `get_failure_distribution(days: int) -> dict`
- `get_pending_remediations() -> list[dict]`
- `get_system_health_score() -> float`
- `update_remediation_status(remediation_id: str, status: str, outcome: str)`
- `get_probe_by_id(probe_id: str) -> dict`

Health score formula:
```
retrieval_norm  = avg_retrieval / 3.0
utilization_norm = avg_utilization / 100.0
faithfulness_norm = avg_faithfulness
factuality_norm = avg_factuality
refusal_norm = avg_refusal
health = ((retrieval_norm + utilization_norm + faithfulness_norm
           + factuality_norm + refusal_norm) / 5.0) * 100
```

When run directly: initialise database and print confirmation
with database path and table count.

### utils/logger.py

Centralised logging used by every module.

Implement three handlers:
1. Rotating file: `logs/ragops.log`, max 10MB, keep 5 backups
2. Structured JSONL: `logs/ragops_structured.jsonl`, one JSON per line
3. Console: colour output in development, plain in production

JSONL format:
```json
{
  "timestamp": "2026-06-01T07:00:00.000Z",
  "level": "INFO",
  "logger": "monitoring.probe_engine",
  "message": "Probe cycle started",
  "extra": {}
}
```

Expose: `get_logger(name: str) -> logging.Logger`
All modules: `logger = get_logger(__name__)`

### Phase 1 Verification

```bash
python -c "from config.settings import settings; print(settings.model_dump())"
python database/db_client.py
python -c "from utils.logger import get_logger; l = get_logger('test'); l.info('Logger OK')"
```

All three must exit with code 0 before proceeding to Phase 2.

---

## 5. Phase 2 — Data Ingestion

### ingestion/fetch_wikipedia.py

Library: wikipedia-api
User agent: `RAGOps-Monitor/1.0`
Output directory: `data/raw/wikipedia/`
File header before content:
```
TITLE: {title}
URL: {url}
FETCHED: {timestamp}
---
```

Articles to fetch (30 total):
```
Transformer (machine learning model)
Attention mechanism
BERT (language model)
GPT (language model)
Reinforcement learning from human feedback
Constitutional AI
Mixture of experts
Retrieval-augmented generation
Fine-tuning (machine learning)
Prompt engineering
Chain-of-thought prompting
Hallucination (artificial intelligence)
Large language model
Generative pre-trained transformer
Neural scaling law
OpenAI
Anthropic
Google DeepMind
Meta AI
Hugging Face
EleutherAI
Mistral AI
Yoshua Bengio
Geoffrey Hinton
Yann LeCun
Demis Hassabis
Ilya Sutskever
BLEU
Word embedding
Tokenization (machine learning)
```

Behaviour:
- Skip if file already exists
- Sleep 0.5s between requests
- Handle missing pages gracefully
- Return (success_count, failed_list)

### ingestion/fetch_huggingface.py

Library: huggingface_hub ModelCard.load()
Output directory: `data/raw/huggingface/`
Filename format: replace `/` with `__` — example: `meta-llama__Llama-2-7b-hf.txt`

Models to fetch (30 total):
```
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-13b-hf
meta-llama/Llama-2-70b-hf
meta-llama/Meta-Llama-3-8B
meta-llama/Meta-Llama-3-70B
meta-llama/Meta-Llama-3.1-8B-Instruct
meta-llama/Meta-Llama-3.1-70B-Instruct
mistralai/Mistral-7B-v0.1
mistralai/Mistral-7B-Instruct-v0.2
mistralai/Mixtral-8x7B-v0.1
mistralai/Mixtral-8x22B-v0.1
microsoft/phi-2
microsoft/Phi-3-mini-4k-instruct
microsoft/Phi-3-medium-4k-instruct
google/gemma-2b
google/gemma-7b
google/gemma-2-9b
google/gemma-2-27b
tiiuae/falcon-7b
tiiuae/falcon-40b
Qwen/Qwen2-7B-Instruct
Qwen/Qwen2-72B-Instruct
deepseek-ai/deepseek-llm-7b-base
deepseek-ai/deepseek-llm-67b-base
databricks/dbrx-base
CohereForAI/c4ai-command-r-plus
01-ai/Yi-1.5-34B
allenai/OLMo-7B
EleutherAI/gpt-neox-20b
bigscience/bloom
```

Behaviour:
- Skip if file already exists
- Handle EntryNotFoundError and RepositoryNotFoundError gracefully
- Sleep 0.3s between requests
- Return (success_count, failed_list)

### ingestion/fetch_paperswithcode.py

API base: `https://paperswithcode.com/api/v1/`
No API key required.
Output directory: `data/raw/paperswithcode/`

Tasks to fetch:
```
language-modelling
question-answering
common-sense-reasoning
math-word-problem-solving
code-generation
text-summarization
information-retrieval
```

Named benchmarks for SOTA results:
```
mmlu hellaswag arc truthfulqa gsm8k humaneval mbpp
```

Format output as structured plain text showing benchmark name,
description, and top 5 models with scores.
Handle 404 responses gracefully.
Sleep 0.5s between requests.

### ingestion/clean_text.py

```python
def clean_text(text: str) -> str:
    # Remove HTML tags
    # Normalise whitespace — max 2 consecutive newlines
    # Remove markdown table pipe formatting
    # Replace URLs with [URL]
    # Normalise unicode quotes and dashes
    # Remove control characters
    # Strip leading and trailing whitespace

def extract_metadata(filepath: Path) -> dict:
    # Infer source from directory name: huggingface | paperswithcode | wikipedia
    # Infer doc_type: model_card | benchmark | article
    # Return dict with source and doc_type
```

### ingestion/chunk_documents.py

Tokeniser: tiktoken cl100k_base
Strategy: sentence-aware with token-based splitting

```python
def chunk_text(text: str, doc_id: str, metadata: dict) -> list[dict]:
    # Split into chunks of settings.chunk_size tokens
    # With settings.chunk_overlap token overlap
    # Discard chunks below settings.min_chunk_size tokens
    # Each chunk dict must contain:
    #   chunk_id: uuid4 string
    #   document_id: str
    #   content: str
    #   token_count: int
    #   position: int (index within document)
    #   date_indexed: ISO UTC timestamp string
    #   source: str
    #   doc_type: str
    #   filename: str

def process_all_documents() -> list[dict]:
    # Read all .txt files from all three raw subdirectories
    # Clean each with clean_text()
    # Chunk each with chunk_text()
    # Log count per directory and total
    # Return complete list of all chunk dicts
```

When run directly: save to `data/processed/all_chunks.json`.

### ingestion/build_index.py

```python
def get_chroma_client() -> chromadb.PersistentClient:
    # PersistentClient at settings paths

def get_collection(client) -> chromadb.Collection:
    # get_or_create_collection with SentenceTransformer embedding function
    # cosine similarity metric

def build_index(batch_size: int = 50) -> int:
    # Load all_chunks.json
    # Log existing chunk count
    # Upsert all chunks in batches with tqdm progress bar
    # Return final chunk count

def verify_index() -> None:
    # Run 3 test queries
    # Print top result for each
    # Confirm retrieval is working
```

When run directly: chunk → index → verify. Print summary.

### Phase 2 Verification

```bash
python ingestion/fetch_wikipedia.py
python ingestion/fetch_huggingface.py
python ingestion/fetch_paperswithcode.py
python ingestion/build_index.py
```

Verify:
- `data/raw/` directories contain .txt files
- `data/processed/all_chunks.json` exists with 1000+ chunks
- `data/chromadb/` exists with ChromaDB files
- `verify_index()` prints sensible results for 3 queries

---

## 6. Phase 3 — RAG System

### rag_system/prompt_templates.py

Single source of truth for all prompts. Module-level string constants.
Every prompt has a comment explaining what it does and which model uses it.

Required prompts:

**RAG_SYSTEM_PROMPT** — instructs 70B to answer only from context,
refuse with exact phrase "I cannot find this information in my
knowledge base." when context is insufficient, never guess.

**RAG_QUERY_TEMPLATE** — slots: `{context}` `{question}`

**RETRIEVAL_RELEVANCE_PROMPT** — slots: `{query}` `{chunk}`
Scoring 0-3 with clear definitions.
Returns exactly two lines:
```
SCORE: [integer]
CONFIDENCE: [float]
```

**CONTEXT_UTILIZATION_PROMPT** — slots: `{answer}` `{context}`
Returns exactly two lines:
```
PERCENTAGE: [integer 0-100]
CONFIDENCE: [float 0.0-1.0]
```

**FAITHFULNESS_PROMPT** — slots: `{question}` `{answer}` `{context}`
Returns JSON only with exactly these keys:
`supported_claims`, `unsupported_claims`, `faithfulness_score`,
`confidence`, `reasoning`

**FACTUALITY_JUDGE_PROMPT** — slots: `{question}` `{correct_answer}` `{system_answer}`
Returns exactly two lines:
```
VERDICT: correct|partial|incorrect
CONFIDENCE: [float 0.0-1.0]
```

**REMEDIATION_PROMPT** — slots: `{alert_type}` `{current_value}`
`{baseline_value}` `{drop_pct}` `{affected_categories}`
`{failure_distribution}` `{source_breakdown}` `{chunk_size}`
`{top_k}` `{days_since_reindex}`
Returns JSON with exactly these keys:
`root_cause`, `confidence`, `remediation`, `specific_steps`,
`estimated_impact`, `priority`, `test_to_verify`

### rag_system/retriever.py

```python
def retrieve(query: str, top_k: int = settings.top_k) -> list[dict]:
    # Query ChromaDB
    # Include documents, metadatas, distances
    # Convert cosine distance to similarity: similarity = 1 - distance
    # Return list of dicts each containing:
    #   content, metadata, similarity_score, chunk_id, source, document_id

def format_context(chunks: list[dict]) -> str:
    # Format chunks as numbered sections
    # Each header: [Source N | source | document_id]
    # Join with separator line
    # Return "No relevant context found." if chunks is empty
```

### rag_system/generator.py

Singleton Groq client initialised once.

```python
def generate(question: str, context: str) -> dict:
    # Build prompt from RAG_SYSTEM_PROMPT and RAG_QUERY_TEMPLATE
    # Call settings.llm_model with settings.generation_temperature
    # Apply settings.llm_timeout_seconds timeout
    # Return: {answer: str, usage: dict, error: str | None}
    # Handle all Groq exceptions, return error in dict
    # Log token usage
```

### rag_system/pipeline.py

```python
def query(question: str) -> dict:
    # Time retrieval and generation separately with time.perf_counter
    # Call retrieve() then format_context() then generate()
    # Return dict with:
    #   question, answer, retrieved_chunks, context_used,
    #   latency_retrieval_ms, latency_generation_ms,
    #   latency_total_ms, error
```

When run directly, test with 5 questions:
```
"What company developed the Llama 2 model?"
"What does RLHF stand for?"
"Which organization created BLOOM?"
"What will be the best AI model in 2030?"
"Is GPT-4 open source?"
```

Print: question, answer, chunk count, top similarity score, total latency.

### Phase 3 Verification

```bash
python rag_system/pipeline.py
```

Verify:
- Questions 1-3 get factual answers using retrieved context
- Question 4 triggers exact refusal phrase
- Question 5 correctly answers GPT-4 is not open source
- Latency prints for each question

---

## 7. Phase 4 — Monitoring

### Terminology — Use These Exact Terms Everywhere

| Term | Definition |
|------|------------|
| Faithfulness | Is the answer supported by the retrieved chunks? |
| Factuality | Is the answer correct compared to ground truth? |
| Groundedness | Are the retrieved chunks relevant to the true answer? |
| Hallucination | Faithfulness failure AND factuality failure together — not a metric name |

Never create a metric called `hallucination_score`.

### Failure Taxonomy

| Label | Meaning | Primary Signal |
|-------|---------|----------------|
| RETRIEVAL_FAILURE | Wrong chunks retrieved | Measure 1 below threshold |
| CONTEXT_BYPASS | Right chunks, model ignored them | Measure 2 below threshold |
| FAITHFULNESS_FAILURE | Unsupported claims in answer | Measure 3 below threshold |
| FACTUAL_ERROR | Answer wrong vs ground truth | Measure 4 below threshold |
| REFUSAL_FAILURE | Should refuse but answered | Measure 5 — should_refuse=True, did not refuse |
| FALSE_REFUSAL | Should answer but refused | Measure 5 — should_refuse=False, refused |
| LATENCY_DEGRADATION | Correct but >3x baseline time | Latency metric |
| PARTIAL_ANSWER | Incomplete but not wrong | Measure 4 returns 0.5 |
| PASS | No failure detected | All measures within thresholds |

### monitoring/probe_engine.py

```python
def load_ground_truth() -> list[dict]:
    # Read config/ground_truth.json

def run_single_probe(gt_item: dict, run_id: str) -> dict:
    # Call pipeline.query()
    # Detect refusal using these phrases (case insensitive):
    #   "cannot find", "don't have", "no information",
    #   "not available", "unable to", "i cannot"
    # Set refused_when_should based on should_refuse field
    # Build probe dict with all required fields
    # Call insert_probe_result()
    # Return probe dict

def run_probe_cycle() -> dict:
    # Generate run_id with uuid4
    # Call run_single_probe for every ground truth item
    # Log progress: "Probe N/total: query_id"
    # Return: {run_id, total, successful, failed}
```

### monitoring/measure_retrieval.py

```python
def score_retrieval_relevance(query: str, chunks: list[dict]) -> dict:
    # For each chunk call 8B model with RETRIEVAL_RELEVANCE_PROMPT
    # Temperature 0, timeout 30s
    # Parse SCORE line and CONFIDENCE line
    # Default to score=0, confidence=0.5 on parse failure
    # Final score = average of chunk scores
    # Log judge model version with every call
    # Return: {score: float 0-3, confidence: float, per_chunk_scores: list}
```

### monitoring/measure_utilization.py

```python
def score_context_utilization(answer: str, chunks: list[dict]) -> dict:
    # Format chunks into context string
    # Call 8B with CONTEXT_UTILIZATION_PROMPT temperature 0
    # Parse PERCENTAGE and CONFIDENCE lines
    # Default to 50, 0.5 on parse failure
    # Return: {score: float 0-100, confidence: float}
```

### monitoring/measure_faithfulness.py

```python
def score_faithfulness(question: str, answer: str,
                       chunks: list[dict]) -> dict:
    # Format chunks into context
    # Call 8B with FAITHFULNESS_PROMPT temperature 0
    # Parse JSON response
    # Default to score=0.5 on any parse failure
    # Return all parsed fields: faithfulness_score, supported_claims,
    #   unsupported_claims, confidence, reasoning
```

### monitoring/measure_factuality.py

This is the most reliable measure. No LLM for most queries.

```python
def score_factuality(question: str, answer: str,
                     correct_answer: str,
                     acceptable_answers: list[str],
                     query_id: str,
                     refused_when_should: int | None) -> dict:

    # Method selection:
    # If should_refuse (refused_when_should is not None):
    #   If model refused: score=1.0, method=exact_match
    #   If model did not refuse: score=0.0, method=exact_match
    # Elif answer under 100 chars OR correct_answer under 50 chars:
    #   Method A — exact string matching, NO LLM
    #   Check if answer.lower() contains any acceptable_answer.lower()
    #   Score: 1.0 if match, 0.0 if not
    #   method="exact_match"
    # Else:
    #   Method B — call 8B with FACTUALITY_JUDGE_PROMPT temperature 0
    #   Parse VERDICT and CONFIDENCE
    #   correct=1.0, partial=0.5, incorrect=0.0
    #   method="llm_judge"
    #
    # Always log which method was used
    # Return: {score: float, method: str, confidence: float, verdict: str}
```

### monitoring/measure_refusal.py

Refusal detection phrases (case insensitive):
`"cannot find"`, `"don't have"`, `"no information"`,
`"not available"`, `"unable to"`, `"i cannot"`

```python
def score_refusal_calibration(answer: str, should_refuse: bool) -> dict:
    # Detect if answer contains any refusal phrase
    # If should_refuse and refused: score=1.0, label=PASS
    # If should_refuse and not refused: score=0.0, label=REFUSAL_FAILURE
    # If not should_refuse and refused: score=0.0, label=FALSE_REFUSAL
    # If not should_refuse and not refused: score=1.0, label=PASS
    # Return: {score: float, label: str}
```

### monitoring/classify_failure.py

```python
def classify_failure(scores: dict, probe: dict) -> str:
    # Apply in this exact priority order:
    # 1. If refusal score 0 and should_refuse: REFUSAL_FAILURE
    # 2. If refusal score 0 and not should_refuse: FALSE_REFUSAL
    # 3. If retrieval score < settings.alert_retrieval_threshold: RETRIEVAL_FAILURE
    # 4. If utilization score < settings.alert_utilization_threshold: CONTEXT_BYPASS
    # 5. If faithfulness score < settings.alert_faithfulness_threshold: FAITHFULNESS_FAILURE
    # 6. If factuality score < settings.alert_factuality_threshold: FACTUAL_ERROR
    # 7. If latency > baseline * settings.alert_latency_multiplier: LATENCY_DEGRADATION
    # 8. If factuality score == 0.5: PARTIAL_ANSWER
    # 9. Otherwise: PASS
```

### monitoring/run_probe_cycle.py

```python
def run_full_cycle() -> dict:
    # Step 1: run_probe_cycle() — get run_id
    # Step 2: fetch all probes from this run_id
    # Step 3: for each probe call measure_probe() then insert_measurement()
    # Step 4: check all thresholds — if any breached call
    #         remediation_proposer.run_remediation_check()
    # Step 5: log health score
    # Return: {run_id, health_score, probes summary, measured count}

def measure_probe(probe: dict, run_id: str) -> dict:
    # Parse retrieved_chunks JSON string to list
    # Call all 5 measurement functions
    # Call classify_failure()
    # Average judge confidence across non-None measures
    # Build and return complete measurement dict
```

When run directly: call run_full_cycle() and print summary.

### Phase 4 Verification

```bash
python monitoring/run_probe_cycle.py
```

Verify:
- 20 rows in probe_results table
- 20 rows in measurements table
- All rows have failure_category populated
- Health score printed at end

---

## 8. Phase 5 — Analytics

### analysis/trend_analysis.py

```python
def calculate_moving_average(values: list[float], window: int) -> list[float]:
    # Standard moving average

def detect_trend(current_avg: float, baseline_avg: float,
                 threshold_pct: float) -> dict:
    # Returns: {direction: "up"|"down"|"stable", pct_change: float, is_alert: bool}

def run_statistical_test(series_a: list[float],
                         series_b: list[float]) -> dict:
    # Paired t-test via scipy.stats
    # Returns: {t_statistic: float, p_value: float, is_significant: bool}

def get_dimension_trends(days: int = 7) -> dict:
    # For each of 5 dimensions:
    #   Get daily averages for last 30 days
    #   Calculate 7-day moving average
    #   Compare to 30-day baseline
    #   Run t-test
    # Return nested dict with results per dimension
```

### analysis/pattern_detector.py

```python
def run_pattern_detection() -> dict:
    # Analysis 1 — Dimension trends
    #   Dimension averages today vs 7-day baseline
    #   Flag any dimension below alert threshold

    # Analysis 2 — Failure distribution shift
    #   Failure category counts today vs last 7 days
    #   Flag categories spiked more than 20%

    # Analysis 3 — Query category breakdown
    #   Failure rate per query category today
    #   Identify worst performing category

    # Analysis 4 — Source attribution
    #   Failure rate per data source
    #   Flag if one source causes disproportionate failures

    # Analysis 5 — Temporal pattern
    #   Group failures by hour of day
    #   Identify time-based clustering

    # Analysis 6 — Retrieval gap
    #   For multi_hop queries: was expected chunk retrieved?
    #   Flag queries where expected chunk never in top 5

    # After SQL: call 70B with aggregated data
    # Ask for one paragraph identifying most important finding
    # Store as top_finding

    # Call insert_pattern_report()
    # Return report dict
```

### analysis/remediation_proposer.py

```python
def check_thresholds() -> list[dict]:
    # Get dimension averages for last 24 hours
    # Compare each against settings alert threshold
    # Return list of breached threshold dicts

def propose_remediation(alert_type: str, diagnostic_data: dict) -> dict:
    # Build diagnostic payload
    # Call 70B with REMEDIATION_PROMPT
    # Parse JSON response
    # Call insert_remediation() with status=pending
    # Return remediation dict

def run_remediation_check() -> None:
    # Call check_thresholds()
    # For each breach call propose_remediation()
    # Log number of remediations proposed
```

### analysis/reporter.py

```python
def generate_daily_report() -> str:
    # Collect from database:
    #   health score, probe counts, pass rate
    #   all 5 dimension scores today and 7d average
    #   trend direction per dimension
    #   failure distribution with percentages
    #   top failing category
    #   top_finding from last pattern report
    #   pending remediations
    #   cumulative totals

    # Format as structured text with these exact sections:
    #   Header with date and time
    #   SYSTEM HEALTH SCORE with status badge
    #   LAST 24 HOURS summary stats
    #   FIVE DIMENSION SCORECARD table
    #   FAILURE BREAKDOWN TODAY with percentages
    #   TOP FAILING QUERY CATEGORY
    #   MOST INTERESTING FINDING
    #   ACTIVE REMEDIATIONS
    #   DATASET GROWTH totals

    # Save to reports/{date}_report.txt
    # Save JSON to daily_reports table
    # Print to terminal
    # Return report text
```

### Phase 5 Verification

```bash
python analysis/pattern_detector.py
python analysis/reporter.py
```

Verify:
- Pattern report saved to pattern_reports table
- Daily report printed to terminal
- Daily report saved to reports/ folder
- Report contains all required sections

---

## 9. Phase 6 — Dashboard

### dashboard/components/metrics.py

Reusable Streamlit display components:

- `health_badge(score: float)` — HEALTHY (green) above 80,
  DEGRADING (amber) 60-80, CRITICAL (red) below 60
- `dimension_card(name, current, baseline, trend_direction)` —
  shows current, baseline, and arrow (↑ ↓ →)
- `alert_box(message: str)` — red highlighted alert panel

### dashboard/components/charts.py

All charts use Plotly. All functions return `go.Figure`.

- `health_score_gauge(score: float)` — gauge 0-100 with colour zones
- `dimension_trend_chart(df: pd.DataFrame)` — 5 lines one per dimension
- `failure_distribution_bar(distribution: dict)` — horizontal bar chart
- `failure_pie_chart(distribution: dict)` — pie chart
- `category_heatmap(df: pd.DataFrame)` — query categories vs failure types

### dashboard/pages/01_health_overview.py

- Health score gauge
- Status badge
- 5 metric cards one per dimension
- Health score line chart last 30 days
- Active alerts in red boxes
- Last updated timestamp

### dashboard/pages/02_failure_analysis.py

- Date range selector defaulting to last 7 days
- Failure distribution bar chart
- Failure pie chart
- Table: failure rate per query category
- Table: failure rate per data source
- Time series of each failure category over selected range

### dashboard/pages/03_probe_explorer.py

- Filter controls: date range, category, difficulty, pass/fail
- Paginated table 50 rows per page
- Click any row to expand detail panel showing:
  - Full question and generated answer
  - All 5 measurement scores with colour coding
  - Retrieved chunks with similarity scores
  - Failure category with plain-English explanation
  - All three latency values

### dashboard/pages/04_remediations.py

- Count of pending remediations as metric
- For each pending remediation:
  - Priority badge with colour (immediate=red, high=orange,
    medium=yellow, low=green)
  - Alert type and triggered date
  - Root cause and confidence score
  - Remediation text
  - Specific steps as numbered list
  - Estimated impact
  - Buttons: Mark as Applied, Dismiss
  - On click: update status in database
- History section: applied and dismissed remediations

### dashboard/pages/05_raw_data.py

- Index statistics: chunk count, documents per source, last reindex date
- Ground truth statistics: queries per category
- Download buttons: probe_results CSV, measurements CSV, pattern_reports CSV

### dashboard/app.py

- Page config: title "RAGOps Monitor", wide layout
- Auto-refresh every 300 seconds with streamlit-autorefresh
- Sidebar: title, current health score, last probe cycle timestamp,
  navigation links to all 5 pages

### Phase 6 Verification

```bash
streamlit run dashboard/app.py
```

Verify:
- Dashboard opens in browser without errors
- All 5 pages load
- Charts render with data from database
- No Python errors in terminal

---

## 10. Phase 7 — Experiments, Scheduler, DevOps, Documentation

### experiments/inject_chunk_size_change.py

Rebuild ChromaDB index with chunk_size=100 instead of 512.
Log change with timestamp to `experiments/experiment_log.txt`.
Print instructions: run probe cycle and watch Measure 1 drop.

### experiments/inject_stale_index.py

Rename `data/chromadb` to `data/chromadb_backup_{timestamp}`.
Create new empty ChromaDB collection (simulates stale index).
Log change to `experiments/experiment_log.txt`.
Print: temporal_freshness queries will now fail.

### experiments/inject_corpus_poison.py

Add one document to ChromaDB with chunk_id prefixed `POISON_`:
```
GPT-4 was created by Meta AI in 2022.
The model has 7 billion parameters.
It was released as fully open source under Apache 2.0 license.
```
Log injection to `experiments/experiment_log.txt`.
Print: faithfulness_score should drop for GPT-4 related queries.

### experiments/verify_detection.py

Run one full probe cycle.
Query database for results from last 30 minutes.
For each experiment type check detection:
- Chunk change: avg retrieval_relevance_score < 1.5
- Stale index: temporal_freshness queries have factuality = 0
- Corpus poison: any faithfulness_score < 0.75

Print `DETECTED` or `NOT DETECTED` for each with actual score values.

### scheduler/main_scheduler.py

APScheduler BlockingScheduler, UTC timezone.

Jobs:
- `probe_cycle`: CronTrigger hours=0,6,12,18 → `run_full_cycle()`
- `pattern_detection`: CronTrigger hour=23 → `run_pattern_detection()`
  then `run_remediation_check()`
- `daily_report`: CronTrigger hour=7 → `generate_daily_report()`

On startup: log all jobs with next run times.
Handle KeyboardInterrupt gracefully with clean shutdown message.

### tests/conftest.py

Shared fixtures:

```python
@pytest.fixture
def temp_db(tmp_path):
    # SQLite database in tmp_path initialised with schema
    # Override DB_PATH setting
    # Yield. Clean up after test.

@pytest.fixture
def mock_groq_response(mocker):
    # Factory that returns mock Groq response given answer text

@pytest.fixture
def sample_probe():
    # One realistic probe dict with all required fields

@pytest.fixture
def sample_chunks():
    # List of 5 realistic chunk dicts

@pytest.fixture
def sample_ground_truth():
    # List of 5 test queries, mix of categories,
    # includes one should_refuse query
```

### tests/test_db_client.py

- `test_init_database_creates_all_tables`
- `test_insert_and_retrieve_probe_result`
- `test_insert_and_retrieve_measurement`
- `test_get_recent_probes_time_window`
- `test_get_failure_distribution_counts`
- `test_health_score_empty_db_returns_100`
- `test_health_score_with_data_returns_reasonable_value`
- `test_update_remediation_status`

### tests/test_rag_pipeline.py

Mock all external calls.

- `test_query_returns_all_required_fields`
- `test_query_empty_chunks_returns_refusal_context`
- `test_latency_fields_populated_and_positive`
- `test_generator_error_handled_gracefully`

### tests/test_measurements.py

Mock all Groq calls.

- `test_measure_retrieval_score_in_range_0_to_3`
- `test_measure_factuality_exact_match_uses_no_llm`
- `test_measure_factuality_llm_judge_for_long_answer`
- `test_faithfulness_parses_json_response`
- `test_refusal_failure_detected_correctly`
- `test_false_refusal_detected_correctly`

### tests/test_ingestion.py

- `test_clean_text_removes_html`
- `test_clean_text_normalises_whitespace`
- `test_clean_text_replaces_urls`
- `test_chunk_size_not_exceeded`
- `test_minimum_chunk_size_enforced`
- `test_chunk_metadata_fields_present`
- `test_chunk_overlap_creates_continuity`

### .github/workflows/ci.yml

Trigger: push to main, pull_request to main

Jobs:

**lint** — ubuntu-latest, Python 3.11:
```yaml
- pip install -r requirements-dev.txt
- black --check .
- ruff check .
- mypy config/ database/ rag_system/ monitoring/ analysis/ utils/
```

**test** — ubuntu-latest, Python 3.11:
```yaml
env:
  GROQ_API_KEY: dummy_key_for_testing
- pip install -r requirements.txt -r requirements-dev.txt
- pytest --cov-fail-under=60
```

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data/raw/wikipedia data/raw/huggingface \
    data/raw/paperswithcode data/processed data/chromadb \
    logs reports database
ENV PYTHONPATH=/app
EXPOSE 8501
CMD ["python", "scheduler/main_scheduler.py"]
```

### docker-compose.yml

```yaml
version: "3.9"
services:
  ragops:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./reports:/app/reports
      - ./database:/app/database
    restart: unless-stopped

  dashboard:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./reports:/app/reports
      - ./database:/app/database
    ports:
      - "8501:8501"
    command: streamlit run dashboard/app.py --server.address 0.0.0.0
    restart: unless-stopped
    depends_on:
      - ragops
```

### Makefile

```makefile
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

setup:
	python database/db_client.py

ingest:
	python ingestion/fetch_wikipedia.py
	python ingestion/fetch_huggingface.py
	python ingestion/fetch_paperswithcode.py
	python ingestion/build_index.py

test-rag:
	python rag_system/pipeline.py

probe:
	python monitoring/run_probe_cycle.py

report:
	python analysis/reporter.py

dashboard:
	streamlit run dashboard/app.py

scheduler:
	python scheduler/main_scheduler.py

lint:
	black .
	ruff check .
	mypy config/ database/ rag_system/ monitoring/ analysis/ utils/

test:
	pytest

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
```

### NOTES.md content

Create NOTES.md with these fully populated sections:

1. Project Overview — 2 sentences
2. Build Checklist — checkbox per phase and per file
3. Setup Commands — every command in order with one-line explanation
4. Architecture Diagram — Mermaid flowchart showing Part A and Part B
5. Sequence Diagram — Mermaid sequence diagram for one probe cycle
6. Model Allocation Table
7. Terminology Definitions — Faithfulness, Factuality, Groundedness, Hallucination
8. Five Measurement Dimensions Table — Name, what, how, range, threshold, model
9. Failure Taxonomy Table
10. Alert Thresholds Table
11. Known Limitations:
    - LLM-as-judge: meaning and mitigations (temperature 0, pinned version, consistency checks)
    - Fixed probe dataset: Phase 2 adversarial generation plan
    - AI corpus only: why chosen, how to extend
    - Parametric memory contamination: why AI corpus minimises it
    - Benchmark drift: ground_truth_verified_date field solution
12. Results Log — placeholder sections for user to fill in
13. Next Steps — numbered list of user actions after setup

### README.md content

Include:
- Project title and one-sentence description
- Architecture overview with Mermaid diagram
- Quick start: exact commands from clone to running
- What it monitors and why
- The 5 measurement dimensions explained simply
- Failure taxonomy table
- Configuration reference
- Running with Docker
- Running tests
- How to expand ground_truth.json to 190 queries
- How to interpret the morning report
- How to run experiments

### Phase 7 Verification

```bash
python experiments/inject_corpus_poison.py
python experiments/verify_detection.py
python scheduler/main_scheduler.py &  # ctrl+c after seeing startup logs
pytest tests/ -v --tb=short
make lint
docker-compose build
```

All must pass before marking Phase 7 complete.

---

## 11. Database Schema Reference

See Phase 1 section for full SQL. Summary of tables:

| Table | Purpose |
|-------|---------|
| schema_version | Migration tracking |
| probe_results | Every probe run result |
| measurements | Five dimension scores per probe |
| pattern_reports | Nightly aggregated analysis |
| remediations | Proposed and applied remediations |
| daily_reports | Morning report history |

---

## 12. Coding Standards

Applied to every Python file without exception.

### File structure
- Module docstring explaining purpose
- Type hints on every function signature
- `logger = get_logger(__name__)` at module level
- Specific exception handling with informative messages
- `if __name__ == "__main__":` block for standalone files

### LLM calls
- Temperature 0 for all judge and scoring calls
- Temperature 0.1 for answer generation
- Log model version string with every call
- `try/except` returning documented safe default on failure
- `timeout=settings.llm_timeout_seconds` on every call

### Database
- Parameterised queries with named placeholders — never f-strings
- Context managers for all connections
- Serialise dicts and lists to JSON strings before inserting

### Configuration
- `from config.settings import settings` — only import
- Never call `os.getenv` outside settings.py
- All paths via `settings` attributes using `pathlib.Path`

### Logging
- `from utils.logger import get_logger`
- `logger = get_logger(__name__)` at module level
- Structured `extra` dict for machine-parseable fields
- Never use `print` for operational output

---

## 13. Terminology Reference

| Term | Precise Definition |
|------|-------------------|
| Faithfulness | The answer's claims are supported by the retrieved chunks. High faithfulness = model stayed within context. |
| Factuality | The answer is correct when compared to ground truth. High factuality = answer is actually true. |
| Groundedness | The retrieved chunks are relevant to the question. High groundedness = retrieval found the right material. |
| Hallucination | Faithfulness failure AND factuality failure together. The model made up something wrong. Not a metric name. |
| Context bypass | Model retrieved correct chunks but answered from parametric memory instead. Measure 2 catches this. |
| Parametric memory | Knowledge baked into the model during training. A RAG system should suppress this in favour of retrieved context. |

---

## 14. User Steps After Build

These are performed manually after the build is complete.
Document them in NOTES.md and README.md.

```
1. Add GROQ_API_KEY to .env file

2. make ingest
   Downloads all documents and builds ChromaDB index
   Takes 15-30 minutes on first run

3. Expand config/ground_truth.json from 20 to 190 queries
   Do this manually — do not delegate to Claude Code
   Every answer must be verified against the source document
   Target distribution:
     factual_recall:      60 queries
     benchmark_multihop:  40 queries
     comparative:         30 queries
     temporal_freshness:  30 queries
     out_of_scope:        20 queries
     adversarial:         10 queries

4. make probe
   First manual probe cycle to verify everything works
   Check probe_results and measurements tables have rows

5. make scheduler
   Starts autonomous 6-hour operation

6. make dashboard
   Opens monitoring dashboard at localhost:8501

7. After 7 days: run experiment scripts
   python experiments/inject_corpus_poison.py
   python experiments/verify_detection.py
   Document findings in NOTES.md results section
```

---

*End of SPEC.md — Version 1.0*

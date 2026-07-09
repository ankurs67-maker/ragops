##
## RAGOps Makefile
## Usage: make <target>
##

PYTHON := python
PYTHONPATH := $(shell pwd)
export PYTHONPATH

# ── Data Pipeline ─────────────────────────────────────────────────────────────
.PHONY: ingest
ingest: ## Fetch all documents from Wikipedia, HuggingFace, PapersWithCode
	$(PYTHON) ingestion/fetch_wikipedia.py
	$(PYTHON) ingestion/fetch_huggingface.py
	$(PYTHON) ingestion/fetch_paperswithcode.py

.PHONY: chunk
chunk: ## Chunk all documents with Contextual RAG and save to all_chunks.json
	$(PYTHON) ingestion/chunk_documents.py

.PHONY: index
index: ## Build ChromaDB vector index from all_chunks.json
	$(PYTHON) ingestion/build_index.py

.PHONY: pipeline
pipeline: ingest chunk index ## Full data pipeline: ingest → chunk → index

# ── Database ──────────────────────────────────────────────────────────────────
.PHONY: db-init
db-init: ## Initialise the SQLite database schema
	$(PYTHON) database/db_client.py

.PHONY: db-reset
db-reset: ## Drop and recreate the database (WARNING: destroys all data)
	rm -f database/ragops.db
	$(MAKE) db-init

# ── Monitoring ────────────────────────────────────────────────────────────────
.PHONY: probe
probe: ## Run a complete probe cycle across all 20 ground truth queries
	$(PYTHON) monitoring/run_probe_cycle.py

.PHONY: probe-quick
probe-quick: ## Run probe cycle with only 5 queries (quick test)
	$(PYTHON) -c "from monitoring.run_probe_cycle import run_probe_cycle; s=run_probe_cycle(max_queries=5); [print(f'  {k}: {v}') for k,v in s.items()]"

# ── Analysis ──────────────────────────────────────────────────────────────────
.PHONY: analyze
analyze: ## Run all 9 pattern analyses and generate remediations
	$(PYTHON) -c "from analysis.pattern_detector import run_all_analyses; import json; print(json.dumps(run_all_analyses(), indent=2))"

.PHONY: report
report: ## Generate daily report
	$(PYTHON) analysis/reporter.py

.PHONY: trends
trends: ## Show trend analysis
	$(PYTHON) -c "from analysis.trend_analysis import analyze_trends; import json; print(json.dumps(analyze_trends(), indent=2))"

# ── Advanced Technique Reports ─────────────────────────────────────────────────
.PHONY: reflexion-status
reflexion-status: ## Show Reflexion effectiveness (analysis 7)
	$(PYTHON) -c "from analysis.pattern_detector import analysis_7_reflexion_effectiveness; import json; r=analysis_7_reflexion_effectiveness(); print(json.dumps(r,indent=2))"

.PHONY: self-rag-report
self-rag-report: ## Show Self-RAG effectiveness (analysis 8)
	$(PYTHON) -c "from analysis.pattern_detector import analysis_8_self_rag_effectiveness; import json; r=analysis_8_self_rag_effectiveness(); print(json.dumps(r,indent=2))"

.PHONY: loop-report
loop-report: ## Show loop engineering effectiveness (analysis 9)
	$(PYTHON) -c "from analysis.pattern_detector import analysis_9_loop_effectiveness; import json; r=analysis_9_loop_effectiveness(); print(json.dumps(r,indent=2))"

.PHONY: failure-memory
failure-memory: ## Show current Reflexion failure memory contents
	$(PYTHON) -c "from monitoring.probe_engine import load_reflexion_lessons; print(load_reflexion_lessons() or 'No lessons yet')"

# ── Dashboard ─────────────────────────────────────────────────────────────────
.PHONY: dashboard
dashboard: ## Launch the Streamlit dashboard
	streamlit run dashboard/app.py --server.port 8501 --server.headless false

.PHONY: dashboard-headless
dashboard-headless: ## Launch Streamlit in headless mode (for CI/remote)
	streamlit run dashboard/app.py --server.port 8501 --server.headless true

# ── Scheduler ─────────────────────────────────────────────────────────────────
.PHONY: schedule
schedule: ## Start the APScheduler (blocks until Ctrl+C)
	$(PYTHON) scheduler/main_scheduler.py

# ── Testing ───────────────────────────────────────────────────────────────────
.PHONY: test
test: ## Run the full test suite
	pytest tests/ -v

.PHONY: test-fast
test-fast: ## Run tests excluding slow API tests
	pytest tests/ -v -m "not slow"

.PHONY: test-unit
test-unit: ## Run unit tests only
	pytest tests/test_chunking.py tests/test_db.py tests/test_classifier.py -v

# ── Experiments ───────────────────────────────────────────────────────────────
.PHONY: exp-baseline
exp-baseline: ## Run baseline retrieval experiment (no advanced techniques)
	$(PYTHON) experiments/01_baseline_retrieval.py

.PHONY: exp-contextual
exp-contextual: ## Run contextual RAG vs baseline experiment
	$(PYTHON) experiments/02_contextual_rag_ablation.py

.PHONY: exp-self-rag
exp-self-rag: ## Run Self-RAG ablation experiment
	$(PYTHON) experiments/03_self_rag_ablation.py

.PHONY: exp-reflexion
exp-reflexion: ## Run Reflexion effectiveness experiment
	$(PYTHON) experiments/04_reflexion_effectiveness.py

# ── Quality ───────────────────────────────────────────────────────────────────
.PHONY: lint
lint: ## Run flake8 linter
	flake8 . --max-line-length=100 --exclude=.venv,data,logs,reports

.PHONY: typecheck
typecheck: ## Run mypy type checker
	mypy . --ignore-missing-imports --exclude=.venv

.PHONY: format
format: ## Format code with black
	black . --line-length 100 --exclude ".venv|data|logs|reports"

# ── Utilities ─────────────────────────────────────────────────────────────────
.PHONY: clean-data
clean-data: ## Remove processed chunks and ChromaDB index (keeps raw data)
	rm -f data/processed/all_chunks.json
	rm -rf data/chromadb/

.PHONY: clean-logs
clean-logs: ## Remove all log files
	rm -f logs/*.log logs/*.jsonl

.PHONY: status
status: ## Show system status (health score + recent failures)
	$(PYTHON) -c "\
from database.db_client import get_system_health_score, get_failure_distribution; \
h = get_system_health_score(); \
d = get_failure_distribution(days=7); \
print(f'Health: {h:.1f}/100'); \
print(f'7d failures: {d}') \
"

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

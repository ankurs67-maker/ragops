"""Centralised configuration for RAGOps using Pydantic Settings v2.

Single source of truth for all settings. Every other module imports `settings` from here.
Never call os.getenv anywhere else in the codebase.
"""

from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API keys ─────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    deepseek_api_key: str = ""
    openrouter_api_key: str = ""

    # ── Environment ───────────────────────────────────────────────────────────
    ragops_env: str = "development"
    log_level: str = "INFO"

    # ── Provider selection ────────────────────────────────────────────────────
    llm_provider: str = "groq"         # groq | deepseek | openrouter
    scoring_provider: str = "deepseek" # deepseek | groq | openrouter

    # ── LLM models (Groq) ────────────────────────────────────────────────────
    llm_model: str = "llama-3.1-70b-versatile"
    scoring_model: str = "llama-3.1-8b-instant"
    judge_model_version: str = "llama-3.1-8b-instant"

    # ── DeepSeek settings ─────────────────────────────────────────────────────
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # ── OpenRouter settings ───────────────────────────────────────────────────
    openrouter_model: str = "deepseek/deepseek-chat"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Generation parameters ─────────────────────────────────────────────────
    generation_temperature: float = 0.1
    judge_temperature: float = 0.0
    max_answer_tokens: int = 500
    max_judge_tokens: int = 200
    llm_timeout_seconds: int = 30

    # ── Embeddings and vector store ───────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    chroma_collection: str = "llm_intelligence_corpus"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    top_k: int = 5

    # ── Alert thresholds ──────────────────────────────────────────────────────
    alert_retrieval_threshold: float = 1.5
    alert_utilization_threshold: float = 60.0
    alert_faithfulness_threshold: float = 0.75
    alert_factuality_threshold: float = 0.60
    alert_refusal_threshold: float = 0.70
    alert_latency_multiplier: float = 3.0

    # ── Trend detection ───────────────────────────────────────────────────────
    trend_window_days: int = 7
    baseline_window_days: int = 30
    trend_alert_percent: float = 15.0

    # ── Self-RAG mode ─────────────────────────────────────────────────────────
    self_rag_blocking: bool = True  # False = advisory mode (checks run, never block)

    # ── Retrieval: sibling-chunk expansion ────────────────────────────────────
    # When a benchmark (PapersWithCode) chunk is retrieved, also pull its
    # adjacent chunk(s) from the same document. Targets the diagnosed failure
    # where leaderboard chunks embed poorly and the wrong half of a benchmark
    # file is retrieved (NOTES.md follow-up 6). Toggle off for A/B comparison.
    sibling_expansion_enabled: bool = True

    # ── Scheduling (2 cycles/day to stay within free tier limits) ─────────────
    probe_schedule_hours: List[int] = [0, 12]
    pattern_schedule_hour: int = 23
    report_schedule_hour: int = 7

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator("groq_api_key", "deepseek_api_key", "openrouter_api_key", mode="before")
    @classmethod
    def strip_api_key_whitespace(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    # ── Derived paths (all via pathlib.Path from BASE_DIR) ───────────────────
    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def raw_dir(self) -> Path:
        return BASE_DIR / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return BASE_DIR / "data" / "processed"

    @property
    def chroma_dir(self) -> Path:
        return BASE_DIR / "data" / "chromadb"

    @property
    def db_path(self) -> Path:
        return BASE_DIR / "database" / "ragops.db"

    @property
    def schema_path(self) -> Path:
        return BASE_DIR / "database" / "schema.sql"

    @property
    def ground_truth_path(self) -> Path:
        return BASE_DIR / "config" / "ground_truth.json"

    @property
    def failure_memory_path(self) -> Path:
        return BASE_DIR / "data" / "processed" / "failure_memory.jsonl"

    @property
    def logs_dir(self) -> Path:
        return BASE_DIR / "logs"

    @property
    def reports_dir(self) -> Path:
        return BASE_DIR / "reports"

    @property
    def experiment_log_path(self) -> Path:
        return BASE_DIR / "experiments" / "experiment_log.txt"

    def get_active_llm_key(self) -> str:
        """Return the API key for the active LLM provider."""
        return {
            "groq": self.groq_api_key,
            "deepseek": self.deepseek_api_key,
            "openrouter": self.openrouter_api_key,
        }.get(self.llm_provider, self.groq_api_key)

    def get_scoring_key(self) -> str:
        """Return the API key for the active scoring provider."""
        return {
            "groq": self.groq_api_key,
            "deepseek": self.deepseek_api_key,
            "openrouter": self.openrouter_api_key,
        }.get(self.scoring_provider, self.deepseek_api_key)


settings = Settings()

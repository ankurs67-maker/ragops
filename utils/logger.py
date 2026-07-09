"""Centralised logging for RAGOps.

Three handlers:
  1. Rotating file — logs/ragops.log, max 10 MB, 5 backups
  2. Structured JSONL — logs/ragops_structured.jsonl, one JSON per line
  3. Console — coloured in development, plain in production
"""

import json
import logging
import logging.handlers
import os
from datetime import datetime, timezone
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
_LOGS_DIR = _BASE_DIR / "logs"
_LOG_FILE = _LOGS_DIR / "ragops.log"
_JSONL_FILE = _LOGS_DIR / "ragops_structured.jsonl"

_CONFIGURED_LOGGERS: set[str] = set()
_ROOT_CONFIGURED = False


class _JsonlHandler(logging.Handler):
    """Emit each log record as a single JSON line."""

    def __init__(self, filepath: Path) -> None:
        super().__init__()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._filepath = filepath

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "extra": {
                    k: v
                    for k, v in record.__dict__.items()
                    if k
                    not in {
                        "name", "msg", "args", "levelname", "levelno",
                        "pathname", "filename", "module", "exc_info",
                        "exc_text", "stack_info", "lineno", "funcName",
                        "created", "msecs", "relativeCreated", "thread",
                        "threadName", "processName", "process", "message",
                        "taskName",
                    }
                },
            }
            with open(self._filepath, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception:
            self.handleError(record)


def _get_log_level() -> int:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def _configure_root() -> None:
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return
    _ROOT_CONFIGURED = True

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(_get_log_level())

    # 1. Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.addHandler(file_handler)

    # 2. Structured JSONL handler
    root.addHandler(_JsonlHandler(_JSONL_FILE))

    # 3. Console handler
    console_handler = logging.StreamHandler()
    env = os.environ.get("RAGOPS_ENV", "development").lower()
    if env == "development":
        fmt = "%(asctime)s \033[1;32m[%(levelname)s]\033[0m %(name)s — %(message)s"
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    console_handler.setFormatter(
        logging.Formatter(fmt, datefmt="%H:%M:%S")
    )
    root.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    _configure_root()
    return logging.getLogger(name)

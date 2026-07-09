"""Text cleaning utilities for RAGOps data ingestion.

Cleans raw text from Wikipedia, Hugging Face, and Papers With Code.
Normalises whitespace, removes HTML, replaces URLs, fixes unicode.
"""

import re
import unicodedata
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TABLE_PIPE_RE = re.compile(r"\|")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Return cleaned plain text suitable for chunking."""
    # Remove HTML tags
    text = _HTML_TAG_RE.sub(" ", text)

    # Normalise unicode quotes and dashes
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("…", "...")

    # Remove markdown table pipe formatting
    text = _TABLE_PIPE_RE.sub(" ", text)

    # Replace URLs with placeholder
    text = _URL_RE.sub("[URL]", text)

    # Remove control characters (keep \n and \t)
    text = _CONTROL_CHAR_RE.sub("", text)

    # Normalise whitespace — max 2 consecutive newlines
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # Strip leading and trailing whitespace
    return text.strip()


def extract_metadata(filepath: Path) -> dict:
    """Infer source and doc_type from the file's parent directory."""
    parent_name = filepath.parent.name.lower()

    source_map = {
        "huggingface": "huggingface",
        "paperswithcode": "paperswithcode",
        "wikipedia": "wikipedia",
    }
    doc_type_map = {
        "huggingface": "model_card",
        "paperswithcode": "benchmark",
        "wikipedia": "article",
    }

    source = source_map.get(parent_name, parent_name)
    doc_type = doc_type_map.get(parent_name, "document")

    return {
        "source": source,
        "doc_type": doc_type,
        "filename": filepath.name,
    }

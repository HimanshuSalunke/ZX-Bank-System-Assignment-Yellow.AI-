"""
ZX Bank AI — Shared Utilities.

Common helper functions used across multiple modules.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return utc_now().isoformat()


def generate_session_id() -> str:
    """Generate a unique session identifier based on timestamp + random bytes."""
    import secrets
    raw = f"{utc_now_iso()}-{secrets.token_hex(8)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def sanitize_text(text: str) -> str:
    """
    Normalize and clean user input text.

    - Strip leading / trailing whitespace
    - Collapse multiple spaces / newlines
    - Normalize unicode characters
    """
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text


def truncate(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis if it exceeds *max_length*."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def safe_json_dump(data: Any, path: Path, *, indent: int = 2) -> None:
    """Atomically write JSON data to *path* (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=False, default=str)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def safe_json_load(path: Path) -> Any:
    """Read and parse a JSON file; return ``None`` if it does not exist."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

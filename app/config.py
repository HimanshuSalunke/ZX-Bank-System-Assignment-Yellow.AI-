"""
ZX Bank AI — Application Configuration.

Centralised, type-safe configuration using Pydantic Settings.
Only secrets and deployment settings come from ``.env``.
All engineering defaults are hardcoded here as they are design
decisions, not deployment variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

# ── Path Constants ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
ESCALATIONS_DIR = PROJECT_ROOT / "data" / "escalations"
INDEXES_DIR = PROJECT_ROOT / "indexes"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class Settings(BaseSettings):
    """Application settings loaded from .env + hardcoded defaults."""

    # ── Secrets & Deployment (from .env) ───────────────────────────────
    requesty_api_key: str = ""
    llm_model: str = "openai/gpt-4.1-nano"
    llm_base_url: str = "https://router.requesty.ai/v1"
    environment: str = "development"

    # ── Engineering Defaults (hardcoded — design decisions) ────────────
    llm_temperature: float = 0.3
    llm_max_tokens: int = 512
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cuda"
    retrieval_top_k: int = 5
    confidence_threshold: float = 0.45
    rrf_k: int = 60
    log_level: str = "INFO"

    # ── Derived Paths ──────────────────────────────────────────────────
    @property
    def documents_dir(self) -> Path:
        return DOCUMENTS_DIR

    @property
    def escalations_dir(self) -> Path:
        return ESCALATIONS_DIR

    @property
    def indexes_dir(self) -> Path:
        return INDEXES_DIR

    @property
    def frontend_dir(self) -> Path:
        return FRONTEND_DIR

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()

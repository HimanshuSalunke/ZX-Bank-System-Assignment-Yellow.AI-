"""
ZX Bank AI — Structured Logging.

Enterprise-grade observability using *structlog* with *rich* console
rendering. Every log event carries structured fields that satisfy the
assignment's logging requirements:

    • query classification result
    • retrieval trigger (yes / no)
    • documents retrieved (count + IDs)
    • response path (handler name)
    • confidence score
    • escalation events
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from rich.console import Console
from rich.logging import RichHandler

from app.config import get_settings

# ---------------------------------------------------------------------------
# Module-level console (reused for all rich output)
# ---------------------------------------------------------------------------
_console = Console(stderr=True, force_terminal=True)


def setup_logging() -> None:
    """
    Configure *structlog* + *rich* for the entire application.

    Call this **once** at application startup (inside the FastAPI lifespan).
    Subsequent calls are idempotent.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level, logging.INFO)

    # ---- Standard-library root logger (captured by structlog) ------------
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=_console,
                rich_tracebacks=True,
                tracebacks_show_locals=settings.environment == "development",
                markup=True,
                show_path=False,
            ),
        ],
        force=True,
    )

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "uvicorn.access", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ---- Structlog pipeline ----------------------------------------------
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Formatter for stdlib handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        foreign_pre_chain=shared_processors,
    )

    # Apply formatter to root handler
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)


def get_logger(name: str | None = None, **initial_binds: Any) -> structlog.stdlib.BoundLogger:
    """
    Return a *structlog* bound logger with optional initial context.

    Usage::

        logger = get_logger(__name__, component="retrieval")
        logger.info("search_complete", docs_found=5, latency_ms=42)
    """
    log: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if initial_binds:
        log = log.bind(**initial_binds)
    return log

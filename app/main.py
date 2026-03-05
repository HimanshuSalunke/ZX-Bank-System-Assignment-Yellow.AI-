"""
ZX Bank AI — FastAPI Application Entry-Point.

Bootstraps the full conversational AI pipeline:
  1. Configures structured logging.
  2. Loads FAISS + BM25 indexes from disk.
  3. Initialises the hybrid retriever, conversation manager, and LLM client.
  4. Mounts API routes and optional static-file serving.
  5. Gracefully tears down resources on shutdown.

Run with:
    python -m uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.core.conversation import ConversationManager
from app.retrieval.bm25_index import BM25Index
from app.retrieval.embeddings import init_model as preload_embeddings
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_store import VectorStore
from app.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# ── Application State (shared across requests) ────────────────────────────
vector_store = VectorStore()
bm25_index = BM25Index()
hybrid_retriever: HybridRetriever | None = None
conversation_manager = ConversationManager()


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: load indexes on startup, cleanup on shutdown."""
    global hybrid_retriever

    # ── Startup ───────────────────────────────────────────────────────
    setup_logging()
    settings = get_settings()

    logger.info(
        "application_startup",
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        embedding_model=settings.embedding_model,
        device=settings.embedding_device,
        environment=settings.environment,
    )

    # Ensure directories exist
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    settings.escalations_dir.mkdir(parents=True, exist_ok=True)
    settings.indexes_dir.mkdir(parents=True, exist_ok=True)

    # Load persisted indexes
    faiss_loaded = vector_store.load()
    bm25_loaded = bm25_index.load()

    if faiss_loaded and bm25_loaded:
        hybrid_retriever = HybridRetriever(vector_store, bm25_index)
        logger.info(
            "indexes_loaded",
            faiss_vectors=vector_store.size,
            bm25_documents=bm25_index.size,
        )
    else:
        logger.warning(
            "indexes_not_found",
            hint="Run 'python setup_index.py' to build indexes.",
        )

    # Preload SBERT embedding model into GPU memory (eliminates cold start)
    preload_embeddings()

    yield  # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("application_shutdown")


# ---------------------------------------------------------------------------
# FastAPI instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ZX Bank Conversational AI",
    description=(
        "A lightweight conversational AI backend for ZX Bank using hybrid RAG "
        "(BM25 + FAISS), multi-turn conversations, adversarial safety, and "
        "human escalation — powered by GPT-4.1-mini via Requesty, with SSE streaming."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routes ─────────────────────────────────────────────────────────────
from app.api.routes import router as api_router  # noqa: E402

app.include_router(api_router, prefix="/api")

# ── Static Frontend ────────────────────────────────────────────────────────
_settings = get_settings()
if _settings.frontend_dir.exists() and (_settings.frontend_dir / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_settings.frontend_dir), html=True), name="frontend")

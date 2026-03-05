"""
ZX Bank AI — Index Builder Script.

Run this script to process documents and build FAISS + BM25 indexes.

Usage:
    python setup_index.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import get_settings
from app.retrieval.bm25_index import BM25Index
from app.retrieval.document_processor import load_and_split_documents
from app.retrieval.tfidf_extractor import extract_keywords
from app.retrieval.vector_store import VectorStore
from app.utils.logger import get_logger, setup_logging


def main() -> None:
    """Build all indexes from scratch."""
    setup_logging()
    logger = get_logger(__name__, component="index_builder")
    settings = get_settings()

    start = time.perf_counter()
    logger.info("index_build_started", documents_dir=str(settings.documents_dir))

    # ── Step 1: Load & split documents ──
    logger.info("step_1_loading_documents")
    chunks = load_and_split_documents()

    if not chunks:
        logger.error("no_chunks_produced")
        sys.exit(1)

    # ── Step 2: Extract TF-IDF keywords ──
    logger.info("step_2_extracting_keywords")
    chunks = extract_keywords(chunks, top_n=10)

    # ── Step 3: Build FAISS vector index ──
    logger.info("step_3_building_faiss_index")
    vector_store = VectorStore()
    vector_store.build(chunks)
    vector_store.save()

    # ── Step 4: Build BM25 index ──
    logger.info("step_4_building_bm25_index")
    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save()

    elapsed = time.perf_counter() - start

    logger.info(
        "index_build_complete",
        total_chunks=len(chunks),
        faiss_vectors=vector_store.size,
        bm25_documents=bm25.size,
        elapsed_seconds=round(elapsed, 2),
        index_dir=str(settings.indexes_dir),
    )
    print(f"\n✅ Indexes built successfully in {elapsed:.1f}s")
    print(f"   → {len(chunks)} chunks indexed")
    print(f"   → FAISS vectors: {vector_store.size}")
    print(f"   → BM25 documents: {bm25.size}")
    print(f"   → Saved to: {settings.indexes_dir}")


if __name__ == "__main__":
    main()

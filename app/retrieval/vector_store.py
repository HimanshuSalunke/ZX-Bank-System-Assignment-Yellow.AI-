"""
ZX Bank AI — FAISS Vector Store.

Manages the FAISS index for dense (semantic) retrieval. Supports
building, persisting, loading, and querying the index.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.config import INDEXES_DIR
from app.retrieval.document_processor import DocumentChunk
from app.retrieval.embeddings import encode_query, encode_texts, get_embedding_dimension
from app.utils.logger import get_logger

logger = get_logger(__name__, component="vector_store")

_FAISS_INDEX_FILE = "faiss_index.bin"
_CHUNK_MAP_FILE = "chunk_map.pkl"


class VectorStore:
    """FAISS-backed dense vector store."""

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[DocumentChunk] = []
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.index is not None

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0

    # ── Build ──────────────────────────────────────────────────────────

    def build(self, chunks: list[DocumentChunk]) -> None:
        """
        Build the FAISS index from document chunks.

        Uses Inner Product (IP) on L2-normalised vectors, which is
        equivalent to cosine similarity but faster.
        """
        if not chunks:
            logger.warning("build_skipped_empty_chunks")
            return

        texts = [chunk.content for chunk in chunks]
        embeddings = encode_texts(texts, show_progress=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks
        self._is_loaded = True

        logger.info(
            "vector_store_built",
            num_vectors=self.index.ntotal,
            dimensions=dim,
        )

    # ── Search ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Search the index for the most similar chunks.

        Returns list of (chunk, similarity_score) tuples sorted by
        descending relevance.
        """
        if not self.is_loaded:
            logger.warning("search_on_unloaded_index")
            return []

        query_vec = encode_query(query).reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, min(top_k, self.size))

        results: list[tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append((self.chunks[idx], float(score)))

        logger.info(
            "vector_search_complete",
            query=query[:80],
            results_found=len(results),
            top_score=round(results[0][1], 4) if results else 0.0,
        )
        return results

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, index_dir: Path | None = None) -> None:
        """Persist index and chunk map to disk."""
        save_dir = index_dir or INDEXES_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            logger.warning("save_skipped_no_index")
            return

        faiss.write_index(self.index, str(save_dir / _FAISS_INDEX_FILE))

        chunk_data = [c.to_dict() for c in self.chunks]
        with open(save_dir / _CHUNK_MAP_FILE, "wb") as f:
            pickle.dump(chunk_data, f)

        logger.info(
            "vector_store_saved",
            path=str(save_dir),
            num_vectors=self.index.ntotal,
        )

    def load(self, index_dir: Path | None = None) -> bool:
        """Load index and chunk map from disk. Returns True on success."""
        load_dir = index_dir or INDEXES_DIR

        faiss_path = load_dir / _FAISS_INDEX_FILE
        chunk_path = load_dir / _CHUNK_MAP_FILE

        if not faiss_path.exists() or not chunk_path.exists():
            logger.info("no_persisted_index_found", path=str(load_dir))
            return False

        self.index = faiss.read_index(str(faiss_path))

        with open(chunk_path, "rb") as f:
            chunk_dicts = pickle.load(f)

        self.chunks = [
            DocumentChunk(**{k: v for k, v in d.items() if k != "chunk_id"})
            for d in chunk_dicts
        ]
        self._is_loaded = True

        logger.info(
            "vector_store_loaded",
            num_vectors=self.index.ntotal,
            num_chunks=len(self.chunks),
        )
        return True

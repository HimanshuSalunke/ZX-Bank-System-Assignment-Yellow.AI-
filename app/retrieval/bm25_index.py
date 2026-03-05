"""
ZX Bank AI — BM25 Sparse Index.

Okapi BM25 index for sparse keyword-based retrieval. Combined with
FAISS dense search via Reciprocal Rank Fusion (RRF) in the hybrid
retriever.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import INDEXES_DIR
from app.retrieval.document_processor import DocumentChunk
from app.utils.logger import get_logger

logger = get_logger(__name__, component="bm25_index")

_BM25_INDEX_FILE = "bm25_index.pkl"
_BM25_CHUNKS_FILE = "bm25_chunks.pkl"


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    # Remove very short tokens
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """BM25 sparse retrieval index."""

    def __init__(self) -> None:
        self.index: BM25Okapi | None = None
        self.chunks: list[DocumentChunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.index is not None

    @property
    def size(self) -> int:
        return len(self.chunks)

    # ── Build ──────────────────────────────────────────────────────────

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Build BM25 index from document chunks."""
        if not chunks:
            logger.warning("bm25_build_skipped_empty")
            return

        self.chunks = chunks

        # Enrich corpus with metadata for better keyword matching
        enriched_texts = []
        for chunk in chunks:
            parts = [chunk.content]
            if chunk.doc_title:
                parts.append(chunk.doc_title)
            if chunk.section_heading:
                parts.append(chunk.section_heading)
            keywords = chunk.metadata.get("keywords", [])
            if keywords:
                parts.append(" ".join(keywords))
            enriched_texts.append(" ".join(parts))

        self.tokenized_corpus = [_tokenize(text) for text in enriched_texts]
        self.index = BM25Okapi(self.tokenized_corpus)
        self._is_loaded = True

        logger.info(
            "bm25_index_built",
            num_documents=len(chunks),
            avg_doc_length=round(
                sum(len(t) for t in self.tokenized_corpus) / max(len(self.tokenized_corpus), 1)
            ),
        )

    # ── Search ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Search for relevant chunks using BM25 scoring.

        Returns list of (chunk, bm25_score) tuples sorted by descending
        relevance.
        """
        if not self.is_loaded:
            logger.warning("bm25_search_on_unloaded_index")
            return []

        tokenized_query = _tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        results: list[tuple[DocumentChunk, float]] = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunks[idx], float(scores[idx])))

        logger.info(
            "bm25_search_complete",
            query=query[:80],
            results_found=len(results),
            top_score=round(results[0][1], 4) if results else 0.0,
        )
        return results

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, index_dir: Path | None = None) -> None:
        """Persist BM25 index to disk."""
        save_dir = index_dir or INDEXES_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / _BM25_INDEX_FILE, "wb") as f:
            pickle.dump(
                {
                    "tokenized_corpus": self.tokenized_corpus,
                    "chunks": [c.to_dict() for c in self.chunks],
                },
                f,
            )

        logger.info("bm25_index_saved", path=str(save_dir))

    def load(self, index_dir: Path | None = None) -> bool:
        """Load BM25 index from disk. Returns True on success."""
        load_dir = index_dir or INDEXES_DIR
        bm25_path = load_dir / _BM25_INDEX_FILE

        if not bm25_path.exists():
            logger.info("no_persisted_bm25_found", path=str(load_dir))
            return False

        with open(bm25_path, "rb") as f:
            data = pickle.load(f)

        self.tokenized_corpus = data["tokenized_corpus"]
        self.chunks = [
            DocumentChunk(**{k: v for k, v in d.items() if k != "chunk_id"})
            for d in data["chunks"]
        ]
        self.index = BM25Okapi(self.tokenized_corpus)
        self._is_loaded = True

        logger.info(
            "bm25_index_loaded",
            num_documents=len(self.chunks),
        )
        return True

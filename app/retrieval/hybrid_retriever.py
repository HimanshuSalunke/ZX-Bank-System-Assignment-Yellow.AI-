"""
ZX Bank AI — Hybrid Retriever.

Combines BM25 (sparse) and FAISS (dense) search results using
Reciprocal Rank Fusion (RRF) for superior retrieval quality.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.retrieval.bm25_index import BM25Index
from app.retrieval.document_processor import DocumentChunk
from app.retrieval.vector_store import VectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__, component="hybrid_retriever")


@dataclass
class RetrievalResult:
    """A single result from the hybrid retrieval engine."""

    chunk: DocumentChunk
    rrf_score: float
    bm25_rank: int | None = None
    faiss_rank: int | None = None
    bm25_score: float = 0.0
    faiss_score: float = 0.0

    @property
    def source_display(self) -> str:
        return self.chunk.display_source


class HybridRetriever:
    """
    Hybrid retrieval engine combining BM25 + FAISS with RRF re-ranking.

    Reciprocal Rank Fusion formula:
        RRF_score(d) = Σ  1 / (k + rank_i(d))

    where k is a smoothing constant (default 60) and rank_i(d) is the
    rank of document d in the i-th retrieval system.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    @property
    def is_ready(self) -> bool:
        return self.vector_store.is_loaded and self.bm25_index.is_loaded

    def search(
        self,
        query: str,
        top_k: int | None = None,
        bm25_weight: float = 1.0,
        faiss_weight: float = 1.0,
    ) -> list[RetrievalResult]:
        """
        Execute hybrid search with RRF re-ranking.

        Args:
            query: User's search query.
            top_k: Number of results to return (defaults to config).
            bm25_weight: Weight for BM25 RRF contribution.
            faiss_weight: Weight for FAISS RRF contribution.

        Returns:
            Sorted list of RetrievalResult by descending RRF score.
        """
        settings = get_settings()
        k = settings.rrf_k
        top_k = top_k or settings.retrieval_top_k
        fetch_k = top_k * 3  # Over-fetch for better fusion

        # ── Retrieve from both engines ──
        bm25_results = self.bm25_index.search(query, top_k=fetch_k)
        faiss_results = self.vector_store.search(query, top_k=fetch_k)

        # ── Build RRF score map ──
        rrf_scores: dict[str, dict] = {}

        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            cid = chunk.chunk_id
            if cid not in rrf_scores:
                rrf_scores[cid] = {
                    "chunk": chunk,
                    "rrf": 0.0,
                    "bm25_rank": None,
                    "faiss_rank": None,
                    "bm25_score": 0.0,
                    "faiss_score": 0.0,
                }
            rrf_scores[cid]["rrf"] += bm25_weight * (1.0 / (k + rank))
            rrf_scores[cid]["bm25_rank"] = rank
            rrf_scores[cid]["bm25_score"] = score

        for rank, (chunk, score) in enumerate(faiss_results, start=1):
            cid = chunk.chunk_id
            if cid not in rrf_scores:
                rrf_scores[cid] = {
                    "chunk": chunk,
                    "rrf": 0.0,
                    "bm25_rank": None,
                    "faiss_rank": None,
                    "bm25_score": 0.0,
                    "faiss_score": 0.0,
                }
            rrf_scores[cid]["rrf"] += faiss_weight * (1.0 / (k + rank))
            rrf_scores[cid]["faiss_rank"] = rank
            rrf_scores[cid]["faiss_score"] = score

        # ── Sort by RRF score and return top-k ──
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["rrf"],
            reverse=True,
        )[:top_k]

        results = [
            RetrievalResult(
                chunk=r["chunk"],
                rrf_score=r["rrf"],
                bm25_rank=r["bm25_rank"],
                faiss_rank=r["faiss_rank"],
                bm25_score=r["bm25_score"],
                faiss_score=r["faiss_score"],
            )
            for r in sorted_results
        ]

        logger.info(
            "hybrid_search_complete",
            query=query[:80],
            bm25_candidates=len(bm25_results),
            faiss_candidates=len(faiss_results),
            unique_chunks=len(rrf_scores),
            results_returned=len(results),
            top_rrf=round(results[0].rrf_score, 6) if results else 0.0,
        )
        return results

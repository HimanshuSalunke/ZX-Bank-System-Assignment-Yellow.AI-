"""
ZX Bank AI — Document Query Handler.

Handles document-type queries by:
  1. Retrieving relevant chunks via hybrid search
  2. Building a grounded prompt with citations
  3. Generating an LLM response anchored to retrieved context
"""

from __future__ import annotations

import time

from app.api.schemas import ChatResponse, SourceDocument
from app.config import get_settings
from app.core.llm import BANKING_SYSTEM_PROMPT, generate
from app.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from app.utils.logger import get_logger

logger = get_logger(__name__, component="document_handler")


def _build_context_prompt(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    if not results:
        return "No relevant documents found in the knowledge base."

    sections: list[str] = []
    for i, r in enumerate(results, 1):
        source = r.source_display
        sections.append(
            f"[Source {i}: {source}]\n{r.chunk.content}"
        )

    return (
        "Use ONLY the following context from ZX Bank's knowledge base to answer "
        "the customer's question. Cite sources using [Source N] notation.\n\n"
        + "\n\n---\n\n".join(sections)
    )


async def handle_document_query(
    message: str,
    session_id: str,
    retriever: HybridRetriever,
    history_messages: list[dict[str, str]],
) -> ChatResponse:
    """
    Process a document query: retrieve → generate → respond.
    """
    settings = get_settings()

    # ── Retrieve ──
    t0 = time.perf_counter()
    results = retriever.search(message)
    retrieval_time = time.perf_counter() - t0

    logger.info(
        "retrieval_timing",
        query=message[:80],
        results=len(results),
        elapsed_seconds=round(retrieval_time, 3),
    )

    if not results:
        logger.info("no_retrieval_results", query=message[:80])
        return ChatResponse(
            response=(
                "I couldn't find specific information about that in our knowledge base. "
                "Please try rephrasing your question, or contact our helpline at "
                "**1800-200-9925** for assistance."
            ),
            session_id=session_id,
            query_type="document_query",
            confidence=0.2,
            sources=[],
        )

    # ── Build context-enriched prompt ──
    context = _build_context_prompt(results)

    # Combine history + contextualised query
    messages = list(history_messages)
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{message}",
    })

    # ── Generate grounded response ──
    llm_response = await generate(
        messages,
        system_prompt=BANKING_SYSTEM_PROMPT,
    )

    # ── Calculate confidence from retrieval scores ──
    top_score = results[0].rrf_score if results else 0.0
    confidence = min(max(top_score * 30, 0.3), 0.98)  # Normalise RRF to 0-1 range

    # ── Build source citations ──
    sources = [
        SourceDocument(
            doc_title=r.chunk.doc_title,
            section=r.chunk.section_heading or r.chunk.subsection_heading or "General",
            relevance_score=round(min(r.rrf_score * 30, 1.0), 3),
        )
        for r in results[:3]  # Top 3 sources
    ]

    logger.info(
        "document_response_generated",
        query=message[:80],
        sources_used=len(sources),
        confidence=round(confidence, 3),
        tokens_used=llm_response["usage"]["total_tokens"],
    )

    return ChatResponse(
        response=llm_response["content"],
        session_id=session_id,
        query_type="document_query",
        confidence=round(confidence, 3),
        sources=sources,
    )

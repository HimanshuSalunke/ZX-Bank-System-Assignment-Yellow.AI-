"""
ZX Bank AI — API Routes.

Full production endpoints with SSE streaming:
  • POST /api/chat          — Full response (non-streaming)
  • POST /api/chat/stream   — Server-Sent Events (word-by-word)
  • GET  /api/health        — System health
  • GET  /api/history/{id}  — Conversation history
  • GET  /api/escalations   — List escalation records
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationTurn,
    HealthResponse,
    HistoryResponse,
    SourceDocument,
)
from app.config import ESCALATIONS_DIR, get_settings
from app.core.classifier import classify_query
from app.core.llm import BANKING_SYSTEM_PROMPT, SMALL_TALK_SYSTEM_PROMPT, generate_stream
from app.core.safety import get_safe_response, is_adversarial
from app.handlers.document_handler import handle_document_query
from app.handlers.escalation_handler import handle_escalation
from app.handlers.smalltalk_handler import handle_small_talk
from app.utils.helpers import utc_now_iso
from app.utils.logger import get_logger

logger = get_logger(__name__, component="api")
router = APIRouter(tags=["Chat"])


def _get_app_state():
    """Import app state from main module."""
    from app.main import conversation_manager, hybrid_retriever
    return hybrid_retriever, conversation_manager


# ── Health Check ────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return system health status."""
    settings = get_settings()
    retriever, _ = _get_app_state()

    return HealthResponse(
        status="healthy" if retriever and retriever.is_ready else "degraded",
        model=settings.llm_model,
        indexes_loaded=retriever is not None and retriever.is_ready,
        environment=settings.environment,
        timestamp=utc_now_iso(),
    )


# ── Chat (Non-Streaming — kept for compatibility) ──────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a user message and return the full response at once."""
    retriever, conversation = _get_app_state()

    session_id = conversation.get_or_create_session(request.session_id)
    message = request.message.strip()

    # Check for active escalation
    esc_state = conversation.get_escalation_state(session_id)
    if esc_state is not None:
        conversation.add_user_message(session_id, message, "escalation")
        response = await handle_escalation(message, session_id, conversation)
        conversation.add_assistant_message(session_id, response.response)
        response.session_id = session_id
        return response

    # Classify
    query_type, _ = classify_query(message)

    # Deep adversarial check
    if query_type in ("adversarial", "document_query"):
        is_adv, adv_reason = is_adversarial(message)
        if is_adv:
            query_type = "adversarial"

    logger.info("query_classified", session_id=session_id, query_type=query_type, message=message[:100])

    conversation.add_user_message(session_id, message, query_type)
    history_messages = conversation.get_history_messages(session_id)

    if query_type == "adversarial":
        _, adv_reason = is_adversarial(message)
        response = ChatResponse(
            response=get_safe_response(adv_reason),
            session_id=session_id, query_type="adversarial", confidence=0.95, sources=[],
        )
    elif query_type == "escalation":
        response = await handle_escalation(message, session_id, conversation)
        response.session_id = session_id
    elif query_type == "small_talk":
        response = await handle_small_talk(message, session_id, history_messages)
    elif query_type == "document_query" and retriever and retriever.is_ready:
        response = await handle_document_query(message, session_id, retriever, history_messages)
    else:
        response = ChatResponse(
            response="I'm not sure how to help with that. Could you rephrase your question?",
            session_id=session_id, query_type="unknown", confidence=0.3, sources=[],
        )

    conversation.add_assistant_message(session_id, response.response)
    logger.info("chat_response", session_id=session_id, query_type=response.query_type, confidence=response.confidence)
    return response


# ── Chat (SSE Streaming — real-time word-by-word) ───────────────────────────

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream the response token-by-token via Server-Sent Events.

    First token arrives in ~1-2 seconds with gpt-4.1-mini.
    """
    retriever, conversation = _get_app_state()

    session_id = conversation.get_or_create_session(request.session_id)
    message = request.message.strip()

    # Check for active escalation (non-streaming — short responses)
    esc_state = conversation.get_escalation_state(session_id)
    if esc_state is not None:
        conversation.add_user_message(session_id, message, "escalation")
        response = await handle_escalation(message, session_id, conversation)
        conversation.add_assistant_message(session_id, response.response)
        response.session_id = session_id

        async def _esc_stream():
            yield _sse_event("meta", {
                "session_id": session_id,
                "query_type": "escalation",
                "confidence": response.confidence,
                "sources": [],
                "needs_input": response.needs_input,
            })
            yield _sse_event("token", response.response)
            yield _sse_event("done", {})

        return StreamingResponse(_esc_stream(), media_type="text/event-stream")

    # Classify
    query_type, _ = classify_query(message)

    if query_type in ("adversarial", "document_query"):
        is_adv, adv_reason = is_adversarial(message)
        if is_adv:
            query_type = "adversarial"

    logger.info("stream_query_classified", session_id=session_id, query_type=query_type)

    conversation.add_user_message(session_id, message, query_type)
    history_messages = conversation.get_history_messages(session_id)

    # ── Adversarial (instant refusal, no streaming needed) ──
    if query_type == "adversarial":
        _, adv_reason = is_adversarial(message)
        safe_resp = get_safe_response(adv_reason)
        conversation.add_assistant_message(session_id, safe_resp)

        async def _adv_stream():
            yield _sse_event("meta", {
                "session_id": session_id, "query_type": "adversarial",
                "confidence": 0.95, "sources": [],
            })
            yield _sse_event("token", safe_resp)
            yield _sse_event("done", {})

        return StreamingResponse(_adv_stream(), media_type="text/event-stream")

    # ── Escalation (start flow) ──
    if query_type == "escalation":
        response = await handle_escalation(message, session_id, conversation)
        response.session_id = session_id
        conversation.add_assistant_message(session_id, response.response)

        async def _esc2_stream():
            yield _sse_event("meta", {
                "session_id": session_id, "query_type": "escalation",
                "confidence": response.confidence, "sources": [],
                "needs_input": response.needs_input,
            })
            yield _sse_event("token", response.response)
            yield _sse_event("done", {})

        return StreamingResponse(_esc2_stream(), media_type="text/event-stream")

    # ── Small Talk (stream from LLM) ──
    if query_type == "small_talk":
        async def _smalltalk_stream():
            yield _sse_event("meta", {
                "session_id": session_id, "query_type": "small_talk",
                "confidence": 0.95, "sources": [],
            })
            messages = list(history_messages)
            messages.append({"role": "user", "content": message})
            full_response = []
            async for token in generate_stream(messages, system_prompt=SMALL_TALK_SYSTEM_PROMPT, max_tokens=256):
                full_response.append(token)
                yield _sse_event("token", token)
            conversation.add_assistant_message(session_id, "".join(full_response))
            yield _sse_event("done", {})

        return StreamingResponse(_smalltalk_stream(), media_type="text/event-stream")

    # ── Document Query (retrieve then stream LLM) ──
    if retriever and retriever.is_ready:
        t0 = time.perf_counter()
        results = retriever.search(message)
        retrieval_time = time.perf_counter() - t0

        logger.info("retrieval_timing", elapsed_seconds=round(retrieval_time, 3), results=len(results))

        sources = [
            {"doc_title": r.chunk.doc_title,
             "section": r.chunk.section_heading or r.chunk.subsection_heading or "General",
             "relevance_score": round(min(r.rrf_score * 30, 1.0), 3)}
            for r in results[:3]
        ]
        top_score = results[0].rrf_score if results else 0.0
        confidence = min(max(top_score * 30, 0.3), 0.98)

        # Build context
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[Source {i}: {r.source_display}]\n{r.chunk.content}")
        context = (
            "Use ONLY the following context from ZX Bank's knowledge base to answer "
            "the customer's question. Cite sources using [Source N] notation.\n\n"
            + "\n\n---\n\n".join(context_parts)
        )

        async def _doc_stream():
            yield _sse_event("meta", {
                "session_id": session_id, "query_type": "document_query",
                "confidence": round(confidence, 3), "sources": sources,
            })
            messages = list(history_messages)
            messages.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{message}"})
            full_response = []
            async for token in generate_stream(messages, system_prompt=BANKING_SYSTEM_PROMPT):
                full_response.append(token)
                yield _sse_event("token", token)
            conversation.add_assistant_message(session_id, "".join(full_response))
            yield _sse_event("done", {})

        return StreamingResponse(_doc_stream(), media_type="text/event-stream")

    # Fallback
    async def _fallback_stream():
        msg = "Knowledge base is loading. Please try again shortly, or call **1800-200-9925**."
        yield _sse_event("meta", {"session_id": session_id, "query_type": "document_query", "confidence": 0.3, "sources": []})
        yield _sse_event("token", msg)
        yield _sse_event("done", {})

    return StreamingResponse(_fallback_stream(), media_type="text/event-stream")


def _sse_event(event_type: str, data) -> str:
    """Format a Server-Sent Event."""
    payload = json.dumps(data) if isinstance(data, (dict, list)) else json.dumps(data)
    return f"event: {event_type}\ndata: {payload}\n\n"


# ── History ─────────────────────────────────────────────────────────────────

@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str) -> HistoryResponse:
    """Retrieve conversation history for a session."""
    _, conversation = _get_app_state()
    history = conversation.get_history(session_id)
    turns = [
        ConversationTurn(role=t["role"], content=t["content"],
                         timestamp=t.get("timestamp", ""), query_type=t.get("query_type"))
        for t in history
    ]
    return HistoryResponse(session_id=session_id, turns=turns, turn_count=len(turns))


# ── Escalations ─────────────────────────────────────────────────────────────

@router.get("/escalations")
async def list_escalations() -> dict:
    """List all stored escalation records."""
    escalation_files = sorted(ESCALATIONS_DIR.glob("escalation_*.json"))
    escalations = []
    for f in escalation_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                escalations.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return {"escalations": escalations, "total": len(escalations)}

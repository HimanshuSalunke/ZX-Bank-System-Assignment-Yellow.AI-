"""
ZX Bank AI — API Request / Response Schemas.

Pydantic v2 models for every API boundary. Strict validation ensures
invalid payloads are rejected with clear error messages before reaching
business logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming chat message from the user."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's message text.",
        examples=["What are the interest rates for home loans?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for multi-turn conversations. "
        "If omitted, a new session is created automatically.",
    )


# ── Response Models ─────────────────────────────────────────────────────────


class SourceDocument(BaseModel):
    """A single retrieved document chunk used to ground the answer."""

    doc_title: str = Field(description="Title of the source document.")
    section: str = Field(description="Section heading within the document.")
    relevance_score: float = Field(
        ge=0.0, le=1.0,
        description="Relevance score from the retrieval engine.",
    )


class EscalationData(BaseModel):
    """Data collected during a human-escalation flow."""

    name: str | None = Field(default=None, description="Customer name.")
    phone: str | None = Field(default=None, description="Customer phone number.")
    reason: str | None = Field(default=None, description="Reason for escalation.")
    timestamp: str | None = Field(default=None, description="ISO timestamp.")


class ChatResponse(BaseModel):
    """Outgoing response to the user."""

    response: str = Field(description="The assistant's reply text.")
    session_id: str = Field(description="Session ID for follow-up messages.")
    query_type: Literal[
        "document_query",
        "small_talk",
        "escalation",
        "adversarial",
        "unknown",
    ] = Field(description="Classification of the user's query.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for the response.",
    )
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Source documents used to generate the answer.",
    )
    escalation: EscalationData | None = Field(
        default=None,
        description="Non-null when an escalation flow is active.",
    )
    needs_input: str | None = Field(
        default=None,
        description="If the system needs additional info (e.g. name, phone).",
    )


class HealthResponse(BaseModel):
    """Response from the health-check endpoint."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(default="healthy")
    model: str = Field(description="Active LLM model identifier.")
    indexes_loaded: bool = Field(description="Whether FAISS/BM25 indexes are loaded.")
    environment: str = Field(description="Deployment environment.")
    timestamp: str = Field(description="Server UTC timestamp.")


class ConversationTurn(BaseModel):
    """A single turn in the conversation history."""

    role: Literal["user", "assistant"] = Field(description="Message author.")
    content: str = Field(description="Message content.")
    timestamp: str = Field(description="ISO timestamp of the message.")
    query_type: str | None = Field(default=None)


class HistoryResponse(BaseModel):
    """Response containing conversation history for a session."""

    session_id: str
    turns: list[ConversationTurn] = Field(default_factory=list)
    turn_count: int = Field(default=0)

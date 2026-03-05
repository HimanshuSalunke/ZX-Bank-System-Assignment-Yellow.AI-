"""
ZX Bank AI — Conversation Memory Manager.

Manages multi-turn conversation state with a sliding window to keep
context within LLM token limits while preserving conversational
coherence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.utils.helpers import generate_session_id, utc_now_iso
from app.utils.logger import get_logger

logger = get_logger(__name__, component="conversation")

_MAX_TURNS = 10  # Keep last N turns per session


@dataclass
class Turn:
    """A single conversation turn."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str = ""
    query_type: str | None = None

    def to_message(self) -> dict[str, str]:
        """Convert to OpenAI message format."""
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "query_type": self.query_type,
        }


class ConversationManager:
    """
    In-memory conversation state manager.

    Stores conversation history per session with a sliding window.
    Thread-safe for single-process async usage (FastAPI).
    """

    def __init__(self, max_turns: int = _MAX_TURNS) -> None:
        self._sessions: dict[str, list[Turn]] = defaultdict(list)
        self._max_turns = max_turns
        # Track escalation state per session
        self._escalation_state: dict[str, dict[str, Any]] = {}

    def get_or_create_session(self, session_id: str | None) -> str:
        """Return existing session ID or create a new one."""
        if session_id and session_id in self._sessions:
            return session_id
        new_id = session_id or generate_session_id()
        self._sessions[new_id]  # initialise empty list via defaultdict
        logger.info("session_created", session_id=new_id)
        return new_id

    def add_user_message(
        self,
        session_id: str,
        content: str,
        query_type: str | None = None,
    ) -> None:
        """Record a user message."""
        turn = Turn(
            role="user",
            content=content,
            timestamp=utc_now_iso(),
            query_type=query_type,
        )
        self._sessions[session_id].append(turn)
        self._trim(session_id)

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
    ) -> None:
        """Record an assistant message."""
        turn = Turn(
            role="assistant",
            content=content,
            timestamp=utc_now_iso(),
        )
        self._sessions[session_id].append(turn)
        self._trim(session_id)

    def get_history_messages(
        self,
        session_id: str,
    ) -> list[dict[str, str]]:
        """
        Return conversation history as OpenAI-format messages.

        Used to provide multi-turn context to the LLM.
        """
        turns = self._sessions.get(session_id, [])
        return [t.to_message() for t in turns]

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return full history with metadata."""
        turns = self._sessions.get(session_id, [])
        return [t.to_dict() for t in turns]

    def get_turn_count(self, session_id: str) -> int:
        return len(self._sessions.get(session_id, []))

    # ── Escalation State ───────────────────────────────────────────────

    def set_escalation_state(
        self,
        session_id: str,
        state: dict[str, Any],
    ) -> None:
        """Store escalation flow state for a session."""
        self._escalation_state[session_id] = state

    def get_escalation_state(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve escalation state; None if no active escalation."""
        return self._escalation_state.get(session_id)

    def clear_escalation_state(self, session_id: str) -> None:
        """Clear escalation state after completion."""
        self._escalation_state.pop(session_id, None)

    # ── Internal ───────────────────────────────────────────────────────

    def _trim(self, session_id: str) -> None:
        """Keep only the last max_turns turns."""
        turns = self._sessions[session_id]
        if len(turns) > self._max_turns * 2:
            self._sessions[session_id] = turns[-self._max_turns * 2:]

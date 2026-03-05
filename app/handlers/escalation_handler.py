"""
ZX Bank AI — Escalation Handler.

Manages the multi-step human escalation flow:
  1. Detect escalation intent
  2. Collect customer name
  3. Collect customer phone number
  4. Collect reason for escalation
  5. Save escalation record as JSON
  6. Confirm to customer

Escalation state is tracked per-session in the ConversationManager.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.api.schemas import ChatResponse, EscalationData
from app.config import ESCALATIONS_DIR
from app.core.conversation import ConversationManager
from app.utils.helpers import safe_json_dump, utc_now_iso
from app.utils.logger import get_logger

logger = get_logger(__name__, component="escalation_handler")

# Escalation flow steps
_STEP_ASK_NAME = "ask_name"
_STEP_ASK_PHONE = "ask_phone"
_STEP_ASK_REASON = "ask_reason"
_STEP_COMPLETE = "complete"

_PHONE_PATTERN = re.compile(r"^[\d\s\-\+\(\)]{7,15}$")


def _is_valid_phone(text: str) -> bool:
    """Validate phone number format."""
    cleaned = re.sub(r"[\s\-\(\)]", "", text.strip())
    return bool(re.match(r"^\+?\d{7,15}$", cleaned))


async def handle_escalation(
    message: str,
    session_id: str,
    conversation: ConversationManager,
) -> ChatResponse:
    """
    Process the escalation flow step by step.

    The flow is stateful and tracked across messages within a session.
    """
    state = conversation.get_escalation_state(session_id)

    # ── New escalation: start the flow ──
    if state is None:
        state = {"step": _STEP_ASK_NAME, "name": None, "phone": None, "reason": None}
        conversation.set_escalation_state(session_id, state)

        logger.info("escalation_started", session_id=session_id)
        return ChatResponse(
            response=(
                "I understand you'd like to speak with a human agent. "
                "I'll connect you with our support team. First, may I have your **full name** please?"
            ),
            session_id=session_id,
            query_type="escalation",
            confidence=0.95,
            sources=[],
            escalation=EscalationData(),
            needs_input="name",
        )

    step = state["step"]

    # ── Step: Collecting name ──
    if step == _STEP_ASK_NAME:
        name = message.strip()
        if len(name) < 2:
            return ChatResponse(
                response="Please provide a valid name (at least 2 characters).",
                session_id=session_id,
                query_type="escalation",
                confidence=0.90,
                sources=[],
                escalation=EscalationData(name=name),
                needs_input="name",
            )

        state["name"] = name
        state["step"] = _STEP_ASK_PHONE
        conversation.set_escalation_state(session_id, state)

        return ChatResponse(
            response=f"Thank you, **{name}**. Could you please share your **phone number** so our team can reach you?",
            session_id=session_id,
            query_type="escalation",
            confidence=0.95,
            sources=[],
            escalation=EscalationData(name=name),
            needs_input="phone",
        )

    # ── Step: Collecting phone ──
    if step == _STEP_ASK_PHONE:
        phone = message.strip()
        if not _is_valid_phone(phone):
            return ChatResponse(
                response="Please enter a valid phone number (7-15 digits, e.g., +91-9876543210).",
                session_id=session_id,
                query_type="escalation",
                confidence=0.90,
                sources=[],
                escalation=EscalationData(name=state["name"], phone=phone),
                needs_input="phone",
            )

        state["phone"] = phone
        state["step"] = _STEP_ASK_REASON
        conversation.set_escalation_state(session_id, state)

        return ChatResponse(
            response="Got it. Could you briefly describe the **reason** for your request? This helps our team assist you faster.",
            session_id=session_id,
            query_type="escalation",
            confidence=0.95,
            sources=[],
            escalation=EscalationData(name=state["name"], phone=phone),
            needs_input="reason",
        )

    # ── Step: Collecting reason & completing ──
    if step == _STEP_ASK_REASON:
        reason = message.strip()
        state["reason"] = reason if reason else "Not specified"
        state["step"] = _STEP_COMPLETE

        # Save escalation record
        timestamp = utc_now_iso()
        escalation_data = {
            "name": state["name"],
            "phone": state["phone"],
            "reason": state["reason"],
            "session_id": session_id,
            "timestamp": timestamp,
            "status": "pending",
        }

        # Persist to disk
        filename = f"escalation_{session_id}_{timestamp.replace(':', '-').split('.')[0]}.json"
        save_path = ESCALATIONS_DIR / filename
        safe_json_dump(escalation_data, save_path)

        # Clear escalation state
        conversation.clear_escalation_state(session_id)

        logger.info(
            "escalation_completed",
            session_id=session_id,
            name=state["name"],
            saved_to=str(save_path),
        )

        return ChatResponse(
            response=(
                f"Thank you, **{state['name']}**! Your escalation request has been recorded.\n\n"
                f"📋 **Escalation Summary:**\n"
                f"- **Name**: {state['name']}\n"
                f"- **Phone**: {state['phone']}\n"
                f"- **Reason**: {state['reason']}\n"
                f"- **Reference**: {session_id[:8].upper()}\n\n"
                f"Our team will contact you within **2 hours** during business hours "
                f"(Mon-Fri, 10 AM - 6 PM IST). For urgent matters, please call "
                f"**1800-200-9925** (24/7).\n\n"
                f"Is there anything else I can help you with?"
            ),
            session_id=session_id,
            query_type="escalation",
            confidence=0.98,
            sources=[],
            escalation=EscalationData(
                name=state["name"],
                phone=state["phone"],
                reason=state["reason"],
                timestamp=timestamp,
            ),
        )

    # Fallback — shouldn't reach here
    conversation.clear_escalation_state(session_id)
    return ChatResponse(
        response="Something went wrong with the escalation process. Please try again or call 1800-200-9925.",
        session_id=session_id,
        query_type="escalation",
        confidence=0.5,
        sources=[],
    )

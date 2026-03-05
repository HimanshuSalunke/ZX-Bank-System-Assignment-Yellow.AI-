"""
ZX Bank AI — Query Classifier.

Classifies user queries into one of four types to determine the
appropriate response handler:

  • ``document_query``  → Retrieval + grounded answer
  • ``small_talk``      → Direct LLM response (no retrieval)
  • ``escalation``      → Human handoff flow
  • ``adversarial``     → Safety refusal
"""

from __future__ import annotations

import re
from typing import Literal

from app.utils.logger import get_logger

logger = get_logger(__name__, component="classifier")

QueryType = Literal["document_query", "small_talk", "escalation", "adversarial"]

# ── Pattern Definitions ────────────────────────────────────────────────────

_ESCALATION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(speak|talk|connect)\s+(to|with)\s+(a\s+)?(human|agent|person|manager|officer|representative|executive)\b",
        r"\b(human|agent|live)\s+(support|assistance|help|chat)\b",
        r"\b(escalat|complain|grievance)\b",
        r"\bnot\s+satisf(ied|actory)\b",
        r"\b(want|need|require)\s+(a\s+)?(callback|call\s*back)\b",
        r"\b(raise|file|lodge)\s+(a\s+)?(complaint|grievance|ticket)\b",
    ]
]

_ADVERSARIAL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bignore\s+(all\s+)?(your|previous|above|prior)\s+(instructions?|rules?|guidelines?)\b",
        r"\byou\s+are\s+now\s+(DAN|evil|unrestricted)\b",
        r"\b(jailbreak|bypass|override)\b",
        r"\bact\s+as\s+(if|though)\s+you\s+(have|had)\s+no\s+(restrictions?|rules?|limits?)\b",
        r"\b(reveal|share|expose|disclose)\s+(your|the|all)\s+(system|internal|hidden|secret)\s+(prompt|instructions?|password|credentials?)\b",
        r"\b(transfer|withdraw|debit|move)\s+.*?(money|funds|balance|amount|lakh|crore)\s+(from|to)\b",
        r"\bprovide\s+(me\s+)?(all\s+)?(customer|account|user)\s+(data|details|information|records)\b",
        r"\bdo\s+anything\s+now\b",
        r"\bpretend\s+(you\s+)?(are|have)\s+(no|unlimited)\b",
    ]
]

_SMALL_TALK_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|namaste|greetings)\s*[!?.]*$",
        r"^(how\s+are\s+you|what('s|\s+is)\s+up|howdy)\s*[!?.]*$",
        r"^(thank(s|\s+you)|bye|goodbye|see\s+you|take\s+care)\s*[!?.]*$",
        r"^(who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|help)\s*[!?.]*$",
        r"^(ok|okay|alright|sure|fine|got\s+it|understood)\s*[!?.]*$",
        r"^(yes|no|maybe|nope|yep|yeah)\s*[!?.]*$",
    ]
]


def classify_query(message: str) -> tuple[QueryType, float]:
    """
    Classify a user message into a query type.

    Returns:
        Tuple of (query_type, confidence).
    """
    text = message.strip()

    if not text:
        return "small_talk", 1.0

    # ── Check adversarial patterns first (highest priority) ──
    for pattern in _ADVERSARIAL_PATTERNS:
        if pattern.search(text):
            logger.warning(
                "adversarial_detected",
                pattern=pattern.pattern[:60],
                message=text[:100],
            )
            return "adversarial", 0.95

    # ── Check escalation patterns ──
    for pattern in _ESCALATION_PATTERNS:
        if pattern.search(text):
            logger.info(
                "escalation_detected",
                pattern=pattern.pattern[:60],
                message=text[:100],
            )
            return "escalation", 0.90

    # ── Check small talk patterns ──
    for pattern in _SMALL_TALK_PATTERNS:
        if pattern.search(text):
            return "small_talk", 0.95

    # ── Default: document query ──
    return "document_query", 0.80

"""
ZX Bank AI — Adversarial Safety Filter.

Provides multi-layer adversarial detection and safe response
generation. Works alongside the classifier for defense-in-depth.
"""

from __future__ import annotations

import re

from app.utils.logger import get_logger

logger = get_logger(__name__, component="safety")

# ── Sensitive Topic Patterns ───────────────────────────────────────────────
_SENSITIVE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(password|pin|cvv|otp|secret\s+key|api\s+key)\b",
        r"\b(hack|exploit|vulnerability|breach)\b",
        r"\binternal\s+(system|database|server|network|api)\b",
        r"\b(admin|root)\s+(access|credential|password|login)\b",
        r"\b(money\s+laundering|terrorist\s+financing|hawala)\b",
    ]
]

_SAFE_REFUSAL = (
    "I appreciate your message, but I'm unable to assist with that request. "
    "As ZX Bank's AI assistant, I'm designed to help with legitimate banking "
    "queries such as account information, loan details, card services, and "
    "digital banking features.\n\n"
    "If you have a genuine banking question, I'd be happy to help! "
    "For urgent matters, please contact our 24/7 helpline: **1800-200-9925**."
)

_SOCIAL_ENGINEERING_REFUSAL = (
    "I understand your request, but I cannot provide customer data, internal "
    "system details, or perform account operations through this chat interface. "
    "This applies regardless of claimed authority or urgency.\n\n"
    "For legitimate official requests, please use ZX Bank's designated "
    "regulatory compliance channels. For customer support, call **1800-200-9925**."
)


def is_adversarial(message: str) -> tuple[bool, str]:
    """
    Check if a message contains adversarial or sensitive content.

    Returns:
        Tuple of (is_adversarial, reason).
    """
    text = message.strip().lower()

    # Check for prompt injection markers
    injection_markers = [
        "ignore your instructions",
        "ignore all previous",
        "disregard your",
        "you are now",
        "act as if you",
        "do anything now",
        "jailbreak",
        "bypass",
    ]
    for marker in injection_markers:
        if marker in text:
            return True, "prompt_injection"

    # Check for data exfiltration attempts
    exfil_markers = [
        "customer data",
        "account details",
        "all customers",
        "database",
        "internal system",
        "admin credentials",
    ]
    for marker in exfil_markers:
        if marker in text:
            return True, "data_exfiltration"

    # Check for social engineering
    social_markers = [
        "i am an auditor",
        "i am from rbi",
        "regulatory requirement",
        "section 35a",
        "urgent compliance",
        "immediate access",
    ]
    for marker in social_markers:
        if marker in text:
            return True, "social_engineering"

    # Check sensitive pattern matches
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(message):
            return True, "sensitive_topic"

    return False, ""


def get_safe_response(reason: str) -> str:
    """Return an appropriate safe response based on the adversarial type."""
    if reason == "social_engineering":
        return _SOCIAL_ENGINEERING_REFUSAL
    return _SAFE_REFUSAL

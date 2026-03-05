"""
ZX Bank AI — Small Talk Handler.

Handles casual conversation (greetings, thanks, farewells) without
triggering the retrieval pipeline.
"""

from __future__ import annotations

from app.api.schemas import ChatResponse
from app.core.llm import SMALL_TALK_SYSTEM_PROMPT, generate
from app.utils.logger import get_logger

logger = get_logger(__name__, component="smalltalk_handler")


async def handle_small_talk(
    message: str,
    session_id: str,
    history_messages: list[dict[str, str]],
) -> ChatResponse:
    """
    Generate a friendly response to casual conversation.

    No retrieval pipeline is triggered — this goes directly to the LLM
    with the small talk system prompt.
    """
    messages = list(history_messages)
    messages.append({"role": "user", "content": message})

    llm_response = await generate(
        messages,
        system_prompt=SMALL_TALK_SYSTEM_PROMPT,
        max_tokens=256,
    )

    logger.info(
        "small_talk_response",
        message=message[:50],
        tokens_used=llm_response["usage"]["total_tokens"],
    )

    return ChatResponse(
        response=llm_response["content"],
        session_id=session_id,
        query_type="small_talk",
        confidence=0.95,
        sources=[],
    )

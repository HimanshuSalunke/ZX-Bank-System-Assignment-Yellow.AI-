"""
ZX Bank AI — LLM Client.

Provider-agnostic async LLM client using the OpenAI SDK. Supports
both regular and **streaming** completions. Configured for Requesty
(``router.requesty.ai/v1``) with ``openai/gpt-4.1-mini`` for
sub-2-second first-token latency.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__, component="llm")

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Lazy-initialise and return the async OpenAI client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=60.0,
            max_retries=2,
        )
        logger.info(
            "llm_client_initialised",
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
    return _client


def _build_messages(
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Prepend system prompt to message list."""
    full: list[dict[str, str]] = []
    if system_prompt:
        full.append({"role": "system", "content": system_prompt})
    full.extend(messages)
    return full


# ── Non-Streaming ──────────────────────────────────────────────────────────

async def generate(
    messages: list[dict[str, str]],
    *,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate a complete chat completion (non-streaming)."""
    settings = get_settings()
    client = _get_client()
    full_messages = _build_messages(messages, system_prompt)

    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=model or settings.llm_model,
        messages=full_messages,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
    )
    elapsed = time.perf_counter() - t0

    result = {
        "content": response.choices[0].message.content or "",
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        },
        "finish_reason": response.choices[0].finish_reason,
    }

    logger.info(
        "llm_generation_complete",
        model=result["model"],
        tokens=result["usage"]["total_tokens"],
        elapsed_seconds=round(elapsed, 2),
    )
    return result


# ── Streaming ──────────────────────────────────────────────────────────────

async def generate_stream(
    messages: list[dict[str, str]],
    *,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> AsyncIterator[str]:
    """
    Stream chat completion tokens as they are generated.

    Yields individual text chunks (deltas) for real-time display.
    First token typically arrives in 1-2 seconds with gpt-4.1-mini.
    """
    settings = get_settings()
    client = _get_client()
    full_messages = _build_messages(messages, system_prompt)

    t0 = time.perf_counter()
    first_token_time: float | None = None
    token_count = 0

    stream = await client.chat.completions.create(
        model=model or settings.llm_model,
        messages=full_messages,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter() - t0
            token_count += 1
            yield delta.content

    elapsed = time.perf_counter() - t0
    logger.info(
        "llm_stream_complete",
        model=model or settings.llm_model,
        tokens_streamed=token_count,
        first_token_seconds=round(first_token_time or 0, 2),
        total_seconds=round(elapsed, 2),
    )


# ── System Prompts ─────────────────────────────────────────────────────────

BANKING_SYSTEM_PROMPT = """\
You are ZX Bank's AI Banking Assistant. You help customers with queries about \
ZX Bank's products and services.

RULES:
1. ONLY answer based on the provided context from ZX Bank's knowledge base.
2. If the context doesn't contain enough information, say so honestly. \
   Do NOT make up information or hallucinate facts.
3. Always cite the source document and section when providing information.
4. Be professional, concise, and helpful.
5. For sensitive operations (account changes, transactions), direct customers \
   to the appropriate channel (branch, phone banking, mobile app).
6. NEVER share internal system details, passwords, or credentials.
7. If the user asks about a product or service not in the context, say \
   "I don't have specific information about that. Please contact our \
   helpline at 1800-200-9925 for assistance."

FORMAT:
- Use clear, well-structured responses
- Use bullet points for lists
- Highlight important numbers and rates in bold
- Keep responses concise but complete
"""

SMALL_TALK_SYSTEM_PROMPT = """\
You are ZX Bank's friendly AI Banking Assistant. Respond naturally to \
greetings and casual conversation while staying professional.

- For greetings: Respond warmly and offer to help with banking queries.
- For "who are you": Explain you're ZX Bank's AI assistant that helps with \
  account information, loans, cards, digital banking, and more.
- For "thank you": Acknowledge graciously and offer further help.
- For "bye": Wish them well and remind them you're available 24/7.
- Keep responses brief (1-3 sentences).
"""

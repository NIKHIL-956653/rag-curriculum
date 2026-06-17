"""Groq HTTP client — uses OpenAI-compatible Groq API.

Groq runs Llama 3.3 70B at 500+ tokens/sec for free.
Free tier: 14,400 requests/day, 30 requests/minute.

When we hit the per-minute rate limit, raise a special exception so the
router knows to failover to Gemini instead of just sleeping.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

from dotenv import load_dotenv
from groq import Groq, RateLimitError

# Auto-load .env so GROQ_API_KEY is available
load_dotenv()

log = logging.getLogger(__name__)


class GroqRateLimitError(Exception):
    """Raised when Groq says we're rate-limited. Router catches this to failover."""
    pass


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            )
        self.client = Groq(api_key=key)
        self.model_name = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> str:
        """Generate a completion. Raises GroqRateLimitError on rate limit."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=model or self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except RateLimitError as e:
            log.warning("Groq rate-limited: %s", e)
            raise GroqRateLimitError(str(e)) from e
        except Exception as e:
            # Check error message for rate-limit indicators
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "quota" in msg:
                log.warning("Groq rate-limited (detected from message): %s", e)
                raise GroqRateLimitError(str(e)) from e
            raise

    def stream_generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterable[str]:
        """Stream tokens one at a time."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self.client.chat.completions.create(
                model=model or self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except RateLimitError as e:
            raise GroqRateLimitError(str(e)) from e


# Module-level singleton
_default_client: GroqClient | None = None


def get_groq() -> GroqClient:
    global _default_client
    if _default_client is None:
        _default_client = GroqClient()
    return _default_client
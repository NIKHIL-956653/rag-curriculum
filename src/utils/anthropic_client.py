"""Anthropic (Claude) HTTP client — same call shape as the other backends.

Why this exists: the contextual-retrieval step makes one LLM call per child
chunk. On a local 7B that is the slowest stage in the pipeline by a wide
margin (hours to days at 1-2 concurrent requests). Claude Haiku answers the
same prompt in ~1s and tolerates 30-50 concurrent requests, which turns that
stage from "overnight" into "minutes".

Cost reference (Haiku 5.5, Oct 2026): $0.10 / Mtok in, $0.50 / Mtok out.
A ~800-token prompt + ~60-token blurb is roughly $0.00011 per chunk, so
10,000 chunks lands near $1.

The SDK retries 429s and 5xx internally with exponential backoff, so bulk
runs at high concurrency mostly self-regulate. We still raise
AnthropicRateLimitError once the SDK gives up, so LLMRouter can cool this
backend down and fail over like it does for Groq/Gemini.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

from dotenv import load_dotenv

# Auto-load .env so ANTHROPIC_API_KEY is available
load_dotenv()

log = logging.getLogger(__name__)

# Default model for bulk work. Haiku is deliberate: the contextualize task is
# "write a 2-sentence locator blurb", which does not need a frontier model,
# and Sonnet would cost ~20x for no measurable gain on this task.
_DEFAULT_MODEL = "claude-haiku-5-5"


class AnthropicRateLimitError(Exception):
    """Raised when Anthropic rate-limits us and the SDK's own retries ran out.

    LLMRouter catches this to cool the backend down and fail over.
    """

    pass


class AnthropicCreditExhaustedError(Exception):
    """Raised when the account is out of credit.

    Distinct from a rate limit on purpose: a rate limit clears in a minute,
    an empty balance does not. The router retires this backend for the rest
    of the run instead of retrying it every 65 seconds.
    """

    pass


class AnthropicClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 5,
        timeout: float = 60.0,
    ) -> None:
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Create a key at "
                "https://platform.claude.com/settings/keys and put it in .env"
            )

        # Imported lazily-ish (module level is fine) so the rest of the project
        # still runs for anyone who hasn't installed the anthropic package.
        from anthropic import Anthropic

        # One client, reused across threads. The SDK pools connections and is
        # safe to share; max_retries covers 429/5xx with exponential backoff,
        # which is what makes 40-way concurrency survivable.
        self.client = Anthropic(
            api_key=key,
            max_retries=max_retries,
            timeout=timeout,
        )
        self.model_name = model or os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODEL)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        """True for 429 and for 529 (overloaded), which behaves the same for us."""
        status = getattr(exc, "status_code", None)
        if status in (429, 529):
            return True
        msg = str(exc).lower()
        return "rate" in msg or "429" in msg or "529" in msg or "overloaded" in msg

    @staticmethod
    def _is_credit_exhausted(exc: Exception) -> bool:
        """True when the account has no credit left.

        Anthropic returns this as a 400 invalid_request_error whose message
        mentions the credit balance — not a 429 — so it must be checked
        before the rate-limit test or it falls through as a generic error.
        """
        msg = str(exc).lower()
        return (
            "credit balance" in msg
            or "insufficient credit" in msg
            or "billing" in msg
            or "quota exceeded" in msg
        )

    def _classify_and_raise(self, exc: Exception) -> None:
        """Re-raise exc as the most specific error type we recognise."""
        if self._is_credit_exhausted(exc):
            log.error("Anthropic credit exhausted: %s", exc)
            raise AnthropicCreditExhaustedError(str(exc)) from exc
        if self._is_rate_limit(exc):
            log.warning("Anthropic rate-limited: %s", exc)
            raise AnthropicRateLimitError(str(exc)) from exc
        raise exc

    # ------------------------------------------------------------------ generate
    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> str:
        """Generate a completion. Raises AnthropicRateLimitError on rate limit.

        `temperature` is accepted for interface parity with the Groq/Gemini/
        Ollama clients but NOT forwarded: the anthropic SDK removed it from
        messages.create() in the 1.x line, and passing it raises TypeError.
        """
        from anthropic import RateLimitError

        kwargs: dict = {
            "model": model or self.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Anthropic takes the system prompt as a top-level arg, not a message.
        if system:
            kwargs["system"] = system

        try:
            response = self.client.messages.create(**kwargs)
        except RateLimitError as e:
            log.warning("Anthropic rate-limited: %s", e)
            raise AnthropicRateLimitError(str(e)) from e
        except Exception as e:
            self._classify_and_raise(e)
            raise  # unreachable; keeps type checkers happy

        # content is a list of blocks; concatenate the text ones.
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    # ------------------------------------------------------------------ stream
    def stream_generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterable[str]:
        """Stream text deltas one at a time."""
        from anthropic import RateLimitError

        kwargs: dict = {
            "model": model or self.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        try:
            with self.client.messages.stream(**kwargs) as stream_resp:
                for text in stream_resp.text_stream:
                    if text:
                        yield text
        except RateLimitError as e:
            raise AnthropicRateLimitError(str(e)) from e
        except Exception as e:
            self._classify_and_raise(e)
            raise  # unreachable; keeps type checkers happy


# Module-level singleton
_default_client: AnthropicClient | None = None


def get_anthropic() -> AnthropicClient:
    global _default_client
    if _default_client is None:
        _default_client = AnthropicClient()
    return _default_client

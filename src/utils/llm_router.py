"""Routes LLM calls between Groq (primary) and Gemini (failover).

Strategy:
1. Try Groq first (fast, larger free tier, smarter model)
2. On rate limit → switch to Gemini for the next N calls
3. After cooldown → retry Groq
4. If both fail repeatedly → exponential backoff

This pattern is common in production systems with multi-vendor LLM stacks.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)


class LLMRouter:
    """Unified interface with automatic failover.
    Same call shape as OllamaClient/GeminiClient — drop-in replacement.
    """

    def __init__(self):
        self.backend_pref = os.getenv("LLM_BACKEND", "router").lower()
        self.groq_cooldown_until = 0.0  # timestamp when groq is usable again
        self.gemini_cooldown_until = 0.0  # timestamp when gemini is usable again
        self.cooldown_seconds = 65  # how long to avoid a failed backend (rate limits reset per-minute)

        # Lazy-load clients on first use
        self._groq = None
        self._gemini = None

        log.info("LLM router initialized. Preference: %s", self.backend_pref)

    # ---------- Lazy client loaders ----------
    def _get_groq(self):
        if self._groq is None:
            from src.utils.groq_client import get_groq
            self._groq = get_groq()
        return self._groq

    def _get_gemini(self):
        if self._gemini is None:
            from src.utils.gemini_client import get_gemini
            self._gemini = get_gemini()
        return self._gemini

    # ---------- Backend availability tracking ----------
    def _groq_available(self) -> bool:
        return time.time() >= self.groq_cooldown_until

    def _gemini_available(self) -> bool:
        return time.time() >= self.gemini_cooldown_until

    def _mark_groq_cooled_down(self):
        self.groq_cooldown_until = time.time() + self.cooldown_seconds
        log.warning("Groq cooling down for %ds", self.cooldown_seconds)

    def _mark_gemini_cooled_down(self):
        self.gemini_cooldown_until = time.time() + self.cooldown_seconds
        log.warning("Gemini cooling down for %ds", self.cooldown_seconds)

    # ---------- Public API ----------
    def generate(self, prompt: str, **kwargs) -> str:
        """Try Groq first, fall back to Gemini, retry with backoff if both fail."""
        # If user pinned a specific backend, honor it
        if self.backend_pref == "groq":
            return self._get_groq().generate(prompt, **kwargs)
        if self.backend_pref == "gemini":
            return self._get_gemini().generate(prompt, **kwargs)
        if self.backend_pref == "ollama":
            from src.utils.ollama_client import get_ollama
            return get_ollama().generate(prompt, **kwargs)

        # Router mode: try Groq → Gemini → backoff
        for _ in range(3):
            # 1. Try Groq if available
            if self._groq_available():
                try:
                    return self._get_groq().generate(prompt, **kwargs)
                except Exception as e:
                    from src.utils.groq_client import GroqRateLimitError
                    if isinstance(e, GroqRateLimitError):
                        self._mark_groq_cooled_down()
                    else:
                        log.error("Groq failed (non-rate-limit): %s", e)
                        self._mark_groq_cooled_down()

            # 2. Try Gemini if available
            if self._gemini_available():
                try:
                    return self._get_gemini().generate(prompt, **kwargs)
                except Exception as e:
                    msg = str(e).lower()
                    if "quota" in msg or "rate" in msg or "429" in msg:
                        self._mark_gemini_cooled_down()
                    else:
                        log.error("Gemini failed (non-rate-limit): %s", e)
                        self._mark_gemini_cooled_down()

            # 3. Both cooling down → wait it out
            wait = min(self.groq_cooldown_until, self.gemini_cooldown_until) - time.time()
            wait = max(5, int(wait))
            log.warning("Both backends cooling. Sleeping %ds...", wait)
            time.sleep(wait)

        log.error("Router failed after 3 attempts")
        return ""

    def stream_generate(self, prompt: str, **kwargs) -> Iterable[str]:
        """Streaming: use Groq if available, else Gemini. No retry loop for streams."""
        if self._groq_available():
            try:
                yield from self._get_groq().stream_generate(prompt, **kwargs)
                return
            except Exception:
                self._mark_groq_cooled_down()

        if self._gemini_available():
            yield from self._get_gemini().stream_generate(prompt, **kwargs)
            return

        log.error("No backend available for streaming")
        yield ""

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Embeddings stay local on Ollama (nomic-embed-text is fast + free)."""
        from src.utils.ollama_client import get_ollama
        return get_ollama().embed(text, model=model)


# Module singleton
_router: LLMRouter | None = None


def get_llm() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router

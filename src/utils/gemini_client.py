"""Gemini HTTP client — wraps embedding and generation calls.

Mirrors the OllamaClient interface so callers can swap backends via env flag.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Iterable

import google.generativeai as genai

from src.utils.settings import settings

log = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=key)
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._model = genai.GenerativeModel(self.model_name)

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> str:
        """Generate a completion. Returns the full text."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        m = self._model if not model else genai.GenerativeModel(model)

        # Free-tier rate limit: 15 req/min for Flash, ~4 sec safety pause
        # only if we hit a quota error
        for attempt in range(3):
            try:
                response = m.generate_content(
                    full_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text or ""
            except Exception as e:
                msg = str(e).lower()
                if "quota" in msg or "rate" in msg or "429" in msg:
                    wait = 5 * (attempt + 1)
                    log.warning("Gemini rate-limited; sleeping %ds...", wait)
                    time.sleep(wait)
                    continue
                raise
        log.error("Gemini failed after 3 retries")
        return ""

    def stream_generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterable[str]:
        """Stream tokens one at a time."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        m = self._model if not model else genai.GenerativeModel(model)
        response = m.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text


# Module-level singleton
_default_client: GeminiClient | None = None


def get_gemini() -> GeminiClient:
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client
"""Routes LLM calls to Gemini or Ollama based on LLM_BACKEND env var.

This lets us flip between cloud (Gemini, fast) and local (Ollama, free, offline)
without changing call sites.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

log = logging.getLogger(__name__)


class LLMRouter:
    """Unified interface — same shape as OllamaClient/GeminiClient."""

    def __init__(self):
        self.backend = os.getenv("LLM_BACKEND", "ollama").lower()
        log.info("LLM backend: %s", self.backend)

        if self.backend == "gemini":
            from src.utils.gemini_client import get_gemini
            self._client = get_gemini()
        else:
            from src.utils.ollama_client import get_ollama
            self._client = get_ollama()

    def generate(self, prompt: str, **kwargs) -> str:
        return self._client.generate(prompt, **kwargs)

    def stream_generate(self, prompt: str, **kwargs) -> Iterable[str]:
        return self._client.stream_generate(prompt, **kwargs)

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Embeddings always go to Ollama (nomic-embed-text is fast + local)."""
        from src.utils.ollama_client import get_ollama
        return get_ollama().embed(text, model=model)


_router: LLMRouter | None = None


def get_llm() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
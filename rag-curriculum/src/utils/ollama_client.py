"""Ollama HTTP client — wraps embedding and generation calls.

We use httpx directly rather than the ollama package so we have explicit
control over timeouts and connection pooling (important when batch-embedding
20,000 chunks).
"""
from __future__ import annotations

import logging
from typing import Iterable

import httpx

from src.utils.settings import settings

log = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.host = (host or settings.ollama_host).rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------ embeddings
    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Embed a single piece of text."""
        m = model or settings.embedding_model
        r = self._client.post(
            f"{self.host}/api/embeddings",
            json={"model": m, "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]

    def embed_batch(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Embed many texts. Ollama doesn't support true batching yet,
        so we send sequentially — but the HTTP keep-alive makes this fast
        enough for our scale (~10 min for 20K chunks on a 4060)."""
        return [self.embed(t, model=model) for t in texts]

    # ------------------------------------------------------------------ generation
    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> str:
        """Generate a completion. Returns the full text (non-streaming) or
        a concatenated string (streaming consumed internally)."""
        m = model or settings.llm_model
        payload: dict = {
            "model": m,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        if not stream:
            r = self._client.post(f"{self.host}/api/generate", json=payload)
            r.raise_for_status()
            return r.json()["response"]

        # Streaming branch — useful for the UI but for simplicity we
        # accumulate here. Streaming is exposed separately via stream_generate().
        chunks: list[str] = []
        with self._client.stream("POST", f"{self.host}/api/generate", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                import json
                obj = json.loads(line)
                chunks.append(obj.get("response", ""))
                if obj.get("done"):
                    break
        return "".join(chunks)

    def stream_generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterable[str]:
        """Stream tokens one at a time. Use this in the Streamlit UI."""
        import json
        m = model or settings.llm_model
        payload: dict = {
            "model": m,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        with self._client.stream("POST", f"{self.host}/api/generate", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                token = obj.get("response", "")
                if token:
                    yield token
                if obj.get("done"):
                    break

    def close(self) -> None:
        self._client.close()


# Module-level singleton — saves connection setup cost on every call.
_default_client: OllamaClient | None = None


def get_ollama() -> OllamaClient:
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client

"""Answer generation — strict citation-enforced prompting for Qwen 2.5.

Qwen 2.5 7B is one of the best small models at following structured
output rules. We give it a tight prompt with examples and require every
factual claim to cite a chunk by ID like [C1], [C2], etc.

We also implement a refusal pattern: if the retrieved context doesn't
contain the answer, the model says so rather than hallucinating.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable

from src.retrieve.pipeline import ContextChunk, RetrievalResult
from src.utils.db import pg_cursor
from src.utils.ollama_client import get_ollama
from src.utils.settings import settings

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are a curriculum study assistant for university students. \
You answer questions using ONLY the provided context chunks from the student's \
study materials.

STRICT RULES:
1. Every factual claim MUST cite the chunk(s) it came from, like [C1] or [C2,C3].
2. If the context does NOT contain enough information to answer, say so explicitly. \
Do NOT make up information or use outside knowledge.
3. Be concise. Prefer direct, well-organized answers. Use plain prose unless the \
question benefits from a short list.
4. Do not start with phrases like "Based on the context" or "According to the chunks". \
Just answer.
5. If the chunks contradict each other, point this out and cite both."""


def _format_context(chunks: list[ContextChunk]) -> str:
    """Format chunks as a numbered context block for the LLM."""
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"<chunk id=\"C{i}\" source=\"{c.citation}\">")
        lines.append(c.content.strip())
        lines.append("</chunk>")
        lines.append("")
    return "\n".join(lines)


def _build_prompt(query: str, chunks: list[ContextChunk]) -> str:
    if not chunks:
        return (
            f"No relevant context was found in the curriculum for this question:\n\n"
            f"Question: {query}\n\n"
            f"Respond by stating that no relevant material was found, and suggest "
            f"the student check the subject/unit selection."
        )

    return f"""<context>
{_format_context(chunks)}
</context>

Question: {query}

Answer using ONLY the chunks above. Cite every factual claim with [C#]."""


@dataclass
class GenerationResult:
    query: str
    answer: str
    chunks: list[ContextChunk]
    latency_ms: int


def generate_answer(
    retrieval: RetrievalResult,
    stream: bool = False,
) -> GenerationResult | Iterable[str]:
    """Generate a cited answer from retrieval results.

    If stream=True, returns an iterator of tokens for UI streaming.
    Otherwise returns a complete GenerationResult.
    """
    t0 = time.time()
    chunks = retrieval.context_chunks
    prompt = _build_prompt(retrieval.query, chunks)

    client = get_ollama()

    if stream:
        # Return generator that yields tokens
        return client.stream_generate(
            prompt=prompt,
            system=_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=1024,
        )

    answer = client.generate(
        prompt=prompt,
        system=_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=1024,
    )
    latency_ms = int((time.time() - t0) * 1000)
    return GenerationResult(
        query=retrieval.query,
        answer=answer.strip(),
        chunks=chunks,
        latency_ms=latency_ms,
    )


def log_query(
    user_id: int | None,
    query: str,
    subject_filter: str | None,
    retrieval: RetrievalResult,
    answer: str,
    total_latency_ms: int,
) -> None:
    """Persist the query + answer + retrieval to the audit log."""
    try:
        with pg_cursor() as cur:
            cur.execute(
                """INSERT INTO query_log
                     (user_id, query, subject_filter, retrieved_chunk_ids,
                      rerank_scores, answer, latency_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    user_id,
                    query,
                    subject_filter,
                    [c.chunk_id for c in retrieval.context_chunks],
                    [c.score for c in retrieval.context_chunks],
                    answer,
                    total_latency_ms,
                ),
            )
    except Exception as e:
        log.warning("Failed to log query: %s", e)


def answer_question(
    query: str,
    user_acl_groups: list[str] | None = None,
    subject_slug: str | None = None,
    unit_number: int | None = None,
    user_id: int | None = None,
) -> GenerationResult:
    """Top-level convenience: retrieve + generate + log."""
    from src.retrieve.pipeline import retrieve

    t0 = time.time()
    retrieval = retrieve(
        query=query,
        user_acl_groups=user_acl_groups,
        subject_slug=subject_slug,
        unit_number=unit_number,
    )
    result = generate_answer(retrieval, stream=False)
    assert isinstance(result, GenerationResult)
    total_ms = int((time.time() - t0) * 1000)
    log_query(user_id, query, subject_slug, retrieval, result.answer, total_ms)
    result.latency_ms = total_ms
    return result

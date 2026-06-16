"""Contextual retrieval — generates a 1-2 sentence context per chunk.

This is the Anthropic-published technique: prepend a small chunk-specific
context blurb explaining where this chunk sits within the broader document.
Without context, embeddings often miss the topic; with context, retrieval
accuracy jumps measurably (35-50% reduction in retrieval errors in the
original paper).

We use Qwen 2.5 7B locally — it's strong enough at this task.

Memory note: on an 8GB GPU, run this step BEFORE the embedding step (since
Qwen needs ~5.5GB VRAM; unload before loading nomic-embed).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.ingest.chunk import Chunk
from src.utils.llm_router import get_llm
from src.utils.settings import settings

log = logging.getLogger(__name__)


_CONTEXT_SYSTEM = """You write extremely short context blurbs that locate a chunk \
within a larger document. Output ONLY the blurb itself — no preamble, no \
explanation, no quotes. Maximum 2 sentences, 50 words."""

_CONTEXT_TEMPLATE = """<document_summary>
Subject: {subject_name}
Unit: {unit_number}
Section: {section}
</document_summary>

<chunk>
{chunk_text}
</chunk>

Write a 1-2 sentence context blurb (max 50 words) that situates this chunk \
within the document. State what the chunk is about and how it fits into the \
section/unit. Do not summarise the chunk content itself; situate it."""


def _build_prompt(chunk: Chunk, subject_name: str) -> str:
    return _CONTEXT_TEMPLATE.format(
        subject_name=subject_name,
        unit_number=chunk.unit_number,
        section=chunk.section_path or "(unspecified)",
        chunk_text=chunk.content[:2000],   # cap to keep prompt small
    )


def _contextualize_one(chunk: Chunk, subject_name: str) -> str:
    """Generate context for one chunk; returns the context string."""
    client = get_llm()
    try:
        result = client.generate(
            prompt=_build_prompt(chunk, subject_name),
            system=_CONTEXT_SYSTEM,
            temperature=0.1,
            max_tokens=120,
        )
        # Clean up the response
        result = result.strip().strip('"').strip("'")
        # Truncate to ~50 words as a safety net
        words = result.split()
        if len(words) > 60:
            result = " ".join(words[:60])
        return result
    except Exception as e:
        log.warning("Context generation failed for %s: %s", chunk.chunk_id, e)
        return ""


def contextualize_chunks(
    chunks: list[Chunk],
    subject_name: str,
    only_children: bool = True,
    max_workers: int = 2,
) -> list[Chunk]:
    """Add context_prefix to chunks in-place.

    Args:
        chunks: chunks to contextualize.
        subject_name: human-readable subject name for the prompt.
        only_children: if True, only contextualize child chunks (which are
            what gets embedded). Parents are retrieved by ID, not by similarity,
            so they don't need context. This halves the cost.
        max_workers: concurrent Ollama requests. Keep low (1-2) — Ollama
            queues requests internally and high concurrency just wastes RAM.
    """
    if not settings.enable_contextual_retrieval:
        log.info("Contextual retrieval disabled; skipping.")
        return chunks

    targets = [c for c in chunks if (c.chunk_type == "child" or not only_children)]
    if not targets:
        return chunks

    log.info("Contextualizing %d chunks with %s...", len(targets), settings.llm_model)

    # Sequential is fine on a single GPU; threads help slightly with HTTP overhead.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_chunk = {
            pool.submit(_contextualize_one, c, subject_name): c
            for c in targets
        }
        for fut in tqdm(as_completed(future_to_chunk), total=len(targets), desc="contextualize"):
            chunk = future_to_chunk[fut]
            chunk.context_prefix = fut.result()

    return chunks

"""Cross-encoder reranker — refines top-K from hybrid retrieval.

Bi-encoders (embeddings) are fast but coarse; they compare embeddings
independently. Cross-encoders look at (query, document) together so they
catch nuance bi-encoders miss. Cost: slower (~50ms-300ms per pair on CPU),
so we only rerank the top 30-50 from hybrid search.

BGE Reranker v2-m3: free, open source, small (~300MB), strong cross-encoder.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.retrieve.hybrid import RetrievedHit
from src.utils.db import pg_cursor
from src.utils.settings import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    log.info("Loading reranker model: %s on %s", settings.reranker_model, settings.reranker_device)
    return CrossEncoder(settings.reranker_model, device=settings.reranker_device)


def _fetch_chunk_texts(chunk_ids: list[str]) -> dict[str, str]:
    """Fetch content (with context prefix) for a batch of chunks from Postgres."""
    if not chunk_ids:
        return {}
    with pg_cursor(dict_rows=True) as cur:
        cur.execute(
            """SELECT chunk_id, content, context_prefix
               FROM chunks WHERE chunk_id = ANY(%s)""",
            (chunk_ids,),
        )
        rows = cur.fetchall()
    out: dict[str, str] = {}
    for r in rows:
        prefix = r["context_prefix"] or ""
        content = r["content"]
        out[r["chunk_id"]] = f"{prefix}\n\n{content}".strip()
    return out


def rerank(
    query: str,
    hits: list[RetrievedHit],
    top_k: int | None = None,
) -> list[RetrievedHit]:
    """Score (query, chunk) pairs and return top_k by reranker score."""
    if not hits:
        return hits
    if not settings.enable_rerank:
        return hits[: (top_k or settings.top_k_rerank)]

    top_k = top_k or settings.top_k_rerank

    # Fetch text for all candidate chunks
    texts_by_id = _fetch_chunk_texts([h.chunk_id for h in hits])
    pairs = []
    valid_hits: list[RetrievedHit] = []
    for h in hits:
        text = texts_by_id.get(h.chunk_id)
        if not text:
            continue
        pairs.append((query, text))
        valid_hits.append(h)

    if not pairs:
        return []

    reranker = _get_reranker()
    scores = reranker.predict(pairs, show_progress_bar=False)

    # Pair scores with hits; write into hit.score (overwriting RRF score)
    for hit, score in zip(valid_hits, scores):
        hit.score = float(score)

    reranked = sorted(valid_hits, key=lambda h: h.score, reverse=True)
    log.info("Reranked %d → top %d", len(valid_hits), top_k)
    return reranked[:top_k]

"""Hybrid retrieval — dense + sparse + RRF fusion.

Qdrant runs both dense (nomic-embed) and sparse (BM25-like) searches in a
single call. We run multiple queries (from multi-query expansion) and fuse
all the ranked lists with Reciprocal Rank Fusion.

ACL filtering: every query MUST be filtered by the user's ACL groups —
this is the multi-tenancy pattern.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    SparseVector,
)

from src.ingest.embed_index import _build_sparse_vector
from src.utils.db import get_qdrant
from src.utils.ollama_client import get_ollama
from src.utils.settings import settings

log = logging.getLogger(__name__)


@dataclass
class RetrievedHit:
    chunk_id: str
    parent_chunk_id: str | None
    subject_slug: str
    unit_number: int
    page_number: int
    section_path: str | None
    score: float


def _build_filter(
    user_acl_groups: list[str],
    subject_slug: str | None = None,
    unit_number: int | None = None,
) -> Filter:
    """Always filter to child chunks + user's ACL; optionally narrow further."""
    must: list = [
        FieldCondition(key="chunk_type", match=MatchValue(value="child")),
        FieldCondition(key="acl_groups", match=MatchAny(any=user_acl_groups)),
    ]
    if subject_slug:
        must.append(FieldCondition(key="subject_slug", match=MatchValue(value=subject_slug)))
    if unit_number is not None:
        must.append(FieldCondition(key="unit_number", match=MatchValue(value=unit_number)))
    return Filter(must=must)


def _single_hybrid_search(
    query_text: str,
    user_acl_groups: list[str],
    subject_slug: str | None,
    unit_number: int | None,
    top_k: int,
) -> tuple[list[RetrievedHit], list[RetrievedHit]]:
    """Return (dense_hits, sparse_hits) for one query."""
    qdrant = get_qdrant()
    client = get_ollama()

    dense_vec = client.embed(query_text)
    sparse_vec = _build_sparse_vector(query_text)

    flt = _build_filter(user_acl_groups, subject_slug, unit_number)

    # Dense
    dense_resp = qdrant.query_points(
        collection_name=settings.qdrant_collection,
        query=dense_vec,
        using="dense",
        query_filter=flt,
        limit=top_k,
        with_payload=True,
    ).points

    # Sparse (BM25-like)
    sparse_resp = qdrant.query_points(
        collection_name=settings.qdrant_collection,
        query=SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
        using="bm25",
        query_filter=flt,
        limit=top_k,
        with_payload=True,
    ).points

    def to_hit(p) -> RetrievedHit:
        pl = p.payload or {}
        return RetrievedHit(
            chunk_id=pl.get("chunk_id", str(p.id)),
            parent_chunk_id=pl.get("parent_chunk_id"),
            subject_slug=pl.get("subject_slug", ""),
            unit_number=pl.get("unit_number", 0),
            page_number=pl.get("page_number", 0),
            section_path=pl.get("section_path"),
            score=float(p.score),
        )

    return [to_hit(p) for p in dense_resp], [to_hit(p) for p in sparse_resp]


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedHit]],
    k: int = 60,
) -> list[RetrievedHit]:
    """Standard RRF: score(d) = sum over lists of 1 / (k + rank).

    k=60 is the value from the original RRF paper; small enough that top-rank
    docs strongly dominate, large enough that rank-50 docs still contribute.
    """
    scored: dict[str, tuple[float, RetrievedHit]] = {}
    for hit_list in ranked_lists:
        for rank, hit in enumerate(hit_list, start=1):
            contribution = 1.0 / (k + rank)
            if hit.chunk_id in scored:
                cur_score, cur_hit = scored[hit.chunk_id]
                scored[hit.chunk_id] = (cur_score + contribution, cur_hit)
            else:
                # Store with new score, keep first-seen object
                new_hit = RetrievedHit(**{**hit.__dict__, "score": contribution})
                scored[hit.chunk_id] = (contribution, new_hit)

    # Sort by accumulated score, write back as final score
    ordered = sorted(scored.values(), key=lambda x: x[0], reverse=True)
    result: list[RetrievedHit] = []
    for score, hit in ordered:
        hit.score = score
        result.append(hit)
    return result


def hybrid_retrieve(
    queries: list[str],
    user_acl_groups: list[str],
    subject_slug: str | None = None,
    unit_number: int | None = None,
    top_k: int | None = None,
) -> list[RetrievedHit]:
    """Run hybrid search across multiple query variants and fuse with RRF."""
    top_k = top_k or settings.top_k_dense
    all_lists: list[list[RetrievedHit]] = []
    for q in queries:
        dense, sparse = _single_hybrid_search(
            q, user_acl_groups, subject_slug, unit_number, top_k=top_k,
        )
        if dense:
            all_lists.append(dense)
        if sparse:
            all_lists.append(sparse)

    if not all_lists:
        return []

    fused = reciprocal_rank_fusion(all_lists)
    log.info(
        "Hybrid retrieve: %d queries × 2 (dense+sparse) → %d unique hits",
        len(queries), len(fused),
    )
    return fused

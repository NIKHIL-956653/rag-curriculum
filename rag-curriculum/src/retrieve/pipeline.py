"""End-to-end retrieval pipeline orchestrator.

Flow:
  query
   → multi-query expansion (3 variants)
   → hybrid search (dense + BM25, ACL-filtered)  per variant
   → RRF merge across all lists
   → cross-encoder rerank top-50 → top-5
   → parent chunk expansion (children → parents)
   → ContextChunk list for the generator
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.retrieve.hybrid import RetrievedHit, hybrid_retrieve
from src.retrieve.multi_query import expand_query
from src.retrieve.rerank import rerank
from src.utils.db import pg_cursor
from src.utils.settings import settings

log = logging.getLogger(__name__)


@dataclass
class ContextChunk:
    """A parent chunk ready for the LLM's prompt, with its citation metadata."""
    chunk_id: str            # the PARENT chunk id
    matched_child_id: str    # which child chunk matched the query
    subject_slug: str
    unit_number: int
    page_number: int
    section_path: str | None
    score: float             # rerank score
    content: str             # full parent chunk text

    @property
    def citation(self) -> str:
        sec = f" › {self.section_path}" if self.section_path else ""
        return f"[{self.subject_slug} › Unit {self.unit_number}{sec} › Page {self.page_number}]"


@dataclass
class RetrievalResult:
    query: str
    expanded_queries: list[str]
    raw_hit_count: int
    reranked_hits: list[RetrievedHit]
    context_chunks: list[ContextChunk]
    latency_ms: int


def _expand_to_parents(hits: list[RetrievedHit]) -> list[ContextChunk]:
    """Resolve each child hit to its parent chunk. Dedupes — multiple children
    pointing to the same parent become a single context chunk."""
    if not hits:
        return []

    # Map: parent_id -> first-matching child hit
    parent_to_child: dict[str, RetrievedHit] = {}
    for h in hits:
        pid = h.parent_chunk_id or h.chunk_id
        if pid not in parent_to_child:
            parent_to_child[pid] = h

    parent_ids = list(parent_to_child.keys())
    with pg_cursor(dict_rows=True) as cur:
        cur.execute(
            """SELECT chunk_id, content, context_prefix,
                      subject_slug, unit_number, page_number, section_path
               FROM chunks WHERE chunk_id = ANY(%s)""",
            (parent_ids,),
        )
        rows = cur.fetchall()

    rows_by_id = {r["chunk_id"]: r for r in rows}

    out: list[ContextChunk] = []
    for pid in parent_ids:
        row = rows_by_id.get(pid)
        if not row:
            log.warning("Parent chunk not found in Postgres: %s", pid)
            continue
        child = parent_to_child[pid]
        out.append(ContextChunk(
            chunk_id=pid,
            matched_child_id=child.chunk_id,
            subject_slug=row["subject_slug"],
            unit_number=row["unit_number"],
            page_number=row["page_number"],
            section_path=row["section_path"],
            score=child.score,
            content=row["content"],
        ))
    # Sort by rerank score
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def retrieve(
    query: str,
    user_acl_groups: list[str] | None = None,
    subject_slug: str | None = None,
    unit_number: int | None = None,
) -> RetrievalResult:
    """End-to-end retrieval for one query."""
    t0 = time.time()
    user_acl_groups = user_acl_groups or ["all_students"]

    # 1. Multi-query expansion
    queries = expand_query(query)

    # 2. Hybrid search across all query variants → RRF
    hits = hybrid_retrieve(
        queries=queries,
        user_acl_groups=user_acl_groups,
        subject_slug=subject_slug,
        unit_number=unit_number,
    )

    # Cap to top-50 before reranking (rerank is the most expensive step)
    pre_rerank = hits[:50]

    # 3. Rerank
    reranked = rerank(query, pre_rerank, top_k=settings.top_k_rerank)

    # 4. Parent expansion → final context chunks
    context_chunks = _expand_to_parents(reranked)

    latency_ms = int((time.time() - t0) * 1000)
    log.info("Retrieve: %dms (q=%r)", latency_ms, query[:60])

    return RetrievalResult(
        query=query,
        expanded_queries=queries,
        raw_hit_count=len(hits),
        reranked_hits=reranked,
        context_chunks=context_chunks,
        latency_ms=latency_ms,
    )

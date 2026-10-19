"""Embed chunks and push to Qdrant (dense + sparse vectors) + Postgres metadata.

We embed only CHILD chunks (those are used for retrieval). Parent chunks
are retrieved by ID and live only in Postgres.

Sparse vectors (BM25-like) are computed via simple TF over the chunk text
plus context prefix. Qdrant handles the sparse-vector search natively.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from typing import Iterable

from qdrant_client.models import PointStruct, SparseVector
from tqdm import tqdm

from src.ingest.chunk import Chunk
from src.utils.db import ensure_collection, get_qdrant, pg_cursor
from src.utils.ollama_client import get_ollama
from src.utils.settings import settings

log = logging.getLogger(__name__)


# -------------------------------------------------------------------- sparse vec
# Very small vocabulary; for production swap to a learned sparse encoder (SPLADE)
# or rely on Elasticsearch BM25. This is intentionally simple — it works and
# pairs nicely with dense for the hybrid search.
_TOKEN_RX = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
_STOPWORDS = set("""
a an and are as at be by for from has he in is it its of on or that the to was were will with
this these those there their they them then than which who whom whose what when where why how
""".split())


def _tokenize_for_bm25(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RX.findall(text) if t.lower() not in _STOPWORDS and len(t) > 2]


def _term_to_id(term: str) -> int:
    """Hash term to a stable 32-bit ID. Collisions are tolerable for our scale.

    MUST NOT use Python's built-in hash(): PYTHONHASHSEED randomises str
    hashing per process, so each ingest run would write a DIFFERENT term->id
    mapping into the same Qdrant collection and the sparse (BM25) half of
    hybrid search would compare incompatible vocabularies across subjects.
    blake2b is stable across processes, machines and Python versions.
    """
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _build_sparse_vector(text: str) -> SparseVector:
    tokens = _tokenize_for_bm25(text)
    if not tokens:
        return SparseVector(indices=[], values=[])
    counts = Counter(tokens)
    total = sum(counts.values())
    indices: list[int] = []
    values: list[float] = []
    seen: set[int] = set()
    for term, n in counts.items():
        tid = _term_to_id(term)
        if tid in seen:
            continue
        seen.add(tid)
        indices.append(tid)
        # Simple normalised TF; Qdrant ranks by dot product so this works as a
        # cheap BM25 stand-in for the hybrid step.
        values.append(n / total)
    return SparseVector(indices=indices, values=values)


# -------------------------------------------------------------------- Postgres
def _ensure_subject(slug: str, name: str, content_type: str, parser: str, acl_groups: list[str]) -> None:
    """Upsert a subject row."""
    with pg_cursor() as cur:
        cur.execute(
            """INSERT INTO subjects (slug, name, content_type, parser, acl_groups)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (slug) DO UPDATE
                 SET name = EXCLUDED.name,
                     content_type = EXCLUDED.content_type,
                     parser = EXCLUDED.parser,
                     acl_groups = EXCLUDED.acl_groups""",
            (slug, name, content_type, parser, acl_groups),
        )


def _ensure_document(subject_slug: str, unit_number: int, file_path: str, file_hash: str, page_count: int) -> int:
    """Insert document row if new, return its ID. Returns existing ID if file_hash matches."""
    with pg_cursor() as cur:
        cur.execute(
            """SELECT id FROM documents
               WHERE subject_slug=%s AND unit_number=%s AND file_hash=%s""",
            (subject_slug, unit_number, file_hash),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            """INSERT INTO documents (subject_slug, unit_number, file_path, file_hash, page_count)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (subject_slug, unit_number, file_path, file_hash, page_count),
        )
        return cur.fetchone()[0]


def _persist_chunks_pg(chunks: list[Chunk]) -> None:
    """Bulk-insert chunks into Postgres. Idempotent on chunk_id."""
    if not chunks:
        return
    rows = [
        (
            c.chunk_id, c.document_id, c.parent_chunk_id, c.subject_slug,
            c.unit_number, c.page_number, c.section_path, c.chunk_type,
            c.token_count, c.content, c.context_prefix, c.acl_groups,
        )
        for c in chunks
    ]
    with pg_cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks
                 (chunk_id, document_id, parent_chunk_id, subject_slug, unit_number,
                  page_number, section_path, chunk_type, token_count, content,
                  context_prefix, acl_groups)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (chunk_id) DO UPDATE SET
                 content = EXCLUDED.content,
                 context_prefix = EXCLUDED.context_prefix,
                 token_count = EXCLUDED.token_count""",
            rows,
        )


# -------------------------------------------------------------------- Qdrant push
def _embed_and_index_children(
    children: list[Chunk],
    collection: str,
    batch_size: int = 64,
) -> None:
    """Embed children with Ollama (nomic-embed-text) and push to Qdrant.

    Sparse vectors are computed locally from chunk text — no API call.
    """
    if not children:
        return

    client = get_ollama()
    qdrant = get_qdrant()

    points_buffer: list[PointStruct] = []
    for chunk in tqdm(children, desc="embed"):
        try:
            dense_vec = client.embed(chunk.embed_text)
        except Exception as e:
            log.error("Embedding failed for %s: %s", chunk.chunk_id, e)
            continue
        sparse_vec = _build_sparse_vector(chunk.embed_text)

        # Qdrant point ID must be uint or UUID. Same rule as _term_to_id:
        # built-in hash() is per-process random, so re-ingesting a document
        # would upsert to a NEW point instead of replacing the old one,
        # leaving duplicates in the collection.
        point_id = int.from_bytes(
            hashlib.blake2b(chunk.chunk_id.encode("utf-8"), digest_size=8).digest(),
            "big",
        ) % (2**63 - 1)

        points_buffer.append(PointStruct(
            id=point_id,
            vector={
                "dense": dense_vec,
                "bm25": sparse_vec,
            },
            payload={
                "chunk_id": chunk.chunk_id,
                "parent_chunk_id": chunk.parent_chunk_id,
                "subject_slug": chunk.subject_slug,
                "unit_number": chunk.unit_number,
                "page_number": chunk.page_number,
                "section_path": chunk.section_path,
                "chunk_type": chunk.chunk_type,
                "acl_groups": chunk.acl_groups,
            },
        ))

        if len(points_buffer) >= batch_size:
            qdrant.upsert(collection_name=collection, points=points_buffer)
            points_buffer = []

    if points_buffer:
        qdrant.upsert(collection_name=collection, points=points_buffer)


# -------------------------------------------------------------------- public API
def index_chunks(
    chunks: list[Chunk],
    subject_slug: str,
    subject_name: str,
    content_type: str,
    parser: str,
    acl_groups: list[str],
    document_path: str,
    document_hash: str,
    document_unit: int,
    document_page_count: int,
) -> None:
    """Full persistence: subject + document + chunks → Postgres + Qdrant."""
    ensure_collection()

    _ensure_subject(subject_slug, subject_name, content_type, parser, acl_groups)
    doc_id = _ensure_document(
        subject_slug, document_unit, document_path, document_hash, document_page_count,
    )

    # Stamp document_id on chunks before persisting
    for c in chunks:
        c.document_id = doc_id

    _persist_chunks_pg(chunks)

    # Only children get embedded
    children = [c for c in chunks if c.chunk_type == "child"]
    _embed_and_index_children(children, collection=settings.qdrant_collection)

    log.info(
        "Indexed %d chunks (%d children embedded) for %s u%d",
        len(chunks), len(children), subject_slug, document_unit,
    )

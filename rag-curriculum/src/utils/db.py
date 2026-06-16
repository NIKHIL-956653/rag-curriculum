"""Database client helpers — Postgres (metadata, ACLs, audit) + Qdrant (vectors)."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVectorParams,
    VectorParams,
    SparseIndexParams,
)

from src.utils.settings import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- Postgres
@contextmanager
def pg_conn() -> Iterator[psycopg2.extensions.connection]:
    """Yield a Postgres connection with autocommit off; caller commits."""
    conn = psycopg2.connect(settings.postgres_url)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def pg_cursor(dict_rows: bool = False) -> Iterator[psycopg2.extensions.cursor]:
    """Yield a cursor; commits on success, rolls back on error."""
    with pg_conn() as conn:
        cursor_factory = RealDictCursor if dict_rows else None
        cur = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


# ---------------------------------------------------------------------- Qdrant
_qdrant_client: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    """Lazy-init Qdrant client singleton."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    return _qdrant_client


def ensure_collection(collection_name: str | None = None) -> None:
    """Create the Qdrant collection with both dense and sparse vectors.

    Sparse vectors hold BM25-style term-frequency representations; Qdrant
    handles them natively so we get hybrid search through one query.
    """
    name = collection_name or settings.qdrant_collection
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        log.info("Collection '%s' already exists; skipping creation.", name)
        return

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )
    # Payload indexes — make metadata filtering fast
    for field in ("subject_slug", "unit_number", "chunk_type", "acl_groups"):
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema="keyword" if field != "unit_number" else "integer",
        )
    log.info("Created collection '%s' with dense+sparse vectors.", name)

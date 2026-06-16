"""Multi-query expansion — generate paraphrases of the user's query.

Why: a single phrasing often misses semantically similar chunks. Generating
3 variants and searching with each (then merging via RRF) gives a 15-30%
recall lift on hard queries.

Uses Qwen 2.5 7B locally. Cached in Redis so we don't re-generate for
repeated queries.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re

import redis

from src.utils.llm_router import get_llm
from src.utils.settings import settings

log = logging.getLogger(__name__)

_redis_client: redis.Redis | None = None


def _redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
    return _redis_client


_MULTI_QUERY_SYSTEM = """You generate alternative phrasings of a user's question \
to improve search recall. Output ONLY a JSON array of strings, nothing else. \
No preamble, no explanation, no markdown fences."""

_MULTI_QUERY_TEMPLATE = """Generate {n} alternative phrasings of this question. \
Vary the vocabulary and angle while keeping the meaning identical. The original \
question should NOT be in the output.

Question: {query}

Output a JSON array of {n} strings."""


def _cache_key(query: str, n: int) -> str:
    h = hashlib.sha256(f"{query}|{n}".encode()).hexdigest()[:16]
    return f"mq:{h}"


def _parse_variants(raw: str, expected: int) -> list[str]:
    """Best-effort parse of the model's JSON output."""
    raw = raw.strip()
    # Strip markdown fence if present
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M).strip()
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(s).strip() for s in arr if str(s).strip()][:expected]
    except json.JSONDecodeError:
        pass
    # Fallback — extract quoted lines
    lines = re.findall(r'"([^"]{5,200})"', raw)
    return lines[:expected]


def expand_query(query: str, n: int | None = None) -> list[str]:
    """Return [original_query, *variants]. Always returns at least the original."""
    n = n or settings.multi_query_variants
    if not settings.enable_multi_query or n <= 0:
        return [query]

    # Check cache
    key = _cache_key(query, n)
    try:
        cached = _redis().get(key)
        if cached:
            variants = json.loads(cached)
            return [query, *variants]
    except Exception as e:
        log.debug("Redis unavailable (%s); skipping cache.", e)

    client = get_llm()
    try:
        raw = client.generate(
            prompt=_MULTI_QUERY_TEMPLATE.format(n=n, query=query),
            system=_MULTI_QUERY_SYSTEM,
            temperature=0.5,
            max_tokens=300,
        )
        variants = _parse_variants(raw, expected=n)
    except Exception as e:
        log.warning("Multi-query expansion failed: %s", e)
        variants = []

    if not variants:
        return [query]

    # Cache (1 hour TTL — queries repeat)
    try:
        _redis().setex(key, 3600, json.dumps(variants))
    except Exception:
        pass

    return [query, *variants]

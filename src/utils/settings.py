"""Centralised settings loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    # Paths
    root_dir: Path = ROOT_DIR
    data_raw_dir: Path = ROOT_DIR / "data" / "raw"
    data_processed_dir: Path = ROOT_DIR / "data" / "processed"
    configs_dir: Path = ROOT_DIR / "configs"

    # Postgres
    postgres_host: str = _env("POSTGRES_HOST", "localhost")
    postgres_port: int = _env_int("POSTGRES_PORT", 5432)
    postgres_user: str = _env("POSTGRES_USER", "rag")
    postgres_password: str = _env("POSTGRES_PASSWORD", "rag_dev_password")
    postgres_db: str = _env("POSTGRES_DB", "rag_curriculum")

    # Qdrant
    qdrant_host: str = _env("QDRANT_HOST", "localhost")
    qdrant_port: int = _env_int("QDRANT_PORT", 6333)
    qdrant_collection: str = _env("QDRANT_COLLECTION", "curriculum")

    # Redis
    redis_host: str = _env("REDIS_HOST", "localhost")
    redis_port: int = _env_int("REDIS_PORT", 6379)

    # Ollama
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = _env("LLM_MODEL", "qwen2.5:7b")
    embedding_model: str = _env("EMBEDDING_MODEL", "nomic-embed-text")
    embedding_dim: int = _env_int("EMBEDDING_DIM", 768)

    # Anthropic (optional paid backend — used for the bulk contextualize pass)
    anthropic_model: str = _env("ANTHROPIC_MODEL", "claude-haiku-5-5")

    # Ingest concurrency. Local Ollama wants 1-2; a hosted API wants 30-50.
    # Raise this when LLM_BACKEND=anthropic or contextualize stays slow.
    contextualize_workers: int = _env_int("CONTEXTUALIZE_WORKERS", 2)

    # Reranker
    reranker_model: str = _env("RERANKER_MODEL", "BAAI/bge-reranker-base")
    reranker_device: str = _env("RERANKER_DEVICE", "cpu")

    # Retrieval
    top_k_dense: int = _env_int("TOP_K_DENSE", 30)
    top_k_bm25: int = _env_int("TOP_K_BM25", 30)
    top_k_rerank: int = _env_int("TOP_K_RERANK", 5)
    multi_query_variants: int = _env_int("MULTI_QUERY_VARIANTS", 3)

    # Behaviour flags
    enable_contextual_retrieval: bool = _env_bool("ENABLE_CONTEXTUAL_RETRIEVAL", True)
    enable_multi_query: bool = _env_bool("ENABLE_MULTI_QUERY", True)
    enable_rerank: bool = _env_bool("ENABLE_RERANK", True)

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Module-level singleton
settings = Settings()

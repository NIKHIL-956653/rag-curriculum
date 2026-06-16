# Enterprise RAG Pipeline — 18-Subject Curriculum

A fully local, production-architecture RAG system over an 18-subject curriculum (~6,800 pages, ~20K chunks). Zero external API dependencies, designed to demonstrate the enterprise RAG patterns that scale from this laptop to a distributed cluster.

## Why this project

Most RAG tutorials show LangChain `from_documents()` and call it done. This repo demonstrates the patterns that actually matter at enterprise scale:

- **Hybrid retrieval** (BM25 + dense + RRF), not just vector search
- **Parent-child chunking** for precise retrieval with rich context
- **Contextual retrieval** (Anthropic's technique) — adds chunk-level context before embedding
- **Multi-query expansion** — generates semantic variants of every query
- **Cross-encoder reranking** — top-50 → top-5 with a real reranker
- **Per-chunk ACL metadata** — multi-tenancy pattern, even with one user
- **Continuous evaluation** — RAGAS metrics on every code change
- **Observability** — every retrieval/generation traced end-to-end

The pipeline is identical to what you'd run on millions of documents; only the scale differs.

## Architecture

```
PDFs → Parse → Chunk (P-C) → Contextualize → Embed → Index
                                                        ↓
Query → Multi-query → Hybrid (Dense + BM25) → RRF → Rerank → Parent Expand
                                                                    ↓
                                                              Generate (cited)
```

See `docs/architecture.md` for the full diagram and tradeoffs discussion.

## Stack

| Layer | Tool | Why |
|-------|------|-----|
| Parsing | pymupdf4llm, pdfplumber | Best free PDF parsers; pdfplumber for tables |
| Chunking | Custom recursive + parent-child | Standard splitters don't preserve hierarchy |
| Contextual retrieval | Qwen 2.5 7B via Ollama | Generates chunk-level context, local |
| Embeddings | nomic-embed-text via Ollama | 768-dim, 8K context, beats text-embedding-3-small |
| Vector DB | Qdrant | Native hybrid search, Docker, scales to cluster |
| BM25 | Qdrant native sparse vectors | Same API as dense, simpler ops |
| Reranker | BAAI/bge-reranker-base | Free, CPU, strong cross-encoder |
| Generation | Qwen 2.5 7B via Ollama | Strong instruction following, citations |
| Metadata + ACL | PostgreSQL | Standard enterprise pattern |
| UI | Streamlit | Fast iteration; would be Next.js in prod |
| Evaluation | RAGAS | Standard RAG eval suite |
| Tracing | Langfuse (self-hosted) | Open source, runs in Docker |

## Quickstart

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Pull Ollama models (if not already)
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 3. Install Python deps
pip install -r requirements.txt

# 4. Drop PDFs into data/raw/<subject_name>/
# (Matching the slugs in configs/subjects.yaml)

# 5. Ingest a single subject (pilot)
python -m src.ingest.run --subject programming_in_data_science

# 6. Launch UI
streamlit run app/streamlit_app.py
```

## Project structure

```
rag-curriculum/
├── docker-compose.yml          # Qdrant, Postgres, Redis, Langfuse
├── requirements.txt
├── configs/
│   └── subjects.yaml           # 18 subjects → folders, content-type, ACL
├── src/
│   ├── ingest/
│   │   ├── parse.py            # PDF → text + metadata
│   │   ├── chunk.py            # Recursive + parent-child
│   │   ├── contextualize.py    # Qwen generates chunk context
│   │   ├── embed_index.py      # nomic-embed → Qdrant
│   │   └── run.py              # Orchestrates the full ingest
│   ├── retrieve/
│   │   ├── multi_query.py      # Generate query variants
│   │   ├── hybrid.py           # Dense + BM25 via Qdrant
│   │   ├── rerank.py           # BGE cross-encoder
│   │   └── pipeline.py         # Full retrieval orchestration
│   ├── generate/
│   │   └── answer.py           # Prompt + Qwen + citations
│   ├── eval/
│   │   ├── eval_set.jsonl      # Hand-curated Q/A pairs
│   │   └── run_eval.py         # RAGAS metrics
│   └── utils/
│       ├── ollama_client.py    # Lifecycle-aware Ollama wrapper
│       └── db.py               # Postgres + Qdrant clients
├── app/
│   └── streamlit_app.py        # UI: subject picker, citations, eval scores
├── scripts/
│   ├── run_ingest.sh
│   └── run_eval.sh
└── tests/
```

## Tradeoffs made

- **Qwen 2.5 7B instead of Claude Sonnet**: 90% of the quality, $0 cost, fully local. For production, swap `src/generate/answer.py` to call Anthropic API — that's it.
- **Single-node Qdrant instead of cluster**: 20K chunks is trivial; cluster would be over-engineering. The Qdrant API is identical across both, so migration is zero-cost.
- **BGE reranker on CPU**: Doesn't fight 8GB GPU memory with Qwen + nomic. ~300ms latency vs <100ms on GPU — acceptable for demo.
- **Skip Langfuse for v1**: Optional. Adds observability complexity; enable later when needed.

## How this scales

| Component | Now (laptop) | Enterprise (10M+ docs) |
|-----------|--------------|------------------------|
| Vector DB | Qdrant single node | Qdrant cluster, sharded |
| Index | HNSW (full memory) | IVF-PQ or DiskANN |
| Embedding | Ollama local | Voyage/Cohere batch API or GPU pool |
| LLM | Qwen 2.5 7B local | Claude Sonnet / Llama 70B on GPU |
| Ingestion | Python script | Kafka + worker pool on k8s |
| Metadata | Postgres single | Postgres cluster + read replicas |
| Caching | Redis local | Redis cluster |
| Observability | Langfuse single | Langfuse + Datadog + Prometheus |

The application code stays identical; only the infrastructure changes.

## Eval results (pilot: Programming in Data Science)

Target: Recall@5 ≥ 0.85, MRR ≥ 0.70

(Will populate after first run)

## License

MIT

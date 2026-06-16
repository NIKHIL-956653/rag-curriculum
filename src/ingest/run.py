"""Ingestion orchestrator — CLI to ingest one or all subjects.

Usage:
    python -m src.ingest.run --subject programming_in_data_science
    python -m src.ingest.run --subject all
    python -m src.ingest.run --subject all --skip-contextualize
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.logging import RichHandler

from src.ingest.chunk import chunk_document
from src.ingest.contextualize import contextualize_chunks
from src.ingest.embed_index import index_chunks
from src.ingest.parse import parse_subject_folder
from src.utils.settings import settings

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)],
)
log = logging.getLogger("ingest")


def load_subjects_config() -> tuple[dict, list[dict]]:
    cfg_path = settings.configs_dir / "subjects.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    defaults = cfg.get("defaults", {})
    subjects = cfg.get("subjects", [])
    # Merge defaults into each subject
    merged = []
    for s in subjects:
        m = {**defaults, **s}
        merged.append(m)
    return defaults, merged


def ingest_subject(subject_cfg: dict, skip_contextualize: bool = False) -> None:
    slug = subject_cfg["slug"]
    name = subject_cfg["name"]
    folder = settings.data_raw_dir / slug

    console.rule(f"[bold cyan]Ingesting: {name}[/bold cyan]")

    if not folder.exists():
        log.warning("Folder not found: %s — skipping", folder)
        console.print(f"[yellow]Folder not found: {folder}[/yellow]")
        return

    parser = subject_cfg.get("parser", "default")
    content_type = subject_cfg.get("content_type", "unknown")
    acl_groups = subject_cfg.get("acl_groups", ["all_students"])
    child_size = subject_cfg.get("chunk_size", 512)
    parent_size = subject_cfg.get("parent_chunk_size", 1536)
    overlap = subject_cfg.get("chunk_overlap", 100)

    # 1. Parse
    docs = parse_subject_folder(folder, parser_strategy=parser)
    if not docs:
        console.print(f"[yellow]No PDFs found in {folder}[/yellow]")
        return
    console.print(f"  [green]✓[/green] Parsed {len(docs)} PDFs")

    # 2. Chunk
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(
            doc,
            subject_slug=slug,
            child_size=child_size,
            parent_size=parent_size,
            overlap=overlap,
            acl_groups=acl_groups,
        )
        all_chunks.append((doc, chunks))
    total_chunks = sum(len(c) for _, c in all_chunks)
    console.print(f"  [green]✓[/green] Created {total_chunks} chunks")

    # 3. Contextualize (per-document, so we can free memory between)
    if not skip_contextualize and settings.enable_contextual_retrieval:
        console.print("  [cyan]→[/cyan] Generating context (Qwen 2.5 7B)...")
        for doc, chunks in all_chunks:
            contextualize_chunks(chunks, subject_name=name)
        console.print(f"  [green]✓[/green] Contextualized")
    else:
        console.print("  [yellow]→[/yellow] Skipping contextual retrieval")

    # 4. Embed + Index
    console.print("  [cyan]→[/cyan] Embedding + indexing (nomic-embed-text)...")
    for doc, chunks in all_chunks:
        index_chunks(
            chunks=chunks,
            subject_slug=slug,
            subject_name=name,
            content_type=content_type,
            parser=parser,
            acl_groups=acl_groups,
            document_path=str(doc.file_path),
            document_hash=doc.file_hash,
            document_unit=doc.unit_number,
            document_page_count=doc.page_count,
        )
    console.print(f"  [green]✓[/green] Indexed all chunks")
    console.print(f"[bold green]Done: {name}[/bold green]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest curriculum PDFs.")
    parser.add_argument(
        "--subject",
        required=True,
        help="Subject slug (e.g. 'programming_in_data_science') or 'all'",
    )
    parser.add_argument(
        "--skip-contextualize",
        action="store_true",
        help="Skip the contextual retrieval step (saves ~80%% of time)",
    )
    args = parser.parse_args()

    _, subjects = load_subjects_config()

    if args.subject == "all":
        # Order by priority field
        ordered = sorted(subjects, key=lambda s: s.get("priority", 999))
    else:
        ordered = [s for s in subjects if s["slug"] == args.subject]
        if not ordered:
            log.error("Unknown subject: %s", args.subject)
            available = ", ".join(s["slug"] for s in subjects)
            log.error("Available: %s", available)
            return 1

    for subject_cfg in ordered:
        try:
            ingest_subject(subject_cfg, skip_contextualize=args.skip_contextualize)
        except Exception:
            log.exception("Failed to ingest %s", subject_cfg["slug"])

    return 0


if __name__ == "__main__":
    sys.exit(main())

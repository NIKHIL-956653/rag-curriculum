"""Chunking — recursive splitting with parent-child relationships.

Strategy:
- Parent chunks: ~1500 tokens. Used as final context fed to the LLM.
- Child chunks: ~512 tokens (configurable per subject). Used for retrieval.
- Each child stores its parent's ID — retrieval finds child, we return parent.

This gives us: precise retrieval (small chunks embed better) + rich generation
context (parent has enough room for full thoughts).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingest.parse import ParsedDocument

log = logging.getLogger(__name__)

# Token counter — using GPT-4's tokenizer as a stable approximation.
# Qwen and nomic use different tokenizers, but the relative counts are close
# enough for chunking purposes.
_ENC = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass
class Chunk:
    """One chunk of text with full metadata."""
    chunk_id: str
    document_id: int | None        # set by DB after insert
    parent_chunk_id: str | None    # None for parent chunks
    subject_slug: str
    unit_number: int
    page_number: int
    section_path: str | None
    chunk_type: str                # "parent" | "child"
    content: str
    context_prefix: str = ""       # filled in by contextualize step
    token_count: int = 0
    acl_groups: list[str] = field(default_factory=lambda: ["all_students"])

    @property
    def embed_text(self) -> str:
        """Text fed to the embedder — includes context prefix if set."""
        if self.context_prefix:
            return f"{self.context_prefix}\n\n{self.content}"
        return self.content


# --------------------------------------------------------------------- splitter
def _make_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=count_tokens,
        separators=["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )


def chunk_document(
    doc: ParsedDocument,
    subject_slug: str,
    child_size: int = 512,
    parent_size: int = 1536,
    overlap: int = 100,
    acl_groups: list[str] | None = None,
) -> list[Chunk]:
    """Produce parent + child chunks for one document.

    Approach:
    1. Concatenate page text (with page markers preserved in metadata).
    2. First pass: split into parent-sized chunks.
    3. Second pass: split each parent into children.
    """
    acl = acl_groups or ["all_students"]

    # Build a single text body, tracking which character ranges belong to each page.
    body_parts: list[str] = []
    page_offsets: list[tuple[int, int, int, str | None]] = []  # (start, end, page_no, section)
    cursor = 0
    for page in doc.pages:
        page_text = page.text
        # Append tables as a separate block at end of page text (preserves them)
        if page.tables_markdown:
            tables_block = "\n\n" + "\n\n".join(page.tables_markdown)
            page_text = page_text + tables_block
        if not page_text.strip():
            continue
        body_parts.append(page_text)
        page_offsets.append((cursor, cursor + len(page_text), page.page_number, page.section_path))
        cursor += len(page_text) + 2  # account for join "\n\n"
    body = "\n\n".join(body_parts)

    if not body.strip():
        return []

    def page_for_offset(offset: int) -> tuple[int, str | None]:
        for start, end, pg, sec in page_offsets:
            if start <= offset < end:
                return pg, sec
        # Fallback: last page
        return page_offsets[-1][2], page_offsets[-1][3]

    # First pass — parents
    parent_splitter = _make_splitter(parent_size, overlap // 2)
    parent_texts = parent_splitter.split_text(body)

    chunks: list[Chunk] = []
    child_splitter = _make_splitter(child_size, overlap)

    body_cursor = 0
    for parent_text in parent_texts:
        # Find approximate offset of this parent within body
        offset = body.find(parent_text, body_cursor)
        if offset < 0:
            offset = body_cursor
        body_cursor = offset + len(parent_text)

        page_no, section = page_for_offset(offset)

        parent_id = f"{subject_slug}:u{doc.unit_number:02d}:p{page_no:03d}:par:{uuid.uuid4().hex[:8]}"
        parent_chunk = Chunk(
            chunk_id=parent_id,
            document_id=None,
            parent_chunk_id=None,
            subject_slug=subject_slug,
            unit_number=doc.unit_number,
            page_number=page_no,
            section_path=section,
            chunk_type="parent",
            content=parent_text,
            token_count=count_tokens(parent_text),
            acl_groups=acl,
        )
        chunks.append(parent_chunk)

        # Children
        child_texts = child_splitter.split_text(parent_text)
        for ci, child_text in enumerate(child_texts):
            child_id = f"{parent_id}:c{ci:02d}"
            chunks.append(Chunk(
                chunk_id=child_id,
                document_id=None,
                parent_chunk_id=parent_id,
                subject_slug=subject_slug,
                unit_number=doc.unit_number,
                page_number=page_no,
                section_path=section,
                chunk_type="child",
                content=child_text,
                token_count=count_tokens(child_text),
                acl_groups=acl,
            ))

    log.info(
        "Chunked %s u%d: %d parents, %d children",
        doc.file_path.name, doc.unit_number,
        sum(1 for c in chunks if c.chunk_type == "parent"),
        sum(1 for c in chunks if c.chunk_type == "child"),
    )
    return chunks

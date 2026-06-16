"""PDF parsing — multiple strategies based on subject content type.

Parser strategies:
- default:      pymupdf4llm (fast, good general markdown)
- math_heavy:   pymupdf4llm + LaTeX detection (Advanced ML)
- table_heavy:  pdfplumber for tables, pymupdf4llm for prose (Financial, EDA)
- code_heavy:   pymupdf4llm + code-block detection (Programming, Scraping)
- legal:       pymupdf4llm with strict whitespace preservation (Legal)
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf4llm
import pdfplumber

log = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """One parsed page with all metadata needed downstream."""
    page_number: int
    text: str                       # full page text as markdown
    tables_markdown: list[str]      # any extracted tables (markdown format)
    has_code: bool                  # whether code blocks were detected
    has_math: bool                  # whether LaTeX/math was detected
    section_path: str | None        # nearest heading (best effort)


@dataclass
class ParsedDocument:
    """One parsed PDF."""
    file_path: Path
    file_hash: str
    unit_number: int
    pages: list[ParsedPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)


# --------------------------------------------------------------------- helpers
_CODE_RX = re.compile(r"^\s{4,}\S|```|def\s+\w+\(|class\s+\w+|import\s+\w+", re.M)
_MATH_RX = re.compile(r"\$[^$]+\$|\\frac|\\sum|\\int|\\alpha|\\beta")
_HEADING_RX = re.compile(r"^(#{1,4})\s+(.+)$", re.M)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _extract_unit_number(path: Path) -> int:
    """Best-effort extraction of unit number from filename.
    Looks for: 'unit_5.pdf', 'Unit5.pdf', 'U05.pdf', '5.pdf', etc.
    """
    name = path.stem.lower()
    m = re.search(r"(?:unit[_\s-]?|u)(\d{1,2})", name)
    if m:
        return int(m.group(1))
    m = re.search(r"^(\d{1,2})\b", name)
    if m:
        return int(m.group(1))
    # Fallback: 0 (caller can detect and assign)
    return 0


def _detect_section_path(text: str) -> str | None:
    """Pull the most recent markdown heading as section path."""
    headings = _HEADING_RX.findall(text)
    if not headings:
        return None
    # Last heading on the page is most likely the section start
    return headings[-1][1].strip()


def _extract_tables(path: Path, page_number: int) -> list[str]:
    """Extract tables from a specific page as markdown."""
    tables: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            if page_number > len(pdf.pages):
                return tables
            page = pdf.pages[page_number - 1]
            for tbl in page.extract_tables() or []:
                if not tbl or not tbl[0]:
                    continue
                # Build markdown table
                header = tbl[0]
                rows = tbl[1:] if len(tbl) > 1 else []
                md = "| " + " | ".join(str(c or "").strip() for c in header) + " |\n"
                md += "|" + "|".join(["---"] * len(header)) + "|\n"
                for row in rows:
                    md += "| " + " | ".join(str(c or "").strip() for c in row) + " |\n"
                tables.append(md)
    except Exception as e:
        log.warning("Table extraction failed for %s p%d: %s", path.name, page_number, e)
    return tables


# --------------------------------------------------------------------- public API
def parse_pdf(
    path: Path,
    parser_strategy: str = "default",
    unit_number: int | None = None,
) -> ParsedDocument:
    """Parse one PDF into a ParsedDocument."""
    if not path.exists():
        raise FileNotFoundError(path)

    file_hash = _file_hash(path)
    unit = unit_number or _extract_unit_number(path)

    # Get markdown text per page from pymupdf4llm
    pages_md = pymupdf4llm.to_markdown(str(path), page_chunks=True)

    parsed_pages: list[ParsedPage] = []
    for i, page in enumerate(pages_md, start=1):
        text = page.get("text", "")

        has_code = bool(_CODE_RX.search(text))
        has_math = bool(_MATH_RX.search(text))

        # Extract tables for table_heavy parsers (or any page with table syntax)
        tables: list[str] = []
        if parser_strategy == "table_heavy" or "|" in text and "---" in text:
            tables = _extract_tables(path, i)

        section = _detect_section_path(text)

        parsed_pages.append(ParsedPage(
            page_number=i,
            text=text,
            tables_markdown=tables,
            has_code=has_code,
            has_math=has_math,
            section_path=section,
        ))

    log.info(
        "Parsed %s (unit=%d, pages=%d, strategy=%s)",
        path.name, unit, len(parsed_pages), parser_strategy,
    )

    return ParsedDocument(
        file_path=path,
        file_hash=file_hash,
        unit_number=unit,
        pages=parsed_pages,
    )


def parse_subject_folder(
    folder: Path,
    parser_strategy: str = "default",
) -> list[ParsedDocument]:
    """Parse every PDF in a subject folder."""
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        log.warning("No PDFs found in %s", folder)
        return []

    docs: list[ParsedDocument] = []
    for i, pdf in enumerate(pdfs, start=1):
        try:
            doc = parse_pdf(pdf, parser_strategy=parser_strategy)
            # If unit number couldn't be extracted from filename, assign by order
            if doc.unit_number == 0:
                doc.unit_number = i
            docs.append(doc)
        except Exception as e:
            log.error("Failed to parse %s: %s", pdf, e)

    return docs

"""Generate an eval set from the chunks actually in the index.

Why this exists: the shipped eval_set.jsonl has 10 hand-written questions, all
for one subject, and they ask about pandas/numpy while that subject's PDFs are
about R. Running it measures nothing. An eval set is only meaningful when every
question is answerable from a chunk that is really in the corpus.

Method:
  1. Sample child chunks per subject straight from Postgres.
  2. Drop junk (too short, mostly punctuation/digits, separator rows, endnote
     and bibliography fragments) -- a question generated from a page of
     "Ibid, pp431" teaches the eval nothing.
  3. Ask Claude, per chunk, for a question answerable ONLY from that chunk,
     plus a verbatim substring of the chunk and the keywords a correct answer
     must contain.
  4. VALIDATE that the returned substring really occurs in the source chunk,
     and reject the case if it does not. This is the whole point: the harness
     scores retrieval by looking for that substring in retrieved chunks, so a
     hallucinated substring would make a question permanently unanswerable and
     silently depress Recall@k.

Output matches the format src/eval/run_eval.py already reads, plus a
source_chunk_id field for traceability (the loader ignores unknown keys).

Usage:
    python -m src.eval.generate_eval_set --per-subject 20
    python -m src.eval.generate_eval_set --per-subject 30 \
        --subjects programming_in_data_science,financial_management \
        --output src/eval/eval_set_generated.jsonl

The generated set is a DRAFT. Spot-check a sample by hand before trusting the
numbers it produces -- an eval set nobody has read is just a different kind of
guess.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

from src.utils.db import pg_cursor
from src.utils.llm_router import get_llm

console = Console()
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_time=False)],
)
log = logging.getLogger("gen-eval")


# --------------------------------------------------------------- quality filter
_WORD_RX = re.compile(r"[A-Za-z]{3,}")
# Lines that are mostly citation apparatus: "Ibid, pp 431", "op. cit.", bare
# page ranges, "See note 14". These dominate scanned academic back-matter.
_CITATION_RX = re.compile(
    r"\b(ibid|op\.?\s*cit|loc\.?\s*cit|et\.?\s*al|pp?\.?\s*\d+|see\s+note)\b",
    re.I,
)


def is_usable(text: str, min_chars: int) -> bool:
    """True if a chunk is substantive enough to build a question from."""
    t = text.strip()
    if len(t) < min_chars:
        return False

    words = _WORD_RX.findall(t)
    if len(words) < 40:
        return False

    # Mostly letters? Tables of numbers and separator rows are not question material.
    alpha = sum(c.isalpha() or c.isspace() for c in t)
    if alpha / len(t) < 0.75:
        return False

    # Citation-dense back-matter: more than two markers in a short span.
    if len(_CITATION_RX.findall(t)) >= 3:
        return False

    # Separator-only or heading-only fragments.
    if t.replace("-", "").replace("=", "").replace("_", "").strip() == "":
        return False

    return True


# --------------------------------------------------------------------- sampling
@dataclass
class Candidate:
    chunk_id: str
    subject_slug: str
    unit_number: int
    page_number: int | None
    content: str


def fetch_candidates(subjects: list[str] | None, pool_multiplier: int, per_subject: int,
                     min_chars: int) -> dict[str, list[Candidate]]:
    """Pull a generous pool per subject, then filter down to usable chunks.

    We over-fetch because the quality filter rejects a lot on scanned or
    compiled sources.
    """
    where = ["chunk_type = 'child'", "content IS NOT NULL"]
    params: list = []
    if subjects:
        where.append("subject_slug = ANY(%s)")
        params.append(subjects)

    limit_per_subject = per_subject * pool_multiplier

    sql = f"""
        SELECT chunk_id, subject_slug, unit_number, page_number, content
        FROM (
            SELECT chunk_id, subject_slug, unit_number, page_number, content,
                   row_number() OVER (PARTITION BY subject_slug ORDER BY random()) AS rn
            FROM chunks
            WHERE {' AND '.join(where)}
        ) t
        WHERE rn <= %s
    """
    params.append(limit_per_subject)

    by_subject: dict[str, list[Candidate]] = {}
    with pg_cursor() as cur:
        cur.execute(sql, params)
        for chunk_id, slug, unit, page, content in cur.fetchall():
            if not is_usable(content, min_chars):
                continue
            by_subject.setdefault(slug, []).append(
                Candidate(chunk_id, slug, unit, page, content)
            )
    return by_subject


# ------------------------------------------------------------------ generation
_SYSTEM = """You write evaluation questions for a retrieval system. You are given \
ONE passage from a textbook. Write a question whose answer is contained in that \
passage and nowhere else in general knowledge.

Rules:
- The question must be answerable from the passage alone.
- The question must NOT reference "the passage", "the text", "this chunk" or \
"the document" - it should read like something a student would type into a search box.
- Avoid questions answerable from general knowledge without the passage.
- Prefer specific facts, definitions, mechanisms, classifications, or named concepts.

Return ONLY a JSON object, no markdown fence, no commentary:
{"question": "...", "substring": "...", "keywords": ["...", "..."]}

- "substring": 4-10 consecutive words copied EXACTLY from the passage, containing \
the heart of the answer. Copy character for character. Do not paraphrase.
- "keywords": 2-4 short terms that must appear in any correct answer."""

_TEMPLATE = """Subject: {subject}
Unit: {unit}

<passage>
{content}
</passage>"""

_JSON_RX = re.compile(r"\{.*\}", re.S)


def _parse(raw: str) -> dict | None:
    """Pull a JSON object out of the model's reply, tolerating stray prose."""
    m = _JSON_RX.search(raw or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def generate_one(cand: Candidate) -> dict | None:
    """Make one eval case. Returns None if the model's output fails validation."""
    client = get_llm()
    prompt = _TEMPLATE.format(
        subject=cand.subject_slug.replace("_", " "),
        unit=cand.unit_number,
        content=cand.content[:2500],
    )
    try:
        raw = client.generate(prompt=prompt, system=_SYSTEM, temperature=0.3, max_tokens=300)
    except Exception as e:
        log.warning("generation failed for %s: %s", cand.chunk_id, e)
        return None

    obj = _parse(raw)
    if not obj:
        log.warning("unparseable reply for %s", cand.chunk_id)
        return None

    question = (obj.get("question") or "").strip()
    substring = (obj.get("substring") or "").strip().strip('"').strip("'")
    keywords = [str(k).strip() for k in (obj.get("keywords") or []) if str(k).strip()]

    if not question or not substring:
        return None

    # THE validation that matters: the substring must really be in the chunk,
    # or the harness can never score a hit for this question.
    haystack = " ".join(cand.content.split()).lower()
    needle = " ".join(substring.split()).lower()
    if needle not in haystack:
        log.warning("substring not found in source chunk %s - rejected", cand.chunk_id)
        return None

    # Too-short substrings match everything and inflate Recall@k.
    if len(needle) < 20:
        log.warning("substring too short for %s - rejected", cand.chunk_id)
        return None

    return {
        "question": question,
        "subject": cand.subject_slug,
        "expected_chunk_substring": substring,
        "expected_answer_keywords": keywords[:4],
        "source_chunk_id": cand.chunk_id,
        "source_unit": cand.unit_number,
        "source_page": cand.page_number,
    }


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Generate an eval set from indexed chunks.")
    ap.add_argument("--per-subject", type=int, default=20,
                    help="target questions per subject (default 20)")
    ap.add_argument("--subjects", type=str, default=None,
                    help="comma-separated subject slugs (default: all)")
    ap.add_argument("--output", type=Path, default=Path("src/eval/eval_set_generated.jsonl"))
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent generation requests (default 12)")
    ap.add_argument("--min-chars", type=int, default=400,
                    help="minimum chunk length to consider (default 400)")
    ap.add_argument("--pool-multiplier", type=int, default=4,
                    help="how many candidates to fetch per wanted question (default 4)")
    ap.add_argument("--seed", type=int, default=None, help="shuffle seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None

    console.rule("[bold cyan]Sampling chunks[/bold cyan]")
    pools = fetch_candidates(subjects, args.pool_multiplier, args.per_subject, args.min_chars)
    if not pools:
        console.print("[red]No usable chunks found. Is the corpus ingested?[/red]")
        return 1

    selected: list[Candidate] = []
    for slug in sorted(pools):
        pool = pools[slug]
        random.shuffle(pool)
        take = pool[: args.per_subject]
        selected.extend(take)
        flag = "" if len(take) == args.per_subject else f"  [yellow](only {len(take)} usable)[/yellow]"
        console.print(f"  {slug:<40} {len(take):3} selected from {len(pool):4} usable{flag}")

    console.print(f"\n[bold]{len(selected)}[/bold] questions to generate "
                  f"across {len(pools)} subjects\n")

    console.rule("[bold cyan]Generating[/bold cyan]")
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
        futures = {pool_exec.submit(generate_one, c): c for c in selected}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="questions"):
            out = fut.result()
            if out:
                results.append(out)

    rejected = len(selected) - len(results)

    # Stable ids, grouped by subject.
    results.sort(key=lambda r: (r["subject"], r["source_unit"] or 0))
    counters: dict[str, int] = {}
    for r in results:
        n = counters.get(r["subject"], 0) + 1
        counters[r["subject"]] = n
        r["id"] = f"{r['subject']}_{n:03d}"

    # id first, for readability
    ordered = [{"id": r.pop("id"), **r} for r in results]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    console.rule("[bold green]Done[/bold green]")
    console.print(f"  wrote    [bold]{len(ordered)}[/bold] questions -> {args.output}")
    console.print(f"  rejected [bold]{rejected}[/bold] "
                  f"(unparseable reply, or substring not found in the source chunk)")
    console.print("\n[yellow]This is a draft.[/yellow] Read 15-20 of them before you "
                  "quote any number it produces.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

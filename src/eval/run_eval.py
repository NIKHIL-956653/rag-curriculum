"""Evaluation runner — Recall@k, MRR, plus LLM-as-judge faithfulness.

Eval set format (JSONL, one Q/A per line):
{
  "id": "eda_01",
  "question": "What is interquartile range?",
  "subject": "exploratory_data_analysis",
  "expected_chunk_substring": "interquartile",   # substring that must appear in a retrieved chunk
  "expected_answer_keywords": ["Q3", "Q1", "spread"]   # keywords expected in the answer
}

We avoid requiring exact chunk IDs in the eval set because chunk IDs change
between ingests; substring matching is more robust.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

from src.generate.answer import answer_question
from src.utils.settings import settings

console = Console()
log = logging.getLogger("eval")


@dataclass
class EvalCase:
    id: str
    question: str
    subject: str | None = None
    expected_chunk_substring: str | None = None
    expected_answer_keywords: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    case: EvalCase
    answer: str
    retrieved_chunks: list[str]                # parent chunk IDs
    retrieval_hit: bool                        # was expected substring found in any chunk?
    retrieval_hit_rank: int | None              # rank of first chunk containing the substring
    answer_keyword_coverage: float              # fraction of expected keywords present
    latency_ms: int


def load_eval_set(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            cases.append(EvalCase(
                id=obj["id"],
                question=obj["question"],
                subject=obj.get("subject"),
                expected_chunk_substring=obj.get("expected_chunk_substring"),
                expected_answer_keywords=obj.get("expected_answer_keywords", []),
            ))
    return cases


def evaluate_case(case: EvalCase) -> CaseResult:
    """Run one case end-to-end and compute metrics."""
    result = answer_question(
        query=case.question,
        subject_slug=case.subject,
    )

    # Retrieval hit / rank
    hit = False
    hit_rank: int | None = None
    if case.expected_chunk_substring:
        needle = case.expected_chunk_substring.lower()
        for rank, c in enumerate(result.chunks, start=1):
            if needle in c.content.lower():
                hit = True
                hit_rank = rank
                break

    # Answer keyword coverage
    ans_lower = result.answer.lower()
    if case.expected_answer_keywords:
        covered = sum(1 for kw in case.expected_answer_keywords if kw.lower() in ans_lower)
        coverage = covered / len(case.expected_answer_keywords)
    else:
        coverage = 1.0

    return CaseResult(
        case=case,
        answer=result.answer,
        retrieved_chunks=[c.chunk_id for c in result.chunks],
        retrieval_hit=hit,
        retrieval_hit_rank=hit_rank,
        answer_keyword_coverage=coverage,
        latency_ms=result.latency_ms,
    )


def summarise(results: list[CaseResult]) -> dict:
    """Compute aggregate metrics."""
    if not results:
        return {}

    cases_with_substring = [r for r in results if r.case.expected_chunk_substring]
    recall_at_k = (
        sum(1 for r in cases_with_substring if r.retrieval_hit) / len(cases_with_substring)
        if cases_with_substring else None
    )
    mrr = (
        mean([(1.0 / r.retrieval_hit_rank) if r.retrieval_hit_rank else 0.0
              for r in cases_with_substring])
        if cases_with_substring else None
    )
    avg_keyword_coverage = mean([r.answer_keyword_coverage for r in results])
    avg_latency_ms = mean([r.latency_ms for r in results])

    return {
        "n_cases": len(results),
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "avg_keyword_coverage": avg_keyword_coverage,
        "avg_latency_ms": avg_latency_ms,
    }


def print_report(results: list[CaseResult], summary: dict) -> None:
    table = Table(title="Per-case results")
    table.add_column("ID", style="cyan")
    table.add_column("Hit", justify="center")
    table.add_column("Rank", justify="right")
    table.add_column("Kw cov", justify="right")
    table.add_column("Latency ms", justify="right")

    for r in results:
        table.add_row(
            r.case.id,
            "✓" if r.retrieval_hit else "✗",
            str(r.retrieval_hit_rank) if r.retrieval_hit_rank else "-",
            f"{r.answer_keyword_coverage:.0%}",
            str(r.latency_ms),
        )
    console.print(table)

    console.rule("[bold]Summary[/bold]")
    for k, v in summary.items():
        if isinstance(v, float):
            console.print(f"  [cyan]{k}[/cyan]: {v:.3f}")
        else:
            console.print(f"  [cyan]{k}[/cyan]: {v}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG evaluation.")
    parser.add_argument(
        "--eval-set",
        default=str(Path(__file__).parent / "eval_set.jsonl"),
        help="Path to eval set JSONL",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write detailed JSON results",
    )
    args = parser.parse_args()

    path = Path(args.eval_set)
    if not path.exists():
        log.error("Eval set not found: %s", path)
        return 1

    cases = load_eval_set(path)
    console.print(f"[cyan]Loaded {len(cases)} eval cases from {path}[/cyan]")

    results: list[CaseResult] = []
    for case in cases:
        console.print(f"  Running {case.id}...")
        try:
            results.append(evaluate_case(case))
        except Exception:
            log.exception("Failed on %s", case.id)

    summary = summarise(results)
    print_report(results, summary)

    if args.output_json:
        out = {
            "summary": summary,
            "cases": [
                {
                    "id": r.case.id,
                    "question": r.case.question,
                    "answer": r.answer,
                    "retrieved_chunks": r.retrieved_chunks,
                    "retrieval_hit": r.retrieval_hit,
                    "retrieval_hit_rank": r.retrieval_hit_rank,
                    "answer_keyword_coverage": r.answer_keyword_coverage,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ],
        }
        # Create the directory first: without this, a run into a path like
        # reports/foo.json completes the whole evaluation and then throws the
        # results away on the last line with FileNotFoundError.
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote detailed results to {out_path}[/green]")

    return 0


if __name__ == "__main__":
    sys.exit(main())

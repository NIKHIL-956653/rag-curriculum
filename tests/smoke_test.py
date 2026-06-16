"""Smoke test — runs a basic query through the full pipeline.

Run AFTER ingesting at least one subject:
    python -m tests.smoke_test
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.generate.answer import answer_question

console = Console()


def main() -> int:
    test_query = "What is the difference between supervised and unsupervised learning?"

    console.rule("[bold cyan]Smoke test — full pipeline[/bold cyan]")
    console.print(f"Query: [bold]{test_query}[/bold]\n")

    try:
        result = answer_question(test_query)
    except Exception as e:
        console.print(f"[bold red]✗ Failed: {e}[/bold red]")
        console.print_exception()
        return 1

    console.print("[green]✓ Pipeline ran end-to-end[/green]\n")
    console.print("[bold]Answer:[/bold]")
    console.print(result.answer)
    console.print()
    console.print(f"[dim]Latency: {result.latency_ms}ms · Chunks used: {len(result.chunks)}[/dim]")

    if result.chunks:
        console.print("\n[bold]Top sources:[/bold]")
        for i, c in enumerate(result.chunks[:3], start=1):
            console.print(f"  [C{i}] {c.citation} (score={c.score:.3f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

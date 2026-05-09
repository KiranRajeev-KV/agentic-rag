from __future__ import annotations

from typing import Annotated

import typer

from agentic_rag.config import get_settings

app = typer.Typer(help="Agentic RAG local-first CLI.")
eval_app = typer.Typer(help="Run evaluation and ablation commands.")
trace_app = typer.Typer(help="Inspect run traces.")
db_app = typer.Typer(help="Database utility commands.")

app.add_typer(eval_app, name="eval")
app.add_typer(trace_app, name="trace")
app.add_typer(db_app, name="db")


@app.callback()
def main() -> None:
    get_settings()


@app.command("ingest")
def ingest_command(
    limit: Annotated[int, typer.Option("--limit", min=1)] = 20,
) -> None:
    typer.echo(f"[stub] ingest requested with --limit={limit}")


@app.command("ask")
def ask_command(
    question: str,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    typer.echo(f"[stub] ask requested: {question}")
    if debug:
        typer.echo("[stub] debug trace summary will be implemented in a later milestone.")


@eval_app.command("run")
def eval_run_command(
    variant: Annotated[str, typer.Option("--variant")] = "parent_child",
) -> None:
    typer.echo(f"[stub] eval run requested for variant={variant}")


@eval_app.command("compare")
def eval_compare_command(
    baseline: Annotated[str, typer.Option("--baseline")] = "child_only",
    candidate: Annotated[str, typer.Option("--candidate")] = "parent_child",
) -> None:
    typer.echo(f"[stub] eval compare requested: baseline={baseline}, candidate={candidate}")


@trace_app.command("list")
def trace_list_command(
    last: Annotated[int, typer.Option("--last", min=1)] = 10,
) -> None:
    typer.echo(f"[stub] trace list requested with --last={last}")


@trace_app.command("show")
def trace_show_command(trace_id: str) -> None:
    typer.echo(f"[stub] trace show requested for trace_id={trace_id}")


@db_app.command("reset")
def db_reset_command() -> None:
    typer.echo("[stub] db reset requested")


if __name__ == "__main__":
    app()

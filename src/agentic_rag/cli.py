from __future__ import annotations

import sqlite3
from typing import Annotated, Any

import typer

from agentic_rag.agent import AskGraphRunner
from agentic_rag.config import Settings, get_settings
from agentic_rag.evals import EvalRunner
from agentic_rag.ingest.indexing import ChildChunkIndexer
from agentic_rag.ingest.pipeline import IngestPipeline
from agentic_rag.retrieval.types import RetrievalVariant
from agentic_rag.storage.bootstrap import initialize_storage
from agentic_rag.storage.sqlite import SQLiteStore
from agentic_rag.traces.reader import TraceReader

app = typer.Typer(help="Agentic RAG local-first CLI.")
eval_app = typer.Typer(help="Run evaluation and ablation commands.")
trace_app = typer.Typer(help="Inspect run traces.")
db_app = typer.Typer(help="Database utility commands.")
corpus_app = typer.Typer(help="Corpus discovery and ingestion helpers.")

app.add_typer(eval_app, name="eval")
app.add_typer(trace_app, name="trace")
app.add_typer(db_app, name="db")


@app.callback()
def main() -> None:
    get_settings()


@app.command("ingest")
def ingest_command(
    limit: Annotated[
        int, typer.Option("--limit", min=1, help="Number of manifest papers to ingest.")
    ] = 100,
    index: Annotated[
        bool,
        typer.Option(
            "--index",
            help="Optional convenience step: run `app index` behavior after SQLite ingestion.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Re-download and re-parse papers even when cached parse artifacts are unchanged.",
        ),
    ] = False,
) -> None:
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)
    typer.echo("ingest.start sqlite_only=true")
    pipeline = IngestPipeline(settings=settings)
    summary = pipeline.run(limit=limit, force=force)
    typer.echo(
        "ingest.summary "
        f"requested={summary.requested_limit} discovered={summary.discovered} "
        f"downloaded={summary.downloaded} parsed={summary.parsed} "
        f"parse_failed={summary.parse_failed} download_failed={summary.download_failed} "
        f"skipped={summary.skipped}"
    )
    if summary.errors:
        typer.echo("ingest.errors:")
        for err in summary.errors[:10]:
            typer.echo(f"- {err}")
    if summary.download_failed > 0 and summary.parsed == 0:
        typer.echo("ingest.hint run `uv run python scripts/download_corpus_pdfs.py` then retry.")

    if index:
        typer.echo("ingest.index opt-in enabled: running indexing step.")
        _run_index(settings=settings, limit=500, batch_size=32, force=False)


@app.command("ask")
def ask_command(
    question: str,
    thread_id: Annotated[
        str,
        typer.Option("--thread-id", help="Conversation thread id for memory-scoped follow-up."),
    ] = "default",
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)
    runner = AskGraphRunner(settings=settings)
    final_state = runner.run(question, thread_id=thread_id)
    final_answer = final_state.get("final_answer", "")
    typer.echo(final_answer)
    if debug:
        _print_debug_summary(final_state)


@app.command("index")
def index_command(
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Max chunks to index. Omit to index all pending."),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size", min=1, help="Embedding batch size for OpenAI embedding calls."
        ),
    ] = 32,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Re-embed and upsert regardless of prior index metadata (limited by --limit).",
        ),
    ] = False,
) -> None:
    settings = get_settings()
    _run_index(settings=settings, limit=limit, batch_size=batch_size, force=force)


@eval_app.command("run")
def eval_run_command(
    variant: Annotated[str, typer.Option("--variant")] = "parent_child",
) -> None:
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=False)
    eval_runner = EvalRunner(settings=settings)
    chosen_variant = RetrievalVariant(variant)
    summary = eval_runner.run_variant(chosen_variant)
    typer.echo(
        f"eval.summary variant={summary['variant']} cases={summary['cases']} "
        f"raw_score={summary['raw_score']:.2f} normalized={summary['normalized_score']:.2f} "
        f"hard_fail_refusal={summary['hard_fail_refusal']}"
    )


@eval_app.command("compare")
def eval_compare_command(
    baseline: Annotated[str, typer.Option("--baseline")] = "child_only",
    candidate: Annotated[str, typer.Option("--candidate")] = "parent_child",
) -> None:
    settings = get_settings()
    eval_runner = EvalRunner(settings=settings)
    comparison = eval_runner.compare(
        baseline=RetrievalVariant(baseline),
        candidate=RetrievalVariant(candidate),
    )
    typer.echo(
        f"eval.compare baseline={comparison['baseline']} candidate={comparison['candidate']} "
        f"baseline_score={comparison['baseline_score']:.2f} "
        f"candidate_score={comparison['candidate_score']:.2f} "
        f"delta={comparison['delta_score']:+.2f}"
    )


@trace_app.command("list")
def trace_list_command(
    last: Annotated[int, typer.Option("--last", min=1)] = 10,
) -> None:
    settings = get_settings()
    store = SQLiteStore(settings.app_db_path)
    trace_repo = TraceReader(store)
    try:
        traces = trace_repo.list_recent(limit=last)
    except sqlite3.OperationalError as err:
        typer.echo("No trace tables found. Run `app db init` first.")
        raise typer.Exit(code=1) from err

    if not traces:
        typer.echo("No traces found.")
        return

    for trace in traces:
        typer.echo(
            f"{trace['trace_id']} thread={trace['thread_id']} turn={trace['turn_id']} "
            f"mode={trace['run_mode']} started={trace['started_at']}"
        )


@trace_app.command("show")
def trace_show_command(trace_id: str) -> None:
    settings = get_settings()
    store = SQLiteStore(settings.app_db_path)
    reader = TraceReader(store)
    result = reader.show(trace_id)
    if not result:
        typer.echo(f"Trace not found: {trace_id}")
        raise typer.Exit(code=1)
    trace = result["trace"]
    typer.echo(
        f"trace_id={trace['trace_id']}\n"
        f"thread_id={trace['thread_id']}\n"
        f"turn_id={trace['turn_id']}\n"
        f"run_mode={trace['run_mode']}\n"
        f"started_at={trace['started_at']}\n"
        f"completed_at={trace['completed_at']}"
    )
    typer.echo("events:")
    for event in result["events"]:
        event_line = (
            f"- {event['ts']} {event['event']} "
            f"node={event.get('node')} payload={event.get('payload')}"
        )
        typer.echo(event_line)


@db_app.command("init")
def db_init_command(
    with_qdrant: Annotated[bool, typer.Option("--with-qdrant")] = False,
) -> None:
    settings = get_settings()
    initialize_storage(settings=settings, init_qdrant=with_qdrant)
    target = "sqlite+qdrant" if with_qdrant else "sqlite"
    typer.echo(f"Initialized storage: {target}")


@db_app.command("reset")
def db_reset_command(
    yes: Annotated[bool, typer.Option("--yes", help="Confirm destructive reset")] = False,
) -> None:
    if not yes:
        typer.echo("Refusing to reset without --yes")
        raise typer.Exit(code=1)

    settings = get_settings()
    if settings.app_db_path.exists():
        settings.app_db_path.unlink()
    initialize_storage(settings=settings, init_qdrant=False)
    typer.echo(f"Reset SQLite DB at {settings.app_db_path}")


def _run_index(settings: Settings, limit: int, batch_size: int, force: bool) -> None:
    initialize_storage(settings=settings, init_qdrant=False)
    typer.echo(
        "index.warning this will call OpenAI embeddings and may incur API cost. "
        "Use --limit for smaller runs."
    )
    typer.echo(
        f"index.start mode={'force' if force else 'idempotent'} "
        f"limit={limit} batch_size={batch_size}"
    )
    indexer = ChildChunkIndexer(settings=settings)
    summary = indexer.index_unembedded_chunks(limit=limit, batch_size=batch_size, force=force)
    typer.echo(
        f"index.progress total_chunks={summary.total_chunks} pending={summary.pending_chunks} "
        f"up_to_date={summary.skipped_as_up_to_date}"
    )
    typer.echo(
        f"index.summary selected={summary.selected_chunks} indexed={summary.indexed_chunks} "
        f"model={summary.model_name} config_hash={summary.config_hash[:12]} "
        f"dimensions={summary.dimensions} force={summary.force}"
    )


def _print_debug_summary(state: dict[str, Any]) -> None:
    trace_id = state.get("trace_id", "unknown")
    thread_id = state.get("thread_id", "default")
    turn_id = state.get("turn_id", "unknown")
    route = state.get("route_action", "unknown")
    route_mode = state.get("routing_mode", "unknown")
    route_conf = state.get("route_confidence", 0.0)
    route_reason = state.get("route_reason_public", "")
    evidence_status = state.get("evidence_status", "unknown")
    if hasattr(evidence_status, "value"):
        evidence_status = evidence_status.value
    evidence_confidence = state.get("evidence_confidence", "unknown")
    conflict_label = state.get("conflict_label", "NONE")
    contradiction_action = state.get("contradiction_action", "")
    retrieved_children = len(state.get("retrieved_child_ids", []))
    selected_parents = state.get("selected_parent_ids", [])
    parent_scores = state.get("parent_scores", {})
    context_packets = [packet["source_id"] for packet in state.get("context_packets", [])]
    final_action = state.get("final_action", "unknown")
    conversation_count = state.get("conversation_memory_read_count", 0)
    semantic_count = state.get("semantic_memory_read_count", 0)
    episodic_count = state.get("episodic_memory_read_count", 0)
    checkpoint_enabled = True
    message_count = len(state.get("messages", []))
    summary_len = len(state.get("conversation_summary", ""))
    active_focus = state.get("active_focus", "")
    total_latency = state.get("total_latency_ms", 0)
    retrieval_latency = state.get("retrieval_latency_ms", 0)
    llm_latency = state.get("llm_latency_ms", 0)
    tool_latency = state.get("tool_latency_ms", 0)
    typer.echo(
        f"\nTrace: {trace_id}\n"
        f"Thread: {thread_id}\n"
        f"Turn: {turn_id}\n"
        f"Checkpoint enabled: {checkpoint_enabled}\n"
        f"Message count: {message_count}\n"
        f"Conversation summary length: {summary_len}\n"
        f"Active focus: {active_focus}\n"
        "Memory read: "
        f"conversation={conversation_count}, "
        f"semantic={semantic_count}, "
        f"episodic={episodic_count}\n"
        f"Route: {route}\n"
        f"Routing mode: {route_mode}\n"
        f"Route confidence: {route_conf}\n"
        f"Route reason: {route_reason}\n"
        f"Retrieved children: {retrieved_children}\n"
        f"Selected parents: {selected_parents}\n"
        f"Parent scores: {parent_scores}\n"
        f"Evidence status: {evidence_status}\n"
        f"Evidence confidence: {evidence_confidence}\n"
        f"Conflict label: {conflict_label}\n"
        f"Contradiction handler action: {contradiction_action or '(not used)'}\n"
        f"Context packets: {', '.join(context_packets) if context_packets else '(none)'}\n"
        f"Final action: {final_action}\n"
        "Latency ms: "
        f"total={total_latency}, "
        f"retrieval={retrieval_latency}, "
        f"llm={llm_latency}, "
        f"tool={tool_latency}"
    )


if __name__ == "__main__":
    app()

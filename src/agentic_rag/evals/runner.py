from __future__ import annotations

import json
import uuid
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from agentic_rag.agent.graph import AskGraphRunner
from agentic_rag.config import Settings
from agentic_rag.retrieval.types import RetrievalVariant
from agentic_rag.storage.sqlite import SQLiteStore

from .models import EvalCase
from .scoring import score_case


class EvalRunner:
    def __init__(self, settings: Settings, graph_runner: AskGraphRunner | None = None) -> None:
        self.settings = settings
        self.graph_runner = graph_runner or AskGraphRunner(settings=settings)
        self.store = SQLiteStore(settings.app_db_path)

    def run_variant(self, variant: RetrievalVariant) -> dict[str, Any]:
        cases = _load_cases()
        eval_run_id = f"eval_{uuid.uuid4().hex[:10]}"

        case_results = []
        retrieval_metrics = []
        refusal_misses = 0
        for case in cases:
            trace_writer = self.graph_runner.trace_writer
            case_thread_id = f"eval_{variant.value}_{case.id}"
            case_trace = trace_writer.start(thread_id=case_thread_id, run_mode="eval")
            trace_writer.event(
                trace_id=case_trace.trace_id,
                level="info",
                event="eval.case_started",
                payload={"case_id": case.id, "variant": variant.value},
            )
            for history_turn in case.conversation_history:
                _ = self.graph_runner.run(
                    history_turn,
                    thread_id=case_thread_id,
                    retrieval_variant=variant,
                )
            state = self.graph_runner.run(
                case.question,
                thread_id=case_thread_id,
                retrieval_variant=variant,
            )
            result = score_case(case, state)
            trace_writer.event(
                trace_id=case_trace.trace_id,
                level="info",
                event="eval.case_scored",
                payload={
                    "case_id": case.id,
                    "variant": variant.value,
                    "score": result.score,
                    "final_action": result.actual_final_action,
                },
            )
            trace_writer.complete(
                trace_id=case_trace.trace_id,
                final_action=result.actual_final_action,
            )
            case_results.append(result)
            retrieval_metrics.append(_retrieval_metrics(case=case, state=state))
            if case.should_refuse and state.get("final_action") != "REFUSE":
                refusal_misses += 1

        raw_score = sum(item.score for item in case_results)
        normalized_score = (raw_score / (len(cases) * 10.0)) * 100.0 if cases else 0.0
        hard_fail = refusal_misses >= 2
        summary = {
            "eval_run_id": eval_run_id,
            "variant": variant.value,
            "cases": len(cases),
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "hard_fail_refusal": hard_fail,
            "route_accuracy": _accuracy(case_results, "expected_route", "actual_route"),
            "final_action_accuracy": _accuracy(
                case_results, "expected_final_action", "actual_final_action"
            ),
            "intent_metrics": _intent_metrics(case_results),
            "retrieval_metrics": _aggregate_retrieval_metrics(retrieval_metrics),
            "results": [item.model_dump(mode="json") for item in case_results],
        }
        eval_trace = self.graph_runner.trace_writer.start(
            thread_id=f"eval_{variant.value}",
            run_mode="eval",
        )
        self.graph_runner.trace_writer.event(
            trace_id=eval_trace.trace_id,
            level="info",
            event="eval.completed",
            payload={
                "variant": variant.value,
                "cases": len(cases),
                "normalized_score": normalized_score,
            },
        )
        self.graph_runner.trace_writer.complete(trace_id=eval_trace.trace_id, final_action="EVAL")
        self._write_reports(summary)
        self._persist_eval(summary=summary, cases=cases, case_results=case_results)
        return summary

    def compare(self, baseline: RetrievalVariant, candidate: RetrievalVariant) -> dict[str, Any]:
        baseline_summary = self._load_summary(baseline)
        candidate_summary = self._load_summary(candidate)
        comparison = {
            "baseline": baseline.value,
            "candidate": candidate.value,
            "baseline_score": baseline_summary["normalized_score"],
            "candidate_score": candidate_summary["normalized_score"],
            "delta_score": candidate_summary["normalized_score"]
            - baseline_summary["normalized_score"],
            "baseline_retrieval": baseline_summary["retrieval_metrics"],
            "candidate_retrieval": candidate_summary["retrieval_metrics"],
        }
        self._write_comparison_report(comparison)
        return comparison

    def _write_reports(self, summary: dict[str, Any]) -> None:
        report_dir = self.settings.app_runs_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"eval_summary_{summary['variant']}.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    def _load_summary(self, variant: RetrievalVariant) -> dict[str, Any]:
        path = self.settings.app_runs_dir / "reports" / f"eval_summary_{variant.value}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing eval summary for {variant.value}. Run eval first.")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_comparison_report(self, comparison: dict[str, Any]) -> None:
        report_dir = self.settings.app_runs_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Eval Comparison",
            f"Baseline: {comparison['baseline']}",
            f"Candidate: {comparison['candidate']}",
            f"Baseline score: {comparison['baseline_score']:.2f}",
            f"Candidate score: {comparison['candidate_score']:.2f}",
            f"Delta score: {comparison['delta_score']:+.2f}",
            "",
            "Retrieval metrics:",
            f"- baseline paper_hit@k: {comparison['baseline_retrieval']['paper_hit_at_k']:.3f}",
            f"- candidate paper_hit@k: {comparison['candidate_retrieval']['paper_hit_at_k']:.3f}",
            f"- baseline parent_hit@k: {comparison['baseline_retrieval']['parent_hit_at_k']:.3f}",
            f"- candidate parent_hit@k: {comparison['candidate_retrieval']['parent_hit_at_k']:.3f}",
            f"- baseline parent_mrr: {comparison['baseline_retrieval']['parent_mrr']:.3f}",
            f"- candidate parent_mrr: {comparison['candidate_retrieval']['parent_mrr']:.3f}",
        ]
        (report_dir / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")

    def _persist_eval(
        self,
        *,
        summary: dict[str, Any],
        cases: list[EvalCase],
        case_results: list,
    ) -> None:  # noqa: ANN001
        for case in cases:
            self.store.execute(
                """
                INSERT OR REPLACE INTO eval_cases (
                  case_id, question, conversation_history, expected_route, expected_final_action,
                  expected_tool, expected_paper_ids, expected_parent_ids, expected_answer_points,
                  should_cite_sources, should_refuse, should_clarify, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case.id,
                    case.question,
                    json.dumps(case.conversation_history, ensure_ascii=True),
                    case.expected_route,
                    case.expected_final_action,
                    case.expected_tool,
                    json.dumps(case.expected_paper_ids, ensure_ascii=True),
                    json.dumps(case.expected_parent_ids, ensure_ascii=True),
                    json.dumps(case.expected_answer_points, ensure_ascii=True),
                    1 if case.should_cite_sources else 0,
                    1 if case.should_refuse else 0,
                    1 if case.should_clarify else 0,
                    case.notes,
                ),
            )
        self.store.execute(
            """
            INSERT OR REPLACE INTO eval_runs (
              eval_run_id, variant, started_at, completed_at, total_cases,
              raw_score, normalized_score, hard_fail_refusal
            ) VALUES (?, ?, datetime('now'), datetime('now'), ?, ?, ?, ?)
            """,
            (
                summary["eval_run_id"],
                summary["variant"],
                summary["cases"],
                summary["raw_score"],
                summary["normalized_score"],
                1 if summary["hard_fail_refusal"] else 0,
            ),
        )
        for result in case_results:
            score_id = f"score_{summary['eval_run_id']}_{result.case_id}"
            self.store.execute(
                """
                INSERT OR REPLACE INTO eval_scores (
                  score_id, eval_run_id, case_id, score, route_score, retrieval_score,
                  evidence_score, answer_score, citation_score, memory_score, tool_score,
                  trace_score, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    score_id,
                    summary["eval_run_id"],
                    result.case_id,
                    result.score,
                    result.route_score,
                    result.retrieval_score,
                    result.evidence_score,
                    result.answer_score,
                    result.citation_score,
                    result.memory_score,
                    result.tool_score,
                    result.trace_score,
                    result.notes,
                ),
            )


def _load_cases() -> list[EvalCase]:
    path = Path(__file__).with_name("cases.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(item) for item in raw]


def _accuracy(rows: list, expected_key: str, actual_key: str) -> float:  # noqa: ANN001
    if not rows:
        return 0.0
    matches = sum(1 for row in rows if getattr(row, expected_key) == getattr(row, actual_key))
    return matches / len(rows)


def _retrieval_metrics(case: EvalCase, state: dict[str, Any]) -> dict[str, float]:
    selected_parents = state.get("selected_parent_ids", [])
    context_packets = state.get("context_packets", [])
    selected_papers = [packet.get("paper_id", "") for packet in context_packets]

    paper_hit = (
        1.0
        if not case.expected_paper_ids
        else float(any(x in selected_papers for x in case.expected_paper_ids))
    )
    parent_hit = (
        1.0
        if not case.expected_parent_ids
        else float(any(x in selected_parents for x in case.expected_parent_ids))
    )
    parent_mrr = 0.0
    if case.expected_parent_ids:
        for idx, parent_id in enumerate(selected_parents, start=1):
            if parent_id in case.expected_parent_ids:
                parent_mrr = 1.0 / idx
                break
    else:
        parent_mrr = 1.0 if selected_parents else 0.0

    return {
        "paper_hit_at_k": paper_hit,
        "parent_hit_at_k": parent_hit,
        "parent_mrr": parent_mrr,
        "context_token_count": float(_token_count_from_packets(context_packets)),
    }


def _aggregate_retrieval_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "paper_hit_at_k": 0.0,
            "parent_hit_at_k": 0.0,
            "parent_mrr": 0.0,
            "context_token_count": 0.0,
        }
    return {
        "paper_hit_at_k": mean(row["paper_hit_at_k"] for row in rows),
        "parent_hit_at_k": mean(row["parent_hit_at_k"] for row in rows),
        "parent_mrr": mean(row["parent_mrr"] for row in rows),
        "context_token_count": mean(row["context_token_count"] for row in rows),
    }


def _token_count_from_packets(context_packets: list[dict[str, Any]]) -> int:
    total = 0
    for packet in context_packets:
        total += len(str(packet.get("section_context", "")).split())
        total += len(str(packet.get("highlighted_evidence", "")).split())
    return total


def _intent_metrics(rows: list) -> dict[str, dict[str, float]]:  # noqa: ANN001
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(getattr(row, "intent", "unknown"), []).append(row)
    summary: dict[str, dict[str, float]] = {}
    for intent, items in grouped.items():
        n = len(items)
        route_ok = sum(1 for item in items if item.expected_route == item.actual_route)
        final_ok = sum(
            1 for item in items if item.expected_final_action == item.actual_final_action
        )
        avg_score = mean(item.score for item in items) if items else 0.0
        summary[intent] = {
            "count": float(n),
            "route_accuracy": (route_ok / n) if n else 0.0,
            "final_action_accuracy": (final_ok / n) if n else 0.0,
            "avg_score": avg_score,
        }
    return summary

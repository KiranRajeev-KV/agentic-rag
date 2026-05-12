from __future__ import annotations

import re
import time
from dataclasses import dataclass

from agentic_rag.llm.client import OpenAILLMClient
from agentic_rag.llm.prompts import (
    ANSWER_SYSTEM_PROMPT,
    CITATION_SYSTEM_PROMPT,
    CONTRADICTION_SYSTEM_PROMPT,
    EVIDENCE_SYSTEM_PROMPT,
    MEMORY_SYSTEM_PROMPT,
    answer_user_prompt,
    citation_user_prompt,
    contradiction_user_prompt,
    evidence_user_prompt,
    memory_user_prompt,
)
from agentic_rag.llm.schemas import (
    LLMAnswerOutput,
    LLMCitationValidationOutput,
    LLMContradictionOutput,
    LLMEvidenceOutput,
    LLMMemoryWriteOutput,
)
from agentic_rag.retrieval.types import EvidenceStatus, RetrievalVariant
from agentic_rag.tools.arxiv_tools import ArxivToolset
from agentic_rag.tools.schemas import ArxivLookupByIdInput, ArxivSearchInput

from .citations import build_sources_block, validate_citations
from .memory import MemoryService, extract_arxiv_ids, extract_focus
from .router import route_query_with_llm
from .state import AgentState


@dataclass
class AgentDependencies:
    retrieval_service: object
    memory_service: MemoryService
    toolset: ArxivToolset
    trace_writer: object
    llm_client: OpenAILLMClient


class AgentNodes:
    def __init__(self, deps: AgentDependencies) -> None:
        self.deps = deps
        self.settings = deps.llm_client.settings

    def load_state(self, state: AgentState) -> AgentState:
        thread_id = state.get("thread_id", "default")
        semantic = self.deps.memory_service.read_semantic(limit=10)
        episodes = self.deps.memory_service.read_episodes(thread_id=thread_id, limit=6)
        messages = list(state.get("messages", []))
        checkpoint_summary = str(state.get("conversation_summary", ""))
        active_focus = str(state.get("active_focus", ""))
        active_paper_ids = list(state.get("active_paper_ids", []))
        active_arxiv_ids = list(state.get("active_arxiv_ids", []))
        recent_turns = self._recent_turns_from_messages(messages, limit=6)
        self._trace(
            state,
            "memory.checkpoint_read",
            {"count": len(recent_turns), "thread_id": thread_id, "message_count": len(messages)},
            node="load_state",
        )
        self._trace(
            state,
            "memory.checkpoint_enabled",
            {"enabled": True, "thread_id": thread_id},
            node="load_state",
        )
        self._trace(
            state,
            "memory.semantic_read",
            {"count": len(semantic)},
            node="load_state",
        )
        self._trace(
            state,
            "memory.episodic_read",
            {"count": len(episodes)},
            node="load_state",
        )
        return {
            "conversation_summary": checkpoint_summary,
            "recent_turns": recent_turns,
            "active_focus": active_focus,
            "active_paper_ids": active_paper_ids,
            "active_arxiv_ids": active_arxiv_ids,
            "memory_context": semantic,
            "episodic_context": episodes,
            "conversation_memory_read_count": len(recent_turns),
            "semantic_memory_read_count": len(semantic),
            "episodic_memory_read_count": len(episodes),
        }

    def route_query(self, state: AgentState) -> AgentState:
        self._trace(
            state,
            "router.started",
            {"query_len": len(state.get("raw_user_query", ""))},
            node="route_query",
        )
        started = time.monotonic()
        decision, mode = route_query_with_llm(
            query=state["raw_user_query"],
            llm_client=self.deps.llm_client,
            router_model=self.settings.router_model,
            memory_context=state.get("memory_context", []),
            conversation_summary=state.get("conversation_summary", ""),
            recent_turns=state.get("recent_turns", []),
            episodic_context=state.get("episodic_context", []),
        )
        llm_latency_ms = int((time.monotonic() - started) * 1000) if mode.startswith("llm") else 0
        self._trace(
            state,
            "router.completed",
            {
                "action": decision.action,
                "confidence": decision.route_confidence,
                "reason": decision.route_reason_public,
                "routing_mode": mode,
                "model": self.settings.router_model if mode.startswith("llm") else "fallback",
                "schema": (
                    "LLMIntentOutput"
                    if mode == "llm_intent"
                    else "LLMRouterOutput"
                    if mode == "llm"
                    else "heuristic"
                ),
                "latency_ms": llm_latency_ms,
            },
            node="route_query",
        )
        return {
            "route_action": decision.action,
            "route_confidence": decision.route_confidence,
            "route_reason_public": decision.route_reason_public,
            "rewritten_query": decision.rewritten_query,
            "retrieval_filters": decision.retrieval_filters,
            "tool_name": decision.tool_name or "",
            "tool_args": decision.tool_args,
            "clarifying_question": decision.clarifying_question or "",
            "refusal_reason": decision.refusal_reason or "",
            "expected_next_node": decision.expected_next_node,
            "retrieval_variant": decision.retrieval_variant,
            "routing_mode": mode,
            "llm_latency_ms": state.get("llm_latency_ms", 0) + llm_latency_ms,
        }

    def clarify(self, state: AgentState) -> AgentState:
        question = state.get("clarifying_question") or (
            "Could you clarify the scope you want within the indexed corpus?"
        )
        return {
            "final_action": "CLARIFY",
            "final_answer": question,
            "citations": [],
            "sources_block": "",
        }

    def retrieve(self, state: AgentState) -> AgentState:
        variant = state.get(
            "forced_retrieval_variant",
            state.get("retrieval_variant", RetrievalVariant.parent_child),
        )
        self._trace(
            state,
            "retrieval.started",
            {"variant": variant.value},
            node="retrieve",
        )
        started = time.monotonic()
        try:
            result = self.deps.retrieval_service.run(
                query=state.get("rewritten_query", state["raw_user_query"]),
                variant=variant,
                retrieval_filters={
                    str(k): str(v) for k, v in (state.get("retrieval_filters", {}) or {}).items()
                },
            )
        except Exception as err:  # noqa: BLE001
            self._trace(
                state,
                "retrieval.failed",
                {"error": str(err), "variant": variant.value},
                node="retrieve",
            )
            return {
                "retrieved_child_ids": [],
                "selected_parent_ids": [],
                "parent_scores": {},
                "evidence_child_ids": [],
                "context_packets": [],
                "evidence_status": EvidenceStatus.insufficient,
                "evidence_confidence": "LOW",
                "evidence_signals": {"error": str(err)},
                "conflict_label": "NONE",
                "refusal_reason": (
                    "I can’t answer from the indexed corpus right now because "
                    "retrieval dependencies are unavailable."
                ),
            }

        retrieval_latency_ms = int((time.monotonic() - started) * 1000)
        self._trace(
            state,
            "retrieval.children_found",
            {"count": len(result.retrieved_child_ids)},
            node="retrieve",
        )
        self._trace(
            state,
            "retrieval.parents_scored",
            {
                "variant": result.variant.value,
                "selected_parents": result.selected_parent_ids,
                "top_parent_score": result.signals.top_parent_score,
                "evidence_status": result.signals.evidence_status.value,
                "source_ids": [packet.source_id for packet in result.context_packets],
                "latency_ms": retrieval_latency_ms,
            },
            node="retrieve",
        )
        self._trace(
            state,
            "context.assembled",
            {
                "count": len(result.context_packets),
                "source_ids": [packet.source_id for packet in result.context_packets],
            },
            node="retrieve",
        )
        if state.get("trace_id"):
            self.deps.trace_writer.retrieval(
                trace_id=state["trace_id"],
                payload={
                    "top_child_score": result.signals.top_child_score,
                    "top_parent_score": result.signals.top_parent_score,
                    "second_parent_score": result.signals.second_parent_score,
                    "score_margin": result.signals.score_margin,
                    "supporting_child_count": result.signals.supporting_child_count,
                    "distinct_parent_count": result.signals.distinct_parent_count,
                    "distinct_paper_count": result.signals.distinct_paper_count,
                    "section_type_distribution": result.signals.section_type_distribution,
                    "confidence_band": result.signals.confidence_band,
                    "evidence_status": result.signals.evidence_status.value,
                    "thresholds_used": result.signals.thresholds_used,
                },
            )
        return {
            "retrieved_child_ids": result.retrieved_child_ids,
            "selected_parent_ids": result.selected_parent_ids,
            "parent_scores": result.parent_scores,
            "evidence_child_ids": result.evidence_child_ids,
            "context_packets": [packet.__dict__ for packet in result.context_packets],
            "evidence_status": result.signals.evidence_status,
            "evidence_confidence": result.signals.confidence_band,
            "evidence_signals": result.signals.__dict__,
            "conflict_label": "NONE",
            "retrieval_latency_ms": state.get("retrieval_latency_ms", 0) + retrieval_latency_ms,
        }

    def tool(self, state: AgentState) -> AgentState:
        tool_name = state.get("tool_name", "")
        tool_started = time.monotonic()
        if tool_name == "arxiv_search":
            self._trace(
                state,
                "tool.started",
                {"tool_name": tool_name, "tool_args": state.get("tool_args", {})},
                node="tool",
            )
            query = str(
                state.get("tool_args", {}).get(
                    "query", state.get("rewritten_query", state["raw_user_query"])
                )
            )
            payload = ArxivSearchInput(query=query)
            output = self.deps.toolset.arxiv_search(payload)
            self._trace(
                state,
                "tool.called",
                {
                    "tool_name": tool_name,
                    "tool_args": payload.model_dump(mode="json"),
                    "status": output.status.value,
                    "papers": len(output.papers),
                },
                node="tool",
            )
            lines = []
            for idx, paper in enumerate(output.papers[:5], start=2):
                lines.append(
                    _format_tool_paper_line(
                        citation_id=f"T{idx}",
                        paper=paper,
                        include_full_abstract=False,
                    )
                )
            sources = [f"[T1] arXiv API arxiv_search query: {payload.query}"]
            tool_latency_ms = int((time.monotonic() - tool_started) * 1000)
            if output.status.value == "error":
                detail = (
                    "; ".join(output.errors[:2]) if output.errors else "unknown arXiv API error"
                )
                answer = f"arXiv search failed [T1]: {detail}"
                citations = ["T1"]
            elif lines:
                answer = "Here are matching arXiv metadata results [T1]:\n\n" + "\n\n".join(lines)
                for idx, paper in enumerate(output.papers[:5], start=2):
                    sources.append(f"[T{idx}] {paper.title}, arXiv:{paper.arxiv_id}")
                citations = [f"T{idx}" for idx in range(1, len(lines) + 2)]
            else:
                answer = "No matching arXiv records were returned for this query [T1]."
                citations = ["T1"]
            if state.get("trace_id"):
                self.deps.trace_writer.tool(
                    trace_id=state["trace_id"],
                    payload={
                        "tool_name": tool_name,
                        "tool_args": payload.model_dump(mode="json"),
                        "tool_status": output.status.value,
                        "tool_latency_ms": tool_latency_ms,
                        "tool_result_summary": f"papers={len(output.papers)}",
                        "tool_error": "; ".join(output.errors[:2]) if output.errors else "",
                    },
                )
            self._trace(
                state,
                "tool.completed",
                {
                    "tool_name": tool_name,
                    "status": output.status.value,
                    "papers": len(output.papers),
                    "latency_ms": tool_latency_ms,
                },
                node="tool",
            )
            return {
                "tool_result": {"tool_name": tool_name, **output.model_dump(mode="json")},
                "final_action": "ANSWER_FROM_TOOL",
                "final_answer": answer,
                "citations": citations,
                "sources_block": build_sources_block(source_lines=[], tool_lines=sources),
                "tool_latency_ms": state.get("tool_latency_ms", 0) + tool_latency_ms,
                "tool_calls": [
                    {
                        "tool_name": tool_name,
                        "tool_args": payload.model_dump(mode="json"),
                        "status": output.status.value,
                    }
                ],
            }
        if tool_name == "arxiv_lookup_by_id":
            self._trace(
                state,
                "tool.started",
                {"tool_name": tool_name, "tool_args": state.get("tool_args", {})},
                node="tool",
            )
            tool_args = state.get("tool_args", {})
            raw_ids = tool_args.get("arxiv_ids")
            ids = _coerce_arxiv_ids(raw_ids)
            if not ids:
                ids = _coerce_arxiv_ids(state.get("rewritten_query", state["raw_user_query"]))
            include_abstract = _coerce_bool(tool_args.get("include_abstract"), default=True)
            payload = ArxivLookupByIdInput(arxiv_ids=ids, include_abstract=include_abstract)
            output = self.deps.toolset.arxiv_lookup_by_id(payload)
            self._trace(
                state,
                "tool.called",
                {
                    "tool_name": tool_name,
                    "tool_args": payload.model_dump(mode="json"),
                    "status": output.status.value,
                    "papers": len(output.papers),
                },
                node="tool",
            )
            lines = []
            for idx, paper in enumerate(output.papers[:5], start=2):
                lines.append(
                    _format_tool_paper_line(
                        citation_id=f"T{idx}",
                        paper=paper,
                        include_full_abstract=True,
                    )
                )
            sources = [f"[T1] arXiv API arxiv_lookup_by_id ids: {', '.join(payload.arxiv_ids)}"]
            tool_latency_ms = int((time.monotonic() - tool_started) * 1000)
            if lines:
                answer = "Here are arXiv lookup results [T1]:\n\n" + "\n\n".join(lines)
                for idx, paper in enumerate(output.papers[:5], start=2):
                    sources.append(f"[T{idx}] {paper.title}, arXiv:{paper.arxiv_id}")
                citations = [f"T{idx}" for idx in range(1, len(lines) + 2)]
            else:
                answer = "No matching arXiv records were returned for those IDs [T1]."
                citations = ["T1"]
            if state.get("trace_id"):
                self.deps.trace_writer.tool(
                    trace_id=state["trace_id"],
                    payload={
                        "tool_name": tool_name,
                        "tool_args": payload.model_dump(mode="json"),
                        "tool_status": output.status.value,
                        "tool_latency_ms": tool_latency_ms,
                        "tool_result_summary": f"papers={len(output.papers)}",
                        "tool_error": "; ".join(output.errors[:2]) if output.errors else "",
                    },
                )
            self._trace(
                state,
                "tool.completed",
                {
                    "tool_name": tool_name,
                    "status": output.status.value,
                    "papers": len(output.papers),
                    "latency_ms": tool_latency_ms,
                },
                node="tool",
            )
            return {
                "tool_result": {"tool_name": tool_name, **output.model_dump(mode="json")},
                "final_action": "ANSWER_FROM_TOOL",
                "final_answer": answer,
                "citations": citations,
                "sources_block": build_sources_block(source_lines=[], tool_lines=sources),
                "tool_latency_ms": state.get("tool_latency_ms", 0) + tool_latency_ms,
                "tool_calls": [
                    {
                        "tool_name": tool_name,
                        "tool_args": payload.model_dump(mode="json"),
                        "status": output.status.value,
                    }
                ],
            }
        return {
            "final_action": "REFUSE",
            "final_answer": "Unsupported tool route.",
            "refusal_reason": "Router requested an unsupported tool.",
            "citations": [],
            "sources_block": "",
            "tool_calls": [
                {
                    "tool_name": tool_name,
                    "tool_args": state.get("tool_args", {}),
                    "status": "unsupported",
                }
            ],
        }

    def evidence_check(self, state: AgentState) -> AgentState:
        packets = state.get("context_packets", [])
        llm_result = self._classify_evidence_with_llm(state, packets)
        if llm_result is not None:
            evidence_status = EvidenceStatus(llm_result.evidence_status)
            if (
                evidence_status == EvidenceStatus.contradictory
                or llm_result.conflict_label != "NONE"
            ):
                final_action = "CONTRADICTION_HANDLER"
            else:
                final_action = llm_result.recommended_action
            confidence = llm_result.confidence_band
            conflict_label = llm_result.conflict_label
            update = {
                "evidence_status": evidence_status,
                "evidence_confidence": confidence,
                "conflict_label": conflict_label,
            }
        else:
            status = state.get("evidence_status", EvidenceStatus.insufficient)
            conflict_label = state.get("conflict_label", "NONE")
            if status == EvidenceStatus.contradictory or conflict_label != "NONE":
                final_action = "CONTRADICTION_HANDLER"
            elif status == EvidenceStatus.sufficient:
                final_action = "ANSWER_FROM_CONTEXT"
            elif status == EvidenceStatus.ambiguous:
                final_action = "CLARIFY"
            else:
                final_action = "REFUSE"
            update = {"conflict_label": conflict_label}

        self._trace(
            state,
            "evidence.checked",
            {
                "status": str(
                    update.get("evidence_status", state.get("evidence_status", "UNKNOWN"))
                ),
                "confidence": update.get(
                    "evidence_confidence", state.get("evidence_confidence", "LOW")
                ),
                "conflict_label": update.get("conflict_label", "NONE"),
                "final_evidence_action": final_action,
                "signals": state.get("evidence_signals", {}),
            },
            node="evidence_check",
        )
        if state.get("trace_id"):
            self.deps.trace_writer.evidence(
                trace_id=state["trace_id"],
                payload={
                    "evidence_status": str(
                        update.get("evidence_status", state.get("evidence_status", "UNKNOWN"))
                    )
                    .replace("EvidenceStatus.", "")
                    .upper(),
                    "confidence_band": update.get(
                        "evidence_confidence", state.get("evidence_confidence", "LOW")
                    ),
                    "missing_info": state.get("refusal_reason", ""),
                    "contradiction_notes": str(update.get("conflict_label", "NONE")),
                },
            )
        return {**update, "final_action": final_action}

    def answer(self, state: AgentState) -> AgentState:
        packets = state.get("context_packets", [])
        if not packets:
            return {
                "final_action": "REFUSE",
                "final_answer": (
                    "I can’t answer that from the indexed arXiv corpus. "
                    "The retrieved sections do not provide sufficient evidence."
                ),
                "refusal_reason": "No retrieved evidence.",
                "citations": [],
                "sources_block": "",
            }

        llm_started = time.monotonic()
        llm_answer = self._answer_with_llm(state, packets)
        llm_latency_ms = int((time.monotonic() - llm_started) * 1000) if llm_answer else 0
        if llm_answer is None:
            llm_answer = self._fallback_answer(packets)

        if llm_answer.final_action == "CLARIFY":
            return {
                "final_action": "CLARIFY",
                "final_answer": llm_answer.clarifying_question or "Could you clarify your request?",
                "citations": [],
                "sources_block": "",
                "llm_latency_ms": state.get("llm_latency_ms", 0) + llm_latency_ms,
            }
        if llm_answer.final_action == "REFUSE":
            return {
                "final_action": "REFUSE",
                "final_answer": llm_answer.refusal_reason
                or "I can’t answer that from the indexed arXiv corpus.",
                "citations": [],
                "sources_block": "",
                "llm_latency_ms": state.get("llm_latency_ms", 0) + llm_latency_ms,
            }

        source_lines = [self._source_line(packet) for packet in packets]
        sources_block = build_sources_block(source_lines=source_lines, tool_lines=[])
        citations = sorted(set(llm_answer.cited_source_ids + llm_answer.cited_tool_ids))
        self._trace(
            state,
            "answer.generated",
            {
                "final_action": "ANSWER_FROM_CONTEXT",
                "citations": citations,
                "source_ids": [packet["source_id"] for packet in packets],
            },
            node="answer",
        )
        return {
            "final_action": "ANSWER_FROM_CONTEXT",
            "final_answer": llm_answer.answer_text,
            "citations": citations,
            "sources_block": sources_block,
            "llm_latency_ms": state.get("llm_latency_ms", 0) + llm_latency_ms,
        }

    def refuse(self, state: AgentState) -> AgentState:
        reason = state.get("refusal_reason") or (
            "I can’t answer that from the indexed arXiv corpus. "
            "The retrieved sections do not provide sufficient evidence."
        )
        return {
            "final_action": "REFUSE",
            "final_answer": reason,
            "citations": [],
            "sources_block": "",
        }

    def citation_validate(self, state: AgentState) -> AgentState:
        final_action = state.get("final_action", "")
        if final_action in {"CLARIFY", "REFUSE"}:
            return {}

        answer = state.get("final_answer", "")
        sources_block = state.get("sources_block", "")
        if sources_block and "Sources:" not in answer:
            answer = f"{answer}\n\n{sources_block}"

        source_ids = {packet["source_id"] for packet in state.get("context_packets", [])}
        tool_ids = set(re.findall(r"^\[(T\d+)\]", sources_block, flags=re.MULTILINE))
        if final_action == "ANSWER_FROM_TOOL" and not tool_ids:
            tool_ids = {cid for cid in state.get("citations", []) if str(cid).startswith("T")}

        evidence_status = state.get("evidence_status", EvidenceStatus.insufficient)
        if hasattr(evidence_status, "value"):
            evidence_status = evidence_status.value

        ok, message = validate_citations(
            answer=answer,
            sources_block=sources_block,
            allowed_source_ids=source_ids,
            allowed_tool_ids=tool_ids,
            final_action=final_action,
            evidence_status=str(evidence_status),
        )
        self._trace(
            state,
            "citation.deterministic_validated",
            {
                "status": "ok" if ok else "failed",
                "message": message,
                "allowed_source_ids": sorted(source_ids),
                "allowed_tool_ids": sorted(tool_ids),
            },
            node="citation_validate",
        )
        if ok:
            llm_validation = self._llm_citation_validate(
                state=state,
                final_action=final_action,
                answer=answer,
                sources_block=sources_block,
                source_ids=sorted(source_ids),
                tool_ids=sorted(tool_ids),
            )
            if llm_validation is not None and not llm_validation.valid:
                self._trace(
                    state,
                    "citation.llm_validated",
                    {
                        "status": "failed",
                        "verdict": llm_validation.verdict,
                        "unknown_citation_ids": llm_validation.unknown_citation_ids,
                        "missing_citation_spans": llm_validation.missing_citation_spans[:3],
                    },
                    node="citation_validate",
                )
                self._trace(
                    state,
                    "citation.validated",
                    {"status": "failed", "mode": "llm", "verdict": llm_validation.verdict},
                    node="citation_validate",
                )
                return {
                    "final_action": "REFUSE",
                    "final_answer": (
                        "I can’t provide a grounded answer with valid citations for that query."
                    ),
                    "citations": [],
                    "sources_block": "",
                }
            if llm_validation is not None:
                self._trace(
                    state,
                    "citation.llm_validated",
                    {"status": "ok", "verdict": llm_validation.verdict},
                    node="citation_validate",
                )
            else:
                self._trace(
                    state,
                    "citation.llm_validated",
                    {"status": "skipped"},
                    node="citation_validate",
                )
            self._trace(
                state,
                "citation.validated",
                {
                    "status": "ok",
                    "citations": sorted(source_ids | tool_ids),
                    "mode": "deterministic+llm",
                },
                node="citation_validate",
            )
            if state.get("trace_id"):
                self.deps.trace_writer.answer(
                    trace_id=state["trace_id"],
                    payload={
                        "final_action": final_action,
                        "citation_ids": state.get("citations", []),
                        "answer_text": answer,
                        "refusal_reason": state.get("refusal_reason", ""),
                        "clarifying_question": state.get("clarifying_question", ""),
                    },
                )
            return {"final_answer": answer}
        self._trace(
            state,
            "citation.validated",
            {"status": "failed", "message": message, "mode": "deterministic"},
            node="citation_validate",
        )
        self._trace(
            state,
            "citation.llm_validated",
            {"status": "skipped", "reason": "deterministic_failed"},
            node="citation_validate",
        )
        return {
            "final_action": "REFUSE",
            "final_answer": (
                "I can’t provide a grounded answer with valid citations for that query."
            ),
            "citations": [],
            "sources_block": "",
        }

    def _llm_citation_validate(
        self,
        *,
        state: AgentState,
        final_action: str,
        answer: str,
        sources_block: str,
        source_ids: list[str],
        tool_ids: list[str],
    ) -> LLMCitationValidationOutput | None:
        if not self.deps.llm_client.enabled():
            return None
        started = time.monotonic()
        try:
            out = self.deps.llm_client.complete_json(
                model=self.settings.evidence_model,
                schema=LLMCitationValidationOutput,
                system_prompt=CITATION_SYSTEM_PROMPT,
                user_prompt=citation_user_prompt(
                    final_action=final_action,
                    answer_text=answer,
                    sources_block=sources_block,
                    allowed_source_ids=source_ids,
                    allowed_tool_ids=tool_ids,
                ),
            )
            elapsed = int((time.monotonic() - started) * 1000)
            self._trace(
                state,
                "llm.call",
                {
                    "model": self.settings.evidence_model,
                    "schema": "LLMCitationValidationOutput",
                    "status": "ok",
                    "latency_ms": elapsed,
                },
                node="citation_validate",
            )
            return out
        except Exception as err:  # noqa: BLE001
            elapsed = int((time.monotonic() - started) * 1000)
            self._trace(
                state,
                "llm.call",
                {
                    "model": self.settings.evidence_model,
                    "schema": "LLMCitationValidationOutput",
                    "status": "failed",
                    "latency_ms": elapsed,
                    "error": str(err),
                },
                node="citation_validate",
            )
            return None

    def memory_update(self, state: AgentState) -> AgentState:
        final_action = state.get("final_action", "")
        if final_action not in {"ANSWER_FROM_CONTEXT", "ANSWER_FROM_TOOL", "CLARIFY", "REFUSE"}:
            return {}
        if final_action in {"CLARIFY", "REFUSE"}:
            episode_id = self.deps.memory_service.write_episode(
                thread_id=state.get("thread_id", "default"),
                turn_id=state["turn_id"],
                user_query=state.get("raw_user_query", ""),
                route_action=state.get("route_action", ""),
                route_confidence=state.get("route_confidence"),
                retrieved_child_ids=state.get("retrieved_child_ids", []),
                selected_parent_ids=state.get("selected_parent_ids", []),
                tool_calls=state.get("tool_calls", []),
                final_action=final_action,
                final_answer_summary=_truncate_text(str(state.get("final_answer", "")), 280),
            )
            self._trace(
                state,
                "episode.written",
                {"episode_id": episode_id},
                node="memory_update",
            )
            conversation_update = self._build_checkpoint_conversation_update(state)
            return {**conversation_update, "episode_id": episode_id}
        memory_write = self._memory_write_decision(state)
        self._trace(
            state,
            "memory.write_decision",
            {
                "should_write": memory_write.should_write,
                "kind": memory_write.kind,
                "key": memory_write.key,
            },
            node="memory_update",
        )
        if not memory_write.should_write or not memory_write.key or not memory_write.value:
            episode_id = self.deps.memory_service.write_episode(
                thread_id=state.get("thread_id", "default"),
                turn_id=state["turn_id"],
                user_query=state.get("raw_user_query", ""),
                route_action=state.get("route_action", ""),
                route_confidence=state.get("route_confidence"),
                retrieved_child_ids=state.get("retrieved_child_ids", []),
                selected_parent_ids=state.get("selected_parent_ids", []),
                tool_calls=state.get("tool_calls", []),
                final_action=state.get("final_action", ""),
                final_answer_summary=_truncate_text(str(state.get("final_answer", "")), 280),
            )
            self._trace(
                state,
                "episode.written",
                {"episode_id": episode_id},
                node="memory_update",
            )
            conversation_update = self._build_checkpoint_conversation_update(state)
            return {**conversation_update, "episode_id": episode_id}

        memory_id = self.deps.memory_service.write(
            kind=memory_write.kind,
            key=memory_write.key,
            value=memory_write.value,
            confidence=memory_write.confidence,
            source_turn_id=state["turn_id"],
        )
        self._trace(
            state,
            "memory.write",
            {"memory_id": memory_id, "key": memory_write.key, "kind": memory_write.kind},
        )
        episode_id = self.deps.memory_service.write_episode(
            thread_id=state.get("thread_id", "default"),
            turn_id=state["turn_id"],
            user_query=state.get("raw_user_query", ""),
            route_action=state.get("route_action", ""),
            route_confidence=state.get("route_confidence"),
            retrieved_child_ids=state.get("retrieved_child_ids", []),
            selected_parent_ids=state.get("selected_parent_ids", []),
            tool_calls=state.get("tool_calls", []),
            final_action=state.get("final_action", ""),
            final_answer_summary=_truncate_text(str(state.get("final_answer", "")), 280),
        )
        self._trace(
            state,
            "episode.written",
            {"episode_id": episode_id},
            node="memory_update",
        )
        conversation_update = self._build_checkpoint_conversation_update(state)
        return {**conversation_update, "episode_id": episode_id}

    def _trace(
        self,
        state: AgentState,
        event: str,
        payload: dict[str, object],
        node: str | None = None,
    ) -> None:
        trace_id = state.get("trace_id")
        if not trace_id:
            return
        self.deps.trace_writer.event(
            trace_id=trace_id,
            level="info",
            event=event,
            payload=payload,
            node=node,
        )

    def route_branch(self, state: AgentState) -> str:
        action = state.get("route_action", "REFUSE")
        return {
            "CLARIFY": "clarify",
            "TOOL": "tool",
            "RETRIEVE": "retrieve",
            "REFUSE": "refuse",
            "ANSWER_FROM_MEMORY": "answer",
            "ANSWER_FROM_CONTEXT": "answer",
        }.get(action, "refuse")

    def evidence_branch(self, state: AgentState) -> str:
        action = state.get("final_action", "REFUSE")
        return {
            "ANSWER_FROM_CONTEXT": "answer",
            "CLARIFY": "clarify",
            "REFUSE": "refuse",
            "CONTRADICTION_HANDLER": "contradiction_handler",
        }.get(action, "refuse")

    def contradiction_handler(self, state: AgentState) -> AgentState:
        packets = state.get("context_packets", [])
        self._trace(
            state,
            "contradiction.detected",
            {
                "evidence_status": str(state.get("evidence_status", "")),
                "conflict_label": state.get("conflict_label", "NONE"),
            },
            node="contradiction_handler",
        )
        decision = self._classify_contradiction_with_llm(state=state, packets=packets)
        if decision is None:
            decision = LLMContradictionOutput(
                conflict_label="DIRECT_CONTRADICTION",
                recommended_action="CLARIFY",
                summary="Conflicting evidence could not be resolved confidently.",
                side_a_source_ids=[],
                side_b_source_ids=[],
                metadata_ids_to_check=[],
                missing_info="Need a narrower question to resolve conflicting claims.",
                confidence_band="LOW",
            )
        self._trace(
            state,
            "contradiction.checked",
            {
                "conflict_label": decision.conflict_label,
                "recommended_action": decision.recommended_action,
                "confidence_band": decision.confidence_band,
            },
            node="contradiction_handler",
        )
        next_node = {
            "ANSWER_WITH_CONFLICT": "answer",
            "CLARIFY": "clarify",
            "REFUSE": "refuse",
            "TOOL_LOOKUP": "tool",
        }[decision.recommended_action]
        update: AgentState = {
            "conflict_label": decision.conflict_label,
            "contradiction_action": decision.recommended_action,
            "refusal_reason": decision.missing_info or state.get("refusal_reason", ""),
            "clarifying_question": decision.missing_info or state.get("clarifying_question", ""),
            "final_action": (
                "ANSWER_FROM_CONTEXT"
                if decision.recommended_action == "ANSWER_WITH_CONFLICT"
                else "CLARIFY"
                if decision.recommended_action == "CLARIFY"
                else "REFUSE"
            ),
        }
        if decision.recommended_action == "TOOL_LOOKUP":
            update["tool_name"] = "arxiv_lookup_by_id"
            if decision.metadata_ids_to_check:
                update["tool_args"] = {"arxiv_ids": decision.metadata_ids_to_check}
            self._trace(
                state,
                "contradiction.tool_lookup_requested",
                {"metadata_ids_to_check": decision.metadata_ids_to_check},
                node="contradiction_handler",
            )
        self._trace(
            state,
            "contradiction.handled",
            {"next_node": next_node, "conflict_label": decision.conflict_label},
            node="contradiction_handler",
        )
        return update

    def contradiction_branch(self, state: AgentState) -> str:
        action = state.get("contradiction_action", "")
        return {
            "ANSWER_WITH_CONFLICT": "answer",
            "CLARIFY": "clarify",
            "REFUSE": "refuse",
            "TOOL_LOOKUP": "tool",
        }.get(action, "refuse")

    def _classify_evidence_with_llm(
        self, state: AgentState, packets: list[dict[str, object]]
    ) -> LLMEvidenceOutput | None:
        if not packets:
            return None
        if not self.deps.llm_client.enabled():
            return None
        started = time.monotonic()
        try:
            out = self.deps.llm_client.complete_json(
                model=self.settings.evidence_model,
                schema=LLMEvidenceOutput,
                system_prompt=EVIDENCE_SYSTEM_PROMPT,
                user_prompt=evidence_user_prompt(
                    query=state.get("rewritten_query", state["raw_user_query"]),
                    context_packets=packets,
                    retrieval_signals=state.get("evidence_signals", {}),
                ),
            )
            elapsed = int((time.monotonic() - started) * 1000)
            self._trace(
                state,
                "llm.call",
                {
                    "model": self.settings.evidence_model,
                    "schema": "LLMEvidenceOutput",
                    "status": "ok",
                    "latency_ms": elapsed,
                },
                node="evidence_check",
            )
            return out
        except Exception:  # noqa: BLE001
            elapsed = int((time.monotonic() - started) * 1000)
            self._trace(
                state,
                "llm.call",
                {
                    "model": self.settings.evidence_model,
                    "schema": "LLMEvidenceOutput",
                    "status": "failed",
                    "latency_ms": elapsed,
                },
                node="evidence_check",
            )
            return None

    def _answer_with_llm(
        self, state: AgentState, packets: list[dict[str, object]]
    ) -> LLMAnswerOutput | None:
        if not self.deps.llm_client.enabled():
            return None
        evidence_status = state.get("evidence_status", EvidenceStatus.insufficient)
        if hasattr(evidence_status, "value"):
            evidence_status = evidence_status.value
        started = time.monotonic()
        try:
            answer = self.deps.llm_client.complete_json(
                model=self.settings.answer_model,
                schema=LLMAnswerOutput,
                system_prompt=ANSWER_SYSTEM_PROMPT,
                user_prompt=answer_user_prompt(
                    query=state.get("rewritten_query", state["raw_user_query"]),
                    context_packets=packets,
                    evidence_status=str(evidence_status),
                    tool_result=state.get("tool_result"),
                ),
            )
            elapsed = int((time.monotonic() - started) * 1000)
            self._trace(
                state,
                "llm.call",
                {
                    "model": self.settings.answer_model,
                    "schema": "LLMAnswerOutput",
                    "status": "ok",
                    "latency_ms": elapsed,
                },
                node="answer",
            )
            if answer.final_action == "ANSWER_FROM_CONTEXT" and not answer.answer_text:
                return None
            return answer
        except Exception:  # noqa: BLE001
            elapsed = int((time.monotonic() - started) * 1000)
            self._trace(
                state,
                "llm.call",
                {
                    "model": self.settings.answer_model,
                    "schema": "LLMAnswerOutput",
                    "status": "failed",
                    "latency_ms": elapsed,
                },
                node="answer",
            )
            return None

    def _fallback_answer(self, packets: list[dict[str, object]]) -> LLMAnswerOutput:
        claims: list[str] = []
        for packet in packets[:3]:
            evidence = str(packet.get("highlighted_evidence", "")).splitlines()[0:1]
            text = evidence[0] if evidence else str(packet.get("section_context", ""))[:200]
            text = _strip_internal_ids(text)
            claims.append(f"{text} [{packet['source_id']}]")
        answer_text = "Based on the retrieved corpus sections:\n" + "\n".join(
            f"- {claim}" for claim in claims
        )
        return LLMAnswerOutput(
            final_action="ANSWER_FROM_CONTEXT",
            answer_text=answer_text,
            cited_source_ids=[packet["source_id"] for packet in packets[:3]],
        )

    def _source_line(self, packet: dict[str, object]) -> str:
        page_start = packet.get("page_start")
        page_end = packet.get("page_end")
        if page_start is None or page_end is None:
            page_span = "page unavailable"
        else:
            page_span = f"pp. {page_start}-{page_end}"
        return (
            f"[{packet['source_id']}] {packet['title']}, arXiv:{packet['arxiv_id']}, "
            f"{packet['section_path']}, {page_span}"
        )

    def _memory_write_decision(self, state: AgentState) -> LLMMemoryWriteOutput:
        query = state.get("raw_user_query", "")
        lowered = query.lower()
        trigger_terms = ("remember", "lock this", "we choose", "decision", "prefer", "constraint")
        if not any(term in lowered for term in trigger_terms):
            return LLMMemoryWriteOutput(should_write=False)
        if not self.deps.llm_client.enabled():
            return LLMMemoryWriteOutput(
                should_write=True,
                kind="decision",
                key="user_decision",
                value=query[:200],
                confidence=0.8,
            )
        started = time.monotonic()
        try:
            decision = self.deps.llm_client.complete_json(
                model=self.settings.evidence_model,
                schema=LLMMemoryWriteOutput,
                system_prompt=MEMORY_SYSTEM_PROMPT,
                user_prompt=memory_user_prompt(
                    query=query,
                    answer=state.get("final_answer", ""),
                    final_action=state.get("final_action", ""),
                ),
            )
            elapsed = int((time.monotonic() - started) * 1000)
            self._trace(
                state,
                "llm.call",
                {
                    "model": self.settings.evidence_model,
                    "schema": "LLMMemoryWriteOutput",
                    "status": "ok",
                    "latency_ms": elapsed,
                },
                node="memory_update",
            )
            return decision
        except Exception:  # noqa: BLE001
            elapsed = int((time.monotonic() - started) * 1000)
            self._trace(
                state,
                "llm.call",
                {
                    "model": self.settings.evidence_model,
                    "schema": "LLMMemoryWriteOutput",
                    "status": "failed",
                    "latency_ms": elapsed,
                },
                node="memory_update",
            )
            return LLMMemoryWriteOutput(should_write=False)

    def _build_checkpoint_conversation_update(self, state: AgentState) -> AgentState:
        user_query = state.get("raw_user_query", "")
        assistant_answer = state.get("final_answer", "")
        message_count = len(state.get("messages", [])) + 1
        summary = (
            f"User asked: {user_query[:180]} | Assistant: "
            f"{' '.join(assistant_answer.strip().split())[:240]}"
        )
        active_focus = extract_focus(user_query=user_query, assistant_answer=assistant_answer)
        active_arxiv_ids = extract_arxiv_ids(f"{user_query}\n{assistant_answer}")
        active_paper_ids = [f"paper_{aid.lower()}" for aid in active_arxiv_ids]
        self._trace(
            state,
            "conversation.updated",
            {
                "thread_id": state.get("thread_id", "default"),
                "active_focus": active_focus,
                "message_count": message_count,
            },
            node="memory_update",
        )
        self._trace(
            state,
            "checkpoint.persisted",
            {"thread_id": state.get("thread_id", "default"), "message_count": message_count},
            node="memory_update",
        )
        return {
            "messages": [{"role": "assistant", "content": assistant_answer}],
            "conversation_summary": summary,
            "active_focus": active_focus,
            "active_arxiv_ids": active_arxiv_ids,
            "active_paper_ids": active_paper_ids,
            "last_answer_summary": _truncate_text(assistant_answer, 280),
            "last_selected_parent_ids": state.get("selected_parent_ids", []),
            "last_retrieved_child_ids": state.get("retrieved_child_ids", []),
            "last_tool_calls": state.get("tool_calls", []),
        }

    def _recent_turns_from_messages(
        self, messages: list[dict[str, object] | object], limit: int
    ) -> list[dict[str, object]]:
        turns: list[dict[str, object]] = []
        for msg in messages[-limit:]:
            if isinstance(msg, dict):
                role = str(msg.get("role", ""))
                content = str(msg.get("content", ""))
            else:
                role = str(getattr(msg, "type", ""))
                content = str(getattr(msg, "content", ""))
            if role and content:
                turns.append({"role": role, "content": content})
        return turns

    def _classify_contradiction_with_llm(
        self, *, state: AgentState, packets: list[dict[str, object]]
    ) -> LLMContradictionOutput | None:
        if not self.deps.llm_client.enabled():
            return None
        started = time.monotonic()
        try:
            out = self.deps.llm_client.complete_json(
                model=self.settings.evidence_model,
                schema=LLMContradictionOutput,
                system_prompt=CONTRADICTION_SYSTEM_PROMPT,
                user_prompt=contradiction_user_prompt(
                    query=state.get("rewritten_query", state.get("raw_user_query", "")),
                    conflict_label=state.get("conflict_label", "NONE"),
                    context_packets=packets,
                    evidence_status=str(state.get("evidence_status", "")),
                ),
            )
            elapsed = int((time.monotonic() - started) * 1000)
            self._trace(
                state,
                "llm.call",
                {
                    "model": self.settings.evidence_model,
                    "schema": "LLMContradictionOutput",
                    "status": "ok",
                    "latency_ms": elapsed,
                },
                node="contradiction_handler",
            )
            return out
        except Exception as err:  # noqa: BLE001
            elapsed = int((time.monotonic() - started) * 1000)
            self._trace(
                state,
                "llm.call",
                {
                    "model": self.settings.evidence_model,
                    "schema": "LLMContradictionOutput",
                    "status": "failed",
                    "latency_ms": elapsed,
                    "error": str(err),
                },
                node="contradiction_handler",
            )
            return None


def _strip_internal_ids(text: str) -> str:
    text = re.sub(r"\bchunk_[a-zA-Z0-9_]+\b", "", text)
    text = re.sub(r"\bparent_[a-zA-Z0-9_]+\b", "", text)
    text = re.sub(r"\bpaper_[a-zA-Z0-9_]+\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _coerce_arxiv_ids(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if "," in stripped:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        matches = re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", stripped)
        return matches or [stripped]
    return []


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return default


def _format_tool_paper_line(*, citation_id: str, paper: object, include_full_abstract: bool) -> str:
    title = str(getattr(paper, "title", "") or "").strip() or "Untitled"
    arxiv_id = str(getattr(paper, "arxiv_id", "") or "").strip()
    version = str(getattr(paper, "version", "") or "").strip()
    arxiv_label = f"{arxiv_id}{version}" if version else arxiv_id
    primary_category = str(getattr(paper, "primary_category", "") or "").strip()
    categories = getattr(paper, "categories", []) or []
    category = primary_category or (categories[0] if categories else "unknown")
    published = _format_date(getattr(paper, "published_at", None))
    updated = _format_date(getattr(paper, "updated_at", None))
    authors = getattr(paper, "authors", []) or []
    author_line = _format_authors([str(a) for a in authors if str(a).strip()])

    abstract = str(getattr(paper, "abstract", "") or "").strip()
    if abstract and not include_full_abstract:
        abstract = _truncate_text(abstract, max_chars=320)

    line_parts = [
        f"[{citation_id}] {title}",
        f"arXiv:{arxiv_label} | Category: {category} | Published: {published} | Updated: {updated}",
        f"Authors: {author_line}",
    ]
    if abstract:
        line_parts.append(f"Abstract: {abstract}")
    return "\n".join(line_parts)


def _format_authors(authors: list[str]) -> str:
    if not authors:
        return "unknown"
    if len(authors) <= 3:
        return ", ".join(authors)
    return f"{', '.join(authors[:3])}, et al."


def _format_date(value: object) -> str:
    if value is None:
        return "unknown"
    if hasattr(value, "date"):
        try:
            return str(value.date())
        except Exception:  # noqa: BLE001
            pass
    text = str(value).strip()
    return text[:10] if len(text) >= 10 else text or "unknown"


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."

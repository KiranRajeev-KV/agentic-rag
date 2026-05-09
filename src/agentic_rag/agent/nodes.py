from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.retrieval.types import EvidenceStatus, RetrievalVariant
from agentic_rag.tools.arxiv_tools import ArxivToolset
from agentic_rag.tools.schemas import ArxivSearchInput

from .citations import build_sources_block, validate_citations
from .memory import MemoryService
from .router import RouterDecision, route_query
from .state import AgentState


@dataclass
class AgentDependencies:
    retrieval_service: object
    memory_service: MemoryService
    toolset: ArxivToolset
    trace_writer: object


class AgentNodes:
    def __init__(self, deps: AgentDependencies) -> None:
        self.deps = deps

    def load_state(self, state: AgentState) -> AgentState:
        memories = self.deps.memory_service.read_recent(limit=10)
        self._trace(state, "memory.read", {"semantic_count": len(memories)})
        return {"memory_context": memories}

    def route_query(self, state: AgentState) -> AgentState:
        decision: RouterDecision = route_query(state["raw_user_query"])
        self._trace(
            state,
            "router.completed",
            {
                "action": decision.action,
                "confidence": decision.route_confidence,
                "reason": decision.route_reason_public,
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
        }

    def clarify(self, state: AgentState) -> AgentState:
        question = state.get("clarifying_question") or "Could you clarify what you want to compare?"
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
        try:
            result = self.deps.retrieval_service.run(
                query=state.get("rewritten_query", state["raw_user_query"]),
                variant=variant,
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
                "refusal_reason": (
                    "I can’t answer from the indexed corpus right now because "
                    "retrieval dependencies are unavailable "
                    "(embedding model or vector index)."
                ),
            }
        self._trace(
            state,
            "retrieval.parents_scored",
            {
                "variant": result.variant.value,
                "selected_parents": result.selected_parent_ids,
                "top_parent_score": result.signals.top_parent_score,
                "evidence_status": result.signals.evidence_status.value,
            },
            node="retrieve",
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
        }

    def tool(self, state: AgentState) -> AgentState:
        tool_name = state.get("tool_name", "")
        if tool_name == "arxiv_search":
            payload = ArxivSearchInput(query=state.get("rewritten_query", state["raw_user_query"]))
            output = self.deps.toolset.arxiv_search(payload)
            self._trace(
                state,
                "tool.called",
                {
                    "tool_name": tool_name,
                    "status": output.status.value,
                    "papers": len(output.papers),
                },
                node="tool",
            )
            lines = []
            for idx, paper in enumerate(output.papers[:5], start=1):
                lines.append(f"[T{idx}] {paper.title} arXiv:{paper.arxiv_id}")
            answer = (
                "Here are matching arXiv metadata results:\n" + "\n".join(lines)
                if lines
                else "No matches."
            )
            sources = [
                f"[T{idx}] arXiv API arxiv_search query: {payload.query}"
                for idx, _ in enumerate(lines, 1)
            ]
            return {
                "tool_result": output.model_dump(mode="json"),
                "final_action": "ANSWER_FROM_TOOL",
                "final_answer": answer,
                "citations": [f"T{idx}" for idx, _ in enumerate(lines, 1)],
                "sources_block": build_sources_block(source_lines=[], tool_lines=sources),
            }
        return {
            "final_action": "REFUSE",
            "final_answer": "Unsupported tool route.",
            "refusal_reason": "Router requested an unsupported tool.",
            "citations": [],
            "sources_block": "",
        }

    def evidence_check(self, state: AgentState) -> AgentState:
        status = state.get("evidence_status", EvidenceStatus.insufficient)
        if status == EvidenceStatus.sufficient:
            final_action = "ANSWER_FROM_CONTEXT"
        elif status == EvidenceStatus.ambiguous:
            final_action = "CLARIFY"
        else:
            final_action = "REFUSE"
        self._trace(
            state,
            "evidence.checked",
            {
                "status": status.value,
                "confidence": state.get("evidence_confidence", "LOW"),
                "final_evidence_action": final_action,
            },
            node="evidence_check",
        )
        return {"final_action": final_action}

    def answer(self, state: AgentState) -> AgentState:
        packets = state.get("context_packets", [])
        if not packets:
            return {
                "final_action": "REFUSE",
                "final_answer": "I can’t answer that from the indexed arXiv corpus.",
                "refusal_reason": "No retrieved evidence.",
                "citations": [],
                "sources_block": "",
            }

        claims: list[str] = []
        source_lines: list[str] = []
        for packet in packets[:4]:
            source_id = packet["source_id"]
            evidence = (
                packet["highlighted_evidence"].splitlines()[0]
                if packet["highlighted_evidence"]
                else ""
            )
            short = evidence[:200] if evidence else packet["section_context"][:200]
            claims.append(f"{short} [{source_id}]")
            source_lines.append(
                f"[{source_id}] {packet['title']}, arXiv:{packet['arxiv_id']}, "
                f"{packet['section_path']}, pp. {packet['page_start']}-{packet['page_end']}"
            )

        answer = "Evidence from retrieved sections:\n" + "\n".join(f"- {claim}" for claim in claims)
        sources_block = build_sources_block(source_lines=source_lines, tool_lines=[])
        return {
            "final_action": "ANSWER_FROM_CONTEXT",
            "final_answer": answer,
            "citations": [packet["source_id"] for packet in packets[:4]],
            "sources_block": sources_block,
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
        if state.get("sources_block"):
            answer = f"{answer}\n\n{state['sources_block']}"
        source_ids = {packet["source_id"] for packet in state.get("context_packets", [])}
        tool_ids = set(state.get("citations", []))
        require_source = final_action == "ANSWER_FROM_CONTEXT"
        ok, message = validate_citations(
            answer=answer,
            allowed_source_ids=source_ids,
            allowed_tool_ids=tool_ids,
            require_source_citation=require_source,
        )
        if ok:
            return {"final_answer": answer}
        self._trace(
            state,
            "citation.validated",
            {"status": "failed", "message": message},
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

    def memory_update(self, state: AgentState) -> AgentState:
        action = state.get("final_action", "")
        if action == "ANSWER_FROM_CONTEXT":
            key = "last_answer_topic"
            value = state.get("raw_user_query", "")[:120]
            memory_id = self.deps.memory_service.write_decision(
                key=key,
                value=value,
                source_turn_id=state["turn_id"],
            )
            self._trace(state, "memory.write", {"memory_id": memory_id, "key": key})
        return {}

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
        }.get(action, "refuse")

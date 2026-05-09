from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agentic_rag.config import Settings
from agentic_rag.retrieval.service import RetrievalService
from agentic_rag.storage.repositories import SemanticMemoryRepository
from agentic_rag.storage.sqlite import SQLiteStore
from agentic_rag.tools.arxiv_tools import ArxivToolset
from agentic_rag.traces.writer import TraceWriter

from .memory import MemoryService
from .nodes import AgentDependencies, AgentNodes
from .state import AgentState


class AskGraphRunner:
    def __init__(self, settings: Settings) -> None:
        store = SQLiteStore(settings.app_db_path)
        deps = AgentDependencies(
            retrieval_service=RetrievalService(settings=settings),
            memory_service=MemoryService(SemanticMemoryRepository(store)),
            toolset=ArxivToolset(settings=settings),
            trace_writer=TraceWriter(store),
        )
        self.nodes = AgentNodes(deps)
        self.trace_writer = deps.trace_writer
        self.graph = self._compile_graph()

    def run(self, query: str, thread_id: str = "default") -> AgentState:
        trace = self.trace_writer.start(thread_id=thread_id, run_mode="cli")
        initial_state: AgentState = {
            "thread_id": thread_id,
            "turn_id": trace.turn_id,
            "trace_id": trace.trace_id,
            "raw_user_query": query,
            "normalized_query": query.strip(),
        }
        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as err:  # noqa: BLE001
            self.trace_writer.event(
                trace_id=trace.trace_id,
                level="error",
                event="turn.failed",
                payload={"error": str(err)},
                node="graph.invoke",
            )
            final_state = {
                **initial_state,
                "final_action": "REFUSE",
                "final_answer": (
                    "I can’t answer from the indexed corpus right now because required runtime "
                    "dependencies are unavailable."
                ),
            }
        self.trace_writer.complete(
            trace_id=trace.trace_id, final_action=final_state.get("final_action", "")
        )
        return final_state

    def _compile_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("load_state", self.nodes.load_state)
        graph.add_node("route_query", self.nodes.route_query)
        graph.add_node("clarify", self.nodes.clarify)
        graph.add_node("retrieve", self.nodes.retrieve)
        graph.add_node("tool", self.nodes.tool)
        graph.add_node("evidence_check", self.nodes.evidence_check)
        graph.add_node("answer", self.nodes.answer)
        graph.add_node("refuse", self.nodes.refuse)
        graph.add_node("citation_validate", self.nodes.citation_validate)
        graph.add_node("memory_update", self.nodes.memory_update)

        graph.add_edge(START, "load_state")
        graph.add_edge("load_state", "route_query")
        graph.add_conditional_edges(
            "route_query",
            self.nodes.route_branch,
            {
                "clarify": "clarify",
                "retrieve": "retrieve",
                "tool": "tool",
                "answer": "answer",
                "refuse": "refuse",
            },
        )

        graph.add_edge("retrieve", "evidence_check")
        graph.add_conditional_edges(
            "evidence_check",
            self.nodes.evidence_branch,
            {"answer": "answer", "clarify": "clarify", "refuse": "refuse"},
        )

        graph.add_edge("answer", "citation_validate")
        graph.add_edge("tool", "citation_validate")
        graph.add_edge("citation_validate", "memory_update")
        graph.add_edge("clarify", END)
        graph.add_edge("refuse", END)
        graph.add_edge("memory_update", END)
        return graph.compile()

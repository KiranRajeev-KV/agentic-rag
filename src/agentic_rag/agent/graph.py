from __future__ import annotations

import sqlite3
import time

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agentic_rag.config import Settings
from agentic_rag.llm.client import OpenAILLMClient
from agentic_rag.retrieval.service import RetrievalService
from agentic_rag.retrieval.types import RetrievalVariant
from agentic_rag.storage.repositories import EpisodeRepository, SemanticMemoryRepository
from agentic_rag.storage.sqlite import SQLiteStore
from agentic_rag.tools.arxiv_tools import ArxivToolset
from agentic_rag.traces.writer import TraceWriter

from .memory import MemoryService
from .nodes import AgentDependencies, AgentNodes
from .state import AgentState


class AskGraphRunner:
    def __init__(self, settings: Settings) -> None:
        store = SQLiteStore(settings.app_db_path)
        checkpoint_conn = sqlite3.connect(settings.app_db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(checkpoint_conn)
        self.checkpointer.setup()
        deps = AgentDependencies(
            retrieval_service=RetrievalService(settings=settings),
            memory_service=MemoryService(
                semantic_repo=SemanticMemoryRepository(store),
                episode_repo=EpisodeRepository(store),
            ),
            toolset=ArxivToolset(settings=settings),
            trace_writer=TraceWriter(store),
            llm_client=OpenAILLMClient(settings=settings),
        )
        self.nodes = AgentNodes(deps)
        self.trace_writer = deps.trace_writer
        self.graph = self._compile_graph()

    def run(
        self,
        query: str,
        thread_id: str = "default",
        retrieval_variant: RetrievalVariant | None = None,
    ) -> AgentState:
        started = time.monotonic()
        trace = self.trace_writer.start(thread_id=thread_id, run_mode="cli")
        initial_state: AgentState = {
            "thread_id": thread_id,
            "turn_id": trace.turn_id,
            "trace_id": trace.trace_id,
            "raw_user_query": query,
            "normalized_query": query.strip(),
            "messages": [{"role": "user", "content": query}],
        }
        if retrieval_variant is not None:
            initial_state["forced_retrieval_variant"] = retrieval_variant
        try:
            final_state = self.graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
            )
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
        if not final_state.get("episode_id"):
            episode_id = self.nodes.deps.memory_service.write_episode(
                thread_id=thread_id,
                turn_id=trace.turn_id,
                user_query=query,
                route_action=final_state.get("route_action", ""),
                route_confidence=final_state.get("route_confidence"),
                retrieved_child_ids=final_state.get("retrieved_child_ids", []),
                selected_parent_ids=final_state.get("selected_parent_ids", []),
                tool_calls=final_state.get("tool_calls", []),
                final_action=final_state.get("final_action", ""),
                final_answer_summary=str(final_state.get("final_answer", ""))[:280],
            )
            final_state["episode_id"] = episode_id
            self.trace_writer.event(
                trace_id=trace.trace_id,
                level="info",
                event="episode.written",
                payload={"episode_id": episode_id},
                node="graph.finalize",
            )
            self.trace_writer.event(
                trace_id=trace.trace_id,
                level="info",
                event="checkpoint.persisted",
                payload={"thread_id": thread_id},
                node="graph.finalize",
            )
        total_latency_ms = int((time.monotonic() - started) * 1000)
        final_state["total_latency_ms"] = total_latency_ms
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
        graph.add_node("contradiction_handler", self.nodes.contradiction_handler)
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
            {
                "answer": "answer",
                "clarify": "clarify",
                "refuse": "refuse",
                "contradiction_handler": "contradiction_handler",
            },
        )
        graph.add_conditional_edges(
            "contradiction_handler",
            self.nodes.contradiction_branch,
            {
                "answer": "answer",
                "clarify": "clarify",
                "refuse": "refuse",
                "tool": "tool",
            },
        )

        graph.add_edge("answer", "citation_validate")
        graph.add_edge("tool", "citation_validate")
        graph.add_edge("citation_validate", "memory_update")
        graph.add_edge("clarify", "memory_update")
        graph.add_edge("refuse", "memory_update")
        graph.add_edge("memory_update", END)
        return graph.compile(checkpointer=self.checkpointer)

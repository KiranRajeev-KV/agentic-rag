from __future__ import annotations

from agentic_rag.config import Settings
from agentic_rag.storage.repositories import ChunkRepository, PaperRepository, ParentRepository
from agentic_rag.storage.sqlite import SQLiteStore

from .child_retriever import ChildRetriever
from .context_assembly import assemble_context_packets, dynamic_parent_budget
from .evidence_gate import evaluate_evidence
from .parent_scoring import score_parent_groups
from .types import ChildHit, ParentGroup, RetrievalResult, RetrievalVariant


class RetrievalService:
    def __init__(self, settings: Settings, child_retriever: ChildRetriever | None = None) -> None:
        self.settings = settings
        self.child_retriever = child_retriever or ChildRetriever(settings=settings)
        store = SQLiteStore(settings.app_db_path)
        self.paper_repo = PaperRepository(store)
        self.parent_repo = ParentRepository(store)
        self.chunk_repo = ChunkRepository(store)

    def run(
        self,
        query: str,
        variant: RetrievalVariant,
        top_n_children: int = 30,
        max_parents: int | None = None,
        include_references: bool = False,
    ) -> RetrievalResult:
        child_hits = self.child_retriever.retrieve(
            query=query,
            top_n=top_n_children,
            include_references=include_references,
        )

        if variant == RetrievalVariant.child_only:
            parent_groups = _child_only_groups(child_hits)
        else:
            parent_groups = score_parent_groups(child_hits, top_m=max_parents or 6)

        budget = max_parents if max_parents is not None else dynamic_parent_budget(query)
        packets = assemble_context_packets(
            parent_groups=parent_groups,
            paper_repo=self.paper_repo,
            parent_repo=self.parent_repo,
            chunk_repo=self.chunk_repo,
            max_parents=budget,
        )
        signals = evaluate_evidence(parent_groups=parent_groups, child_hits=child_hits)

        parent_scores = {group.parent_id: group.parent_score for group in parent_groups}
        evidence_ids = [
            chunk_id for group in parent_groups for chunk_id in group.evidence_child_ids
        ]
        return RetrievalResult(
            variant=variant,
            selected_parent_ids=[group.parent_id for group in parent_groups[:budget]],
            retrieved_child_ids=[hit.chunk_id for hit in child_hits],
            evidence_child_ids=evidence_ids,
            parent_scores=parent_scores,
            context_packets=packets,
            signals=signals,
        )


def _child_only_groups(child_hits: list[ChildHit]) -> list[ParentGroup]:
    parent_groups: list[ParentGroup] = []
    for hit in child_hits:
        parent_groups.append(
            ParentGroup(
                parent_id=hit.parent_id,
                paper_id=hit.paper_id,
                section_path=hit.section_path,
                section_type=hit.section_type,
                parent_score=hit.score,
                max_child_score=hit.score,
                support_count=1,
                evidence_child_ids=[hit.chunk_id],
                child_hits=[hit],
            )
        )
    return parent_groups

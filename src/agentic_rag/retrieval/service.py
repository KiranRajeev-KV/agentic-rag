from __future__ import annotations

import re

from agentic_rag.config import Settings
from agentic_rag.storage.repositories import ChunkRepository, PaperRepository, ParentRepository
from agentic_rag.storage.sqlite import SQLiteStore

from .child_retriever import ChildRetriever
from .context_assembly import assemble_context_packets, dynamic_parent_budget
from .evidence_gate import evaluate_evidence
from .parent_scoring import score_parent_groups
from .types import ChildHit, EvidenceStatus, ParentGroup, RetrievalResult, RetrievalVariant


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
        retrieval_filters: dict[str, str] | None = None,
    ) -> RetrievalResult:
        try:
            child_hits = self.child_retriever.retrieve(
                query=query,
                top_n=top_n_children,
                include_references=include_references,
                metadata_filters=retrieval_filters,
            )
        except TypeError:
            # Backward compatibility for test doubles/older retrievers without metadata_filters.
            child_hits = self.child_retriever.retrieve(
                query=query,
                top_n=top_n_children,
                include_references=include_references,
            )
        if len(child_hits) < 8:
            child_hits = self._merge_with_lexical_fallback(query, child_hits, top_n_children)

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
        if signals.evidence_status in {EvidenceStatus.ambiguous, EvidenceStatus.insufficient}:
            recovered = self.child_retriever.retrieve(
                query=query,
                top_n=max(60, top_n_children),
                include_references=True,
                metadata_filters=retrieval_filters,
            )
            if len(recovered) > len(child_hits):
                child_hits = recovered
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

    def _merge_with_lexical_fallback(
        self,
        query: str,
        vector_hits: list[ChildHit],
        top_n_children: int,
    ) -> list[ChildHit]:
        terms = [token for token in re.split(r"[^a-zA-Z0-9]+", query.lower()) if len(token) >= 5]
        if not hasattr(self.chunk_repo, "lexical_search"):
            return vector_hits
        lexical_rows = self.chunk_repo.lexical_search(query_terms=terms, limit=top_n_children)
        seen = {hit.chunk_id for hit in vector_hits}
        merged = list(vector_hits)
        for idx, row in enumerate(lexical_rows, start=1):
            chunk_id = str(row.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(
                ChildHit(
                    chunk_id=chunk_id,
                    parent_id=str(row.get("parent_id", "")),
                    paper_id=str(row.get("paper_id", "")),
                    section_path=str(row.get("section_path", "")),
                    section_type=str(row.get("section_type", "content")),
                    content_type=str(row.get("content_type", "text")),
                    score=max(0.30, 0.55 - (idx * 0.01)),
                    payload={},
                )
            )
            if len(merged) >= top_n_children:
                break
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_n_children]


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

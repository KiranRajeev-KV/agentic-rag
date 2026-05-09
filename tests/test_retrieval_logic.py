from agentic_rag.retrieval.context_assembly import (
    assemble_context_packets,
    dynamic_parent_budget,
    source_block,
)
from agentic_rag.retrieval.evidence_gate import evaluate_evidence
from agentic_rag.retrieval.parent_scoring import score_parent_groups
from agentic_rag.retrieval.service import RetrievalService
from agentic_rag.retrieval.types import ChildHit, EvidenceStatus, ParentGroup, RetrievalVariant


def test_parent_scoring_support_bonus_and_order() -> None:
    hits = [
        ChildHit(
            chunk_id="c1",
            parent_id="p1",
            paper_id="paper1",
            section_path="1",
            section_type="content",
            content_type="text",
            score=0.6,
        ),
        ChildHit(
            chunk_id="c2",
            parent_id="p1",
            paper_id="paper1",
            section_path="1",
            section_type="content",
            content_type="text",
            score=0.55,
        ),
        ChildHit(
            chunk_id="c3",
            parent_id="p2",
            paper_id="paper2",
            section_path="2",
            section_type="references",
            content_type="text",
            score=0.7,
        ),
    ]
    groups = score_parent_groups(hits, top_m=5)
    assert groups[0].parent_id == "p1"
    assert groups[0].support_count == 2
    assert groups[0].parent_score > 0.62


def test_evidence_gate_status() -> None:
    groups = [
        ParentGroup(
            parent_id="p1",
            paper_id="paper1",
            section_path="1",
            section_type="content",
            parent_score=0.65,
            max_child_score=0.63,
            support_count=3,
            evidence_child_ids=["c1", "c2", "c3"],
            child_hits=[],
        ),
        ParentGroup(
            parent_id="p2",
            paper_id="paper2",
            section_path="2",
            section_type="content",
            parent_score=0.54,
            max_child_score=0.54,
            support_count=1,
            evidence_child_ids=["c4"],
            child_hits=[],
        ),
    ]
    hits = [
        ChildHit("c1", "p1", "paper1", "1", "content", "text", 0.63),
        ChildHit("c2", "p1", "paper1", "1", "content", "text", 0.61),
    ]
    signals = evaluate_evidence(groups, hits)
    assert signals.evidence_status == EvidenceStatus.sufficient
    assert signals.confidence_band in {"HIGH", "MEDIUM"}


def test_context_packets_and_source_block() -> None:
    groups = [
        ParentGroup(
            parent_id="parent_1",
            paper_id="paper_1",
            section_path="1 Intro",
            section_type="content",
            parent_score=0.62,
            max_child_score=0.6,
            support_count=1,
            evidence_child_ids=["chunk_1"],
            child_hits=[],
        )
    ]

    class _PaperRepo:
        def get_by_ids(self, paper_ids):  # noqa: ANN001
            del paper_ids
            return {"paper_1": {"paper_id": "paper_1", "title": "T", "arxiv_id": "2501.00001"}}

    class _ParentRepo:
        def get_by_ids(self, parent_ids):  # noqa: ANN001
            del parent_ids
            return {
                "parent_1": {
                    "parent_id": "parent_1",
                    "paper_id": "paper_1",
                    "section_path": "1 Intro",
                    "page_start": 1,
                    "page_end": 2,
                    "parent_text": "A long section text",
                }
            }

    class _ChunkRepo:
        def get_by_ids(self, chunk_ids):  # noqa: ANN001
            del chunk_ids
            return {"chunk_1": {"chunk_id": "chunk_1", "chunk_text": "evidence text"}}

    packets = assemble_context_packets(
        groups, _PaperRepo(), _ParentRepo(), _ChunkRepo(), max_parents=3
    )
    assert len(packets) == 1
    assert packets[0].source_id == "S1"
    block = source_block(packets[0])
    assert '<SOURCE id="S1">' in block


def test_dynamic_budget() -> None:
    assert dynamic_parent_budget("compare method A vs method B") == 5
    assert dynamic_parent_budget("overall synthesis") == 6
    assert dynamic_parent_budget("what is title of paper") == 0


def test_retrieval_service_variants() -> None:
    class _FakeRetriever:
        def retrieve(self, query: str, top_n: int = 30, include_references: bool = False):  # noqa: ARG002
            return [
                ChildHit("c1", "p1", "paper1", "1", "content", "text", 0.7),
                ChildHit("c2", "p1", "paper1", "1", "content", "text", 0.65),
                ChildHit("c3", "p2", "paper2", "2", "content", "text", 0.6),
            ]

    service = object.__new__(RetrievalService)
    service.settings = None
    service.child_retriever = _FakeRetriever()

    class _PaperRepo:
        def get_by_ids(self, ids):  # noqa: ANN001
            return {
                paper_id: {"paper_id": paper_id, "title": paper_id, "arxiv_id": "x"}
                for paper_id in ids
            }

    class _ParentRepo:
        def get_by_ids(self, ids):  # noqa: ANN001
            return {
                parent_id: {
                    "parent_id": parent_id,
                    "paper_id": "paper1" if parent_id == "p1" else "paper2",
                    "section_path": "s",
                    "page_start": 1,
                    "page_end": 1,
                    "parent_text": "context",
                }
                for parent_id in ids
            }

    class _ChunkRepo:
        def get_by_ids(self, ids):  # noqa: ANN001
            return {chunk_id: {"chunk_id": chunk_id, "chunk_text": "chunk"} for chunk_id in ids}

    service.paper_repo = _PaperRepo()
    service.parent_repo = _ParentRepo()
    service.chunk_repo = _ChunkRepo()

    baseline = service.run("q", variant=RetrievalVariant.child_only, max_parents=2)
    candidate = service.run("q", variant=RetrievalVariant.parent_child, max_parents=2)

    assert baseline.variant == RetrievalVariant.child_only
    assert candidate.variant == RetrievalVariant.parent_child
    assert candidate.selected_parent_ids

from types import SimpleNamespace

from agentic_rag.corpus.discovery import discover_relevant_papers
from agentic_rag.tools.schemas import ArxivPaperMetadata, ArxivToolOutput, ToolStatus


def test_discovery_filters_and_deduplicates() -> None:
    relevant = ArxivPaperMetadata(
        arxiv_id="2501.00001",
        version="v1",
        title="Agentic memory planning",
        authors=["A"],
        abstract="A retrieval workflow for agents.",
        categories=["cs.AI"],
    )
    duplicate = ArxivPaperMetadata(
        arxiv_id="2501.00001",
        version="v2",
        title="Agentic memory planning",
        authors=["A"],
        abstract="A retrieval workflow for agents.",
        categories=["cs.AI"],
    )
    irrelevant = ArxivPaperMetadata(
        arxiv_id="2501.00002",
        version="v1",
        title="Graph coloring in practice",
        authors=["B"],
        abstract="Combinatorics and heuristics.",
        categories=["cs.AI"],
    )
    fake_toolset = SimpleNamespace(
        arxiv_get_recent=lambda payload: ArxivToolOutput(  # noqa: ARG005
            status=ToolStatus.ok,
            papers=[relevant, duplicate, irrelevant],
            errors=[],
        )
    )

    papers, errors = discover_relevant_papers(toolset=fake_toolset, limit=10, days_back=90)
    assert not errors
    assert len(papers) == 1
    assert papers[0].metadata.arxiv_id == "2501.00001"
    assert "agentic" in papers[0].filter_terms

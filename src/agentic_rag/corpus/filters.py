from __future__ import annotations

FILTER_TERMS: tuple[str, ...] = (
    "agent",
    "agents",
    "agentic",
    "tool use",
    "tool-use",
    "retrieval",
    "rag",
    "retrieval augmented",
    "memory",
    "planning",
    "reasoning",
    "workflow",
    "llm",
    "large language model",
    "evaluation",
    "self-reflection",
    "multi-agent",
    "autonomous",
    "orchestration",
)


def matched_filter_terms(title: str, abstract: str) -> list[str]:
    return matched_filter_details(title=title, abstract=abstract)["terms"]


def matched_filter_details(title: str, abstract: str) -> dict[str, list[str]]:
    lowered_title = title.lower()
    lowered_abstract = abstract.lower()
    terms = sorted(
        {term for term in FILTER_TERMS if term in lowered_title or term in lowered_abstract}
    )
    title_terms = sorted({term for term in FILTER_TERMS if term in lowered_title})
    abstract_terms = sorted({term for term in FILTER_TERMS if term in lowered_abstract})
    return {
        "terms": terms,
        "title_terms": title_terms,
        "abstract_terms": abstract_terms,
    }

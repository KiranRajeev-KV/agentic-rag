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
    haystack = f"{title}\n{abstract}".lower()
    matches = [term for term in FILTER_TERMS if term in haystack]
    return sorted(set(matches))

from __future__ import annotations

import re


def validate_citations(
    answer: str,
    sources_block: str,
    allowed_source_ids: set[str],
    allowed_tool_ids: set[str],
    final_action: str,
    evidence_status: str = "",
) -> tuple[bool, str]:
    cited_ids = set(re.findall(r"\[(S\d+|T\d+)\]", answer))
    allowed_ids = allowed_source_ids | allowed_tool_ids
    unknown = cited_ids - allowed_ids
    if unknown:
        return False, f"Unknown citation IDs found: {sorted(unknown)}"

    source_ids_in_sources = set(re.findall(r"^\[(S\d+)\]", sources_block, flags=re.MULTILINE))
    tool_ids_in_sources = set(re.findall(r"^\[(T\d+)\]", sources_block, flags=re.MULTILINE))
    listed_ids = source_ids_in_sources | tool_ids_in_sources
    if cited_ids - listed_ids:
        return False, "Every cited ID must appear in the Sources block."

    if final_action == "ANSWER_FROM_CONTEXT" and not any(x.startswith("S") for x in cited_ids):
        return False, "Answer from corpus context requires at least one [S#] citation."
    if final_action == "ANSWER_FROM_TOOL" and not any(x.startswith("T") for x in cited_ids):
        return False, "Answer from tool route requires at least one [T#] citation."
    if (
        evidence_status == "CONTRADICTORY"
        and final_action == "ANSWER_FROM_CONTEXT"
        and len([x for x in cited_ids if x.startswith("S")]) < 2
    ):
        return False, "Contradiction answers must cite at least two [S#] sources."
    if _contains_internal_ids(answer):
        return False, "Answer exposes internal IDs."
    return True, "ok"


def build_sources_block(source_lines: list[str], tool_lines: list[str]) -> str:
    lines = ["Sources:"]
    lines.extend(source_lines)
    lines.extend(tool_lines)
    return "\n".join(lines)


def _contains_internal_ids(text: str) -> bool:
    patterns = (r"\bchunk_[a-zA-Z0-9_]+\b", r"\bparent_[a-zA-Z0-9_]+\b", r"\bpaper_[a-zA-Z0-9_]+\b")
    return any(re.search(pattern, text) for pattern in patterns)

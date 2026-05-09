from __future__ import annotations

import re


def validate_citations(
    answer: str,
    allowed_source_ids: set[str],
    allowed_tool_ids: set[str],
    require_source_citation: bool,
) -> tuple[bool, str]:
    cited_ids = set(re.findall(r"\[(S\d+|T\d+)\]", answer))
    allowed_ids = allowed_source_ids | allowed_tool_ids
    unknown = cited_ids - allowed_ids
    if unknown:
        return False, f"Unknown citation IDs found: {sorted(unknown)}"
    if require_source_citation and not any(citation.startswith("S") for citation in cited_ids):
        return False, "Answer from corpus context requires at least one [S#] citation."
    return True, "ok"


def build_sources_block(source_lines: list[str], tool_lines: list[str]) -> str:
    lines = ["Sources:"]
    lines.extend(source_lines)
    lines.extend(tool_lines)
    return "\n".join(lines)

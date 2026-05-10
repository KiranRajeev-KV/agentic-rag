import json

ROUTER_SYSTEM_PROMPT = """You are a strict router for a local arXiv corpus QA agent.
Return JSON only.
You must choose exactly one action from:
CLARIFY, RETRIEVE, TOOL, REFUSE, ANSWER_FROM_MEMORY, ANSWER_FROM_CONTEXT.
Use TOOL only for arXiv metadata/search requests.
Use RETRIEVE for paper-content questions.
Use REFUSE for out-of-domain questions.
Use CLARIFY only when required ambiguity blocks grounded answering.
"""

INTENT_SYSTEM_PROMPT = """You classify user query intent for a local arXiv corpus QA agent.
Return JSON only.
Allowed intents:
- content_qa: direct question answerable from corpus content
- comparison: asks to compare methods/findings/tradeoffs
- follow_up: refers to prior turns/context
- ambiguous: underspecified question requiring clarification
- refusal: out-of-scope request
- tool_arxiv: request for arXiv metadata/search/ID lookup
- general: fallback when uncertain
Set confidence between 0 and 1.
"""

EVIDENCE_SYSTEM_PROMPT = """You classify evidence quality for grounded QA
from provided source packets.
Return JSON only.
Allowed evidence_status: SUFFICIENT, AMBIGUOUS, INSUFFICIENT, CONTRADICTORY.
Allowed conflict_label: NONE, DIFFERENCE, TENSION, DIRECT_CONTRADICTION, METADATA_CONFLICT.
recommended_action must be one of ANSWER_FROM_CONTEXT, CLARIFY, REFUSE.
"""

ANSWER_SYSTEM_PROMPT = """You generate a grounded answer from provided context packets.
Return JSON only.
Rules:
- Use only provided SOURCE/TOOL_RESULT context.
- Do not use outside knowledge for paper-content claims.
- Every factual claim about paper content must cite [S#].
- Tool metadata claims must cite [T#].
- Never expose internal ids: chunk_id, parent_id, paper_id, raw scores, evidence ids.
- If insufficient evidence, choose REFUSE.
- If query is ambiguous, choose CLARIFY.
"""

MEMORY_SYSTEM_PROMPT = """You decide whether semantic memory should be written for this turn.
Return JSON only.
Default is should_write=false.
Write only for explicit durable user decisions/preferences/constraints or stable project facts.
"""

CITATION_SYSTEM_PROMPT = """You validate citation grounding for a local-corpus answer.
Return JSON only.
Rules:
- Mark valid=true only if every factual claim is supported by cited IDs.
- Reject unknown IDs, missing citations, source-block mismatches, or internal ID leaks.
- Treat unsupported claims as invalid.
"""

CONTRADICTION_SYSTEM_PROMPT = """You handle contradiction/tension decisions for grounded QA.
Return JSON only.
Choose conflict_label in: NONE, DIFFERENCE, TENSION, DIRECT_CONTRADICTION, METADATA_CONFLICT.
Choose recommended_action in: ANSWER_WITH_CONFLICT, CLARIFY, REFUSE, TOOL_LOOKUP.
For DIRECT_CONTRADICTION do not resolve truth; surface both sides with citations.
For METADATA_CONFLICT prefer TOOL_LOOKUP when metadata IDs are available.
"""


def router_user_prompt(
    query: str,
    memory_context: list[dict[str, str]],
    conversation_summary: str = "",
    recent_turns: list[dict[str, str]] | None = None,
    episodic_context: list[dict[str, str]] | None = None,
) -> str:
    return (
        "User query:\n"
        f"{query}\n\n"
        "Conversation summary:\n"
        f"{conversation_summary}\n\n"
        "Recent turns:\n"
        f"{json.dumps(recent_turns or [], ensure_ascii=True)}\n\n"
        "Recent semantic memory:\n"
        f"{json.dumps(memory_context, ensure_ascii=True)}\n\n"
        "Recent episodic memory:\n"
        f"{json.dumps(episodic_context or [], ensure_ascii=True)}"
    )


def intent_user_prompt(
    query: str,
    conversation_summary: str = "",
    recent_turns: list[dict[str, str]] | None = None,
) -> str:
    return (
        "User query:\n"
        f"{query}\n\n"
        "Conversation summary:\n"
        f"{conversation_summary}\n\n"
        "Recent turns:\n"
        f"{json.dumps(recent_turns or [], ensure_ascii=True)}"
    )


def evidence_user_prompt(
    query: str,
    context_packets: list[dict[str, object]],
    retrieval_signals: dict[str, object],
) -> str:
    return (
        "User query:\n"
        f"{query}\n\n"
        "Retrieval signals:\n"
        f"{json.dumps(retrieval_signals, ensure_ascii=True)}\n\n"
        "Context packets:\n"
        f"{json.dumps(context_packets, ensure_ascii=True)}"
    )


def answer_user_prompt(
    query: str,
    context_packets: list[dict[str, object]],
    evidence_status: str,
    tool_result: dict[str, object] | None = None,
) -> str:
    return (
        "User query:\n"
        f"{query}\n\n"
        f"Evidence status: {evidence_status}\n\n"
        "SOURCE packets:\n"
        f"{json.dumps(context_packets, ensure_ascii=True)}\n\n"
        "TOOL_RESULT packets:\n"
        f"{json.dumps(tool_result or {}, ensure_ascii=True)}"
    )


def memory_user_prompt(query: str, answer: str, final_action: str) -> str:
    return f"Turn data:\nquery={query}\nfinal_action={final_action}\nanswer={answer}"


def citation_user_prompt(
    *,
    final_action: str,
    answer_text: str,
    sources_block: str,
    allowed_source_ids: list[str],
    allowed_tool_ids: list[str],
) -> str:
    return (
        "Final action:\n"
        f"{final_action}\n\n"
        "Answer text:\n"
        f"{answer_text}\n\n"
        "Sources block:\n"
        f"{sources_block}\n\n"
        "Allowed source IDs:\n"
        f"{json.dumps(allowed_source_ids, ensure_ascii=True)}\n\n"
        "Allowed tool IDs:\n"
        f"{json.dumps(allowed_tool_ids, ensure_ascii=True)}"
    )


def contradiction_user_prompt(
    *,
    query: str,
    conflict_label: str,
    context_packets: list[dict[str, object]],
    evidence_status: str,
) -> str:
    return (
        "User query:\n"
        f"{query}\n\n"
        f"Evidence status: {evidence_status}\n"
        f"Conflict label hint: {conflict_label}\n\n"
        "Context packets:\n"
        f"{json.dumps(context_packets, ensure_ascii=True)}"
    )

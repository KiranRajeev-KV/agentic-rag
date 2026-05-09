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


def router_user_prompt(query: str, memory_context: list[dict[str, str]]) -> str:
    return (
        "User query:\n"
        f"{query}\n\n"
        "Recent semantic memory:\n"
        f"{json.dumps(memory_context, ensure_ascii=True)}"
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

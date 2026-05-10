from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from docling_core.transforms.chunker import HybridChunker

CHUNKER_NAME = "docling_hybrid_chunker"
CHUNKER_CONFIG_HASH = "default"


@dataclass(frozen=True)
class ChunkingResult:
    parent_rows: list[dict[str, Any]]
    chunk_rows: list[dict[str, Any]]


def build_parent_child_rows(paper_id: str, conversion: object) -> ChunkingResult:
    hybrid_chunker = HybridChunker()
    doc = conversion.document
    raw_chunks = list(hybrid_chunker.chunk(dl_doc=doc))

    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for chunk_index, chunk in enumerate(raw_chunks):
        section_path = _section_path(chunk)
        grouped.setdefault(section_path, []).append(_chunk_row_base(chunk, chunk_index))

    now = datetime.now(UTC).isoformat()
    parent_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []

    for parent_index, (section_path, base_rows) in enumerate(grouped.items()):
        parent_id = _stable_id("parent", paper_id, section_path, str(parent_index))
        section_heading = (
            section_path.split(" > ")[-1] if section_path != "__root__" else "__root__"
        )
        section_type = _section_type(section_path)
        page_start, page_end = _page_span(base_rows)

        child_ids: list[str] = []
        content_types: set[str] = set()
        parent_text_parts: list[str] = []
        token_count = 0
        char_count = 0

        for chunk_row in base_rows:
            chunk_id = _stable_id(
                "chunk",
                paper_id,
                parent_id,
                str(chunk_row["chunk_index"]),
                chunk_row["chunk_text"],
            )
            child_ids.append(chunk_id)
            content_types.add(chunk_row["content_type"])
            parent_text_parts.append(chunk_row["chunk_text"])
            token_count += chunk_row["token_count"]
            char_count += len(chunk_row["chunk_text"])
            chunk_rows.append(
                {
                    **chunk_row,
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "paper_id": paper_id,
                    "section_path": section_path,
                    "section_type": section_type,
                    "embedding_model": "",
                    "created_at": now,
                }
            )

        parent_rows.append(
            {
                "parent_id": parent_id,
                "paper_id": paper_id,
                "section_path": section_path,
                "section_heading": section_heading,
                "section_type": section_type,
                "section_level": max(section_path.count(" > ") + 1, 1),
                "parent_index": parent_index,
                "page_start": page_start,
                "page_end": page_end,
                "token_count": token_count,
                "char_count": char_count,
                "content_types": json.dumps(sorted(content_types), ensure_ascii=True),
                "child_chunk_ids": json.dumps(child_ids, ensure_ascii=True),
                "text_hash": _sha256("\n\n".join(parent_text_parts)),
                "parent_text": "\n\n".join(parent_text_parts),
            }
        )

    return ChunkingResult(parent_rows=parent_rows, chunk_rows=chunk_rows)


def current_chunker_name() -> str:
    return CHUNKER_NAME


def current_chunker_config_hash() -> str:
    return CHUNKER_CONFIG_HASH


def _chunk_row_base(chunk: object, chunk_index: int) -> dict[str, Any]:
    text = chunk.text.strip()
    token_count = len(text.split())
    doc_items = getattr(chunk.meta, "doc_items", [])
    page_numbers = [
        prov.page_no
        for item in doc_items
        for prov in getattr(item, "prov", [])
        if getattr(prov, "page_no", None) is not None
    ]
    page_start = min(page_numbers) if page_numbers else None
    page_end = max(page_numbers) if page_numbers else None
    content_type = _infer_content_type(doc_items)
    ref_ids = [getattr(item, "self_ref", "") for item in doc_items if getattr(item, "self_ref", "")]
    return {
        "chunk_index": chunk_index,
        "content_type": content_type,
        "page_start": page_start,
        "page_end": page_end,
        "token_count": token_count,
        "char_start": None,
        "char_end": None,
        "docling_ref_ids": json.dumps(ref_ids, ensure_ascii=True),
        "embedding_text_hash": _sha256(text),
        "chunk_text": text,
    }


def _section_path(chunk: object) -> str:
    raw_headings = getattr(chunk.meta, "headings", None)
    headings_src = raw_headings if isinstance(raw_headings, list) else []
    headings = [h.strip() for h in headings_src if isinstance(h, str) and h.strip()]
    if not headings:
        return "__root__"
    return " > ".join(headings)


def _section_type(section_path: str) -> str:
    lowered = section_path.lower()
    if "reference" in lowered or "bibliograph" in lowered:
        return "references"
    return "content"


def _infer_content_type(doc_items: list[object]) -> str:
    labels = {str(getattr(item, "label", "")).lower() for item in doc_items}
    if not labels:
        return "text"
    if any("table" in label for label in labels) and len(labels) > 1:
        return "mixed"
    if any("table" in label for label in labels):
        return "table"
    return "text"


def _page_span(base_rows: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    starts = [row["page_start"] for row in base_rows if row["page_start"] is not None]
    ends = [row["page_end"] for row in base_rows if row["page_end"] is not None]
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def _stable_id(prefix: str, *parts: str) -> str:
    joined = "::".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]  # noqa: S324
    return f"{prefix}_{digest}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

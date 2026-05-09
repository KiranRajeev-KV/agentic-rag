from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_jsonl_event(path: Path, event: dict[str, Any]) -> None:
    payload = {
        "ts": datetime.now(UTC).isoformat(),
        **event,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")

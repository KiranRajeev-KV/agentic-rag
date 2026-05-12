from pathlib import Path

REQUIRED_RECIPES = [
    "default",
    "setup",
    "qdrant-up",
    "qdrant-down",
    "db-init",
    "db-reset",
    "ingest",
    "index",
    "ask",
    "ask-thread",
    "trace-list",
    "eval",
    "eval-compare",
    "test",
    "lint",
    "format",
    "check",
    "demo-small",
]


def test_justfile_contains_required_recipes() -> None:
    text = Path("justfile").read_text(encoding="utf-8")
    for recipe in REQUIRED_RECIPES:
        assert f"{recipe}:" in text or f"@{recipe} " in text

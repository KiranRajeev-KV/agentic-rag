from pathlib import Path

import pytest

from agentic_rag.config import get_settings
from agentic_rag.llm.client import LLMStructuredOutputError, OpenAILLMClient
from agentic_rag.llm.schemas import LLMRouterOutput


def _setup_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()


def test_openai_llm_client_uses_responses_parse(tmp_path: Path, monkeypatch) -> None:
    _setup_env(tmp_path, monkeypatch)
    settings = get_settings()

    class _Resp:
        output_parsed = LLMRouterOutput(
            action="RETRIEVE",
            route_confidence=0.8,
            route_reason_public="ok",
            rewritten_query="q",
            retrieval_filters={},
            tool_name=None,
            tool_args={},
            clarifying_question=None,
            refusal_reason=None,
            expected_next_node="retrieve",
            retrieval_variant="parent_child",
        )

    class _Responses:
        called_kwargs = None

        def parse(self, **kwargs):  # noqa: ANN003
            self.called_kwargs = kwargs
            return _Resp()

    class _Client:
        responses = _Responses()

    client = OpenAILLMClient(settings=settings)
    client._client = _Client()  # noqa: SLF001

    out = client.complete_json(
        model="gpt-5-nano",
        schema=LLMRouterOutput,
        system_prompt="sys",
        user_prompt="usr",
    )
    assert out.action == "RETRIEVE"
    called = client._client.responses.called_kwargs  # type: ignore[union-attr]  # noqa: SLF001
    assert called is not None
    assert called["model"] == "gpt-5-nano"
    assert called["text_format"] is LLMRouterOutput
    assert "input" in called


def test_openai_llm_client_raises_on_missing_output_parsed(tmp_path: Path, monkeypatch) -> None:
    _setup_env(tmp_path, monkeypatch)
    settings = get_settings()

    class _Resp:
        output_parsed = None
        refusal = None

    class _Responses:
        def parse(self, **kwargs):  # noqa: ANN003
            del kwargs
            return _Resp()

    class _Client:
        responses = _Responses()

    client = OpenAILLMClient(settings=settings)
    client._client = _Client()  # noqa: SLF001
    with pytest.raises(LLMStructuredOutputError):
        client.complete_json(
            model="gpt-5-nano",
            schema=LLMRouterOutput,
            system_prompt="sys",
            user_prompt="usr",
        )

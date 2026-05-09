from pathlib import Path

from agentic_rag.config import get_settings
from agentic_rag.llm.client import OpenAILLMClient
from agentic_rag.llm.schemas import LLMRouterOutput


def test_openai_llm_client_complete_json_with_mocked_response(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_PDF_DIR", str(tmp_path / "raw_pdfs"))
    monkeypatch.setenv("APP_LOG_JSONL", str(tmp_path / "runs" / "logs" / "app.jsonl"))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()
    settings = get_settings()

    class _Message:
        content = (
            '{"action":"RETRIEVE","route_confidence":0.8,"route_reason_public":"ok",'
            '"rewritten_query":"q","retrieval_filters":{},"tool_name":null,'
            '"tool_args":{},"clarifying_question":null,"refusal_reason":null,'
            '"expected_next_node":"retrieve","retrieval_variant":"parent_child"}'
        )

    class _Choice:
        message = _Message()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):  # noqa: ANN003
            del kwargs
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    client = OpenAILLMClient(settings=settings)
    client._client = _Client()  # noqa: SLF001
    out = client.complete_json(
        model="gpt-5-nano",
        schema=LLMRouterOutput,
        system_prompt="sys",
        user_prompt="usr",
    )
    assert out.action == "RETRIEVE"
    assert out.retrieval_variant == "parent_child"

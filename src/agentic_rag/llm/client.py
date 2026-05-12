from __future__ import annotations

from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from agentic_rag.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMStructuredOutputError(RuntimeError):
    """Raised when strict structured output parsing fails."""


class OpenAILLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None

    def enabled(self) -> bool:
        return self.settings.llm_provider.lower() == "openai" and bool(self.settings.openai_api_key)

    def complete_json(
        self,
        *,
        model: str,
        schema: type[T],
        system_prompt: str,
        user_prompt: str,
    ) -> T:
        if not self.enabled():
            raise RuntimeError("OpenAI LLM client is not configured.")

        request = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text_format": schema,
            "temperature": 0,
            "store": False,
        }
        try:
            response = self._client_or_raise().responses.parse(**request)
        except TypeError:
            request.pop("store", None)
            response = self._client_or_raise().responses.parse(**request)

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            refusal = getattr(response, "refusal", None)
            if refusal:
                raise LLMStructuredOutputError(f"Model refused structured output: {refusal}")
            raise LLMStructuredOutputError("No structured output was parsed from response.")

        if isinstance(parsed, schema):
            return parsed
        return schema.model_validate(parsed)

    def _client_or_raise(self) -> OpenAI:
        if self._client is not None:
            return self._client
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM calls.")
        self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

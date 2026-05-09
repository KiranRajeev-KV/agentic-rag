from __future__ import annotations

import json
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from agentic_rag.config import Settings

T = TypeVar("T", bound=BaseModel)


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
        response = self._client_or_raise().chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        try:
            return schema.model_validate_json(content)
        except ValidationError:
            parsed = json.loads(content)
            return schema.model_validate(parsed)

    def _client_or_raise(self) -> OpenAI:
        if self._client is not None:
            return self._client
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM calls.")
        self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

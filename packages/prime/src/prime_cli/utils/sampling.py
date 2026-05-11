"""Shared sampling argument helpers."""

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, model_validator


class SamplingArgsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_tokens: int | None = None
    temperature: float | None = None
    repetition_penalty: float | None = None
    min_tokens: int | None = None
    seed: int | None = None
    extra_body: Dict[str, Any] | None = None
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None

    @model_validator(mode="after")
    def validate_chat_template_kwargs(self) -> "SamplingArgsConfig":
        if self.enable_thinking is None and self.reasoning_effort is None:
            return self
        if self.extra_body is None:
            return self
        chat_template_kwargs = self.extra_body.get("chat_template_kwargs")
        if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
            raise ValueError("extra_body.chat_template_kwargs must be an object")
        return self

    def extra_body_to_api_dict(self) -> Dict[str, Any] | None:
        extra_body = dict(self.extra_body) if self.extra_body is not None else None
        if self.enable_thinking is None and self.reasoning_effort is None:
            return extra_body

        if extra_body is None:
            extra_body = {}

        existing_kwargs = extra_body.get("chat_template_kwargs")
        chat_template_kwargs = dict(existing_kwargs) if isinstance(existing_kwargs, dict) else {}
        if self.enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = self.enable_thinking
        if self.reasoning_effort is not None:
            chat_template_kwargs["reasoning_effort"] = self.reasoning_effort
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        return extra_body

    def to_api_dict(self) -> Dict[str, Any]:
        result = self.model_dump(
            exclude_none=True,
            exclude={"enable_thinking", "reasoning_effort"},
        )
        extra_body = self.extra_body_to_api_dict()
        if extra_body is not None:
            result["extra_body"] = extra_body
        else:
            result.pop("extra_body", None)
        return result

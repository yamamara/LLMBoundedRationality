from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationOptions:
    temperature: float = 0.0
    top_p: float | None = 1.0
    max_output_tokens: int = 800
    timeout_seconds: float = 60.0
    reasoning_effort: str | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderRequest:
    model: str
    system_prompt: str
    messages: list[dict[str, str]]
    json_schema: dict[str, Any]
    options: GenerationOptions


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    provider: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    finish_reason: str = ""


class LLMProvider(Protocol):
    name: str

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        ...

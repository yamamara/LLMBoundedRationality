from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from sandbox.agents.cournot_openai_compatible import (
    parse_quantity_action,
    validate_quantity,
)
from sandbox.cournot_prompts import (
    DEFAULT_COURNOT_AGENT_PROMPT,
    DEFAULT_COURNOT_SYSTEM_PROMPT,
    render_cournot_prompts,
)
from sandbox.models import Agent, AgentDecision, Observation
from sandbox.providers.base import GenerationOptions, LLMProvider, ProviderRequest


def quantity_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action_type": {"type": "string", "enum": ["submit_quantity"]},
            "value": {
                "type": "object",
                "properties": {"quantity": {"type": "number"}},
                "required": ["quantity"],
                "additionalProperties": False,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["action_type", "value", "reasoning"],
        "additionalProperties": False,
    }


@dataclass
class LLMCournotAgentConfig:
    model: str
    temperature: float = 0.0
    top_p: float | None = 1.0
    max_output_tokens: int = 800
    timeout_seconds: float = 60.0
    memory_rounds: int | None = None
    max_retries: int = 2
    reasoning_effort: str | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)
    system_prompt_template: str = DEFAULT_COURNOT_SYSTEM_PROMPT
    agent_prompt_template: str = DEFAULT_COURNOT_AGENT_PROMPT


class LLMCournotAgent(Agent):
    def __init__(
        self,
        provider: LLMProvider,
        config: LLMCournotAgentConfig,
        request_semaphore: threading.Semaphore | None = None,
    ):
        self.provider = provider
        self.config = config
        self.request_semaphore = request_semaphore
        self.name = config.model

    def decide(self, observation: Observation) -> AgentDecision:
        prompt_observation = self._prompt_observation(observation)
        rendered = render_cournot_prompts(
            self.config.system_prompt_template,
            self.config.agent_prompt_template,
            prompt_observation,
        )
        messages = [{"role": "user", "content": rendered.agent}]
        attempts: list[dict[str, Any]] = []
        started = time.monotonic()
        last_error = "no response"

        for attempt_index in range(max(1, self.config.max_retries)):
            request_messages = deepcopy(messages)
            request = ProviderRequest(
                model=self.config.model,
                system_prompt=rendered.system,
                messages=request_messages,
                json_schema=quantity_json_schema(),
                options=GenerationOptions(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_output_tokens=self.config.max_output_tokens,
                    timeout_seconds=self.config.timeout_seconds,
                    reasoning_effort=self.config.reasoning_effort,
                    provider_options=self.config.provider_options,
                ),
            )
            try:
                if self.request_semaphore:
                    with self.request_semaphore:
                        response = self.provider.generate(request)
                else:
                    response = self.provider.generate(request)
                action = parse_quantity_action(response.text)
                last_error = validate_quantity(observation, action) or ""
                attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "request": {
                            "system_prompt": rendered.system,
                            "messages": request_messages,
                        },
                        "response": response.text,
                        "provider": response.provider,
                        "model": response.model,
                        "usage": response.usage,
                        "request_id": response.request_id,
                        "finish_reason": response.finish_reason,
                        "error": last_error,
                    }
                )
                if not last_error:
                    return AgentDecision(
                        action,
                        {
                            "provider": response.provider,
                            "model": response.model,
                            "usage": response.usage,
                            "attempts": attempts,
                            "system_prompt_template": self.config.system_prompt_template,
                            "agent_prompt_template": self.config.agent_prompt_template,
                            "rendered_system_prompt": rendered.system,
                            "rendered_agent_prompt": rendered.agent,
                            "retry_count": attempt_index,
                            "latency_seconds": time.monotonic() - started,
                        },
                    )
                messages.extend(
                    [
                        {"role": "assistant", "content": response.text},
                        {
                            "role": "user",
                            "content": (
                                f"Invalid action: {last_error}. "
                                "Return valid JSON only."
                            ),
                        },
                    ]
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "request": {
                            "system_prompt": rendered.system,
                            "messages": request_messages,
                        },
                        "error": last_error,
                    }
                )

        raise RuntimeError(
            f"{self.provider.name} failed to return a valid quantity after "
            f"{max(1, self.config.max_retries)} attempts: {last_error}"
        )

    def _prompt_observation(self, observation: Observation) -> Observation:
        public_state = deepcopy(observation.public_state)
        history = public_state["completed_rounds"]
        if self.config.memory_rounds is not None:
            public_state["completed_rounds"] = (
                history[-self.config.memory_rounds :]
                if self.config.memory_rounds
                else []
            )
        return Observation(
            player_id=observation.player_id,
            round=observation.round,
            public_state=public_state,
            private_state=deepcopy(observation.private_state),
            legal_actions=observation.legal_actions,
        )

from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from sandbox.agents.openai_compatible import parse_action, validate_inputs
from sandbox.models import Agent, AgentDecision, Observation
from sandbox.prompts import DEFAULT_AGENT_PROMPT, DEFAULT_SYSTEM_PROMPT, render_prompts
from sandbox.providers.base import GenerationOptions, LLMProvider, ProviderRequest


class LLMAuctionDecisionError(RuntimeError):
    def __init__(self, message: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


def bid_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action_type": {"type": "string", "enum": ["submit_bid"]},
            "value": {
                "type": "object",
                "properties": {"bid": {"type": "integer"}},
                "required": ["bid"],
                "additionalProperties": False,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["action_type", "value", "reasoning"],
        "additionalProperties": False,
    }


@dataclass
class LLMAuctionAgentConfig:
    model: str
    temperature: float = 0.0
    top_p: float | None = 1.0
    max_output_tokens: int = 800
    timeout_seconds: float = 60.0
    memory_rounds: int | None = 3
    max_retries: int = 2
    reasoning_effort: str | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT
    agent_prompt_template: str = DEFAULT_AGENT_PROMPT


class LLMAuctionAgent(Agent):
    def __init__(
        self,
        provider: LLMProvider,
        config: LLMAuctionAgentConfig,
        request_semaphore: threading.Semaphore | None = None,
    ):
        self.provider = provider
        self.config = config
        self.request_semaphore = request_semaphore
        self.name = config.model

    def decide(self, observation: Observation) -> AgentDecision:
        observation = self.prompt_observation(observation)
        rendered = render_prompts(
            self.config.system_prompt_template,
            self.config.agent_prompt_template,
            observation,
        )
        messages = [{"role": "user", "content": rendered.agent}]
        attempts: list[dict[str, Any]] = []
        started = time.monotonic()
        last_error = "no response"
        attempts_allowed = max(1, self.config.max_retries)

        for attempt_index in range(attempts_allowed):
            request_messages = deepcopy(messages)
            request = ProviderRequest(
                model=self.config.model,
                system_prompt=rendered.system,
                messages=request_messages,
                json_schema=bid_json_schema(),
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
                action = parse_action(response.text)
                last_error = validate_inputs(observation, action) or ""
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
                            "accepted_prompt_request": {
                                "system_prompt": rendered.system,
                                "messages": request_messages,
                            },
                            "retry_count": attempt_index,
                            "latency_seconds": time.monotonic() - started,
                        },
                    )
                messages.extend(
                    [
                        {"role": "assistant", "content": response.text},
                        {"role": "user", "content": f"Invalid action: {last_error}. Return valid JSON only."},
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

        raise LLMAuctionDecisionError(
            f"{self.provider.name} failed to return a valid bid after {attempts_allowed} attempts: {last_error}",
            {
                "system_prompt_template": self.config.system_prompt_template,
                "agent_prompt_template": self.config.agent_prompt_template,
                "rendered_system_prompt": rendered.system,
                "rendered_agent_prompt": rendered.agent,
                "attempts": attempts,
            },
        )

    def build_messages(self, observation: Observation) -> list[dict[str, str]]:
        observation = self.prompt_observation(observation)
        rendered = render_prompts(
            self.config.system_prompt_template,
            self.config.agent_prompt_template,
            observation,
        )
        return [{"role": "user", "content": rendered.agent}]

    def prompt_observation(self, observation: Observation) -> Observation:
        payload = deepcopy(observation.public_state)
        history = payload["completed_rounds"]
        if self.config.memory_rounds is not None:
            payload["completed_rounds"] = (
                history[-self.config.memory_rounds :] if self.config.memory_rounds else []
            )
        return Observation(
            player_id=observation.player_id,
            round=observation.round,
            public_state=payload,
            private_state=deepcopy(observation.private_state),
            legal_actions=observation.legal_actions,
        )

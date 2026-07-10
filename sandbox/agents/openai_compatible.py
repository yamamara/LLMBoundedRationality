from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import requests

from sandbox.models import Action, AgentDecision, Observation, Agent


@dataclass
class OpenAICompatibleConfig:
    base_url: str
    model: str
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 800
    timeout_seconds: float = 60.0
    memory_rounds: int | None = None
    max_retries: int = 2
    reasoning_effort: str | None = "none"
    response_format: dict[str, Any] | None = None


def auction_bid_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "auction_bid",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "enum": ["submit_bid"]},
                    "value": {
                        "type": "object",
                        "properties": {
                            "bid": {"type": "integer"},
                        },
                        "required": ["bid"],
                        "additionalProperties": False,
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["action_type", "value", "reasoning"],
                "additionalProperties": False,
            },
        },
    }


def parse_action(content: str) -> Action:
    action_json = json.loads(content)
    value = action_json["value"]
    if isinstance(value, int) and not isinstance(value, bool):
        value = {"bid": value}

    return Action(
        action_type=action_json["action_type"],
        value=value,
        reasoning=action_json["reasoning"],
    )


def validate_inputs(observation: Observation, action: Action) -> str | None:
    legal_actions = observation.legal_actions
    bid = action.value.get("bid")

    if action.action_type != legal_actions.action_type:
        return f"action_type must be {legal_actions.action_type}"

    if isinstance(bid, bool) or not isinstance(bid, int):
        return "bid must be an integer"

    if not legal_actions.limits["bid_min"] <= bid <= legal_actions.limits["bid_max"]:
        return f"bid must be between {legal_actions.limits['bid_min']} and {legal_actions.limits['bid_max']}"

    return None


class OpenAICompatibleAgent(Agent):
    def __init__(self, config: OpenAICompatibleConfig):
        self.config = config
        self.name = config.model

    def decide(self, observation: Observation) -> AgentDecision:
        messages = self.build_prompts(observation)
        attempts: list[dict[str, Any]] = []

        # Monotonic better for elapsed time than time.time
        start_time = time.monotonic()

        last_error = None
        for attempt_index in range(self.config.max_retries):
            content, response_metadata = self.generate_response(messages)

            attempts.append(
                {
                    "request_messages": deepcopy(messages),
                    "response": content,
                    **response_metadata,
                }
            )

            try:
                action = parse_action(content)
                error = validate_inputs(observation, action)
            except Exception as exc:
                action = None
                error = str(exc)
            last_error = error

            if action and not error:
                return AgentDecision(
                    action,
                    {
                        "attempts": attempts,
                        "latency_seconds": time.monotonic() - start_time,
                        "retry_count": attempt_index,
                    },
                )

            # --------- Only runs in error condition -------------

            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": f"Invalid action: {error}. Return one valid JSON object only.",
                }
            )

        raise ValueError(
            f"AI failed to return a valid action after {len(attempts)} attempts: {last_error}"
        )

    def build_prompts(self, observation: Observation) -> list[dict[str, str]]:
        observation_dict = deepcopy(observation.to_dict())

        if self.config.memory_rounds is not None:
            history = observation_dict["public_state"]["completed_rounds"]
            observation_dict["public_state"]["completed_rounds"] = (
                history[-self.config.memory_rounds :] if self.config.memory_rounds else []
            )

        user_prompt = {
            **observation_dict,
            "instruction": (
                "Choose your bid. Return JSON only with action_type='submit_bid', "
                "value as an object like {'bid': 5}, where bid is an integer in the legal range, "
                "and a brief reasoning string. "
                "Do not reveal chain-of-thought."
            ),
        }

        return [
            {
                "role": "system",
                "content": "You are a participant in a research auction. Follow the provided rules exactly.",
            },
            {"role": "user", "content": json.dumps(user_prompt, sort_keys=True)},
        ]


    # Boilerplate API calling function
    def generate_response(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            headers["Authorization"] = f"Bearer {os.environ[self.config.api_key_env]}"
        request_body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": self.config.response_format or auction_bid_response_format(),
        }
        if self.config.reasoning_effort and self.config.reasoning_effort != "none":
            request_body["reasoning_effort"] = self.config.reasoning_effort
        response = requests.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=request_body,
            timeout=self.config.timeout_seconds,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"OpenAI-compatible API request failed: {response.status_code} {response.text}"
            ) from exc
        body = response.json()
        return body["choices"][0]["message"]["content"], {
            "response_model": body.get("model", ""),
            "usage": body.get("usage", {}),
        }

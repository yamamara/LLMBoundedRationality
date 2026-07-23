from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from typing import Any

from sandbox.agents.openai_compatible import OpenAICompatibleAgent, OpenAICompatibleConfig
from sandbox.cournot_prompts import (
    DEFAULT_COURNOT_AGENT_PROMPT,
    DEFAULT_COURNOT_SYSTEM_PROMPT,
    render_cournot_prompts,
)
from sandbox.models import Action, AgentDecision, Observation


def cournot_quantity_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "cournot_quantity",
            "strict": True,
            "schema": {
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
            },
        },
    }


def parse_quantity_action(content: str) -> Action:
    action_json = json.loads(content)
    value = action_json["value"]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = {"quantity": value}
    return Action(action_json["action_type"], value, action_json["reasoning"])


def validate_quantity(observation: Observation, action: Action) -> str | None:
    if action.action_type != "submit_quantity":
        return "action_type must be submit_quantity"
    quantity = action.value.get("quantity")
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
        return "quantity must be a number"
    if not math.isfinite(quantity):
        return "quantity must be finite"

    limits = observation.legal_actions.limits
    if not limits["quantity_min"] <= quantity <= limits["quantity_max"]:
        return (
            f"quantity must be between {limits['quantity_min']} "
            f"and {limits['quantity_max']}"
        )
    steps = round((quantity - limits["quantity_min"]) / limits["quantity_step"])
    grid_value = limits["quantity_min"] + steps * limits["quantity_step"]
    if not math.isclose(quantity, grid_value, abs_tol=1e-9):
        return f"quantity must use increments of {limits['quantity_step']}"
    return None


class CournotOpenAICompatibleAgent(OpenAICompatibleAgent):
    def __init__(
        self,
        config: OpenAICompatibleConfig,
        system_prompt_template: str = DEFAULT_COURNOT_SYSTEM_PROMPT,
        agent_prompt_template: str = DEFAULT_COURNOT_AGENT_PROMPT,
    ):
        if config.response_format is None:
            config.response_format = cournot_quantity_response_format()
        super().__init__(config)
        self.system_prompt_template = system_prompt_template
        self.agent_prompt_template = agent_prompt_template

    def decide(self, observation: Observation) -> AgentDecision:
        messages = self.build_prompts(observation)
        attempts: list[dict[str, Any]] = []
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
                action = parse_quantity_action(content)
                error = validate_quantity(observation, action)
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
        rendered = render_cournot_prompts(
            self.system_prompt_template,
            self.agent_prompt_template,
            Observation(
                player_id=observation.player_id,
                round=observation.round,
                public_state=observation_dict["public_state"],
                private_state=observation_dict["private_state"],
                legal_actions=observation.legal_actions,
            ),
        )
        return [
            {"role": "system", "content": rendered.system},
            {"role": "user", "content": rendered.agent},
        ]

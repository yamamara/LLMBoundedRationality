from __future__ import annotations

import json
import math
from collections.abc import Callable

from sandbox.models import Action, AgentDecision, Observation, Agent


class HumanCliAgent(Agent):
    name = "human_cli"

    def __init__(self, input_fn: Callable[[str], str] = input):
        self.input_fn = input_fn

    def decide(self, observation: Observation) -> AgentDecision:
        print(json.dumps(observation.to_dict(), indent=2))
        action_type = observation.legal_actions.action_type
        limits = observation.legal_actions.limits
        invalid_attempts: list[str] = []

        while True:
            if action_type == "submit_bid":
                raw_value = self.input_fn("Bid: ")
                try:
                    value = int(raw_value)
                except ValueError:
                    error = "Bid must be an integer."
                else:
                    error = None
                    if not limits["bid_min"] <= value <= limits["bid_max"]:
                        error = f"Bid must be between {limits['bid_min']} and {limits['bid_max']}."
                value_key = "bid"
            elif action_type == "submit_quantity":
                raw_value = self.input_fn("Quantity: ")
                try:
                    value = float(raw_value)
                except ValueError:
                    error = "Quantity must be a number."
                else:
                    error = None
                    if not math.isfinite(value):
                        error = "Quantity must be finite."
                    elif not limits["quantity_min"] <= value <= limits["quantity_max"]:
                        error = (
                            f"Quantity must be between {limits['quantity_min']} "
                            f"and {limits['quantity_max']}."
                        )
                    else:
                        steps = round(
                            (value - limits["quantity_min"]) / limits["quantity_step"]
                        )
                        grid_value = limits["quantity_min"] + steps * limits["quantity_step"]
                        if not math.isclose(value, grid_value, abs_tol=1e-9):
                            error = (
                                f"Quantity must use increments of "
                                f"{limits['quantity_step']}."
                            )
                value_key = "quantity"
            else:
                raise ValueError(f"Human CLI does not support action type {action_type!r}")

            if error is None:
                return AgentDecision(
                    Action(
                        action_type,
                        {value_key: value},
                        f"Human-entered {value_key}.",
                    ),
                    {
                        "raw_input": raw_value,
                        "invalid_attempts": invalid_attempts,
                        "retry_count": len(invalid_attempts),
                    },
                )

            invalid_attempts.append(error)
            print(error)

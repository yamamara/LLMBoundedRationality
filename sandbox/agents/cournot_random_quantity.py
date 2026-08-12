from __future__ import annotations

import math
import random

from sandbox.models import Action, AgentDecision, Observation


class CournotRandomQuantityAgent:
    """Choose a uniformly random legal quantity between 1 and 100."""

    name = "cournot_random_quantity"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def decide(self, observation: Observation) -> AgentDecision:
        limits = observation.legal_actions.limits
        quantity_min = limits["quantity_min"]
        quantity_max = limits["quantity_max"]
        quantity_step = limits["quantity_step"]

        first_step = math.ceil(
            (max(1.0, quantity_min) - quantity_min) / quantity_step - 1e-12
        )
        last_step = math.floor(
            (min(100.0, quantity_max) - quantity_min) / quantity_step + 1e-12
        )
        if first_step > last_step:
            raise ValueError("No legal Cournot quantity exists between 1 and 100")

        step = self.rng.randint(first_step, last_step)
        quantity = round(quantity_min + step * quantity_step, 10)
        return AgentDecision(
            Action(
                "submit_quantity",
                {"quantity": quantity},
                "Random legal quantity between 1 and 100.",
            )
        )

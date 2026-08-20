from __future__ import annotations

from sandbox.models import Action, AgentDecision, Observation


class CournotPreviousAverageAgent:
    """Repeat the previous round's average quantity across all firms."""

    name = "cournot_previous_average"

    def __init__(self, initial_quantity: float = 20.0):
        self.initial_quantity = initial_quantity

    def decide(self, observation: Observation) -> AgentDecision:
        history = observation.public_state["completed_rounds"]
        if history:
            previous = history[-1]
            firms = observation.public_state["market"]["firms"]
            quantity = (
                previous["own_quantity"] + previous["opponents_total_quantity"]
            ) / firms
            reasoning = "Previous round's average quantity across all firms."
        else:
            quantity = self.initial_quantity
            reasoning = "Initial fallback quantity; no previous round is available."

        limits = observation.legal_actions.limits
        quantity = min(max(quantity, limits["quantity_min"]), limits["quantity_max"])
        steps = round(
            (quantity - limits["quantity_min"]) / limits["quantity_step"]
        )
        quantity = round(
            limits["quantity_min"] + steps * limits["quantity_step"], 10
        )
        return AgentDecision(
            Action("submit_quantity", {"quantity": quantity}, reasoning)
        )

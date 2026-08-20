from __future__ import annotations

from sandbox.models import Action, AgentDecision, Observation


class CournotBestReplyAgent:
    name = "cournot_best_reply"

    def __init__(self, initial_quantity: float = 20.0):
        self.initial_quantity = initial_quantity

    def decide(self, observation: Observation) -> AgentDecision:
        quantity = observation.private_state["best_reply_quantity"]
        if quantity is None:
            quantity = self.initial_quantity
        return AgentDecision(
            Action(
                "submit_quantity",
                {"quantity": quantity},
                "Myopic best reply to the previous total quantity of the other firms.",
            )
        )

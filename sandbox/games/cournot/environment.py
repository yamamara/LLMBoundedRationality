from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

from sandbox.environment import Environment
from sandbox.models import Action, AgentDecision, LegalActions, Observation, Participant
from sandbox.serialization import OutputWriter


@dataclass
class CournotConfig:
    rounds: int
    quantity_min: float
    quantity_max: float
    quantity_step: float
    demand_intercept: float
    marginal_cost: float
    revision_probability: float
    fixed_payment: float
    treatment: str
    seed: int
    institution: str


class CournotEnvironment(Environment):
    def __init__(
        self,
        config: CournotConfig,
        participants: list[Participant],
        output: OutputWriter | None = None,
    ):
        if config.treatment not in {"BEST", "FULL"}:
            raise ValueError("Cournot treatment must be BEST or FULL")
        if len(participants) != 4:
            raise ValueError("The Cournot experiment requires exactly four participants")
        if not 0 <= config.revision_probability <= 1:
            raise ValueError("revision_probability must be between 0 and 1")
        if config.quantity_step <= 0:
            raise ValueError("quantity_step must be positive")

        self.config = config
        self.participants = {participant.player_id: participant for participant in participants}
        if len(self.participants) != 4:
            raise ValueError("Cournot player IDs must be unique")
        self.player_ids = list(self.participants)
        self.output = output
        self.reset()

    def reset(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.round = 0
        self.quantities: dict[str, float] = {}
        self.profits: dict[str, float] = {}
        self.total_profits = {player_id: 0.0 for player_id in self.player_ids}
        self.history: list[dict[str, Any]] = []
        self.current_quantities: dict[str, float] = {}
        self.decision_rows: list[dict[str, Any]] = []
        self.round_rows: list[dict[str, Any]] = []

    def legal_actions(self, player_id: str) -> LegalActions:
        del player_id
        return LegalActions(
            action_type="submit_quantity",
            limits={
                "quantity_min": self.config.quantity_min,
                "quantity_max": self.config.quantity_max,
                "quantity_step": self.config.quantity_step,
            },
        )

    def observe(self, player_id: str) -> Observation:
        previous_quantity = self.quantities.get(player_id)
        opponents_total = (
            sum(quantity for other, quantity in self.quantities.items() if other != player_id)
            if self.quantities
            else None
        )
        return Observation(
            player_id=player_id,
            round=self.round,
            public_state={
                "treatment": self.config.treatment,
                "rounds_total": self.config.rounds,
                "market": {
                    "firms": 4,
                    "inverse_demand": "price = max(100 - total_quantity, 0)",
                    "cost": "cost = quantity",
                },
                "completed_rounds": self.visible_history(player_id),
            },
            private_state={
                "previous_quantity": previous_quantity,
                "previous_profit": self.profits.get(player_id),
                "opponents_total_last_round": opponents_total,
                "best_reply_quantity": (
                    self.best_reply(opponents_total) if opponents_total is not None else None
                ),
            },
            legal_actions=self.legal_actions(player_id),
        )

    def step(self, player_id: str, action: Action) -> None:
        if player_id in self.current_quantities:
            raise ValueError(f"{player_id} has already chosen in round {self.round}")
        valid, reason = self.validate_action(action)
        if not valid:
            raise ValueError(reason)
        self.current_quantities[player_id] = self.normalize_quantity(action.value["quantity"])

    def is_done(self) -> bool:
        return self.round >= self.config.rounds

    def run(self) -> dict[str, list[dict[str, Any]]]:
        while not self.is_done():
            self.run_round()
        return {"decisions": self.decision_rows, "rounds": self.round_rows}

    def run_round(self) -> None:
        contexts: dict[str, dict[str, Any]] = {}
        for player_id, participant in self.participants.items():
            observation = self.observe(player_id)
            revision_allowed = self.round == 0 or self.rng.random() < self.config.revision_probability
            if revision_allowed:
                decision = participant.agent.decide(observation)
                valid, reason = self.validate_action(decision.action)
                fallback = self.quantities.get(player_id, self.config.quantity_min)
                applied_action = (
                    decision.action
                    if valid
                    else Action(
                        "submit_quantity",
                        {"quantity": fallback},
                        "Invalid action replaced with the previous quantity.",
                    )
                )
            else:
                valid, reason = True, None
                applied_action = Action(
                    "submit_quantity",
                    {"quantity": self.quantities[player_id]},
                    "Revision was not allowed; previous quantity retained.",
                )
                decision = AgentDecision(applied_action, {"forced_hold": True})

            self.step(player_id, applied_action)
            contexts[player_id] = {
                "decision": decision,
                "valid": valid,
                "invalid_reason": reason,
                "revision_allowed": revision_allowed,
            }
            self.log(
                {
                    "event_type": "decision",
                    "round": self.round,
                    "player_id": player_id,
                    "observation": observation.to_dict(),
                    "returned_action": decision.action.to_dict(),
                    "applied_action": applied_action.to_dict(),
                    "valid": valid,
                    "invalid_reason": reason,
                    "revision_allowed": revision_allowed,
                    "agent_metadata": decision.metadata,
                }
            )

        self.compute_round(contexts)
        self.quantities = dict(self.current_quantities)
        self.current_quantities = {}
        self.round += 1

    def compute_round(self, contexts: dict[str, dict[str, Any]]) -> None:
        total_quantity = round(sum(self.current_quantities.values()), 10)
        price = round(max(self.config.demand_intercept - total_quantity, 0.0), 10)
        profits = {
            player_id: round(
                (price - self.config.marginal_cost) * quantity,
                4,
            )
            for player_id, quantity in self.current_quantities.items()
        }
        for player_id, profit in profits.items():
            self.total_profits[player_id] = round(self.total_profits[player_id] + profit, 4)

        firm_results = [
            {
                "player_id": player_id,
                "quantity": self.current_quantities[player_id],
                "profit": profits[player_id],
            }
            for player_id in self.player_ids
        ]
        nash_quantity = (
            len(self.player_ids)
            * (self.config.demand_intercept - self.config.marginal_cost)
            / (len(self.player_ids) + 1)
        )
        round_summary = {
            "round": self.round,
            "treatment": self.config.treatment,
            "institution": self.config.institution,
            "total_quantity": total_quantity,
            "price": price,
            "total_market_profit": round(sum(profits.values()), 4),
            "distance_to_nash": abs(total_quantity - nash_quantity),
        }
        self.history.append({**round_summary, "firm_results": firm_results})
        self.round_rows.append(round_summary)

        for player_id in self.player_ids:
            metadata = contexts[player_id]["decision"].metadata
            profit = profits[player_id]
            self.decision_rows.append(
                {
                    "round": self.round,
                    "treatment": self.config.treatment,
                    "institution": self.config.institution,
                    "player_id": player_id,
                    "agent_type": self.participants[player_id].agent_type,
                    "agent": self.participants[player_id].agent.name,
                    "revision_allowed": contexts[player_id]["revision_allowed"],
                    "quantity": self.current_quantities[player_id],
                    "price": price,
                    "profit": profit,
                    "reward": round(profit + self.config.fixed_payment, 4),
                    "cumulative_profit": self.total_profits[player_id],
                    "valid_action": contexts[player_id]["valid"],
                    "invalid_reason": contexts[player_id]["invalid_reason"] or "",
                    "retry_count": metadata.get("retry_count", 0),
                    "latency_seconds": metadata.get("latency_seconds", 0.0),
                }
            )

        self.profits = profits
        self.log({"event_type": "settlement", **round_summary, "firm_results": firm_results})

    def visible_history(self, player_id: str) -> list[dict[str, Any]]:
        if not self.history:
            return []
        result = self.history[-1]
        own = next(item for item in result["firm_results"] if item["player_id"] == player_id)
        visible: dict[str, Any] = {
            "round": result["round"],
            "opponents_total_quantity": result["total_quantity"] - own["quantity"],
            "price": result["price"],
            "own_quantity": own["quantity"],
            "own_profit": own["profit"],
        }
        if self.config.treatment == "FULL":
            visible["firm_results"] = result["firm_results"]
        return [visible]

    def best_reply(self, opponents_total: float) -> float:
        raw = (self.config.demand_intercept - self.config.marginal_cost - opponents_total) / 2
        raw = min(max(raw, self.config.quantity_min), self.config.quantity_max)
        lower_steps = math.floor((raw - self.config.quantity_min) / self.config.quantity_step)
        candidates = {
            self.normalize_quantity(self.config.quantity_min + lower_steps * self.config.quantity_step),
            self.normalize_quantity(
                self.config.quantity_min + (lower_steps + 1) * self.config.quantity_step
            ),
        }
        legal_candidates = [
            quantity
            for quantity in candidates
            if self.config.quantity_min <= quantity <= self.config.quantity_max
        ]
        return max(
            legal_candidates,
            key=lambda quantity: (
                self.quantity_profit(quantity, opponents_total),
                -quantity,
            ),
        )

    def quantity_profit(self, quantity: float, opponents_total: float) -> float:
        price = max(self.config.demand_intercept - opponents_total - quantity, 0.0)
        return (price - self.config.marginal_cost) * quantity

    def validate_action(self, action: Action) -> tuple[bool, str | None]:
        if action.action_type != "submit_quantity":
            return False, "action_type must be submit_quantity"
        if not isinstance(action.value, dict):
            return False, "value must be an object"
        quantity = action.value.get("quantity")
        if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
            return False, "quantity must be a number"
        if not math.isfinite(quantity):
            return False, "quantity must be finite"
        if not self.config.quantity_min <= quantity <= self.config.quantity_max:
            return False, (
                f"quantity must be between {self.config.quantity_min} "
                f"and {self.config.quantity_max}"
            )
        if not math.isclose(quantity, self.normalize_quantity(quantity), abs_tol=1e-9):
            return False, f"quantity must use increments of {self.config.quantity_step}"
        return True, None

    def normalize_quantity(self, quantity: float) -> float:
        steps = round((quantity - self.config.quantity_min) / self.config.quantity_step)
        return round(self.config.quantity_min + steps * self.config.quantity_step, 10)

    def log(self, event: dict[str, Any]) -> None:
        if self.output:
            self.output.log_event(event)

    def get_config_as_dict(self) -> dict[str, Any]:
        return asdict(self.config)

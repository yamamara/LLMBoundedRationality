from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from sandbox.environment import Environment
from sandbox.models import Action, LegalActions, Observation, Participant
from sandbox.serialization import OutputWriter


# Java Record equivalent setup exactly like the auction config in the JSON
@dataclass
class AuctionConfig:
    rounds: int
    starting_budget: int
    true_value_min: int
    true_value_max: int
    mechanism: str
    seed: int
    institution: str


class AuctionEnvironment(Environment):
    def __init__(
        self,
        config: AuctionConfig,
        participants: list[Participant],
        output: OutputWriter | None = None
    ):
        if config.mechanism not in {"first_price", "second_price"}:
            # TODO: Add more? Question for Dr. Cen
            raise ValueError(f"Unsupported auction mechanism: {config.mechanism}")

        self.config = config

        # Participant lookup (with a dictionary) using ID broke and AI told me this was the fix
        # TODO: Make a better solution since this seems jank?
        self.participants = {participant.player_id: participant for participant in participants}

        self.player_ids = list(self.participants)
        self.output = output
        self.reset()


    # The function is called reset but all it does is initialize a new game, hence why we use it in the constructor as well
    # Allows to initialize at startup, and reset at any moment (probably will be useful)
    def reset(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.round = 0
        self.budgets = {player_id: self.config.starting_budget for player_id in self.player_ids}
        self.total_payoffs = {player_id: 0 for player_id in self.player_ids}
        self.history: list[dict[str, Any]] = []
        self.current_bids: dict[str, int] = {}

        self.true_values = {
            player_id: [
                self.rng.randint(self.config.true_value_min, self.config.true_value_max)
                for _ in range(self.config.rounds)
            ]
            for player_id in self.player_ids
        }
        self.decision_rows: list[dict[str, Any]] = []
        self.round_rows: list[dict[str, Any]] = []


    def legal_actions(self, player_id: str) -> LegalActions:
        return LegalActions(
            action_type="submit_bid",
            limits={
                "bid_min": 0,
                "bid_max": self.budgets[player_id],
            },
        )


    def observe(self, player_id: str) -> Observation:
        return Observation(
            player_id=player_id,
            round=self.round,
            public_state={
                "mechanism": self.config.mechanism,
                "rounds_total": self.config.rounds,
                "completed_rounds": self.history,
            },
            private_state={
                "true_value": self.true_values[player_id][self.round],
                "remaining_budget": self.budgets[player_id],
            },
            legal_actions=self.legal_actions(player_id),
        )


    def step(self, player_id: str, action: Action) -> None:
        if player_id in self.current_bids:
            raise ValueError(f"{player_id} has already bid in round {self.round}")

        self.current_bids[player_id] = self.get_bid_value(player_id, action)


    def is_done(self) -> bool:
        return self.round >= self.config.rounds


    def run(self) -> dict[str, list[dict[str, Any]]]:
        while not self.is_done():
            self.run_round()
        return {
            "decisions": self.decision_rows,
            "rounds": self.round_rows,
        }


    def run_round(self) -> None:
        decision_context: dict[str, dict[str, Any]] = {}

        for player_id, participant in self.participants.items():
            observation = self.observe(player_id)
            decision = participant.agent.decide(observation)
            valid, reason = self.validate_action(player_id, decision.action)

            applied_action = (
                decision.action
                if valid
                else Action("submit_bid", {"bid": 0}, "Invalid action replaced with zero bid.")
            )

            self.step(player_id, applied_action)

            decision_context[player_id] = {
                "observation": observation,
                "decision": decision,
                "valid": valid,
                "invalid_reason": reason,
                "applied_action": applied_action,
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
                    "agent_metadata": decision.metadata,
                }
            )

        self.compute_round(decision_context)
        self.current_bids = {}
        self.round += 1


    def compute_round(self, decision_context: dict[str, dict[str, Any]]) -> None:
        bids = self.current_bids
        highest_bid = max(bids.values())
        candidates = sorted(player_id for player_id, bid in bids.items() if bid == highest_bid)
        winner_id = self.rng.choice(candidates) if highest_bid > 0 else None
        price = self.price(winner_id, bids)

        if winner_id:
            self.budgets[winner_id] -= price

        true_values = {
            player_id: self.true_values[player_id][self.round] for player_id in self.player_ids
        }

        payoffs = {
            player_id: true_values[player_id] - price if player_id == winner_id else 0
            for player_id in self.player_ids
        }

        for player_id, payoff in payoffs.items():
            self.total_payoffs[player_id] += payoff

        efficiency = true_values[winner_id] / max(true_values.values()) if winner_id else 0.0

        round_summary = {
            "round": self.round,
            "mechanism": self.config.mechanism,
            "institution": self.config.institution,
            "winner_id": winner_id or "",
            "price": price,
            "highest_bid": highest_bid,
            "allocative_efficiency": efficiency,
        }

        self.history.append(round_summary)
        self.round_rows.append(round_summary)

        for player_id in self.player_ids:
            metadata = decision_context[player_id]["decision"].metadata
            self.decision_rows.append(
                {
                    "round": self.round,
                    "mechanism": self.config.mechanism,
                    "institution": self.config.institution,
                    "player_id": player_id,
                    "agent_type": self.participants[player_id].agent_type,
                    "agent": self.participants[player_id].agent.name,
                    "true_value": true_values[player_id],
                    "bid": bids[player_id],
                    "winner": player_id == winner_id,
                    "price": price if player_id == winner_id else 0,
                    "payoff": payoffs[player_id],
                    "regret": self.regret(player_id, true_values[player_id], winner_id, price),
                    "remaining_budget": self.budgets[player_id],
                    "valid_action": decision_context[player_id]["valid"],
                    "invalid_reason": decision_context[player_id]["invalid_reason"] or "",
                    "retry_count": metadata.get("retry_count", 0),
                    "latency_seconds": metadata.get("latency_seconds", 0.0),
                }
            )

        self.log({"event_type": "settlement", **round_summary, "bids": bids, "true_values": true_values})


    def price(self, winner_id: str | None, bids: dict[str, int]) -> int:
        if winner_id is None:
            return 0

        if self.config.mechanism == "first_price":
            return bids[winner_id]

        return sorted(bids.values(), reverse=True)[1] if len(bids) > 1 else 0


    # Regret measures exactly how much better the player could've done with a better bid (how much they regret the bid they did)
    # If the player won their payoff is true value - price, if they lost then 0 payoff
    # But the BEST payoff they could've gotten is the true value - winning price, where winning price is other bid + 1
    # Regret is then calculated as best_payoff - actual_payoff
    def regret(self, player_id: str, true_value: int, winner_id: str | None, price: int) -> int:
        realized_payoff = true_value - price if player_id == winner_id else 0
        other_highest = max((bid for other, bid in self.current_bids.items() if other != player_id), default=0)

        # TODO: Add more if we expand mechanisms
        if self.config.mechanism == "first_price":
            winning_price = other_highest + 1
        else:
            winning_price = other_highest

        can_win = other_highest + 1 <= self.budgets[player_id] + (price if player_id == winner_id else 0)
        best_payoff = max(0, true_value - winning_price) if can_win else 0
        return max(0, best_payoff - realized_payoff)


    def get_bid_value(self, player_id: str, action: Action) -> int:
        valid, reason = self.validate_action(player_id, action)

        if not valid:
            raise ValueError(reason)

        return action.value["bid"]


    def validate_action(self, player_id: str, action: Action) -> tuple[bool, str | None]:
        if action.action_type != "submit_bid":
            return False, "action_type must be submit_bid"

        bid = action.value.get("bid")

        if isinstance(bid, bool) or not isinstance(bid, int):
            return False, "bid must be an integer"

        if not 0 <= bid <= self.budgets[player_id]:
            return False, f"bid must be between 0 and {self.budgets[player_id]}"

        return True, None


    def log(self, event: dict[str, Any]) -> None:
        if self.output:
            self.output.log_event(event)


    def get_config_as_dict(self) -> dict[str, Any]:
        return asdict(self.config)

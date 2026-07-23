from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Callable

from sandbox.environment import Environment
from sandbox.models import Action, LegalActions, Observation, Participant
from sandbox.serialization import OutputWriter
from sandbox.games.auction.mechanisms import MECHANISM_NAMES, mechanism_description


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
    mode: str = "legacy"
    tie_breaking: str = "seeded_random"
    tie_break_seed: int | None = None
    tie_break_priority: list[str] | None = None
    mechanism_description_treatment: str = "concise"
    player_value_ranges: dict[str, dict[str, int]] | None = None
    valuation_schedule: str = "baseline"
    num_players: int | None = None


class AuctionEnvironment(Environment):
    def __init__(
        self,
        config: AuctionConfig,
        participants: list[Participant],
        output: OutputWriter | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ):
        if config.mechanism not in {"first_price", "second_price"}:
            # TODO: Add more? Question for Dr. Cen
            raise ValueError(f"Unsupported auction mechanism: {config.mechanism}")
        if config.rounds <= 0:
            raise ValueError("rounds must be positive")
        if config.starting_budget < 0:
            raise ValueError("starting_budget must not be negative")
        if config.true_value_min < 0 or config.true_value_min > config.true_value_max:
            raise ValueError("true value bounds must satisfy 0 <= min <= max")
        if config.mode not in {"legacy", "four_ai_sealed_bid", "ai_sealed_bid"}:
            raise ValueError(f"Unsupported auction mode: {config.mode}")
        if config.tie_breaking not in {"seeded_random", "player_priority"}:
            raise ValueError(f"Unsupported tie-breaking rule: {config.tie_breaking}")
        if len({participant.player_id for participant in participants}) != len(participants):
            raise ValueError("Auction participant IDs must be unique")
        if config.num_players is not None and config.num_players != len(participants):
            raise ValueError("num_players must match the participant list")
        if config.mode == "four_ai_sealed_bid":
            if config.mechanism != "first_price":
                raise ValueError("four_ai_sealed_bid mode requires first_price mechanism")
            if len(participants) != 4 or any(p.agent_type != "AI" for p in participants):
                raise ValueError("four_ai_sealed_bid mode requires exactly four AI participants")
        if config.mode == "ai_sealed_bid":
            if len(participants) not in {2, 4, 6, 8} or any(p.agent_type != "AI" for p in participants):
                raise ValueError("ai_sealed_bid mode requires 2, 4, 6, or 8 AI participants")
        mechanism_description(config.mechanism, config.mechanism_description_treatment)

        player_ids = [participant.player_id for participant in participants]
        if config.tie_break_priority is not None and set(config.tie_break_priority) != set(player_ids):
            raise ValueError("tie_break_priority must contain every participant ID exactly once")
        if config.tie_break_priority is not None and len(config.tie_break_priority) != len(player_ids):
            raise ValueError("tie_break_priority must not contain duplicate participant IDs")

        self.config = config

        # Participant lookup (with a dictionary) using ID broke and AI told me this was the fix
        # TODO: Make a better solution since this seems jank?
        self.participants = {participant.player_id: participant for participant in participants}

        self.player_ids = list(self.participants)
        self.output = output
        self.progress_callback = progress_callback
        self.reset()


    # The function is called reset but all it does is initialize a new game, hence why we use it in the constructor as well
    # Allows to initialize at startup, and reset at any moment (probably will be useful)
    def reset(self) -> None:
        self.rng = random.Random(self.config.seed)
        tie_seed = self.config.seed if self.config.tie_break_seed is None else self.config.tie_break_seed
        self.tie_rng = random.Random(tie_seed)
        self.round = 0
        self.budgets = {player_id: self.config.starting_budget for player_id in self.player_ids}
        self.total_payoffs = {player_id: 0 for player_id in self.player_ids}
        self.history: list[dict[str, Any]] = []
        self.current_bids: dict[str, int] = {}

        self.true_values = {
            player_id: [
                self.rng.randint(*self.value_range(player_id))
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
                "mechanism_name": MECHANISM_NAMES[self.config.mechanism],
                "mechanism_description": mechanism_description(
                    self.config.mechanism, self.config.mechanism_description_treatment
                ),
                "mechanism_description_treatment": self.config.mechanism_description_treatment,
                "num_players": len(self.player_ids),
                "num_opponents": len(self.player_ids) - 1,
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
        observations = {player_id: self.observe(player_id) for player_id in self.player_ids}
        self.log({"event_type": "round_started", "round": self.round})

        if self.is_sealed_ai_mode:
            decisions = self._collect_sealed_decisions(observations)
        else:
            decisions = {
                player_id: self.participants[player_id].agent.decide(observations[player_id])
                for player_id in self.player_ids
            }

        for player_id in self.player_ids:
            observation = observations[player_id]
            decision = decisions[player_id]
            valid, reason = self.validate_action(player_id, decision.action)

            if not valid:
                self.log(
                    {
                        "event_type": "decision_rejected",
                        "round": self.round,
                        "player_id": player_id,
                        "observation": observation.to_dict(),
                        "returned_action": decision.action.to_dict(),
                        "valid": False,
                        "invalid_reason": reason,
                        "agent_metadata": decision.metadata,
                    }
                )
                raise ValueError(f"Invalid action from {player_id}: {reason}")

            applied_action = decision.action

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
        if self.progress_callback:
            self.progress_callback(self.round)

    def _collect_sealed_decisions(
        self, observations: dict[str, Observation]
    ) -> dict[str, Any]:
        decisions: dict[str, Any] = {}
        errors: dict[str, Exception] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(self.player_ids)), thread_name_prefix="sealed-bid") as executor:
            future_players = {
                executor.submit(self.participants[player_id].agent.decide, observations[player_id]): player_id
                for player_id in self.player_ids
            }
            for future in as_completed(future_players):
                player_id = future_players[future]
                try:
                    decisions[player_id] = future.result()
                except Exception as exc:
                    errors[player_id] = exc
                    self.log(
                        {
                            "event_type": "decision_error",
                            "round": self.round,
                            "player_id": player_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "agent_metadata": getattr(exc, "metadata", {}),
                        }
                    )
        if errors:
            for player_id, decision in decisions.items():
                self.log(
                    {
                        "event_type": "decision_unsettled",
                        "round": self.round,
                        "player_id": player_id,
                        "observation": observations[player_id].to_dict(),
                        "returned_action": decision.action.to_dict(),
                        "agent_metadata": decision.metadata,
                    }
                )
            player_id = sorted(errors)[0]
            exc = errors[player_id]
            raise RuntimeError(
                f"Agent {player_id} failed in round {self.round}: {exc}"
            ) from exc
        return decisions


    def compute_round(self, decision_context: dict[str, dict[str, Any]]) -> None:
        bids = self.current_bids
        highest_bid = max(bids.values())
        candidates = sorted(player_id for player_id, bid in bids.items() if bid == highest_bid)
        winner_id = self.resolve_tie(candidates) if (highest_bid > 0 or self.is_sealed_ai_mode) else None
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

        maximum_value = max(true_values.values())
        efficiency = (
            true_values[winner_id] / maximum_value
            if winner_id and maximum_value
            else (1.0 if winner_id else 0.0)
        )

        round_summary = {
            "round": self.round,
            "mechanism": self.config.mechanism,
            "mechanism_description_treatment": self.config.mechanism_description_treatment,
            "valuation_schedule": self.config.valuation_schedule,
            "num_players": len(self.player_ids),
            "institution": self.config.institution,
            "winner_id": winner_id or "",
            "price": price,
            "highest_bid": highest_bid,
            "allocative_efficiency": efficiency,
            "revenue": price,
            "total_utility": sum(payoffs.values()),
            "winning_value": true_values[winner_id] if winner_id else 0,
            "maximum_value": maximum_value,
            "tie_breaking": self.config.tie_breaking,
            "tie_candidates": candidates,
        }

        if self.is_sealed_ai_mode:
            self.history.append(
                {
                    "round": self.round,
                    "mechanism": self.config.mechanism,
                    "mechanism_description_treatment": self.config.mechanism_description_treatment,
                    "institution": self.config.institution,
                    "valuation_schedule": self.config.valuation_schedule,
                    "num_players": len(self.player_ids),
                    "winner_id": winner_id or "",
                    "price": price,
                    "highest_bid": highest_bid,
                }
            )
        else:
            self.history.append(round_summary)
        self.round_rows.append(round_summary)

        for player_id in self.player_ids:
            metadata = decision_context[player_id]["decision"].metadata
            self.decision_rows.append(
                {
                    "round": self.round,
                    "mechanism": self.config.mechanism,
                    "mechanism_description_treatment": self.config.mechanism_description_treatment,
                    "valuation_schedule": self.config.valuation_schedule,
                    "num_players": len(self.player_ids),
                    "institution": self.config.institution,
                    "player_id": player_id,
                    "agent_type": self.participants[player_id].agent_type,
                    "agent": self.participants[player_id].agent.name,
                    "true_value": true_values[player_id],
                    "valuation": true_values[player_id],
                    "bid": bids[player_id],
                    "bid_to_value_ratio": (
                        bids[player_id] / true_values[player_id]
                        if true_values[player_id]
                        else None
                    ),
                    "truthfulness_deviation": abs(bids[player_id] - true_values[player_id]),
                    "winner": player_id == winner_id,
                    "price": price if player_id == winner_id else 0,
                    "payment": price if player_id == winner_id else 0,
                    "payoff": payoffs[player_id],
                    "utility": payoffs[player_id],
                    "cumulative_utility": self.total_payoffs[player_id],
                    "regret": self.regret(player_id, true_values[player_id], winner_id, price),
                    "remaining_budget": self.budgets[player_id],
                    "valid_action": decision_context[player_id]["valid"],
                    "invalid_reason": decision_context[player_id]["invalid_reason"] or "",
                    "retry_count": metadata.get("retry_count", 0),
                    "latency_seconds": metadata.get("latency_seconds", 0.0),
                    "provider": metadata.get("provider", ""),
                    "model": metadata.get("model", self.participants[player_id].agent.name),
                    "usage": metadata.get("usage", {}),
                    "system_prompt_template": metadata.get("system_prompt_template", ""),
                    "agent_prompt_template": metadata.get("agent_prompt_template", ""),
                    "rendered_system_prompt": metadata.get("rendered_system_prompt", ""),
                    "rendered_agent_prompt": metadata.get("rendered_agent_prompt", ""),
                    "accepted_prompt_request": metadata.get("accepted_prompt_request", {}),
                    "prompt_attempts": metadata.get("attempts", []),
                }
            )

        self.log({"event_type": "settlement", **round_summary, "bids": bids, "true_values": true_values})

    def resolve_tie(self, candidates: list[str]) -> str:
        if self.config.tie_breaking == "seeded_random":
            return self.tie_rng.choice(sorted(candidates))
        priority = self.config.tie_break_priority or self.player_ids
        candidate_set = set(candidates)
        return next(player_id for player_id in priority if player_id in candidate_set)


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

    @property
    def is_sealed_ai_mode(self) -> bool:
        return self.config.mode in {"four_ai_sealed_bid", "ai_sealed_bid"}

    def value_range(self, player_id: str) -> tuple[int, int]:
        ranges = self.config.player_value_ranges or {}
        selected = ranges.get(player_id)
        if selected is None:
            return self.config.true_value_min, self.config.true_value_max
        minimum = selected["minimum"]
        maximum = selected["maximum"]
        if minimum < 0 or minimum > maximum:
            raise ValueError(f"Invalid valuation range for {player_id}")
        return minimum, maximum

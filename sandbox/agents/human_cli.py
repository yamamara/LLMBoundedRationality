from __future__ import annotations

import json
from collections.abc import Callable

from sandbox.models import Action, AgentDecision, Observation, Agent


class HumanCliAgent(Agent):
    name = "human_cli"

    def __init__(self, input_fn: Callable[[str], str] = input):
        self.input_fn = input_fn

    def decide(self, observation: Observation) -> AgentDecision:
        print(json.dumps(observation.to_dict(), indent=2))
        raw_bid = self.input_fn("Bid: ")

        # Just a fallback in case of bad input
        # TODO: Make this more specific
        try:
            bid = int(raw_bid)
        except ValueError:
            bid = raw_bid

        return AgentDecision(
            Action("submit_bid", {"bid": bid}, "Human-entered bid."),
            {"raw_input": raw_bid},
        )

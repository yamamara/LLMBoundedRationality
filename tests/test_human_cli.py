from __future__ import annotations

import unittest

from sandbox.agents.human_cli import HumanCliAgent
from sandbox.models import LegalActions, Observation


def observation(action_type, limits):
    return Observation("H1", 0, {}, {}, LegalActions(action_type, limits))


class HumanCliTests(unittest.TestCase):
    def test_accepts_an_auction_bid(self):
        agent = HumanCliAgent(input_fn=lambda prompt: "25")

        decision = agent.decide(
            observation("submit_bid", {"bid_min": 0, "bid_max": 100})
        )

        self.assertEqual(decision.action.action_type, "submit_bid")
        self.assertEqual(decision.action.value, {"bid": 25})

    def test_reprompts_until_cournot_quantity_is_valid(self):
        answers = iter(["not-a-number", "100.001", "19.80"])
        agent = HumanCliAgent(input_fn=lambda prompt: next(answers))

        decision = agent.decide(
            observation(
                "submit_quantity",
                {
                    "quantity_min": 0.0,
                    "quantity_max": 100.0,
                    "quantity_step": 0.01,
                },
            )
        )

        self.assertEqual(decision.action.action_type, "submit_quantity")
        self.assertEqual(decision.action.value, {"quantity": 19.8})
        self.assertEqual(decision.metadata["retry_count"], 2)


if __name__ == "__main__":
    unittest.main()

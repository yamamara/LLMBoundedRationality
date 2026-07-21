from __future__ import annotations

import unittest

from sandbox.games.auction.environment import AuctionConfig, AuctionEnvironment
from sandbox.models import Action, AgentDecision, Participant


class InvalidBidAgent:
    name = "invalid_bid"

    def decide(self, observation):
        return AgentDecision(Action("submit_bid", {"bid": -1}, "Invalid test bid."))


class AuctionTests(unittest.TestCase):
    def test_invalid_bid_aborts_instead_of_becoming_zero(self):
        environment = AuctionEnvironment(
            AuctionConfig(
                rounds=1,
                starting_budget=100,
                true_value_min=1,
                true_value_max=100,
                mechanism="first_price",
                seed=7,
                institution="baseline",
            ),
            [Participant("P1", InvalidBidAgent(), "AI")],
        )

        with self.assertRaisesRegex(ValueError, "Invalid action from P1"):
            environment.run()

        self.assertEqual(environment.decision_rows, [])
        self.assertEqual(environment.round_rows, [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from sandbox.agents.cournot_previous_average import CournotPreviousAverageAgent
from sandbox.agents.cournot_random_quantity import CournotRandomQuantityAgent
from sandbox.models import LegalActions, Observation


def observation(completed_rounds=None, quantity_step=0.01):
    return Observation(
        player_id="P1",
        round=1 if completed_rounds else 0,
        public_state={
            "treatment": "BEST",
            "rounds_total": 40,
            "market": {"firms": 4},
            "completed_rounds": completed_rounds or [],
        },
        private_state={},
        legal_actions=LegalActions(
            "submit_quantity",
            {
                "quantity_min": 0.0,
                "quantity_max": 100.0,
                "quantity_step": quantity_step,
            },
        ),
    )


class CournotScriptAgentTests(unittest.TestCase):
    def test_random_quantity_is_seeded_and_legal(self):
        first = CournotRandomQuantityAgent(seed=11)
        second = CournotRandomQuantityAgent(seed=11)

        first_values = [first.decide(observation()).action.value["quantity"] for _ in range(8)]
        second_values = [second.decide(observation()).action.value["quantity"] for _ in range(8)]

        self.assertEqual(first_values, second_values)
        self.assertTrue(all(1.0 <= value <= 100.0 for value in first_values))
        self.assertGreater(len(set(first_values)), 1)

    def test_random_quantity_respects_a_coarser_legal_grid(self):
        agent = CournotRandomQuantityAgent(seed=3)

        values = [
            agent.decide(observation(quantity_step=0.25)).action.value["quantity"]
            for _ in range(20)
        ]

        self.assertTrue(all((value * 4).is_integer() for value in values))

    def test_previous_average_uses_initial_quantity_in_first_round(self):
        decision = CournotPreviousAverageAgent(17.5).decide(observation())

        self.assertEqual(decision.action.value["quantity"], 17.5)

    def test_previous_average_uses_all_four_prior_quantities(self):
        completed = [
            {
                "own_quantity": 10.0,
                "opponents_total_quantity": 71.01,
            }
        ]

        decision = CournotPreviousAverageAgent().decide(observation(completed))

        self.assertEqual(decision.action.value["quantity"], 20.25)


if __name__ == "__main__":
    unittest.main()

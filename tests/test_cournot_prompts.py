from __future__ import annotations

import unittest

from sandbox.cournot_prompts import render_cournot_prompts
from sandbox.models import LegalActions, Observation


class CournotPromptTests(unittest.TestCase):
    def test_custom_templates_render_cournot_observation(self):
        observation = Observation(
            player_id="P2",
            round=3,
            public_state={
                "treatment": "BEST",
                "rounds_total": 40,
                "market": {"firms": 6},
                "completed_rounds": [{"round": 2, "own_profit": 12}],
            },
            private_state={
                "previous_quantity": 20,
                "previous_profit": 12,
                "opponents_total_last_round": 59.2,
                "best_reply_quantity": 19.9,
            },
            legal_actions=LegalActions(
                "submit_quantity",
                {"quantity_min": 0, "quantity_max": 100, "quantity_step": 0.01},
            ),
        )

        rendered = render_cournot_prompts(
            "Firm {{ player_id }} is one of {{ num_players }} and uses {{ treatment }}.",
            "Round {{ round_number }}; reply {{ best_reply }}; data {{ observation }}",
            observation,
        )

        self.assertEqual(rendered.system, "Firm P2 is one of 6 and uses BEST.")
        self.assertIn("Round 4; reply 19.9", rendered.agent)
        self.assertIn('"opponents_total_last_round":59.2', rendered.agent)


if __name__ == "__main__":
    unittest.main()

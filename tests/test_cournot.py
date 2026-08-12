from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from sandbox.games.cournot.environment import CournotConfig, CournotEnvironment
from sandbox.models import Action, AgentDecision, Participant
from sandbox.scorecard import cournot_outcomes
from simulation import run_pipeline


class FixedAgent:
    name = "fixed"

    def __init__(self, quantity: float):
        self.quantity = quantity
        self.calls = 0

    def decide(self, observation):
        self.calls += 1
        return AgentDecision(
            Action("submit_quantity", {"quantity": self.quantity}, "Fixed test action.")
        )


class InvalidAgent:
    name = "invalid"

    def decide(self, observation):
        return AgentDecision(
            Action("submit_quantity", {"quantity": 100.001}, "Invalid test action.")
        )


def config(treatment: str = "BEST", rounds: int = 1, revision_probability: float = 2 / 3):
    return CournotConfig(
        rounds=rounds,
        quantity_min=0.0,
        quantity_max=100.0,
        quantity_step=0.01,
        demand_intercept=100.0,
        marginal_cost=1.0,
        revision_probability=revision_probability,
        fixed_payment=150.0,
        treatment=treatment,
        seed=7,
        institution="baseline",
    )


def participants(agents):
    return [
        Participant(f"P{index + 1}", agent, "AI", "producer")
        for index, agent in enumerate(agents)
    ]


class CournotTests(unittest.TestCase):
    def test_progress_callback_runs_after_each_settled_round(self):
        progress = []
        environment = CournotEnvironment(
            config(rounds=3),
            participants([FixedAgent(20) for _ in range(4)]),
            progress_callback=progress.append,
        )

        environment.run()

        self.assertEqual(progress, [1, 2, 3])

    def test_variable_player_counts_update_market_and_nash_benchmark(self):
        for player_count in (2, 6, 8):
            with self.subTest(player_count=player_count):
                environment = CournotEnvironment(
                    config(),
                    participants([FixedAgent(10) for _ in range(player_count)]),
                )

                self.assertEqual(
                    environment.observe("P1").public_state["market"]["firms"],
                    player_count,
                )
                rows = environment.run()
                expected_nash_total = player_count * 99 / (player_count + 1)
                self.assertAlmostEqual(
                    rows["rounds"][0]["distance_to_nash"],
                    abs(player_count * 10 - expected_nash_total),
                )

    def test_cournot_rejects_player_counts_outside_supported_range(self):
        for player_count in (1, 9):
            with self.subTest(player_count=player_count):
                with self.assertRaisesRegex(ValueError, "between two and eight"):
                    CournotEnvironment(
                        config(),
                        participants([FixedAgent(10) for _ in range(player_count)]),
                    )

    def test_nash_profile_has_expected_price_and_profit(self):
        environment = CournotEnvironment(
            config(),
            participants([FixedAgent(19.8) for _ in range(4)]),
        )

        rows = environment.run()

        self.assertAlmostEqual(rows["rounds"][0]["total_quantity"], 79.2)
        self.assertAlmostEqual(rows["rounds"][0]["price"], 20.8)
        self.assertAlmostEqual(rows["rounds"][0]["distance_to_nash"], 0.0)
        self.assertAlmostEqual(rows["decisions"][0]["profit"], 392.04)

    def test_best_hides_and_full_reveals_individual_results(self):
        best = CournotEnvironment(
            config("BEST", rounds=2),
            participants([FixedAgent(10 + index) for index in range(4)]),
        )
        full = CournotEnvironment(
            config("FULL", rounds=2),
            participants([FixedAgent(10 + index) for index in range(4)]),
        )
        best.run_round()
        full.run_round()

        self.assertNotIn("firm_results", best.observe("P1").public_state["completed_rounds"][0])
        self.assertIn("firm_results", full.observe("P1").public_state["completed_rounds"][0])

    def test_observation_exposes_all_completed_rounds(self):
        best = CournotEnvironment(
            config("BEST", rounds=4, revision_probability=1),
            participants([FixedAgent(10 + index) for index in range(4)]),
        )
        full = CournotEnvironment(
            config("FULL", rounds=4, revision_probability=1),
            participants([FixedAgent(10 + index) for index in range(4)]),
        )
        for _ in range(3):
            best.run_round()
            full.run_round()

        best_history = best.observe("P1").public_state["completed_rounds"]
        full_history = full.observe("P1").public_state["completed_rounds"]

        self.assertEqual([row["round"] for row in best_history], [0, 1, 2])
        self.assertEqual([row["round"] for row in full_history], [0, 1, 2])
        self.assertTrue(all("firm_results" not in row for row in best_history))
        self.assertTrue(all("firm_results" in row for row in full_history))

    def test_inertia_holds_quantity_without_calling_agent(self):
        agents = [FixedAgent(10 + index) for index in range(4)]
        environment = CournotEnvironment(
            config(rounds=3, revision_probability=0),
            participants(agents),
        )

        rows = environment.run()

        self.assertEqual([agent.calls for agent in agents], [1, 1, 1, 1])
        self.assertTrue(all(not row["revision_allowed"] for row in rows["decisions"][4:]))

    def test_invalid_quantity_aborts_instead_of_changing_market_data(self):
        environment = CournotEnvironment(
            config(),
            participants([InvalidAgent(), FixedAgent(20), FixedAgent(20), FixedAgent(20)]),
        )

        with self.assertRaisesRegex(ValueError, "Invalid action from P1"):
            environment.run()

        self.assertEqual(environment.decision_rows, [])
        self.assertEqual(environment.round_rows, [])

    def test_runner_writes_the_existing_pipeline_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_config = {
                "game_name": "cournot-test",
                "run_id": "cournot-test",
                "output_dir": str(root),
                "cournot": {
                    **config(rounds=3).__dict__,
                },
                "participants": [
                    {
                        "player_id": f"P{index + 1}",
                        "agent_type": "AI",
                        "role": "producer",
                        "policy": "cournot_best_reply",
                        "initial_quantity": 10 + index * 10,
                    }
                    for index in range(4)
                ],
            }

            analysis_output = root / "custom-analysis"
            run_dir, analysis_dir = run_pipeline(
                run_config,
                analyze=True,
                analysis_output=analysis_output,
            )

            self.assertTrue((run_dir / "participant_data.csv").exists())
            self.assertTrue((run_dir / "system_data.csv").exists())
            with (run_dir / "participant_data.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 12)
            with (run_dir / "summary.json").open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["game_type"], "cournot")
            self.assertTrue(cournot_outcomes([run_dir]))
            self.assertEqual(analysis_dir, analysis_output)
            self.assertTrue((analysis_output / "scorecard.json").exists())
            self.assertTrue((analysis_output / "scorecard.csv").exists())
            self.assertTrue((analysis_output / "cournot_player_scores.csv").exists())
            self.assertGreater(
                (analysis_output / "cournot_player_scores.svg").stat().st_size,
                0,
            )
            with (analysis_output / "scorecard.json").open(encoding="utf-8") as handle:
                scorecard = json.load(handle)
            self.assertEqual(
                scorecard["measures"]["total_quantity"]["descriptive"]["run_count"],
                1,
            )

    def test_runner_automatically_analyzes_cournot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_config = {
                "game_name": "cournot-auto-analysis-test",
                "run_id": "cournot-auto-analysis-test",
                "output_dir": str(root),
                "cournot": {**config(rounds=1).__dict__},
                "participants": [
                    {
                        "player_id": f"P{index + 1}",
                        "agent_type": "AI",
                        "role": "producer",
                        "policy": "cournot_best_reply",
                        "initial_quantity": 20,
                    }
                    for index in range(4)
                ],
            }

            run_dir, analysis_dir = run_pipeline(run_config)

            self.assertEqual(analysis_dir, run_dir / "analysis")
            self.assertTrue((analysis_dir / "scorecard.json").exists())
            self.assertTrue((analysis_dir / "cournot_player_scores.csv").exists())
            self.assertTrue((analysis_dir / "cournot_player_scores.svg").exists())
            self.assertTrue((analysis_dir / "cournot_playback.html").exists())


if __name__ == "__main__":
    unittest.main()

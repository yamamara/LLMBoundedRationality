from __future__ import annotations

import tempfile
import time
import unittest
import csv
from pathlib import Path

from pydantic import ValidationError

from sandbox.api.cournot_jobs import CournotJobManager
from sandbox.api.models import CournotSimulationRequest


class CournotJobTests(unittest.TestCase):
    def test_frontend_job_produces_its_own_graph_and_playback(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = CournotJobManager(Path(temporary))
            try:
                request = CournotSimulationRequest.model_validate(
                    {
                        "name": "frontend-cournot-test",
                        "cournot": {
                            "rounds": 2,
                            "treatment": "BEST",
                            "seed": 11,
                            "revision_probability": 1.0,
                        },
                        "agents": [
                            {
                                "player_id": f"P{index}",
                                "policy": "cournot_best_reply",
                                "initial_quantity": index * 10,
                            }
                            for index in range(1, 5)
                        ],
                    }
                )
                state = manager.submit(request)
                deadline = time.monotonic() + 5
                while manager.get(state.simulation_id).status not in {
                    "completed",
                    "failed",
                }:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)

                completed = manager.get(state.simulation_id)
                self.assertEqual(completed.status, "completed", completed.error)
                result = manager.get_results(state.simulation_id)
                self.assertEqual(result["simulation_id"], state.simulation_id)
                self.assertEqual(result["run_id"], state.simulation_id)
                self.assertEqual(
                    result["graph_two_sd_url"],
                    f"/api/v1/cournot-simulations/{state.simulation_id}/graph",
                )
                self.assertEqual(
                    result["graph_ci95_url"],
                    f"/api/v1/cournot-simulations/{state.simulation_id}/graph-ci95",
                )
                self.assertTrue(manager.artifact_path(
                    state.simulation_id, "cournot_player_scores.svg"
                ).is_file())
                self.assertTrue(manager.artifact_path(
                    state.simulation_id, "cournot_player_scores_ci95.svg"
                ).is_file())
                self.assertTrue(manager.artifact_path(
                    state.simulation_id, "cournot_playback.html"
                ).is_file())
                self.assertIsNone(manager.artifact_path("another-job", "cournot_playback.html"))
            finally:
                manager.close()

    def test_openai_compatible_agents_require_connection_details(self):
        with self.assertRaises(ValidationError):
            CournotSimulationRequest.model_validate(
                {
                    "agents": [
                        {
                            "player_id": f"P{index}",
                            "policy": "cournot_openai_compatible",
                        }
                        for index in range(1, 5)
                    ]
                }
            )

    def test_mixed_job_accepts_browser_human_quantity(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = CournotJobManager(Path(temporary))
            try:
                request = CournotSimulationRequest.model_validate(
                    {
                        "name": "mixed-frontend-test",
                        "cournot": {
                            "rounds": 1,
                            "treatment": "FULL",
                            "seed": 3,
                            "revision_probability": 1.0,
                        },
                        "agents": [
                            {"player_id": "P1", "policy": "web_human"},
                            *[
                                {
                                    "player_id": f"P{index}",
                                    "policy": "cournot_best_reply",
                                    "initial_quantity": 20,
                                }
                                for index in range(2, 5)
                            ],
                        ],
                    }
                )
                state = manager.submit(request)
                deadline = time.monotonic() + 5
                pending = None
                while pending is None:
                    self.assertLess(time.monotonic(), deadline)
                    pending = manager.pending_human_decision(state.simulation_id)
                    time.sleep(0.01)

                self.assertEqual(pending["player_id"], "P1")
                with self.assertRaisesRegex(ValueError, "increments"):
                    manager.submit_human_decision(
                        state.simulation_id, pending["request_id"], 12.345
                    )
                manager.submit_human_decision(
                    state.simulation_id, pending["request_id"], 12.34
                )
                while manager.get(state.simulation_id).status not in {
                    "completed",
                    "failed",
                }:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)

                completed = manager.get(state.simulation_id)
                self.assertEqual(completed.status, "completed", completed.error)
                with (
                    Path(temporary) / state.simulation_id / "participant_data.csv"
                ).open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                human_row = next(row for row in rows if row["player_id"] == "P1")
                self.assertEqual(human_row["agent_type"], "human")
                self.assertEqual(float(human_row["quantity"]), 12.34)
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()

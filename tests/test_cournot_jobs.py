from __future__ import annotations

import tempfile
import time
import unittest
import csv
from pathlib import Path

from pydantic import ValidationError

from sandbox.api.cournot_jobs import CournotJobManager
from sandbox.api.models import CournotSimulationRequest
from sandbox.configuration import ProviderProfile


class CournotJobTests(unittest.TestCase):
    def test_provider_profile_flows_to_cournot_pipeline_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = ProviderProfile(
                profile_id="local-test",
                provider="local_llama",
                model="llama-test",
                endpoint="http://localhost:11434",
                transport="ollama",
                defaults={"num_ctx": 2048},
            )
            manager = CournotJobManager(
                Path(temporary), profiles={profile.profile_id: profile}
            )
            try:
                request = CournotSimulationRequest.model_validate(
                    {
                        "agents": [
                            {
                                "player_id": "P1",
                                "policy": "cournot_llm",
                                "profile_id": "local-test",
                            },
                            {"player_id": "P2", "policy": "cournot_best_reply"},
                        ]
                    }
                )

                participant = manager._pipeline_config("profile-test", request)[
                    "participants"
                ][0]

                self.assertEqual(participant["policy"], "cournot_llm")
                self.assertEqual(participant["provider"], "local_llama")
                self.assertEqual(participant["model"], "llama-test")
                self.assertEqual(participant["provider_options"]["num_ctx"], 2048)
            finally:
                manager.close()

    def test_per_agent_prompt_overrides_flow_to_pipeline_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = CournotJobManager(Path(temporary))
            try:
                request = CournotSimulationRequest.model_validate(
                    {
                        "agents": [
                            {
                                "player_id": "P1",
                                "policy": "cournot_openai_compatible",
                                "base_url": "http://localhost:11434/v1",
                                "model": "model-one",
                                "system_prompt_template": "Strategy A for {{ player_id }}.",
                                "agent_prompt_template": "Choose A in round {{ round_number }}.",
                            },
                            {
                                "player_id": "P2",
                                "policy": "cournot_openai_compatible",
                                "base_url": "http://localhost:11434/v1",
                                "model": "model-two",
                                "system_prompt_template": "Strategy B for {{ player_id }}.",
                                "agent_prompt_template": "Choose B in round {{ round_number }}.",
                            },
                        ],
                        "prompts": {
                            "system_template": "Shared system for {{ player_id }}.",
                            "agent_template": "Shared round {{ round_number }}.",
                        },
                    }
                )

                config = manager._pipeline_config("prompt-test", request)
                p1, p2 = config["participants"]

                self.assertEqual(p1["system_prompt_template"], "Strategy A for {{ player_id }}.")
                self.assertEqual(p1["agent_prompt_template"], "Choose A in round {{ round_number }}.")
                self.assertEqual(p2["system_prompt_template"], "Strategy B for {{ player_id }}.")
                self.assertEqual(p2["agent_prompt_template"], "Choose B in round {{ round_number }}.")
            finally:
                manager.close()

    def test_agent_prompt_override_rejects_unknown_placeholders(self):
        with self.assertRaises(ValidationError):
            CournotSimulationRequest.model_validate(
                {
                    "agents": [
                        {
                            "player_id": "P1",
                            "policy": "cournot_openai_compatible",
                            "base_url": "http://localhost:11434/v1",
                            "model": "model-one",
                            "agent_prompt_template": "{{ unknown_value }}",
                        },
                        {
                            "player_id": "P2",
                            "policy": "cournot_best_reply",
                        },
                    ]
                }
            )

    def test_blank_agent_prompt_overrides_inherit_shared_prompts(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = CournotJobManager(Path(temporary))
            try:
                request = CournotSimulationRequest.model_validate(
                    {
                        "agents": [
                            {
                                "player_id": "P1",
                                "policy": "cournot_openai_compatible",
                                "base_url": "http://localhost:11434/v1",
                                "model": "model-one",
                                "system_prompt_template": "   ",
                                "agent_prompt_template": "",
                            },
                            {"player_id": "P2", "policy": "cournot_best_reply"},
                        ],
                        "prompts": {
                            "system_template": "Shared system {{ player_id }}.",
                            "agent_template": "Shared decision {{ round_number }}.",
                        },
                    }
                )

                participant = manager._pipeline_config("prompt-test", request)[
                    "participants"
                ][0]
                self.assertEqual(
                    participant["system_prompt_template"],
                    "Shared system {{ player_id }}.",
                )
                self.assertEqual(
                    participant["agent_prompt_template"],
                    "Shared decision {{ round_number }}.",
                )
            finally:
                manager.close()

    def test_request_accepts_two_to_eight_agents(self):
        for player_count in (2, 4, 8):
            with self.subTest(player_count=player_count):
                request = CournotSimulationRequest.model_validate(
                    {
                        "agents": [
                            {
                                "player_id": f"P{index}",
                                "policy": "cournot_best_reply",
                            }
                            for index in range(1, player_count + 1)
                        ]
                    }
                )
                self.assertEqual(len(request.agents), player_count)

        for player_count in (1, 9):
            with self.subTest(player_count=player_count):
                with self.assertRaises(ValidationError):
                    CournotSimulationRequest.model_validate(
                        {
                            "agents": [
                                {
                                    "player_id": f"P{index}",
                                    "policy": "cournot_best_reply",
                                }
                                for index in range(1, player_count + 1)
                            ]
                        }
                    )

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
                self.assertEqual(result["player_count"], 4)
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
                            "rounds": 2,
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
                first_request_id = pending["request_id"]
                pending = None
                while pending is None or pending["request_id"] == first_request_id:
                    self.assertLess(time.monotonic(), deadline)
                    pending = manager.pending_human_decision(state.simulation_id)
                    time.sleep(0.01)

                progress = manager.get(state.simulation_id).public_dict()
                self.assertEqual(progress["completed_rounds"], 1)
                self.assertEqual(progress["progress"], 0.5)
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

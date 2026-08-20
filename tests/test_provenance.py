from __future__ import annotations

import unittest

from sandbox.cournot_prompts import DEFAULT_COURNOT_SYSTEM_PROMPT
from sandbox.provenance import (
    decode_parameter_code,
    encode_parameter_code,
    experiment_parameter_spec,
    parameter_hash,
    sanitize_parameters,
)


class ProvenanceTests(unittest.TestCase):
    def test_hash_is_stable_for_key_order_and_changes_with_player_parameters(self):
        first = {
            "game_parameters": {"rounds": 40, "seed": 1},
            "players": [{"player_id": "P1", "temperature": 0.3}],
        }
        reordered = {
            "players": [{"temperature": 0.3, "player_id": "P1"}],
            "game_parameters": {"seed": 1, "rounds": 40},
        }
        changed = {
            **first,
            "players": [{"player_id": "P1", "temperature": 0.7}],
        }

        self.assertEqual(parameter_hash(first), parameter_hash(reordered))
        self.assertNotEqual(parameter_hash(first), parameter_hash(changed))

    def test_parameter_spec_includes_game_and_player_settings_without_secrets(self):
        parameters = experiment_parameter_spec(
            game_name="study",
            game_type="cournot",
            game_parameters={"rounds": 40, "treatment": "FULL"},
            prompts={"system_template": "maximize profit"},
            participants=[
                {
                    "player_id": "P1",
                    "model": "llama3.1:8b",
                    "temperature": 0.7,
                    "api_key": "do-not-record",
                    "api_key_env": "OLLAMA_API_KEY",
                    "job_id": "execution-only",
                }
            ],
        )

        player = parameters["players"][0]
        self.assertEqual(player["model"], "llama3.1:8b")
        self.assertEqual(player["temperature"], 0.7)
        self.assertEqual(player["api_key"], "<redacted>")
        self.assertEqual(player["api_key_env"], "OLLAMA_API_KEY")
        self.assertNotIn("job_id", player)

    def test_sanitizer_handles_nested_values(self):
        self.assertEqual(
            sanitize_parameters({"provider_options": {"secret": "value"}}),
            {"provider_options": {"secret": "<redacted>"}},
        )

    def test_numeric_code_round_trips_game_players_and_prompts(self):
        parameters = {
            "game_parameters": {
                "rounds": 40,
                "revision_probability": 2 / 3,
                "treatment": "FULL",
            },
            "players": [
                {
                    "player_id": "P1",
                    "policy": "cournot_llm",
                    "model": "llama3.1:8b",
                    "temperature": 0.7,
                    "provider_options": {"num_ctx": 16384},
                },
                {
                    "player_id": "P2",
                    "policy": "cournot_previous_average",
                    "initial_quantity": 20.0,
                },
            ],
            "prompts": {
                "system_template": DEFAULT_COURNOT_SYSTEM_PROMPT,
                "agent_template": "A custom exact prompt.",
            },
        }

        code = encode_parameter_code(parameters)

        self.assertTrue(code.isdecimal())
        self.assertTrue(code.startswith("27182801"))
        self.assertEqual(decode_parameter_code(code), parameters)

    def test_numeric_code_rejects_a_changed_digit(self):
        code = encode_parameter_code({"rounds": 40})
        changed_digit = "0" if code[-1] != "0" else "1"
        corrupted = code[:-1] + changed_digit

        with self.assertRaisesRegex(ValueError, "checksum"):
            decode_parameter_code(corrupted)


if __name__ == "__main__":
    unittest.main()

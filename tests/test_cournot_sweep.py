from __future__ import annotations

import copy
import csv
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from sandbox.cournot_sweep import (
    DEFAULT_SWEEP_CONFIG,
    ManifestStore,
    analyze_sweep,
    build_sweep_plan,
    execute_sweep,
    opponent_compositions,
    run_cli,
)


def reduced_config(model_counts=None):
    config = copy.deepcopy(DEFAULT_SWEEP_CONFIG)
    config["experiment"].update(
        {
            "model_counts": model_counts or [4],
            "temperatures": [0.3],
            "treatments": ["BEST"],
            "trials": 1,
            "rounds": 2,
            "steady_state_window": 1,
        }
    )
    config["execution"]["workers"] = 2
    return config


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fake_run(config: dict, *, fail: bool = False) -> Path:
    run_path = Path(config["output_dir"]) / config["run_id"]
    run_path.mkdir(parents=True)
    if fail:
        raise RuntimeError("intentional failure")
    decisions = []
    events = []
    quantities = (10.0, 20.0, 30.0, 40.0)
    policies = {
        item["player_id"]: item["policy"] for item in config["participants"]
    }
    for round_number in range(2):
        for index, quantity in enumerate(quantities, start=1):
            player_id = f"P{index}"
            profit = -quantity
            is_model = policies[player_id] == "cournot_llm"
            decisions.append(
                {
                    "round": round_number,
                    "player_id": player_id,
                    "revision_allowed": True,
                    "quantity": quantity,
                    "profit": profit,
                    "retry_count": 1 if is_model else 0,
                    "latency_seconds": 0.5 if is_model else 0,
                }
            )
            events.append(
                {
                    "event_type": "decision",
                    "round": round_number,
                    "player_id": player_id,
                    "revision_allowed": True,
                    "observation": {
                        "private_state": {
                            "best_reply_quantity": None if round_number == 0 else 20.0
                        }
                    },
                    "agent_metadata": (
                        {
                            "retry_count": 1,
                            "latency_seconds": 0.5,
                            "usage": {
                                "prompt_eval_count": 10,
                                "eval_count": 4,
                                "total_duration": 1_000_000_000,
                            },
                        }
                        if is_model
                        else {}
                    ),
                }
            )
    rounds = [
        {
            "round": round_number,
            "total_quantity": 100.0,
            "price": 0.0,
            "total_market_profit": -100.0,
            "distance_to_nash": 20.8,
        }
        for round_number in range(2)
    ]
    write_csv(run_path / "participant_data.csv", decisions)
    write_csv(run_path / "system_data.csv", rounds)
    (run_path / "summary.json").write_text(
        json.dumps({"game_type": "cournot"}), encoding="utf-8"
    )
    with (run_path / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    return run_path


class CournotSweepTests(unittest.TestCase):
    def test_default_matrix_has_expected_unique_cells_and_runs(self):
        plan = build_sweep_plan(DEFAULT_SWEEP_CONFIG)

        self.assertEqual(len(plan), 560)
        self.assertEqual(len({row["cell_id"] for row in plan}), 112)
        self.assertEqual(len({row["base_run_id"] for row in plan}), 560)

    def test_opponent_compositions_match_the_decided_design(self):
        self.assertEqual(len(opponent_compositions(1)), 4)
        self.assertEqual(len(opponent_compositions(2)), 6)
        self.assertEqual(len(opponent_compositions(3)), 3)
        self.assertEqual(opponent_compositions(4), [tuple()])
        self.assertIn(
            (
                "cournot_best_reply",
                "cournot_random_quantity",
                "cournot_previous_average",
            ),
            opponent_compositions(1),
        )

    def test_temperature_seed_and_positions_are_paired_and_rotated(self):
        config = reduced_config([1])
        config["experiment"]["temperatures"] = [0.0, 1.0]
        config["experiment"]["trials"] = 2
        plan = build_sweep_plan(config)
        homogeneous = [
            row
            for row in plan
            if row["opponent_composition"]
            == "best-reply+best-reply+best-reply"
        ]
        first_zero = next(
            row for row in homogeneous if row["temperature"] == 0 and row["trial_index"] == 0
        )
        first_one = next(
            row for row in homogeneous if row["temperature"] == 1 and row["trial_index"] == 0
        )
        second_zero = next(
            row for row in homogeneous if row["temperature"] == 0 and row["trial_index"] == 1
        )

        self.assertEqual(first_zero["environment_seed"], first_one["environment_seed"])
        zero_model = next(item for item in first_zero["assignments"] if item["policy"] == "cournot_llm")
        one_model = next(item for item in first_one["assignments"] if item["policy"] == "cournot_llm")
        self.assertEqual(zero_model["provider_options"]["seed"], one_model["provider_options"]["seed"])
        self.assertEqual(zero_model["temperature"], 0.0)
        self.assertEqual(one_model["temperature"], 1.0)
        self.assertIsNone(zero_model["memory_rounds"])
        self.assertEqual(zero_model["provider_options"]["num_ctx"], 16384)
        self.assertNotEqual(zero_model["player_id"], next(
            item["player_id"]
            for item in second_zero["assignments"]
            if item["policy"] == "cournot_llm"
        ))

    def test_execute_sweep_caps_concurrency_continues_failures_and_resumes(self):
        config = reduced_config([3])
        plan = build_sweep_plan(config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = ManifestStore(root, plan)
            store.write()
            lock = threading.Lock()
            active = 0
            maximum = 0
            calls = 0

            def runner(run_config):
                nonlocal active, maximum, calls
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                    calls += 1
                    call_number = calls
                try:
                    time.sleep(0.02)
                    return fake_run(run_config, fail=call_number == 1)
                finally:
                    with lock:
                        active -= 1

            result = execute_sweep(config, root, store, workers=2, run_function=runner)

            self.assertEqual(maximum, 2)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["completed"], 2)
            first_calls = calls
            execute_sweep(config, root, store, workers=2, run_function=runner)
            self.assertEqual(calls, first_calls + 1)
            self.assertTrue(all(row["status"] == "completed" for row in store.rows))
            execute_sweep(config, root, store, workers=2, run_function=runner)
            self.assertEqual(calls, first_calls + 1)

    def test_analysis_writes_metrics_tokens_confidence_inputs_and_dashboard(self):
        config = reduced_config([1])
        plan = build_sweep_plan(config)[:2]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for entry in plan:
                entry["attempts"] = 1
                entry["run_id"] = entry["base_run_id"] + "-attempt-01"
                entry["status"] = "completed"
                entry["elapsed_seconds"] = 2.0
                run_config = {
                    "run_id": entry["run_id"],
                    "output_dir": str(root / "runs"),
                    "participants": entry["assignments"],
                }
                entry["run_path"] = str(fake_run(run_config))

            analysis = analyze_sweep(root, config, plan)

            with (analysis / "run_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                run_rows = list(csv.DictReader(handle))
            with (analysis / "player_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                player_rows = list(csv.DictReader(handle))
            model = next(row for row in player_rows if row["cohort"] == "model")
            self.assertEqual(float(model["prompt_tokens"]), 20)
            self.assertEqual(float(model["output_tokens"]), 8)
            self.assertEqual(float(model["ollama_duration_seconds"]), 2.0)
            self.assertEqual(float(run_rows[0]["zero_price_share"]), 1.0)
            self.assertTrue((analysis / "cell_summary.csv").is_file())
            self.assertTrue((analysis / "metrics_long.csv").is_file())
            dashboard = (analysis / "dashboard.html").read_text(encoding="utf-8")
            self.assertIn('id="treatment"', dashboard)
            self.assertIn('id="temperature"', dashboard)
            self.assertIn("Model mean profit", dashboard)

    def test_dry_run_does_not_call_ollama_or_write_output(self):
        config = reduced_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config["output_dir"] = str(root / "output")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch("sandbox.cournot_sweep.preflight_ollama") as preflight:
                output = StringIO()
                with redirect_stdout(output):
                    result = run_cli([str(config_path), "--dry-run"])

            self.assertEqual(result, 0)
            preflight.assert_not_called()
            self.assertFalse((root / "output").exists())
            self.assertIn("Estimated Llama decisions", output.getvalue())

    def test_resume_dry_run_does_not_modify_manifest(self):
        config = reduced_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "resolved_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            plan = build_sweep_plan(config)
            plan[0]["status"] = "failed"
            plan[0]["error"] = "preserve this failure"
            store = ManifestStore(root, plan)
            store.write()
            before = (root / "manifest.json").read_text(encoding="utf-8")

            with patch("sandbox.cournot_sweep.preflight_ollama") as preflight:
                with redirect_stdout(StringIO()):
                    result = run_cli(["--resume", str(root), "--dry-run"])

            self.assertEqual(result, 0)
            preflight.assert_not_called()
            self.assertEqual(
                (root / "manifest.json").read_text(encoding="utf-8"), before
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import csv
import json
import re
import tempfile
import threading
import time
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from itertools import product
from pathlib import Path
from unittest.mock import patch

from sandbox.cournot_parameter_study import (
    DEFAULT_STUDY_CONFIG,
    MEMORY_CATEGORY,
    MEMORY_ONLY_VECTORS,
    MIXED_CATEGORY,
    TEMPERATURE_CATEGORY,
    StudyManifest,
    aggregate_trial_scores,
    build_configurations,
    build_study_plan,
    execute_study,
    memory_configurations,
    mixed_configurations,
    run_cli,
    selected_pending_rows,
    temperature_configurations,
    write_combination_catalog,
)
from sandbox.provenance import decode_parameter_code


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fake_run(config: dict, *, delay: float = 0.0, fail: bool = False) -> Path:
    if delay:
        time.sleep(delay)
    run_path = Path(config["output_dir"]) / config["run_id"]
    run_path.mkdir(parents=True)
    if fail:
        raise RuntimeError("intentional failure")
    decisions = []
    trial_adjustment = int(config["cournot"]["seed"])
    for round_number in range(2):
        for player_index, participant in enumerate(config["participants"], start=1):
            decisions.append(
                {
                    "run_id": config["run_id"],
                    "round": round_number,
                    "player_id": participant["player_id"],
                    "agent": participant["model"],
                    "agent_type": "AI",
                    "profit": player_index * 10 + trial_adjustment + round_number,
                }
            )
    _write_csv(run_path / "participant_data.csv", decisions)
    _write_csv(
        run_path / "system_data.csv",
        [
            {
                "run_id": config["run_id"],
                "round": round_number,
                "total_quantity": 80,
            }
            for round_number in range(2)
        ],
    )
    (run_path / "summary.json").write_text(
        json.dumps(
            {
                "run_id": config["run_id"],
                "game_name": config["game_name"],
                "game_type": "cournot",
                "cournot": config["cournot"],
                "participants": config["participants"],
            }
        ),
        encoding="utf-8",
    )
    return run_path


class CournotParameterStudyTests(unittest.TestCase):
    def test_compact_design_has_expected_unique_counts(self):
        configurations = build_configurations()
        plan = build_study_plan(DEFAULT_STUDY_CONFIG)

        self.assertEqual(len(configurations), 73)
        self.assertEqual(len(plan), 219)
        self.assertEqual(len({row["base_run_id"] for row in plan}), 219)
        self.assertEqual(
            Counter(row["category"] for row in configurations),
            {
                TEMPERATURE_CATEGORY: 17,
                MEMORY_CATEGORY: 10,
                MIXED_CATEGORY: 46,
            },
        )
        self.assertEqual(len(plan) + len(configurations), 292)

    def test_temperature_design_uses_midpoint_only_for_uniform_baseline(self):
        configurations = temperature_configurations()
        binary = [
            row for row in configurations if set(row["temperatures"]) <= {0.3, 0.7}
        ]
        midpoint = [row for row in configurations if 0.5 in row["temperatures"]]

        self.assertEqual(
            {tuple(row["temperatures"]) for row in binary},
            set(product((0.3, 0.7), repeat=4)),
        )
        self.assertEqual(len(midpoint), 1)
        self.assertEqual(midpoint[0]["temperatures"], [0.5] * 4)
        self.assertTrue(midpoint[0]["memory_study_baseline"])

    def test_memory_design_matches_explicit_vectors(self):
        configurations = memory_configurations()

        self.assertEqual(
            [tuple(row["memory_rounds"]) for row in configurations],
            list(MEMORY_ONLY_VECTORS),
        )
        self.assertTrue(all(row["temperatures"] == [0.5] * 4 for row in configurations))
        self.assertNotIn((None, None, None, None), MEMORY_ONLY_VECTORS)

    def test_catalog_records_every_profile_and_shared_baselines(self):
        configurations = build_configurations()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_combination_catalog(root, configurations)
            with (root / "combination_catalog.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 73)
        memory_baseline = next(
            row for row in rows if row["memory_study_baseline"] == "True"
        )
        self.assertEqual(memory_baseline["category"], TEMPERATURE_CATEGORY)
        self.assertEqual(memory_baseline["p1_temperature"], "0.5")
        self.assertEqual(memory_baseline["p1_memory_rounds"], "null")
        self.assertEqual(
            sum(row["mixed_study_control"] == "True" for row in rows), 2
        )

    def test_mixed_design_is_unique_balanced_and_excludes_two_controls(self):
        configurations = mixed_configurations()
        profiles = {
            (tuple(row["temperatures"]), tuple(row["memory_rounds"]))
            for row in configurations
        }

        self.assertEqual(len(configurations), 46)
        self.assertEqual(len(profiles), 46)
        self.assertNotIn(((0.3,) * 4, (None,) * 4), profiles)
        self.assertNotIn(((0.7,) * 4, (None,) * 4), profiles)
        for position in range(4):
            self.assertEqual(
                Counter(row["memory_rounds"][position] for row in configurations),
                {None: 14, 0: 16, 1: 16},
            )
        alternating = [
            tuple(row["memory_rounds"])
            for row in configurations
            if row["temperatures"] == [0.3, 0.7, 0.3, 0.7]
        ]
        self.assertEqual(
            alternating,
            [(1, 0, 1, 0), (None, 1, None, 1), (0, None, 0, None)],
        )

    def test_plan_propagates_full_revision_memory_temperature_and_paired_seeds(self):
        plan = build_study_plan(DEFAULT_STUDY_CONFIG)
        first = plan[0]
        same_trial_other_configuration = plan[3]

        self.assertEqual(first["treatment"], "FULL")
        self.assertEqual(first["environment_seed"], same_trial_other_configuration["environment_seed"])
        for first_player, other_player in zip(
            first["assignments"], same_trial_other_configuration["assignments"]
        ):
            self.assertEqual(
                first_player["provider_options"]["seed"],
                other_player["provider_options"]["seed"],
            )
        self.assertEqual(
            [item["temperature"] for item in first["assignments"]],
            first["temperatures"],
        )
        self.assertEqual(
            [item["memory_rounds"] for item in first["assignments"]],
            first["memory_rounds"],
        )

    def test_sharding_is_deterministic_and_disjoint(self):
        plan = build_study_plan(DEFAULT_STUDY_CONFIG)
        shards = [
            selected_pending_rows(plan, shard_index=index, shard_count=3)
            for index in range(3)
        ]
        ids = [{row["base_run_id"] for row in shard} for shard in shards]

        self.assertEqual(sum(len(shard) for shard in shards), len(plan))
        self.assertFalse(ids[0] & ids[1])
        self.assertFalse(ids[0] & ids[2])
        self.assertFalse(ids[1] & ids[2])
        self.assertEqual(
            [row["base_run_id"] for row in shards[0]],
            [row["base_run_id"] for row in selected_pending_rows(
                plan, shard_index=0, shard_count=3
            )],
        )

    def test_execution_caps_concurrency_writes_graphs_and_resumes(self):
        config = copy.deepcopy(DEFAULT_STUDY_CONFIG)
        plan = build_study_plan(config)[:6]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = StudyManifest(root, plan)
            store.write()
            lock = threading.Lock()
            active = 0
            maximum = 0
            calls = 0

            def runner(run_config):
                nonlocal active, maximum, calls
                semaphore = run_config["_cournot_request_semaphore"]
                with semaphore:
                    with lock:
                        active += 1
                        maximum = max(maximum, active)
                        calls += 1
                    try:
                        return fake_run(run_config, delay=0.01)
                    finally:
                        with lock:
                            active -= 1

            result = execute_study(
                config, root, store, workers=2, run_function=runner
            )

            self.assertEqual(result["completed"], 6)
            self.assertEqual(maximum, 2)
            self.assertEqual(calls, 6)
            self.assertTrue(
                all((root / row["trial_graph_path"]).is_file() for row in plan)
            )
            aggregate_paths = {root / row["aggregate_graph_path"] for row in plan}
            self.assertEqual(len(aggregate_paths), 2)
            self.assertTrue(all(path.is_file() for path in aggregate_paths))
            first_svg = (root / plan[0]["trial_graph_path"]).read_text(
                encoding="utf-8"
            )
            code = re.search(r'data-parameter-code="([0-9]+)"', first_svg).group(1)
            decoded = decode_parameter_code(code)
            self.assertEqual(
                decoded["experiments"][0]["players"][0]["temperature"], 0.3
            )

            execute_study(config, root, store, workers=2, run_function=runner)
            self.assertEqual(calls, 6)
            missing_aggregate = next(iter(aggregate_paths))
            missing_aggregate.unlink()
            execute_study(config, root, store, workers=2, run_function=runner)
            self.assertEqual(calls, 6)
            self.assertTrue(missing_aggregate.is_file())

    def test_aggregate_confidence_interval_uses_trial_level_player_means(self):
        config = copy.deepcopy(DEFAULT_STUDY_CONFIG)
        plan = build_study_plan(config)[:3]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_paths = []
            for entry in plan:
                run_config = {
                    "game_name": config["name"],
                    "run_id": entry["base_run_id"],
                    "output_dir": str(root),
                    "cournot": {
                        "seed": entry["environment_seed"],
                        "treatment": "FULL",
                    },
                    "participants": entry["assignments"],
                }
                run_paths.append(fake_run(run_config))

            p1 = next(
                score
                for score in aggregate_trial_scores(run_paths)
                if score.player_id == "P1"
            )

            self.assertEqual(p1.round_count, 3)
            self.assertEqual(p1.average_profit, 18.5)
            self.assertEqual(p1.standard_deviation, 1.0)
            self.assertAlmostEqual(p1.ci95_half_width, 4.303 / (3 ** 0.5))

    def test_failure_continues_and_resume_reuses_completed_raw_run(self):
        config = copy.deepcopy(DEFAULT_STUDY_CONFIG)
        plan = build_study_plan(config)[:3]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = StudyManifest(root, plan)
            store.write()
            calls = 0

            def runner(run_config):
                nonlocal calls
                calls += 1
                return fake_run(run_config, fail=calls == 1)

            first = execute_study(
                config, root, store, workers=2, run_function=runner
            )
            self.assertEqual(first["failed"], 1)
            self.assertEqual(first["completed"], 2)

            second = execute_study(
                config, root, store, workers=2, run_function=runner
            )
            self.assertEqual(second["completed"], 1)
            self.assertTrue(all(row["status"] == "completed" for row in store.rows))
            self.assertEqual(calls, 4)

    def test_dry_run_does_not_preflight_or_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "output"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(DEFAULT_STUDY_CONFIG), encoding="utf-8")
            with patch(
                "sandbox.cournot_parameter_study.preflight_ollama"
            ) as preflight:
                stdout = StringIO()
                with redirect_stdout(stdout):
                    result = run_cli(
                        [
                            str(config_path),
                            "--output-dir",
                            str(output_dir),
                            "--dry-run",
                        ]
                    )

            self.assertEqual(result, 0)
            preflight.assert_not_called()
            self.assertFalse(output_dir.exists())
            self.assertIn("Planned trial executions: 219", stdout.getvalue())
            self.assertIn("Planned SVG graphs: 292", stdout.getvalue())

    def test_resume_nonexistent_dry_run_directory_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "dry-run-output"
            stderr = StringIO()

            with self.assertRaises(SystemExit) as raised, redirect_stdout(
                StringIO()
            ), redirect_stderr(stderr):
                run_cli(["--resume", str(missing)])

            self.assertEqual(raised.exception.code, 2)
            message = stderr.getvalue()
            self.assertIn("study directory does not exist", message)
            self.assertIn("A --dry-run does not create it", message)
            self.assertIn("--output-dir", message)

    def test_resume_incomplete_directory_lists_missing_control_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stderr = StringIO()
            with self.assertRaises(SystemExit), redirect_stderr(stderr):
                run_cli(["--resume", str(root)])

            self.assertIn("resolved_config.json", stderr.getvalue())
            self.assertIn("manifest.json", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

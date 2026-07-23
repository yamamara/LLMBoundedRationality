from __future__ import annotations

import csv
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from sandbox.playback import cournot_playback_data
from sandbox.scorecard import analyze_runs
from sandbox.visualization import cournot_player_scores


class VisualizationTests(unittest.TestCase):
    def test_cournot_scores_use_all_rounds_and_two_standard_deviations(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            self.write_cournot_run(run_dir)

            scores = cournot_player_scores([run_dir])
            p1 = next(score for score in scores if score.player_id == "P1")
            p2 = next(score for score in scores if score.player_id == "P2")

            self.assertEqual(p1.round_count, 3)
            self.assertEqual(p1.average_profit, 2.0)
            self.assertEqual(p1.standard_deviation, 1.0)
            self.assertEqual(p1.lower_two_sd, 0.0)
            self.assertEqual(p1.upper_two_sd, 4.0)
            self.assertAlmostEqual(p1.ci95_half_width, 4.303 / (3 ** 0.5))
            self.assertAlmostEqual(p1.ci95_lower, 2.0 - 4.303 / (3 ** 0.5))
            self.assertAlmostEqual(p1.ci95_upper, 2.0 + 4.303 / (3 ** 0.5))
            self.assertEqual(p1.total_profit, 6.0)
            self.assertEqual(p2.standard_deviation, 0.0)
            self.assertEqual(p2.lower_two_sd, -2.0)
            self.assertEqual(p2.upper_two_sd, -2.0)
            self.assertEqual(p2.ci95_half_width, 0.0)

    def test_cournot_analysis_writes_csv_and_valid_svg(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            output_dir = root / "analysis"
            run_dir.mkdir()
            self.write_cournot_run(run_dir)

            analyze_runs([run_dir], output_dir)

            with (output_dir / "cournot_player_scores.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["player_id"] for row in rows},
                {"P1", "P2", "P3", "P4"},
            )
            svg_path = output_dir / "cournot_player_scores.svg"
            self.assertGreater(svg_path.stat().st_size, 0)
            root_element = ET.parse(svg_path).getroot()
            self.assertEqual(root_element.tag, "{http://www.w3.org/2000/svg}svg")
            svg = svg_path.read_text(encoding="utf-8")
            self.assertIn("+/- 2 SD", svg)
            self.assertIn("#2563eb", svg)
            self.assertIn("#d97706", svg)
            ci95_svg_path = output_dir / "cournot_player_scores_ci95.svg"
            self.assertGreater(ci95_svg_path.stat().st_size, 0)
            ci95_root = ET.parse(ci95_svg_path).getroot()
            self.assertEqual(ci95_root.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertIn(
                "95% t confidence interval",
                ci95_svg_path.read_text(encoding="utf-8"),
            )
            playback = output_dir / "cournot_playback.html"
            self.assertGreater(playback.stat().st_size, 0)
            html = playback.read_text(encoding="utf-8")
            self.assertIn("Cournot Event Playback", html)
            self.assertIn('id="timeline-slider"', html)
            self.assertIn('id="previous-button"', html)
            self.assertIn('id="play-button"', html)
            self.assertIn('id="next-button"', html)
            self.assertIn('id="speed-select"', html)
            self.assertIn("window.cournotPlayback", html)
            self.assertNotIn("secret repeated request", html)
            self.assertNotIn("</script><img", html)
            self.assertIn("<\\/script><img", html)

    def test_playback_preserves_event_order_and_reconstructs_market_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            self.write_cournot_run(run_dir)

            timeline = cournot_playback_data([run_dir])["runs"][0]["timeline"]

            self.assertEqual(
                [step["event_type"] for step in timeline],
                ["decision", "decision", "decision", "decision", "settlement"],
            )
            self.assertEqual(timeline[0]["players"][0]["state"], "revised")
            self.assertEqual(timeline[0]["players"][1]["state"], "pending")
            p2 = next(
                player for player in timeline[1]["players"] if player["player_id"] == "P2"
            )
            self.assertEqual(p2["state"], "held")
            self.assertEqual(timeline[1]["market"]["stage"], "provisional")
            self.assertEqual(timeline[-1]["market"]["stage"], "settled")
            self.assertEqual(timeline[-1]["market"]["total_quantity"], 100)
            self.assertEqual(timeline[-1]["players"][0]["profit"], 1)
            metadata = timeline[0]["event"]["metadata"]
            self.assertEqual(metadata["retry_count"], 1)
            self.assertNotIn("attempts", metadata)

    def test_auction_analysis_does_not_write_cournot_visualization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            output_dir = root / "analysis"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text(
                json.dumps({"game_type": "auction"}), encoding="utf-8"
            )
            self.write_csv(
                run_dir / "participant_data.csv",
                [
                    {
                        "run_id": "auction-run",
                        "institution": "baseline",
                        "population_composition": "AI_only",
                        "agent_type": "AI",
                        "payoff": 5,
                        "regret": 0,
                        "valid_action": True,
                        "retry_count": 0,
                        "latency_seconds": 0,
                    }
                ],
            )
            self.write_csv(
                run_dir / "system_data.csv",
                [
                    {
                        "run_id": "auction-run",
                        "institution": "baseline",
                        "population_composition": "AI_only",
                        "allocative_efficiency": 1,
                    }
                ],
            )

            analyze_runs([run_dir], output_dir)

            self.assertFalse((output_dir / "cournot_player_scores.csv").exists())
            self.assertFalse((output_dir / "cournot_player_scores.svg").exists())
            self.assertFalse((output_dir / "cournot_player_scores_ci95.svg").exists())
            self.assertFalse((output_dir / "cournot_playback.html").exists())

    def write_cournot_run(self, run_dir: Path) -> None:
        (run_dir / "summary.json").write_text(
            json.dumps({"game_type": "cournot"}), encoding="utf-8"
        )
        participant_rows = []
        for round_number, profit, revision_allowed in (
            (0, 1, True),
            (1, 2, False),
            (2, 3, True),
        ):
            participant_rows.append(
                {
                    "run_id": "cournot-run",
                    "round": round_number,
                    "institution": "baseline",
                    "population_composition": "mixed",
                    "player_id": "P1",
                    "agent": "model",
                    "agent_type": "AI",
                    "revision_allowed": revision_allowed,
                    "profit": profit,
                    "valid_action": True,
                    "retry_count": 0,
                    "latency_seconds": 0,
                }
            )
        participant_rows.append(
            {
                "run_id": "cournot-run",
                "round": 0,
                "institution": "baseline",
                "population_composition": "mixed",
                "player_id": "P2",
                "agent": "human_cli",
                "agent_type": "human",
                "revision_allowed": True,
                "profit": -2,
                "valid_action": True,
                "retry_count": 0,
                "latency_seconds": 0,
            }
        )
        for player_id in ("P3", "P4"):
            participant_rows.append(
                {
                    "run_id": "cournot-run",
                    "round": 0,
                    "institution": "baseline",
                    "population_composition": "mixed",
                    "player_id": player_id,
                    "agent": "model",
                    "agent_type": "AI",
                    "revision_allowed": True,
                    "profit": 0,
                    "valid_action": True,
                    "retry_count": 0,
                    "latency_seconds": 0,
                }
            )
        self.write_csv(run_dir / "participant_data.csv", participant_rows)
        self.write_csv(
            run_dir / "system_data.csv",
            [
                {
                    "run_id": "cournot-run",
                    "institution": "baseline",
                    "population_composition": "mixed",
                    "total_quantity": 80,
                    "total_market_profit": 400,
                    "distance_to_nash": 0.8,
                }
            ],
        )
        events = []
        for index, (player_id, quantity, revision_allowed) in enumerate(
            (
                ("P1", 10, True),
                ("P2", 20, False),
                ("P3", 30, True),
                ("P4", 40, True),
            )
        ):
            reasoning = (
                "</script><img src=x onerror=alert(1)>" if index == 0 else "Test decision."
            )
            action = {
                "action_type": "submit_quantity",
                "value": {"quantity": quantity},
                "reasoning": reasoning,
            }
            events.append(
                {
                    "event_type": "decision",
                    "round": 0,
                    "player_id": player_id,
                    "observation": {
                        "player_id": player_id,
                        "round": 0,
                        "public_state": {"treatment": "FULL", "completed_rounds": []},
                        "private_state": {"best_reply_quantity": 19.8},
                    },
                    "returned_action": action,
                    "applied_action": action,
                    "valid": True,
                    "invalid_reason": None,
                    "revision_allowed": revision_allowed,
                    "agent_metadata": {
                        "retry_count": 1 if index == 0 else 0,
                        "latency_seconds": 0.25,
                        "attempts": [{"request": "secret repeated request"}],
                        **({"forced_hold": True} if not revision_allowed else {}),
                    },
                }
            )
        events.append(
            {
                "event_type": "settlement",
                "round": 0,
                "treatment": "FULL",
                "total_quantity": 100,
                "price": 0,
                "total_market_profit": -1,
                "distance_to_nash": 20.8,
                "firm_results": [
                    {"player_id": "P1", "quantity": 10, "profit": 1},
                    {"player_id": "P2", "quantity": 20, "profit": -2},
                    {"player_id": "P3", "quantity": 30, "profit": 0},
                    {"player_id": "P4", "quantity": 40, "profit": 0},
                ],
            }
        )
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event))
                handle.write("\n")

    @staticmethod
    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()

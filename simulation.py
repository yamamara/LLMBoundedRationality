from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sandbox.models import Agent, Participant
from sandbox.serialization import OutputWriter
from sandbox.agents.cournot_best_reply import CournotBestReplyAgent
from sandbox.agents.cournot_openai_compatible import CournotOpenAICompatibleAgent
from sandbox.agents.human_cli import HumanCliAgent
from sandbox.agents.openai_compatible import OpenAICompatibleAgent, OpenAICompatibleConfig
from sandbox.games.auction.environment import AuctionConfig, AuctionEnvironment
from sandbox.games.cournot.environment import CournotConfig, CournotEnvironment
from sandbox.scorecard import analyze_runs


def load_local_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def population_composition(participants: list[Participant]) -> str:
    agent_types = {participant.agent_type for participant in participants}

    if agent_types == {"human"}:
        return "human_only"
    if agent_types == {"AI"}:
        return "AI_only"
    if agent_types == {"human", "AI"}:
        return "mixed"
    raise ValueError("agent_type must be either 'human' or 'AI'")


def participants_builder(configs: list[dict[str, Any]]) -> list[Participant]:
    participants = []

    for config in configs:
        participant = Participant(
            player_id=config["player_id"],
            agent_type=config["agent_type"],
            role=config.get("role", "bidder"),
            agent=agent_builder(config),
        )

        participants.append(participant)

    return participants


def agent_builder(config: dict[str, Any]) -> Agent:
    agent_type = config["agent_type"]
    policy = config["policy"]

    if agent_type == "human" and policy == "human_cli":
        return HumanCliAgent()
    if agent_type == "AI" and policy == "openai_compatible":
        return OpenAICompatibleAgent(
            OpenAICompatibleConfig(
                base_url=config["base_url"],
                model=config["model"],
                api_key_env=config.get("api_key_env"),
                temperature=config.get("temperature", 0.0),
                max_tokens=config.get("max_tokens", 800),
                timeout_seconds=config.get("timeout_seconds", 60.0),
                memory_rounds=config.get("memory_rounds"),
                max_retries=config.get("max_retries", 2),
                reasoning_effort=config.get("reasoning_effort", "none"),
                response_format=config.get("response_format"),
            )
        )
    if agent_type == "AI" and policy == "cournot_openai_compatible":
        return CournotOpenAICompatibleAgent(
            OpenAICompatibleConfig(
                base_url=config["base_url"],
                model=config["model"],
                api_key_env=config.get("api_key_env"),
                temperature=config.get("temperature", 0.0),
                max_tokens=config.get("max_tokens", 800),
                timeout_seconds=config.get("timeout_seconds", 60.0),
                memory_rounds=config.get("memory_rounds", 1),
                max_retries=config.get("max_retries", 2),
                reasoning_effort=config.get("reasoning_effort", "none"),
                response_format=config.get("response_format"),
            )
        )
    if agent_type == "AI" and policy == "cournot_best_reply":
        return CournotBestReplyAgent(config.get("initial_quantity", 20.0))

    raise ValueError(f"Unsupported agent configuration: agent_type={agent_type!r}, policy={policy!r}")


def run_experiment(config: dict[str, Any]) -> Path:
    load_local_env()

    if config.get("run_id"):
        run_id = config["run_id"]
    else:
        name = config.get("game_name", "unnamed-game")
        time = datetime.now(UTC).strftime("%m-%d-%Y-%H%M%S")
        run_id = f"{name}-{time}"

    # Path class overrides division operator for easier concatenation, woohoo
    run_dir = Path(config.get("output_dir", "runs")) / run_id

    output = OutputWriter(run_dir)
    participants = participants_builder(config["participants"])

    if "auction" in config:
        game_type = "auction"
        environment = AuctionEnvironment(AuctionConfig(**config["auction"]), participants, output)
    elif "cournot" in config:
        game_type = "cournot"
        environment = CournotEnvironment(CournotConfig(**config["cournot"]), participants, output)
    else:
        raise ValueError("Config must contain either auction or cournot settings")
    composition = population_composition(participants)
    rows = environment.run()

    for name, table in rows.items():
        for row in table:
            row["run_id"] = run_id
            row["game_name"] = config.get("game_name", "unnamed-game")
            row["game_type"] = game_type
            row["population_composition"] = composition
        output_name = {
            "decisions": "participant_data",
            "rounds": "system_data",
        }[name]
        output.write_csv(f"{output_name}.csv", table)

    output.write_json(
        "summary.json",
        {
            "run_id": run_id,
            "game_name": config.get("game_name", "unnamed-game"),
            "game_type": game_type,
            "created_at": datetime.now(UTC).isoformat(),
            "population_composition": composition,
            game_type: environment.get_config_as_dict(),
        },
    )

    return run_dir


def run_pipeline(
    config: dict[str, Any],
    analyze: bool = False,
    analysis_output: Path | None = None,
) -> tuple[Path, Path | None]:
    run_dir = run_experiment(config)
    should_analyze = analyze or analysis_output is not None or "cournot" in config
    if not should_analyze:
        return run_dir, None
    output_dir = analysis_output or run_dir / "analysis"
    analyze_runs([run_dir], output_dir)
    return run_dir, output_dir


def main():
    parser = argparse.ArgumentParser(description="Run a sandbox game experiment.")
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run the scorecard after the simulation; automatic for Cournot.",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        help="Analysis directory; providing it also enables --analyze.",
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)

    run_dir, analysis_dir = run_pipeline(
        config,
        analyze=args.analyze or args.analysis_output is not None,
        analysis_output=args.analysis_output,
    )
    print(run_dir)
    if analysis_dir:
        print(f"analysis: {analysis_dir}")


if __name__ == "__main__":
    main()

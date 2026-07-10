from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sandbox.models import Agent, Participant
from sandbox.serialization import OutputWriter
from sandbox.agents.human_cli import HumanCliAgent
from sandbox.agents.openai_compatible import OpenAICompatibleAgent, OpenAICompatibleConfig
from sandbox.games.auction.environment import AuctionConfig, AuctionEnvironment


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

    raise ValueError(f"Unsupported agent configuration: agent_type={agent_type!r}, policy={policy!r}")


def run_experiment(config: dict[str, Any]) -> Path:
    load_local_env()

    if config.get("run_id"):
        run_id = config["run_id"]
    else:
        name = config.get("game_name", "unnamed-auction")
        time = datetime.now(UTC).strftime("%m-%d-%Y-%H%M%S")
        run_id = f"{name}-{time}"

    # Path class overrides division operator for easier concatenation, woohoo
    run_dir = Path(config.get("output_dir", "runs")) / run_id

    output = OutputWriter(run_dir)
    participants = participants_builder(config["participants"])

    # Unpacking dictionary directly into AuctionConfig class since we set up the variables exactly the same
    auction_config = AuctionConfig(**config["auction"])

    environment = AuctionEnvironment(auction_config, participants, output)
    composition = population_composition(participants)
    rows = environment.run()

    for name, table in rows.items():
        for row in table:
            row["run_id"] = run_id
            row["game_name"] = config.get("game_name", "unnamed-auction")
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
            "game_name": config.get("game_name", "unnamed-auction"),
            "created_at": datetime.now(UTC).isoformat(),
            "population_composition": composition,
            "auction": environment.get_config_as_dict(),
        },
    )

    return run_dir


def main():
    # The only arg should point to an initial config JSON
    parser = argparse.ArgumentParser(description="Run a sandbox auction experiment.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)

    print(run_experiment(config))


if __name__ == "__main__":
    main()

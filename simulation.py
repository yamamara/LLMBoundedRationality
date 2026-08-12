from __future__ import annotations

import argparse
import copy
import json
import os
import re
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable

from sandbox.models import Agent, Participant
from sandbox.serialization import OutputWriter
from sandbox.agents.cournot_best_reply import CournotBestReplyAgent
from sandbox.agents.cournot_openai_compatible import CournotOpenAICompatibleAgent
from sandbox.agents.human_cli import HumanCliAgent
from sandbox.agents.web_human import WebHumanAgent
from sandbox.agents.openai_compatible import OpenAICompatibleAgent, OpenAICompatibleConfig
from sandbox.agents.llm_auction import LLMAuctionAgent, LLMAuctionAgentConfig
from sandbox.agents.llm_cournot import LLMCournotAgent, LLMCournotAgentConfig
from sandbox.providers import build_provider
from sandbox.games.auction.environment import AuctionConfig, AuctionEnvironment
from sandbox.games.cournot.environment import CournotConfig, CournotEnvironment
from sandbox.scorecard import analyze_runs
from sandbox.prompts import DEFAULT_AGENT_PROMPT, DEFAULT_SYSTEM_PROMPT, validate_prompt_templates
from sandbox.cournot_prompts import (
    DEFAULT_COURNOT_AGENT_PROMPT,
    DEFAULT_COURNOT_SYSTEM_PROMPT,
    validate_cournot_prompt_templates,
)


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
    if agent_type == "human" and policy == "web_human":
        return WebHumanAgent(
            config["job_id"], config.get("timeout_seconds", 3600.0)
        )
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
    if agent_type == "AI" and policy == "llm_auction":
        provider = build_provider(
            config["provider"],
            endpoint=config.get("endpoint"),
            api_key_env=config.get("api_key_env"),
            transport=config.get("transport"),
        )
        return LLMAuctionAgent(
            provider,
            LLMAuctionAgentConfig(
                model=config["model"],
                temperature=config.get("temperature", 0.0),
                top_p=config.get("top_p", 1.0),
                max_output_tokens=config.get("max_output_tokens", config.get("max_tokens", 800)),
                timeout_seconds=config.get("timeout_seconds", 60.0),
                memory_rounds=config.get("memory_rounds", 3),
                max_retries=config.get("max_retries", 2),
                reasoning_effort=config.get("reasoning_effort"),
                provider_options=config.get("provider_options", {}),
                system_prompt_template=config.get("system_prompt_template", DEFAULT_SYSTEM_PROMPT),
                agent_prompt_template=config.get("agent_prompt_template", DEFAULT_AGENT_PROMPT),
            ),
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
            ),
            system_prompt_template=config.get(
                "system_prompt_template", DEFAULT_COURNOT_SYSTEM_PROMPT
            ),
            agent_prompt_template=config.get(
                "agent_prompt_template", DEFAULT_COURNOT_AGENT_PROMPT
            ),
        )
    if agent_type == "AI" and policy == "cournot_llm":
        provider = build_provider(
            config["provider"],
            endpoint=config.get("endpoint"),
            api_key_env=config.get("api_key_env"),
            transport=config.get("transport"),
        )
        return LLMCournotAgent(
            provider,
            LLMCournotAgentConfig(
                model=config["model"],
                temperature=config.get("temperature", 0.0),
                top_p=config.get("top_p", 1.0),
                max_output_tokens=config.get(
                    "max_output_tokens", config.get("max_tokens", 800)
                ),
                timeout_seconds=config.get("timeout_seconds", 60.0),
                memory_rounds=config.get("memory_rounds", 1),
                max_retries=config.get("max_retries", 2),
                reasoning_effort=config.get("reasoning_effort"),
                provider_options=config.get("provider_options", {}),
                system_prompt_template=config.get(
                    "system_prompt_template", DEFAULT_COURNOT_SYSTEM_PROMPT
                ),
                agent_prompt_template=config.get(
                    "agent_prompt_template", DEFAULT_COURNOT_AGENT_PROMPT
                ),
            ),
        )
    if agent_type == "AI" and policy == "cournot_best_reply":
        return CournotBestReplyAgent(config.get("initial_quantity", 20.0))

    raise ValueError(f"Unsupported agent configuration: agent_type={agent_type!r}, policy={policy!r}")


def run_experiment(
    config: dict[str, Any],
    progress_callback: Callable[[int], None] | None = None,
) -> Path:
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
    prompts = config.get("prompts", {})
    is_cournot = "cournot" in config
    default_system = DEFAULT_COURNOT_SYSTEM_PROMPT if is_cournot else DEFAULT_SYSTEM_PROMPT
    default_agent = DEFAULT_COURNOT_AGENT_PROMPT if is_cournot else DEFAULT_AGENT_PROMPT
    system_template = prompts.get("system_template", default_system)
    agent_template = prompts.get("agent_template", default_agent)
    if is_cournot:
        validate_cournot_prompt_templates(system_template, agent_template)
    else:
        validate_prompt_templates(system_template, agent_template)
    participant_configs = copy.deepcopy(config["participants"])
    for participant in participant_configs:
        if participant.get("policy") == "llm_auction":
            participant["system_prompt_template"] = system_template
            participant["agent_prompt_template"] = agent_template
    participants = participants_builder(participant_configs)

    if "auction" in config:
        game_type = "auction"
        environment = AuctionEnvironment(AuctionConfig(**config["auction"]), participants, output)
    elif "cournot" in config:
        game_type = "cournot"
        environment = CournotEnvironment(
            CournotConfig(**config["cournot"]),
            participants,
            output,
            progress_callback,
        )
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
            "prompts": {"system_template": system_template, "agent_template": agent_template},
            game_type: environment.get_config_as_dict(),
        },
    )

    return run_dir


def run_pipeline(
    config: dict[str, Any],
    analyze: bool = False,
    analysis_output: Path | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[Path, Path | None]:
    if config.get("experiment_matrix"):
        run_dirs, root = run_experiment_matrix(config)
        if not analyze:
            return root, None
        output_dir = analysis_output or root / "analysis"
        analyze_runs(run_dirs, output_dir)
        return root, output_dir
    run_dir = run_experiment(config, progress_callback)
    should_analyze = analyze or analysis_output is not None or "cournot" in config
    if not should_analyze:
        return run_dir, None
    output_dir = analysis_output or run_dir / "analysis"
    analyze_runs([run_dir], output_dir)
    return run_dir, output_dir


def run_experiment_matrix(config: dict[str, Any]) -> tuple[list[Path], Path]:
    matrix = config["experiment_matrix"]
    auction = config["auction"]
    mechanisms = matrix.get("mechanisms") or [auction.get("mechanism", "first_price")]
    descriptions = matrix.get("mechanism_description_treatments") or [
        auction.get("mechanism_description_treatment", "concise")
    ]
    player_counts = matrix.get("player_counts") or [len(config["participants"])]
    schedules = matrix.get("valuation_schedules") or [
        {
            "name": auction.get("valuation_schedule", "baseline"),
            "player_ranges": auction.get("player_value_ranges", {}),
        }
    ]
    trials = int(config.get("n", 1))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    institution = re.sub(r"[^a-z0-9]+", "-", auction.get("institution", "baseline").lower()).strip("-")
    root = Path(config.get("output_dir", "runs")) / f"{institution or 'institution'}-{timestamp}-matrix"
    run_dirs: list[Path] = []
    manifest = []
    for mechanism, description, count, schedule in product(
        mechanisms, descriptions, player_counts, schedules
    ):
        active = config["participants"][:count]
        player_ids = [participant["player_id"] for participant in active]
        base_ranges = schedule.get("player_ranges", {})
        rotations = range(count) if schedule.get("rotate_across_positions") else range(1)
        for rotation in rotations:
            default_range = {
                "minimum": auction["true_value_min"],
                "maximum": auction["true_value_max"],
            }
            source = [base_ranges.get(player_id, default_range) for player_id in player_ids]
            ranges = {
                player_id: copy.deepcopy(source[(index - rotation) % count])
                for index, player_id in enumerate(player_ids)
            }
            schedule_name = schedule["name"] + (
                f"-rotation-{rotation + 1}" if schedule.get("rotate_across_positions") else ""
            )
            treatment = "__".join(
                re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
                for value in (mechanism, description, f"players-{count}", schedule_name)
            )
            for trial in range(trials):
                expanded = copy.deepcopy(config)
                expanded.pop("experiment_matrix", None)
                expanded.pop("n", None)
                expanded["participants"] = active
                expanded["output_dir"] = str(root / "runs")
                expanded["auction"].update(
                    {
                        "mechanism": mechanism,
                        "mechanism_description_treatment": description,
                        "mode": "ai_sealed_bid",
                        "player_value_ranges": ranges,
                        "valuation_schedule": schedule_name,
                        "num_players": count,
                        "seed": auction.get("seed", 7) + trial,
                        "tie_break_seed": auction.get("tie_break_seed", auction.get("seed", 7)) + trial,
                    }
                )
                if expanded["auction"].get("tie_break_priority"):
                    expanded["auction"]["tie_break_priority"] = [
                        value
                        for value in expanded["auction"]["tie_break_priority"]
                        if value in player_ids
                    ]
                expanded["run_id"] = f"{institution}-{timestamp}-{treatment}-trial-{trial + 1:04d}"
                run_dir = run_experiment(expanded)
                run_dirs.append(run_dir)
                manifest.append({"run_id": expanded["run_id"], "treatment_id": treatment, "trial_index": trial})
    root.mkdir(parents=True, exist_ok=True)
    (root / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dirs, root


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

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from sandbox.agents.llm_auction import LLMAuctionAgent, LLMAuctionAgentConfig
from sandbox.api.models import SimulationRequest
from sandbox.configuration import ProviderProfile, load_provider_profiles
from sandbox.experiments import TreatmentCell, expand_treatment_cells
from sandbox.games.auction.environment import AuctionConfig, AuctionEnvironment
from sandbox.models import Participant
from sandbox.power import prevent_system_sleep
from sandbox.providers import build_provider
from sandbox.serialization import OutputWriter
from sandbox.statistics import PRIMARY_METRICS, trial_summary


TERMINAL_STATUSES = {"completed", "completed_with_errors", "failed"}


class _CombinedSemaphore:
    def __init__(self, global_limit: threading.Semaphore, job_limit: threading.Semaphore):
        self.global_limit = global_limit
        self.job_limit = job_limit

    def __enter__(self):
        self.global_limit.acquire()
        self.job_limit.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.job_limit.release()
        self.global_limit.release()


@dataclass
class JobState:
    simulation_id: str
    status: str
    created_at: str
    total_runs: int
    total_rounds: int
    completed_runs: int = 0
    failed_runs: int = 0
    completed_rounds: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    request: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("request", None)
        value["progress"] = (
            self.completed_rounds / self.total_rounds if self.total_rounds else 0.0
        )
        return value


class SimulationJobManager:
    def __init__(
        self,
        output_root: Path | None = None,
        profiles: dict[str, ProviderProfile] | None = None,
        max_jobs: int = 2,
        max_provider_calls: int = 4,
        max_parallel_runs: int = 4,
    ):
        self.output_root = output_root or Path(os.environ.get("SIMULATION_OUTPUT_DIR", "runs/web"))
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.profiles = profiles or load_provider_profiles()
        self.executor = ThreadPoolExecutor(max_workers=max_jobs, thread_name_prefix="simulation-job")
        self.request_semaphore = threading.Semaphore(max_provider_calls)
        self.max_provider_calls = max_provider_calls
        self.max_parallel_runs = max_parallel_runs
        self.lock = threading.RLock()
        self.jobs: dict[str, JobState] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self._load_existing()

    def validate_request(self, request: SimulationRequest) -> None:
        if request.max_parallel_runs > self.max_parallel_runs:
            raise ValueError(
                f"max_parallel_runs exceeds the server limit of {self.max_parallel_runs}"
            )
        allowed_options = {
            "openai": {"service_tier", "store"},
            "anthropic": {"top_k", "stop_sequences", "service_tier"},
            "gemini": {"top_k", "stop_sequences", "candidate_count"},
            "local_llama": {
                "num_ctx", "num_gpu", "seed", "keep_alive", "repeat_penalty",
                "stop", "frequency_penalty", "presence_penalty",
            },
        }
        for agent in request.agents:
            profile = self.profiles.get(agent.profile_id)
            if profile is None:
                raise ValueError(f"Unknown provider profile: {agent.profile_id}")
            if not (agent.model or profile.model):
                raise ValueError(f"No model configured for provider profile {profile.profile_id}")
            if profile.provider != "local_llama" and not (
                profile.api_key_env and os.environ.get(profile.api_key_env)
            ):
                raise ValueError(f"Provider profile {profile.profile_id} is not configured")
            unknown = set(agent.provider_options) - allowed_options[profile.provider]
            if unknown:
                raise ValueError(
                    f"Unsupported options for {profile.provider}: {sorted(unknown)}"
                )
            if "num_ctx" in agent.provider_options and (
                isinstance(agent.provider_options["num_ctx"], bool)
                or not isinstance(agent.provider_options["num_ctx"], int)
                or agent.provider_options["num_ctx"] <= 0
            ):
                raise ValueError("num_ctx must be a positive integer")

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def submit(self, request: SimulationRequest) -> JobState:
        self.validate_request(request)
        cells = expand_treatment_cells(request)
        total_runs = len(cells) * request.n
        simulation_id = uuid.uuid4().hex
        state = JobState(
            simulation_id=simulation_id,
            status="queued",
            created_at=datetime.now(UTC).isoformat(),
            total_runs=total_runs,
            total_rounds=total_runs * request.auction.rounds,
            request=request.model_dump(mode="json"),
        )
        with self.lock:
            self.jobs[simulation_id] = state
            self._persist_state(state)
        self.executor.submit(self._run_job, simulation_id, request)
        return state

    def get(self, simulation_id: str) -> JobState | None:
        with self.lock:
            return self.jobs.get(simulation_id)

    def get_results(self, simulation_id: str) -> dict[str, Any] | None:
        with self.lock:
            if simulation_id in self.results:
                return self.results[simulation_id]
        path = self.output_root / simulation_id / "results.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            with self.lock:
                self.results[simulation_id] = result
            return result
        return None

    def _run_job(self, simulation_id: str, request: SimulationRequest) -> None:
        self._update(simulation_id, status="running")
        with prevent_system_sleep():
            self._run_job_awake(simulation_id, request)

    def _run_job_awake(self, simulation_id: str, request: SimulationRequest) -> None:
        decision_rows: list[dict[str, Any]] = []
        round_rows: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        job_provider_limit = threading.Semaphore(min(4, self.max_provider_calls))
        work = [
            (cell, trial_index)
            for cell in expand_treatment_cells(request)
            for trial_index in range(request.n)
        ]

        with ThreadPoolExecutor(
            max_workers=min(request.max_parallel_runs, len(work)),
            thread_name_prefix="auction-run",
        ) as run_executor:
            futures = {
                run_executor.submit(
                    self._run_one,
                    simulation_id,
                    request,
                    cell,
                    trial_index,
                    run_index,
                    job_provider_limit,
                ): (cell.treatment_id, trial_index, run_index)
                for run_index, (cell, trial_index) in enumerate(work)
            }
            for future in as_completed(futures):
                treatment_id, trial_index, run_index = futures[future]
                try:
                    rows = future.result()
                    decision_rows.extend(rows["decisions"])
                    round_rows.extend(rows["rounds"])
                    self._increment(simulation_id, completed_runs=1)
                except Exception as exc:
                    error = {
                        "run_index": run_index,
                        "trial_index": trial_index,
                        "treatment_id": treatment_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:1000],
                    }
                    errors.append(error)
                    self._increment(simulation_id, failed_runs=1, errors=[error])

        decision_rows.sort(key=lambda row: (row["run_index"], row["round"], row["player_id"]))
        round_rows.sort(key=lambda row: (row["run_index"], row["round"]))
        results = self._build_results(simulation_id, decision_rows, round_rows, errors)
        job_dir = self.output_root / simulation_id
        OutputWriter.write_csv_path(job_dir / "decision_rows.csv", decision_rows)
        OutputWriter.write_csv_path(job_dir / "round_rows.csv", round_rows)
        OutputWriter.write_csv_path(job_dir / "run_aggregates.csv", results["run_aggregates"])
        OutputWriter.write_csv_path(
            job_dir / "treatment_aggregates.csv", results["treatment_aggregates"]
        )
        result_path = self.output_root / simulation_id / "results.json"
        self._write_json_atomic(result_path, results)
        with self.lock:
            self.results[simulation_id] = results
            state = self.jobs[simulation_id]
            state.status = (
                "completed"
                if state.failed_runs == 0
                else "completed_with_errors"
                if state.completed_runs
                else "failed"
            )
            self._persist_state(state)

    def _run_one(
        self,
        simulation_id: str,
        request: SimulationRequest,
        cell: TreatmentCell,
        trial_index: int,
        run_index: int,
        job_provider_limit: threading.Semaphore,
    ) -> dict[str, list[dict[str, Any]]]:
        with self.lock:
            created_at = self.jobs[simulation_id].created_at
        timestamp = datetime.fromisoformat(created_at).astimezone(UTC).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        institution = re.sub(r"[^a-z0-9]+", "-", request.auction.institution.lower()).strip("-")
        institution = institution or "institution"
        run_id = f"{institution}-{timestamp}-{cell.treatment_id}-trial-{trial_index + 1:04d}"
        run_dir = self.output_root / simulation_id / "runs" / run_id
        output = OutputWriter(run_dir)
        participants = [
            self._participant(agent, request, job_provider_limit)
            for agent in request.agents[: cell.num_players]
        ]
        auction = request.auction
        tie_seed_base = auction.tie_break_seed if auction.tie_break_seed is not None else auction.seed
        config = AuctionConfig(
            rounds=auction.rounds,
            starting_budget=auction.starting_budget,
            true_value_min=auction.true_value_min,
            true_value_max=auction.true_value_max,
            mechanism=cell.mechanism,
            seed=auction.seed + trial_index,
            institution=auction.institution,
            mode="ai_sealed_bid",
            tie_breaking=auction.tie_breaking,
            tie_break_seed=tie_seed_base + trial_index,
            tie_break_priority=(
                [value for value in auction.tie_break_priority if value in {p.player_id for p in participants}]
                if auction.tie_break_priority
                else None
            ),
            mechanism_description_treatment=cell.mechanism_description_treatment,
            player_value_ranges=cell.player_value_ranges,
            valuation_schedule=cell.valuation_schedule,
            num_players=cell.num_players,
        )

        def progress(_round: int) -> None:
            self._increment(simulation_id, completed_rounds=1)

        environment = AuctionEnvironment(config, participants, output, progress)
        try:
            rows = environment.run()
        except Exception as exc:
            output.log_event(
                {"event_type": "run_failed", "run_id": run_id, "error_type": type(exc).__name__, "error": str(exc)}
            )
            output.write_json(
                "summary.json",
                {"run_id": run_id, "run_index": run_index, "status": "failed", "error": str(exc), "auction": asdict(config)},
            )
            raise

        for table in rows.values():
            for row in table:
                row.update(
                    {
                        "simulation_id": simulation_id,
                        "run_id": run_id,
                        "run_index": run_index,
                        "trial_index": trial_index,
                        "treatment_id": cell.treatment_id,
                        "game_name": request.name,
                        "game_type": "auction",
                        "population_composition": "AI_only",
                    }
                )
        output.write_csv("participant_data.csv", rows["decisions"])
        output.write_csv("system_data.csv", rows["rounds"])
        output.write_json(
            "summary.json",
            {
                "run_id": run_id,
                "run_index": run_index,
                "trial_index": trial_index,
                "treatment_id": cell.treatment_id,
                "simulation_id": simulation_id,
                "status": "completed",
                "created_at": datetime.now(UTC).isoformat(),
                "game_type": "auction",
                "population_composition": "AI_only",
                "auction": asdict(config),
                "agents": [agent.model_dump(mode="json") for agent in request.agents[: cell.num_players]],
                "prompts": request.prompts.model_dump(mode="json"),
            },
        )
        output.log_event({"event_type": "run_completed", "run_id": run_id})
        return rows

    def _participant(self, spec, request: SimulationRequest, job_provider_limit: threading.Semaphore) -> Participant:
        if spec.profile_id not in self.profiles:
            raise ValueError(f"Unknown provider profile: {spec.profile_id}")
        profile = self.profiles[spec.profile_id]
        model = spec.model or profile.model
        if not model:
            raise ValueError(f"No model configured for provider profile {profile.profile_id}")
        if not profile.configured:
            raise ValueError(f"Provider profile {profile.profile_id} is not configured")
        options = {**profile.defaults, **spec.provider_options}
        for common_key in ("temperature", "top_p"):
            options.pop(common_key, None)
        provider = build_provider(profile.provider, profile.endpoint, profile.api_key_env, profile.transport)
        agent = LLMAuctionAgent(
            provider,
            LLMAuctionAgentConfig(
                model=model,
                temperature=spec.temperature,
                top_p=1.0,
                max_output_tokens=spec.max_output_tokens,
                timeout_seconds=spec.timeout_seconds,
                memory_rounds=spec.memory_rounds,
                max_retries=spec.max_retries,
                reasoning_effort=spec.reasoning_effort,
                provider_options=options,
                system_prompt_template=request.prompts.system_template,
                agent_prompt_template=request.prompts.agent_template,
            ),
            _CombinedSemaphore(self.request_semaphore, job_provider_limit),
        )
        return Participant(spec.player_id, agent, "AI", "bidder")

    def _build_results(self, simulation_id, decisions, rounds, errors):
        agent_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in decisions:
            agent_groups.setdefault((row["run_id"], row["player_id"]), []).append(row)
        agent_aggregates = []
        for (run_id, player_id), rows in sorted(agent_groups.items()):
            ratios = [
                row["bid_to_value_ratio"]
                for row in rows
                if row["bid_to_value_ratio"] is not None
            ]
            agent_aggregates.append(
                {
                    "run_id": run_id,
                    "run_index": rows[0]["run_index"],
                    "player_id": player_id,
                    "provider": rows[0]["provider"],
                    "model": rows[0]["model"],
                    "mean_bid": mean(row["bid"] for row in rows),
                    "mean_bid_to_value_ratio": mean(ratios) if ratios else None,
                    "total_utility": sum(row["utility"] for row in rows),
                    "mean_utility": mean(row["utility"] for row in rows),
                    "mean_payoff": mean(row["payoff"] for row in rows),
                    "mean_truthfulness_deviation": mean(
                        row["truthfulness_deviation"] for row in rows
                    ),
                    "wins": sum(bool(row["winner"]) for row in rows),
                    "win_rate": mean(bool(row["winner"]) for row in rows),
                }
            )
        run_aggregates = []
        for run_id in sorted({row["run_id"] for row in rounds}):
            selected = [row for row in rounds if row["run_id"] == run_id]
            selected_decisions = [row for row in decisions if row["run_id"] == run_id]
            run_aggregates.append(
                {
                    "run_id": run_id,
                    "run_index": selected[0]["run_index"],
                    "trial_index": selected[0]["trial_index"],
                    "treatment_id": selected[0]["treatment_id"],
                    "mechanism": selected[0]["mechanism"],
                    "mechanism_description_treatment": selected[0]["mechanism_description_treatment"],
                    "valuation_schedule": selected[0]["valuation_schedule"],
                    "num_players": selected[0]["num_players"],
                    "total_revenue": sum(row["revenue"] for row in selected),
                    "mean_revenue": mean(row["revenue"] for row in selected),
                    "mean_efficiency": mean(row["allocative_efficiency"] for row in selected),
                    "total_utility": sum(row["total_utility"] for row in selected),
                    "revenue": mean(row["revenue"] for row in selected),
                    "allocative_efficiency": mean(
                        row["allocative_efficiency"] for row in selected
                    ),
                    "payoff": mean(row["payoff"] for row in selected_decisions),
                    "truthfulness_deviation": mean(
                        row["truthfulness_deviation"] for row in selected_decisions
                    ),
                }
            )
        treatment_aggregates = []
        for treatment_id in sorted({row["treatment_id"] for row in run_aggregates}):
            selected = [row for row in run_aggregates if row["treatment_id"] == treatment_id]
            dimensions = {
                key: selected[0][key]
                for key in (
                    "mechanism",
                    "mechanism_description_treatment",
                    "valuation_schedule",
                    "num_players",
                )
            }
            for metric in PRIMARY_METRICS:
                treatment_aggregates.append(
                    {
                        "treatment_id": treatment_id,
                        **dimensions,
                        "metric": metric,
                        "failed_trials": sum(
                            error.get("treatment_id") == treatment_id for error in errors
                        ),
                        **trial_summary(row[metric] for row in selected),
                    }
                )
        return {
            "simulation_id": simulation_id,
            "decision_rows": decisions,
            "round_rows": rounds,
            "agent_aggregates": agent_aggregates,
            "run_aggregates": run_aggregates,
            "treatment_aggregates": treatment_aggregates,
            "metric_definitions": PRIMARY_METRICS,
            "batch_aggregates": {
                "successful_runs": len(run_aggregates),
                "failed_runs": len(errors),
                "mean_revenue": mean(row["mean_revenue"] for row in run_aggregates) if run_aggregates else None,
                "mean_efficiency": mean(row["mean_efficiency"] for row in run_aggregates) if run_aggregates else None,
                "total_utility": sum(row["total_utility"] for row in run_aggregates),
                "mean_payoff": mean(row["payoff"] for row in run_aggregates) if run_aggregates else None,
                "mean_truthfulness_deviation": mean(
                    row["truthfulness_deviation"] for row in run_aggregates
                ) if run_aggregates else None,
            },
            "errors": errors,
        }

    def _increment(self, simulation_id: str, **increments) -> None:
        with self.lock:
            state = self.jobs[simulation_id]
            for key, value in increments.items():
                if key == "errors":
                    state.errors.extend(value)
                else:
                    setattr(state, key, getattr(state, key) + value)
            self._persist_state(state)

    def _update(self, simulation_id: str, **values) -> None:
        with self.lock:
            state = self.jobs[simulation_id]
            for key, value in values.items():
                setattr(state, key, value)
            self._persist_state(state)

    def _persist_state(self, state: JobState) -> None:
        self._write_json_atomic(self.output_root / state.simulation_id / "job.json", asdict(state))

    @staticmethod
    def _write_json_atomic(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _load_existing(self) -> None:
        for path in self.output_root.glob("*/job.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                state = JobState(**value)
                if state.status not in TERMINAL_STATUSES:
                    state.status = "failed"
                    state.errors.append({"message": "Server restarted before this job completed"})
                    self._persist_state(state)
                self.jobs[state.simulation_id] = state
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue

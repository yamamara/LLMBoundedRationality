from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sandbox.api.models import CournotSimulationRequest
from sandbox.agents.web_human import HUMAN_DECISION_BROKER
from sandbox.configuration import ProviderProfile, load_provider_profiles
from sandbox.power import prevent_system_sleep


COURNOT_TERMINAL_STATUSES = {"completed", "failed"}


@dataclass
class CournotJobState:
    simulation_id: str
    status: str
    created_at: str
    total_rounds: int
    completed_rounds: int = 0
    error: str | None = None

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["experiment_type"] = "cournot"
        value["progress"] = (
            self.completed_rounds / self.total_rounds if self.total_rounds else 0.0
        )
        return value


class CournotJobManager:
    """Runs frontend Cournot jobs without coupling them to auction batches."""

    def __init__(
        self,
        output_root: Path | None = None,
        max_jobs: int = 2,
        profiles: dict[str, ProviderProfile] | None = None,
    ):
        self.output_root = output_root or Path(
            os.environ.get("COURNOT_WEB_OUTPUT_DIR", "runs/web/cournot")
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(
            max_workers=max_jobs, thread_name_prefix="cournot-job"
        )
        self.lock = threading.RLock()
        self.profiles = profiles or load_provider_profiles()
        self.jobs: dict[str, CournotJobState] = {}
        self.results: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        with self.lock:
            simulation_ids = list(self.jobs)
        for simulation_id in simulation_ids:
            HUMAN_DECISION_BROKER.finish(simulation_id)
        self.executor.shutdown(wait=False, cancel_futures=True)

    def submit(self, request: CournotSimulationRequest) -> CournotJobState:
        simulation_id = uuid.uuid4().hex
        state = CournotJobState(
            simulation_id=simulation_id,
            status="queued",
            created_at=datetime.now(UTC).isoformat(),
            total_rounds=request.cournot.rounds,
        )
        with self.lock:
            self.jobs[simulation_id] = state
        self.executor.submit(self._run, simulation_id, request)
        return state

    def get(self, simulation_id: str) -> CournotJobState | None:
        with self.lock:
            return self.jobs.get(simulation_id)

    def get_results(self, simulation_id: str) -> dict[str, Any] | None:
        with self.lock:
            return self.results.get(simulation_id)

    def artifact_path(self, simulation_id: str, filename: str) -> Path | None:
        if filename not in {
            "cournot_player_scores.svg",
            "cournot_player_scores_ci95.svg",
            "cournot_playback.html",
        }:
            return None
        state = self.get(simulation_id)
        if state is None or state.status != "completed":
            return None
        path = self.output_root / simulation_id / "analysis" / filename
        return path if path.is_file() else None

    def pending_human_decision(self, simulation_id: str) -> dict[str, Any] | None:
        state = self.get(simulation_id)
        if state is None:
            raise ValueError("Cournot simulation not found")
        if state.status not in {"queued", "running"}:
            return None
        return HUMAN_DECISION_BROKER.pending(simulation_id)

    def submit_human_decision(
        self, simulation_id: str, request_id: str, quantity: float
    ) -> None:
        state = self.get(simulation_id)
        if state is None or state.status not in {"queued", "running"}:
            raise ValueError("Cournot simulation is not waiting for a human decision")
        HUMAN_DECISION_BROKER.submit(simulation_id, request_id, quantity)

    def _run(self, simulation_id: str, request: CournotSimulationRequest) -> None:
        self._update(simulation_id, status="running")
        with prevent_system_sleep():
            self._run_awake(simulation_id, request)

    def _run_awake(
        self, simulation_id: str, request: CournotSimulationRequest
    ) -> None:
        try:
            from simulation import run_pipeline

            config = self._pipeline_config(simulation_id, request)

            def report_progress(completed_rounds: int) -> None:
                self._update(
                    simulation_id,
                    completed_rounds=min(completed_rounds, request.cournot.rounds),
                )

            run_dir, analysis_dir = run_pipeline(
                config,
                analyze=True,
                progress_callback=report_progress,
            )
            if analysis_dir is None:
                raise RuntimeError("Cournot analysis did not produce an output directory")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            result = {
                "simulation_id": simulation_id,
                "experiment_type": "cournot",
                "run_id": summary["run_id"],
                "game_name": summary["game_name"],
                "treatment": summary["cournot"]["treatment"],
                "player_count": len(request.agents),
                "graph_url": f"/api/v1/cournot-simulations/{simulation_id}/graph",
                "graph_two_sd_url": f"/api/v1/cournot-simulations/{simulation_id}/graph",
                "graph_ci95_url": f"/api/v1/cournot-simulations/{simulation_id}/graph-ci95",
                "playback_url": f"/api/v1/cournot-simulations/{simulation_id}/playback",
            }
            with self.lock:
                self.results[simulation_id] = result
            self._update(
                simulation_id,
                status="completed",
                completed_rounds=request.cournot.rounds,
            )
        except Exception as exc:
            self._update(
                simulation_id,
                status="failed",
                error=f"{type(exc).__name__}: {str(exc)[:1000]}",
            )
        finally:
            HUMAN_DECISION_BROKER.finish(simulation_id)

    def _update(self, simulation_id: str, **values: Any) -> None:
        with self.lock:
            state = self.jobs[simulation_id]
            for name, value in values.items():
                setattr(state, name, value)

    def _pipeline_config(
        self, simulation_id: str, request: CournotSimulationRequest
    ) -> dict[str, Any]:
        cournot = request.cournot
        participants = []
        for agent_index, agent in enumerate(request.agents):
            participant = {
                "player_id": agent.player_id,
                "agent_type": "human" if agent.policy == "web_human" else "AI",
                "role": "producer",
                "policy": agent.policy,
            }
            if agent.policy in {"cournot_best_reply", "cournot_previous_average"}:
                participant["initial_quantity"] = agent.initial_quantity
            elif agent.policy == "cournot_random_quantity":
                participant["random_seed"] = cournot.seed + agent_index
            elif agent.policy == "cournot_llm":
                if agent.profile_id not in self.profiles:
                    raise ValueError(
                        f"Unknown provider profile: {agent.profile_id}"
                    )
                profile = self.profiles[agent.profile_id]
                model = agent.model or profile.model
                if not model:
                    raise ValueError(
                        f"No model configured for provider profile {profile.profile_id}"
                    )
                if not profile.configured:
                    raise ValueError(
                        f"Provider profile {profile.profile_id} is not configured"
                    )
                options = {**profile.defaults, **agent.provider_options}
                for common_key in ("temperature", "top_p"):
                    options.pop(common_key, None)
                participant.update(
                    {
                        "provider": profile.provider,
                        "endpoint": profile.endpoint,
                        "api_key_env": profile.api_key_env,
                        "transport": profile.transport,
                        "model": model,
                        "temperature": agent.temperature,
                        "max_output_tokens": agent.max_tokens,
                        "timeout_seconds": agent.timeout_seconds,
                        "memory_rounds": agent.memory_rounds,
                        "max_retries": agent.max_retries,
                        "reasoning_effort": agent.reasoning_effort,
                        "provider_options": options,
                        "system_prompt_template": (
                            agent.system_prompt_template
                            or request.prompts.system_template
                        ),
                        "agent_prompt_template": (
                            agent.agent_prompt_template
                            or request.prompts.agent_template
                        ),
                    }
                )
            elif agent.policy == "cournot_openai_compatible":
                participant.update(
                    {
                        "base_url": agent.base_url,
                        "model": agent.model,
                        "api_key_env": agent.api_key_env,
                        "temperature": agent.temperature,
                        "max_tokens": agent.max_tokens,
                        "timeout_seconds": agent.timeout_seconds,
                        "memory_rounds": agent.memory_rounds,
                        "max_retries": agent.max_retries,
                        "system_prompt_template": (
                            agent.system_prompt_template
                            or request.prompts.system_template
                        ),
                        "agent_prompt_template": (
                            agent.agent_prompt_template
                            or request.prompts.agent_template
                        ),
                    }
                )
            else:
                participant.update(
                    {"job_id": simulation_id, "timeout_seconds": 3600.0}
                )
            participants.append(participant)

        return {
            "game_name": request.name,
            "run_id": simulation_id,
            "output_dir": str(self.output_root),
            "cournot": {
                "rounds": cournot.rounds,
                "quantity_min": 0.0,
                "quantity_max": 100.0,
                "quantity_step": 0.01,
                "demand_intercept": 100.0,
                "marginal_cost": 1.0,
                "revision_probability": cournot.revision_probability,
                "fixed_payment": 150.0,
                "treatment": cournot.treatment,
                "seed": cournot.seed,
                "institution": "frontend",
            },
            "participants": participants,
            "prompts": {
                "system_template": request.prompts.system_template,
                "agent_template": request.prompts.agent_template,
            },
        }

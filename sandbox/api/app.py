from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from sandbox.api.jobs import TERMINAL_STATUSES, SimulationJobManager
from sandbox.api.models import PromptPreviewSpec, SimulationRequest
from sandbox.games.auction.mechanisms import MECHANISM_NAMES, mechanism_description
from sandbox.models import LegalActions, Observation
from sandbox.prompts import (
    DEFAULT_AGENT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    PLACEHOLDER_DESCRIPTIONS,
    render_prompts,
)


def create_api_app(manager: SimulationJobManager | None = None) -> FastAPI:
    job_manager = manager or SimulationJobManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        job_manager.close()

    app = FastAPI(title="LLM Bounded Rationality Auction API", version="1.0.0", lifespan=lifespan)
    app.state.job_manager = job_manager

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/v1/providers")
    def providers():
        return {"profiles": [profile.public_dict() for profile in job_manager.profiles.values()]}

    @app.get("/api/v1/prompts")
    def prompt_configuration():
        return {
            "system_template": DEFAULT_SYSTEM_PROMPT,
            "agent_template": DEFAULT_AGENT_PROMPT,
            "placeholders": PLACEHOLDER_DESCRIPTIONS,
        }

    @app.post("/api/v1/prompts/preview")
    def prompt_preview(spec: PromptPreviewSpec):
        observation = Observation(
            player_id=spec.player_id,
            round=spec.round_number - 1,
            public_state={
                "mechanism": spec.mechanism,
                "mechanism_name": MECHANISM_NAMES[spec.mechanism],
                "mechanism_description": mechanism_description(
                    spec.mechanism, spec.mechanism_description_treatment
                ),
                "mechanism_description_treatment": spec.mechanism_description_treatment,
                "num_players": spec.num_players,
                "num_opponents": spec.num_players - 1,
                "rounds_total": spec.total_rounds,
                "completed_rounds": spec.public_history,
            },
            private_state={
                "true_value": spec.private_value,
                "remaining_budget": spec.remaining_budget,
            },
            legal_actions=LegalActions(
                "submit_bid",
                {"bid_min": spec.minimum_bid, "bid_max": spec.maximum_bid},
            ),
        )
        rendered = render_prompts(
            spec.prompts.system_template, spec.prompts.agent_template, observation
        )
        return {"system_prompt": rendered.system, "agent_prompt": rendered.agent}

    @app.post("/api/v1/simulations", status_code=status.HTTP_202_ACCEPTED)
    def create_simulation(request: SimulationRequest):
        try:
            state = job_manager.submit(request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return state.public_dict()

    @app.get("/api/v1/simulations/{simulation_id}")
    def simulation_status(simulation_id: str):
        state = job_manager.get(simulation_id)
        if state is None:
            raise HTTPException(404, "Simulation not found")
        return state.public_dict()

    @app.get("/api/v1/simulations/{simulation_id}/results")
    def simulation_results(simulation_id: str):
        state = job_manager.get(simulation_id)
        if state is None:
            raise HTTPException(404, "Simulation not found")
        if state.status not in TERMINAL_STATUSES:
            raise HTTPException(409, "Simulation is not complete")
        results = job_manager.get_results(simulation_id)
        if results is None:
            raise HTTPException(404, "Simulation results were not found")
        return results

    return app

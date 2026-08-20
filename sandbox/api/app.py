from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse

from sandbox.api.cournot_jobs import COURNOT_TERMINAL_STATUSES, CournotJobManager
from sandbox.api.jobs import TERMINAL_STATUSES, SimulationJobManager
from sandbox.api.models import (
    CournotSimulationRequest,
    HumanQuantitySubmission,
    PromptPreviewSpec,
    SimulationRequest,
)
from sandbox.games.auction.mechanisms import MECHANISM_NAMES, mechanism_description
from sandbox.models import LegalActions, Observation
from sandbox.prompts import (
    DEFAULT_AGENT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    PLACEHOLDER_DESCRIPTIONS,
    render_prompts,
)


def create_api_app(
    manager: SimulationJobManager | None = None,
    cournot_manager: CournotJobManager | None = None,
) -> FastAPI:
    job_manager = manager or SimulationJobManager()
    cournot_jobs = cournot_manager or CournotJobManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        job_manager.close()
        cournot_jobs.close()

    app = FastAPI(title="LLM Bounded Rationality API", version="1.0.0", lifespan=lifespan)
    app.state.job_manager = job_manager
    app.state.cournot_job_manager = cournot_jobs

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/v1/providers")
    def providers():
        return {"profiles": [profile.public_dict() for profile in job_manager.profiles.values()]}

    @app.post("/api/v1/cournot-simulations", status_code=status.HTTP_202_ACCEPTED)
    def create_cournot_simulation(request: CournotSimulationRequest):
        return cournot_jobs.submit(request).public_dict()

    @app.get("/api/v1/cournot-simulations/{simulation_id}")
    def cournot_simulation_status(simulation_id: str):
        state = cournot_jobs.get(simulation_id)
        if state is None:
            raise HTTPException(404, "Cournot simulation not found")
        return state.public_dict()

    @app.get("/api/v1/cournot-simulations/{simulation_id}/results")
    def cournot_simulation_results(simulation_id: str):
        state = cournot_jobs.get(simulation_id)
        if state is None:
            raise HTTPException(404, "Cournot simulation not found")
        if state.status not in COURNOT_TERMINAL_STATUSES:
            raise HTTPException(409, "Cournot simulation is not complete")
        result = cournot_jobs.get_results(simulation_id)
        if result is None:
            raise HTTPException(404, state.error or "Cournot results not found")
        return result

    @app.get("/api/v1/cournot-simulations/{simulation_id}/human-decision")
    def pending_cournot_human_decision(simulation_id: str):
        try:
            pending = cournot_jobs.pending_human_decision(simulation_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"pending": pending}

    @app.post("/api/v1/cournot-simulations/{simulation_id}/human-decision")
    def submit_cournot_human_decision(
        simulation_id: str, submission: HumanQuantitySubmission
    ):
        try:
            cournot_jobs.submit_human_decision(
                simulation_id, submission.request_id, submission.quantity
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"accepted": True}

    @app.get("/api/v1/cournot-simulations/{simulation_id}/graph")
    def cournot_graph(simulation_id: str):
        path = cournot_jobs.artifact_path(simulation_id, "cournot_player_scores.svg")
        if path is None:
            raise HTTPException(404, "Cournot graph not found")
        return FileResponse(path, media_type="image/svg+xml")

    @app.get("/api/v1/cournot-simulations/{simulation_id}/graph-ci95")
    def cournot_graph_ci95(simulation_id: str):
        path = cournot_jobs.artifact_path(
            simulation_id, "cournot_player_scores_ci95.svg"
        )
        if path is None:
            raise HTTPException(404, "Cournot confidence interval graph not found")
        return FileResponse(path, media_type="image/svg+xml")

    @app.get("/api/v1/cournot-simulations/{simulation_id}/provenance")
    def cournot_provenance(simulation_id: str):
        path = cournot_jobs.artifact_path(simulation_id, "cournot_parameters.json")
        if path is None:
            raise HTTPException(404, "Cournot parameter provenance not found")
        return FileResponse(path, media_type="application/json")

    @app.get("/api/v1/cournot-simulations/{simulation_id}/playback")
    def cournot_playback(simulation_id: str):
        path = cournot_jobs.artifact_path(simulation_id, "cournot_playback.html")
        if path is None:
            raise HTTPException(404, "Cournot playback not found")
        return FileResponse(path, media_type="text/html")

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

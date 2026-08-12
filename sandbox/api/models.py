from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sandbox.prompts import (
    DEFAULT_AGENT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    validate_prompt_templates,
)
from sandbox.cournot_prompts import (
    DEFAULT_COURNOT_AGENT_PROMPT,
    DEFAULT_COURNOT_SYSTEM_PROMPT,
    validate_cournot_prompt_templates,
)

PlayerCount = Literal[2, 4, 6, 8]
Mechanism = Literal["first_price", "second_price"]
DescriptionTreatment = Literal["name_only", "concise", "full"]
CournotTreatment = Literal["BEST", "FULL"]
CournotPolicy = Literal[
    "cournot_best_reply", "cournot_llm", "cournot_openai_compatible", "web_human"
]


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: str = Field(min_length=1, max_length=64)
    profile_id: str = Field(min_length=1, max_length=64)
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=800, ge=32, le=32768)
    timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    memory_rounds: int | None = Field(default=3, ge=0, le=1000)
    max_retries: int = Field(default=2, ge=1, le=10)
    reasoning_effort: str | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)


class PromptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_template: str = Field(default=DEFAULT_SYSTEM_PROMPT, max_length=50_000)
    agent_template: str = Field(default=DEFAULT_AGENT_PROMPT, max_length=50_000)

    @model_validator(mode="after")
    def validate_templates(self):
        validate_prompt_templates(self.system_template, self.agent_template)
        return self


class ValueRangeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum: int = Field(ge=0, le=1_000_000_000)
    maximum: int = Field(ge=0, le=1_000_000_000)

    @model_validator(mode="after")
    def validate_range(self):
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class ValuationScheduleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    player_ranges: dict[str, ValueRangeSpec] = Field(default_factory=dict)
    rotate_across_positions: bool = False


class AuctionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rounds: int = Field(default=10, ge=1, le=500)
    starting_budget: int = Field(default=100, ge=0, le=1_000_000_000)
    true_value_min: int = Field(default=1, ge=0, le=1_000_000_000)
    true_value_max: int = Field(default=100, ge=0, le=1_000_000_000)
    mechanism: Mechanism = "first_price"
    num_players: PlayerCount = 4
    mechanism_description_treatment: DescriptionTreatment = "concise"
    player_value_ranges: dict[str, ValueRangeSpec] = Field(default_factory=dict)
    valuation_schedule: str = Field(default="baseline", min_length=1, max_length=100)
    institution: str = Field(default="baseline", min_length=1, max_length=100)
    seed: int = 7
    tie_breaking: Literal["seeded_random", "player_priority"] = "seeded_random"
    tie_break_seed: int | None = None
    tie_break_priority: list[str] | None = None

    @model_validator(mode="after")
    def validate_range(self):
        if self.true_value_min > self.true_value_max:
            raise ValueError("true_value_min must not exceed true_value_max")
        return self


class ExperimentMatrixSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mechanisms: list[Mechanism] = Field(default_factory=list, max_length=2)
    mechanism_description_treatments: list[DescriptionTreatment] = Field(
        default_factory=list, max_length=3
    )
    player_counts: list[PlayerCount] = Field(default_factory=list, max_length=4)
    valuation_schedules: list[ValuationScheduleSpec] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def unique_dimensions(self):
        for name in ("mechanisms", "mechanism_description_treatments", "player_counts"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        names = [schedule.name for schedule in self.valuation_schedules]
        if len(names) != len(set(names)):
            raise ValueError("valuation schedule names must be unique")
        return self


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="sealed-bid-auction", min_length=1, max_length=100)
    n: int = Field(default=1, ge=1, le=100)
    max_parallel_runs: int = Field(default=1, ge=1, le=8)
    auction: AuctionSpec = Field(default_factory=AuctionSpec)
    agents: list[AgentSpec]
    prompts: PromptSpec = Field(default_factory=PromptSpec)
    experiment_matrix: ExperimentMatrixSpec | None = None

    @model_validator(mode="after")
    def validate_simulation(self):
        player_counts = (
            self.experiment_matrix.player_counts
            if self.experiment_matrix and self.experiment_matrix.player_counts
            else [self.auction.num_players]
        )
        required_agents = max(player_counts)
        if len(self.agents) != required_agents:
            quantity = "four" if required_agents == 4 else str(required_agents)
            raise ValueError(f"Exactly {quantity} agents are required")
        player_ids = [agent.player_id for agent in self.agents]
        if len(set(player_ids)) != len(player_ids):
            raise ValueError("Agent player_id values must be unique")
        known_players = set(player_ids)
        unknown_ranges = set(self.auction.player_value_ranges) - known_players
        if unknown_ranges:
            raise ValueError(f"Unknown players in valuation ranges: {sorted(unknown_ranges)}")
        if self.experiment_matrix:
            for schedule in self.experiment_matrix.valuation_schedules:
                unknown = set(schedule.player_ranges) - known_players
                if unknown:
                    raise ValueError(
                        f"Unknown players in valuation schedule {schedule.name}: {sorted(unknown)}"
                    )
        if self.auction.tie_break_priority is not None:
            if self.auction.tie_breaking != "player_priority":
                raise ValueError("tie_break_priority is only valid for player_priority ties")
            if set(self.auction.tie_break_priority) != known_players or len(
                self.auction.tie_break_priority
            ) != len(player_ids):
                raise ValueError("tie_break_priority must contain every configured player exactly once")
        cells = matrix_cell_count(self)
        total_runs = self.n * cells
        if total_runs > 2_000:
            raise ValueError("A job may contain at most 2,000 total runs")
        if total_runs * self.auction.rounds > 10_000:
            raise ValueError("A job may contain at most 10,000 total rounds")
        if total_runs * self.auction.rounds * required_agents > 50_000:
            raise ValueError("A job may contain at most 50,000 participant decisions")
        return self


class CournotAgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: str = Field(min_length=1, max_length=64)
    policy: CournotPolicy = "cournot_best_reply"
    initial_quantity: float = Field(default=20.0, ge=0.0, le=100.0)
    profile_id: str | None = Field(default=None, min_length=1, max_length=64)
    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=800, ge=32, le=32768)
    timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    memory_rounds: int | None = Field(default=1, ge=0, le=1000)
    max_retries: int = Field(default=2, ge=1, le=10)
    reasoning_effort: str | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
    system_prompt_template: str | None = Field(default=None, max_length=50_000)
    agent_prompt_template: str | None = Field(default=None, max_length=50_000)

    @field_validator("system_prompt_template", "agent_prompt_template", mode="before")
    @classmethod
    def normalize_prompt_override(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_policy_configuration(self):
        if self.policy == "cournot_openai_compatible" and not (
            self.base_url and self.model
        ):
            raise ValueError("OpenAI-compatible Cournot agents require base_url and model")
        if self.policy == "cournot_llm" and not self.profile_id:
            raise ValueError("Provider-backed Cournot agents require profile_id")
        if self.system_prompt_template is not None or self.agent_prompt_template is not None:
            validate_cournot_prompt_templates(
                self.system_prompt_template or DEFAULT_COURNOT_SYSTEM_PROMPT,
                self.agent_prompt_template or DEFAULT_COURNOT_AGENT_PROMPT,
            )
        return self


class CournotPromptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_template: str = Field(
        default=DEFAULT_COURNOT_SYSTEM_PROMPT, max_length=50_000
    )
    agent_template: str = Field(
        default=DEFAULT_COURNOT_AGENT_PROMPT, max_length=50_000
    )

    @model_validator(mode="after")
    def validate_templates(self):
        validate_cournot_prompt_templates(
            self.system_template, self.agent_template
        )
        return self


class CournotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rounds: int = Field(default=40, ge=1, le=500)
    treatment: CournotTreatment = "BEST"
    seed: int = 7
    revision_probability: float = Field(default=2 / 3, ge=0.0, le=1.0)


class CournotSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="huck-cournot", min_length=1, max_length=100)
    cournot: CournotSpec = Field(default_factory=CournotSpec)
    agents: list[CournotAgentSpec] = Field(min_length=2, max_length=8)
    prompts: CournotPromptSpec = Field(default_factory=CournotPromptSpec)

    @model_validator(mode="after")
    def validate_agents(self):
        player_ids = [agent.player_id for agent in self.agents]
        if len(set(player_ids)) != len(player_ids):
            raise ValueError("Cournot player_id values must be unique")
        return self


class HumanQuantitySubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=64)
    quantity: float


class PromptPreviewSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompts: PromptSpec = Field(default_factory=PromptSpec)
    mechanism: Mechanism = "first_price"
    mechanism_description_treatment: DescriptionTreatment = "concise"
    num_players: PlayerCount = 4
    total_rounds: int = Field(default=10, ge=1, le=500)
    player_id: str = "P1"
    round_number: int = Field(default=1, ge=1)
    private_value: int = Field(default=50, ge=0)
    remaining_budget: int = Field(default=100, ge=0)
    minimum_bid: int = Field(default=0, ge=0)
    maximum_bid: int = Field(default=100, ge=0)
    public_history: list[dict[str, Any]] = Field(default_factory=list)


def matrix_cell_count(request: SimulationRequest) -> int:
    matrix = request.experiment_matrix
    if matrix is None:
        return 1
    mechanisms = len(matrix.mechanisms) or 1
    descriptions = len(matrix.mechanism_description_treatments) or 1
    players = matrix.player_counts or [request.auction.num_players]
    schedules = matrix.valuation_schedules
    per_player_counts = 0
    for count in players:
        per_player_counts += (
            sum(count if schedule.rotate_across_positions else 1 for schedule in schedules)
            if schedules
            else 1
        )
    return mechanisms * descriptions * per_player_counts

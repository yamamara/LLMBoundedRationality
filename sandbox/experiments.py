from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product

from sandbox.api.models import AuctionSpec, SimulationRequest, ValuationScheduleSpec


@dataclass(frozen=True)
class TreatmentCell:
    treatment_id: str
    mechanism: str
    mechanism_description_treatment: str
    num_players: int
    valuation_schedule: str
    player_value_ranges: dict[str, dict[str, int]]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "value"


def _schedule_variants(
    schedule: ValuationScheduleSpec,
    player_ids: list[str],
    auction: AuctionSpec,
) -> list[tuple[str, dict[str, dict[str, int]]]]:
    explicit = {
        player_id: value.model_dump()
        for player_id, value in schedule.player_ranges.items()
        if player_id in player_ids
    }
    if not schedule.rotate_across_positions:
        return [(schedule.name, explicit)]
    default_range = {"minimum": auction.true_value_min, "maximum": auction.true_value_max}
    source = [explicit.get(player_id, default_range) for player_id in player_ids]
    variants = []
    for offset in range(len(player_ids)):
        rotated = {
            player_id: dict(source[(index - offset) % len(source)])
            for index, player_id in enumerate(player_ids)
        }
        variants.append((f"{schedule.name}-rotation-{offset + 1}", rotated))
    return variants


def expand_treatment_cells(request: SimulationRequest) -> list[TreatmentCell]:
    matrix = request.experiment_matrix
    mechanisms = matrix.mechanisms if matrix and matrix.mechanisms else [request.auction.mechanism]
    descriptions = (
        matrix.mechanism_description_treatments
        if matrix and matrix.mechanism_description_treatments
        else [request.auction.mechanism_description_treatment]
    )
    player_counts = (
        matrix.player_counts if matrix and matrix.player_counts else [request.auction.num_players]
    )
    cells: list[TreatmentCell] = []
    for mechanism, description, count in product(mechanisms, descriptions, player_counts):
        active_ids = [agent.player_id for agent in request.agents[:count]]
        if matrix and matrix.valuation_schedules:
            variants = [
                variant
                for schedule in matrix.valuation_schedules
                for variant in _schedule_variants(schedule, active_ids, request.auction)
            ]
        else:
            variants = [
                (
                    request.auction.valuation_schedule,
                    {
                        player_id: value.model_dump()
                        for player_id, value in request.auction.player_value_ranges.items()
                        if player_id in active_ids
                    },
                )
            ]
        for schedule_name, ranges in variants:
            treatment_id = "__".join(
                (
                    _slug(mechanism),
                    _slug(description),
                    f"players-{count}",
                    _slug(schedule_name),
                )
            )
            cells.append(
                TreatmentCell(
                    treatment_id,
                    mechanism,
                    description,
                    count,
                    schedule_name,
                    ranges,
                )
            )
    return cells


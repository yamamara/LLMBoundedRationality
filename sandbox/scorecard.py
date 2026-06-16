from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


@dataclass(frozen=True)
class OutcomeSpec:
    name: str
    unit: str
    higher_is_better: bool


@dataclass(frozen=True)
class Outcome:
    run_id: str
    measure: str
    unit: str
    higher_is_better: bool
    institution: str
    population_composition: str
    agent_type: str | None
    value: float


@dataclass(frozen=True)
class Estimate:
    available: bool
    first_count: int | None = None
    second_count: int | None = None
    first_mean: float | None = None
    second_mean: float | None = None
    raw_delta: float | None = None
    favorable_delta: float | None = None
    standardized_delta: float | None = None
    standard_error: float | None = None
    confidence_interval_95: tuple[float, float] | None = None
    reference_standard_deviation: float | None = None
    comparison: str | None = None


AUCTION_OUTCOMES = [
    OutcomeSpec("payoff", "utility_points", True),
    OutcomeSpec("regret", "utility_points", False),
    OutcomeSpec("invalid_action", "share", False),
    OutcomeSpec("retry_count", "count", False),
    OutcomeSpec("latency_seconds", "seconds", False),
]

AUCTION_SYSTEM_OUTCOMES = [
    OutcomeSpec("allocative_efficiency", "share", True),
]


def auction_outcomes(run_dirs: Iterable[Path]) -> list[Outcome]:
    outcomes = []
    for run_dir in run_dirs:
        with (run_dir / "decisions.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for spec in AUCTION_OUTCOMES:
                    value = 0.0 if spec.name == "invalid_action" and row["valid_action"] == "True" else None
                    if spec.name == "invalid_action" and value is None:
                        value = 1.0
                    elif spec.name != "invalid_action":
                        value = float(row[spec.name])
                    outcomes.append(
                        Outcome(
                            run_id=row["run_id"],
                            measure=spec.name,
                            unit=spec.unit,
                            higher_is_better=spec.higher_is_better,
                            institution=row["institution"],
                            population_composition=row["population_composition"],
                            agent_type=row["agent_type"],
                            value=value,
                        )
                    )
        with (run_dir / "rounds.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for spec in AUCTION_SYSTEM_OUTCOMES:
                    outcomes.append(
                        Outcome(
                            run_id=row["run_id"],
                            measure=spec.name,
                            unit=spec.unit,
                            higher_is_better=spec.higher_is_better,
                            institution=row["institution"],
                            population_composition=row["population_composition"],
                            agent_type=None,
                            value=float(row[spec.name]),
                        )
                    )
    return outcomes


def build_scorecard(outcomes: list[Outcome], include_provisional_scalar: bool = False) -> dict[str, Any]:
    grouped: dict[str, list[Outcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.measure].append(outcome)
    scorecard = {"metric": "Institutional Redesign Pressure scorecard", "measures": {}}
    for measure, rows in sorted(grouped.items()):
        components = _measure_components(rows)
        result: dict[str, Any] = {
            "unit": rows[0].unit,
            "higher_is_better": rows[0].higher_is_better,
            "components": {name: asdict(estimate) for name, estimate in components.items()},
        }
        if include_provisional_scalar:
            edge = components["agent_edge"].standardized_delta
            loss = components["human_compatibility_loss"].standardized_delta
            result["provisional_irp_product"] = edge * loss if edge is not None and loss is not None else None
        scorecard["measures"][measure] = result
    return scorecard


def write_scorecard(scorecard: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "irp_scorecard.json").open("w", encoding="utf-8") as handle:
        json.dump(scorecard, handle, indent=2, sort_keys=True)
        handle.write("\n")
    rows = []
    for measure, details in scorecard["measures"].items():
        for component, estimate in details["components"].items():
            rows.append({"measure": measure, "unit": details["unit"], "component": component, **estimate})
    if rows:
        with (output_dir / "irp_scorecard.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _measure_components(rows: list[Outcome]) -> dict[str, Estimate]:
    if rows[0].agent_type is None:
        return _system_components(rows)
    return {
        "agent_edge": _estimate(
            rows,
            ("baseline", "AI_only", "AI"),
            ("baseline", "human_only", "human"),
            "AI-only baseline minus human-only baseline",
        ),
        "mixed_entry_stress": _estimate(
            rows,
            ("baseline", "human_only", "human"),
            ("baseline", "mixed", "human"),
            "human-only baseline minus remaining humans in mixed baseline",
        ),
        "redesign_effectiveness": _redesign_effectiveness(rows),
        "human_compatibility_loss": _estimate(
            rows,
            ("baseline", "human_only", "human"),
            ("redesign", "human_only", "human"),
            "human-only baseline minus human-only redesign",
        ),
    }


def _system_components(rows: list[Outcome]) -> dict[str, Estimate]:
    return {
        "agent_edge": _estimate(
            rows,
            ("baseline", "AI_only", None),
            ("baseline", "human_only", None),
            "AI-only baseline minus human-only baseline",
        ),
        "mixed_entry_stress": _estimate(
            rows,
            ("baseline", "human_only", None),
            ("baseline", "mixed", None),
            "human-only baseline minus mixed baseline",
        ),
        "redesign_effectiveness": _redesign_effectiveness(rows, agent_type=None),
        "human_compatibility_loss": _estimate(
            rows,
            ("baseline", "human_only", None),
            ("redesign", "human_only", None),
            "human-only baseline minus human-only redesign",
        ),
    }


def _redesign_effectiveness(
    rows: list[Outcome],
    agent_type: str | None = "human",
) -> Estimate:
    baseline = _estimate(
        rows,
        ("baseline", "human_only", agent_type),
        ("baseline", "mixed", agent_type),
        "baseline entry stress",
    )
    redesign = _estimate(
        rows,
        ("redesign", "human_only", agent_type),
        ("redesign", "mixed", agent_type),
        "redesign entry stress",
    )
    if baseline.favorable_delta is None or redesign.favorable_delta is None:
        return Estimate(False, comparison="baseline entry stress minus redesign entry stress")
    raw = baseline.favorable_delta - redesign.favorable_delta
    reference_sd = baseline.reference_standard_deviation
    return Estimate(
        available=True,
        raw_delta=raw,
        favorable_delta=raw,
        standardized_delta=raw / reference_sd if reference_sd else None,
        reference_standard_deviation=reference_sd,
        comparison="baseline entry stress minus redesign entry stress",
    )


def _estimate(
    rows: list[Outcome],
    first: tuple[str, str, str | None],
    second: tuple[str, str, str | None],
    comparison: str,
) -> Estimate:
    first_values = _select(rows, first)
    second_values = _select(rows, second)
    reference_agent_type = None if rows[0].agent_type is None else "human"
    reference_values = _select(rows, ("baseline", "human_only", reference_agent_type))
    if not first_values or not second_values:
        return Estimate(False, comparison=comparison)
    raw_delta = mean(first_values) - mean(second_values)
    favorable_delta = raw_delta if rows[0].higher_is_better else -raw_delta
    reference_sd = stdev(reference_values) if len(reference_values) > 1 else 0.0
    standard_error = math.sqrt(_variance(first_values) / len(first_values) + _variance(second_values) / len(second_values))
    return Estimate(
        available=True,
        first_count=len(first_values),
        second_count=len(second_values),
        first_mean=mean(first_values),
        second_mean=mean(second_values),
        raw_delta=raw_delta,
        favorable_delta=favorable_delta,
        standardized_delta=favorable_delta / reference_sd if reference_sd else None,
        standard_error=standard_error,
        confidence_interval_95=(
            favorable_delta - 1.96 * standard_error,
            favorable_delta + 1.96 * standard_error,
        ),
        reference_standard_deviation=reference_sd,
        comparison=comparison,
    )


def _select(
    rows: list[Outcome],
    selector: tuple[str, str, str | None],
) -> list[float]:
    institution, composition, agent_type = selector
    by_run: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.institution == institution
            and row.population_composition == composition
            and row.agent_type == agent_type
        ):
            by_run[row.run_id].append(row.value)
    return [mean(values) for values in by_run.values()]


def _variance(values: list[float]) -> float:
    return stdev(values) ** 2 if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an IRP scorecard from auction run bundles.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"))
    parser.add_argument("--include-provisional-scalar", action="store_true")
    args = parser.parse_args()
    scorecard = build_scorecard(auction_outcomes(args.runs), args.include_provisional_scalar)
    write_scorecard(scorecard, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()

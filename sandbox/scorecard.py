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

from sandbox.playback import write_cournot_playback
from sandbox.visualization import write_cournot_player_visualization


@dataclass(frozen=True)
class OutcomeSpec:
    name: str
    unit: str


@dataclass(frozen=True)
class Outcome:
    run_id: str
    measure: str
    unit: str
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
    percent_reference_value: float | None = None
    percent_reference_available: bool = False
    relative_delta: float | None = None
    relative_percent: float | None = None
    comparison: str | None = None


AUCTION_OUTCOMES = [
    OutcomeSpec("payoff", "utility_points"),
    OutcomeSpec("regret", "utility_points"),
    OutcomeSpec("invalid_action", "share"),
    OutcomeSpec("retry_count", "count"),
    OutcomeSpec("latency_seconds", "seconds"),
]

AUCTION_SYSTEM_OUTCOMES = [
    OutcomeSpec("allocative_efficiency", "share"),
]

COURNOT_OUTCOMES = [
    OutcomeSpec("profit", "utility_points"),
    OutcomeSpec("invalid_action", "share"),
    OutcomeSpec("retry_count", "count"),
    OutcomeSpec("latency_seconds", "seconds"),
]

COURNOT_SYSTEM_OUTCOMES = [
    OutcomeSpec("total_quantity", "units"),
    OutcomeSpec("total_market_profit", "utility_points"),
    OutcomeSpec("distance_to_nash", "units"),
]


def auction_outcomes(run_dirs: Iterable[Path]) -> list[Outcome]:
    outcomes = []
    for run_dir in run_dirs:
        with (run_dir / "participant_data.csv").open(newline="", encoding="utf-8") as handle:
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
                            institution=row["institution"],
                            population_composition=row["population_composition"],
                            agent_type=row["agent_type"],
                            value=value,
                        )
                    )
        with (run_dir / "system_data.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for spec in AUCTION_SYSTEM_OUTCOMES:
                    outcomes.append(
                        Outcome(
                            run_id=row["run_id"],
                            measure=spec.name,
                            unit=spec.unit,
                            institution=row["institution"],
                            population_composition=row["population_composition"],
                            agent_type=None,
                            value=float(row[spec.name]),
                        )
                    )
    return outcomes


def cournot_outcomes(run_dirs: Iterable[Path]) -> list[Outcome]:
    outcomes = []
    for run_dir in run_dirs:
        with (run_dir / "participant_data.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for spec in COURNOT_OUTCOMES:
                    if spec.name == "invalid_action":
                        value = 0.0 if row["valid_action"] == "True" else 1.0
                    else:
                        value = float(row[spec.name])
                    outcomes.append(
                        Outcome(
                            run_id=row["run_id"],
                            measure=spec.name,
                            unit=spec.unit,
                            institution=row["institution"],
                            population_composition=row["population_composition"],
                            agent_type=row["agent_type"],
                            value=value,
                        )
                    )
        with (run_dir / "system_data.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for spec in COURNOT_SYSTEM_OUTCOMES:
                    outcomes.append(
                        Outcome(
                            run_id=row["run_id"],
                            measure=spec.name,
                            unit=spec.unit,
                            institution=row["institution"],
                            population_composition=row["population_composition"],
                            agent_type=None,
                            value=float(row[spec.name]),
                        )
                    )
    return outcomes


def experiment_outcomes(run_dirs: Iterable[Path]) -> list[Outcome]:
    run_dirs = list(run_dirs)
    game_types = set()
    for run_dir in run_dirs:
        with (run_dir / "summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        game_types.add(summary.get("game_type", "auction"))
    if len(game_types) != 1:
        raise ValueError("All scorecard runs must use the same game type")
    if game_types == {"cournot"}:
        return cournot_outcomes(run_dirs)
    return auction_outcomes(run_dirs)


def build_scorecard(outcomes: list[Outcome]) -> dict[str, Any]:
    grouped: dict[str, list[Outcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.measure].append(outcome)
    scorecard = {"metric": "AI Replacement Impact scorecard", "measures": {}}
    for measure, rows in sorted(grouped.items()):
        primary, diagnostics = _measure_scorecard(rows)
        result: dict[str, Any] = {
            "unit": rows[0].unit,
            "descriptive": _descriptive_summary(rows),
            "primary": {name: asdict(estimate) for name, estimate in primary.items()},
            "diagnostics": {name: asdict(estimate) for name, estimate in diagnostics.items()},
        }
        scorecard["measures"][measure] = result
    return scorecard


def write_scorecard(scorecard: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scorecard.json").open("w", encoding="utf-8") as handle:
        json.dump(scorecard, handle, indent=2, sort_keys=True)
        handle.write("\n")
    rows = []
    for measure, details in scorecard["measures"].items():
        rows.append(
            {
                "measure": measure,
                "unit": details["unit"],
                "section": "descriptive",
                "component": "overall",
                **details["descriptive"],
            }
        )
        for section in ("primary", "diagnostics"):
            for component, estimate in details[section].items():
                rows.append(
                    {
                        "measure": measure,
                        "unit": details["unit"],
                        "section": section,
                        "component": component,
                        **estimate,
                    }
                )
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (output_dir / "scorecard.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def analyze_runs(run_dirs: Iterable[Path], output_dir: Path) -> Path:
    run_dirs = list(run_dirs)
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    scorecard = build_scorecard(experiment_outcomes(run_dirs))
    write_scorecard(scorecard, output_dir)
    with (run_dirs[0] / "summary.json").open(encoding="utf-8") as handle:
        game_type = json.load(handle).get("game_type", "auction")
    if game_type == "cournot":
        write_cournot_player_visualization(run_dirs, output_dir)
        write_cournot_playback(run_dirs, output_dir)
    return output_dir


def _descriptive_summary(rows: list[Outcome]) -> dict[str, Any]:
    values = [row.value for row in rows]
    return {
        "observation_count": len(values),
        "run_count": len({row.run_id for row in rows}),
        "mean": mean(values),
        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _measure_scorecard(rows: list[Outcome]) -> tuple[dict[str, Estimate], dict[str, Estimate]]:
    if rows[0].agent_type is None:
        return _system_scorecard(rows)

    ai_introduction_effect = _estimate(
        rows,
        ("baseline", "mixed", "human"),
        ("baseline", "human_only", "human"),
        "humans in mixed AI baseline versus human baseline",
    )
    ai_resistant_redesign_effect = _estimate(
        rows,
        ("redesign", "human_only", "human"),
        ("baseline", "human_only", "human"),
        "humans under AI-resistant redesign versus human baseline",
    )
    primary = {
        "ai_introduction_effect": ai_introduction_effect,
        "ai_resistant_redesign_effect": ai_resistant_redesign_effect,
    }
    diagnostics = {
        "agent_edge": _estimate(
            rows,
            ("baseline", "AI_only", "AI"),
            ("baseline", "human_only", "human"),
            "AI-only baseline minus human-only baseline",
        ),
        "ai_introduction_effect": ai_introduction_effect,
        "ai_resistant_redesign_mitigation": _ai_resistant_redesign_mitigation(rows),
        "ai_resistant_redesign_effect": ai_resistant_redesign_effect,
    }
    return primary, diagnostics


def _system_scorecard(rows: list[Outcome]) -> tuple[dict[str, Estimate], dict[str, Estimate]]:
    ai_introduction_effect = _estimate(
        rows,
        ("baseline", "mixed", None),
        ("baseline", "human_only", None),
        "mixed AI baseline versus human baseline",
    )
    ai_resistant_redesign_effect = _estimate(
        rows,
        ("redesign", "human_only", None),
        ("baseline", "human_only", None),
        "AI-resistant redesign versus human baseline",
    )
    primary = {
        "ai_introduction_effect": ai_introduction_effect,
        "ai_resistant_redesign_effect": ai_resistant_redesign_effect,
    }
    diagnostics = {
        "agent_edge": _estimate(
            rows,
            ("baseline", "AI_only", None),
            ("baseline", "human_only", None),
            "AI-only baseline minus human-only baseline",
        ),
        "ai_introduction_effect": ai_introduction_effect,
        "ai_resistant_redesign_mitigation": _ai_resistant_redesign_mitigation(rows, agent_type=None),
        "ai_resistant_redesign_effect": ai_resistant_redesign_effect,
    }
    return primary, diagnostics


def _ai_resistant_redesign_mitigation(
    rows: list[Outcome],
    agent_type: str | None = "human",
) -> Estimate:
    baseline = _estimate(
        rows,
        ("baseline", "mixed", agent_type),
        ("baseline", "human_only", agent_type),
        "baseline AI introduction effect",
    )
    redesign = _estimate(
        rows,
        ("redesign", "mixed", agent_type),
        ("redesign", "human_only", agent_type),
        "redesign AI introduction effect",
    )
    if baseline.favorable_delta is None or redesign.favorable_delta is None:
        return Estimate(False, comparison="redesign AI introduction effect minus baseline AI introduction effect")
    raw = redesign.favorable_delta - baseline.favorable_delta
    reference_sd = baseline.reference_standard_deviation
    relative_delta = None if baseline.favorable_delta == 0 else raw / abs(baseline.favorable_delta)
    return Estimate(
        available=True,
        raw_delta=raw,
        favorable_delta=raw,
        standardized_delta=raw / reference_sd if reference_sd else None,
        percent_reference_value=baseline.favorable_delta,
        percent_reference_available=baseline.favorable_delta != 0,
        relative_delta=relative_delta,
        relative_percent=relative_delta * 100 if relative_delta is not None else None,
        reference_standard_deviation=reference_sd,
        comparison="redesign AI introduction effect minus baseline AI introduction effect",
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
    reference_mean = mean(reference_values) if reference_values else None
    raw_delta = mean(first_values) - mean(second_values)
    favorable_delta = raw_delta
    reference_sd = stdev(reference_values) if len(reference_values) > 1 else 0.0
    standard_error = math.sqrt(_variance(first_values) / len(first_values) + _variance(second_values) / len(second_values))
    relative_delta = (
        favorable_delta / abs(reference_mean)
        if reference_mean is not None and reference_mean != 0
        else None
    )
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
        percent_reference_value=reference_mean,
        percent_reference_available=reference_mean is not None and reference_mean != 0,
        relative_delta=relative_delta,
        relative_percent=relative_delta * 100 if relative_delta is not None else None,
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
    parser = argparse.ArgumentParser(description="Build an AI Replacement Impact scorecard from game run bundles.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("analysis_output"))
    args = parser.parse_args()
    print(analyze_runs(args.runs, args.output_dir))


if __name__ == "__main__":
    main()

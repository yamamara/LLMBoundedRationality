from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import statistics
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable

from sandbox.cournot_sweep import preflight_ollama
from sandbox.power import prevent_system_sleep
from sandbox.provenance import visualization_provenance
from sandbox.statistics import t_critical_95
from sandbox.visualization import (
    CournotPlayerScore,
    cournot_player_scores,
    cournot_score_svg,
)


TEMPERATURE_CATEGORY = "only_changing_temperature"
MEMORY_CATEGORY = "only_changing_memory_rounds"
MIXED_CATEGORY = "changing_temperature_and_memory"
CATEGORIES = (TEMPERATURE_CATEGORY, MEMORY_CATEGORY, MIXED_CATEGORY)
MEMORY_LEVELS: tuple[int | None, ...] = (None, 0, 1)

DEFAULT_STUDY_CONFIG: dict[str, Any] = {
    "name": "llama31-8b-cournot-temperature-memory",
    "provider": {
        "provider": "local_llama",
        "transport": "ollama",
        "endpoint": "http://localhost:11434",
        "model": "llama3.1:8b",
        "max_output_tokens": 300,
        "timeout_seconds": 120.0,
        "max_retries": 2,
        "top_p": 1.0,
        "provider_options": {"num_ctx": 16384, "keep_alive": "30m"},
    },
    "experiment": {
        "total_players": 4,
        "treatment": "FULL",
        "trials": 3,
        "rounds": 40,
        "revision_probability": 1.0,
        "base_seed": 7,
        "quantity_min": 0.0,
        "quantity_max": 100.0,
        "quantity_step": 0.01,
        "demand_intercept": 100.0,
        "marginal_cost": 1.0,
        "fixed_payment": 150.0,
    },
    "execution": {"workers": 2, "max_concurrent_requests": 2},
}

MEMORY_ONLY_VECTORS: tuple[tuple[int | None, ...], ...] = (
    (None, 0, 0, 0),
    (None, 1, 1, 1),
    (0, None, 0, 1),
    (0, 0, 1, None),
    (0, 1, None, 0),
    (1, None, 1, 0),
    (1, 0, None, 1),
    (1, 1, 0, None),
    (0, 0, 0, 0),
    (1, 1, 1, 1),
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_study_config(path: Path) -> dict[str, Any]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError("Study configuration must be a JSON object")
    config = _deep_merge(DEFAULT_STUDY_CONFIG, body)
    validate_study_config(config)
    return config


def validate_study_config(config: dict[str, Any]) -> None:
    provider = config["provider"]
    experiment = config["experiment"]
    execution = config["execution"]
    if provider.get("provider") != "local_llama" or provider.get("transport") != "ollama":
        raise ValueError("The study requires local_llama with Ollama transport")
    if not provider.get("endpoint") or not provider.get("model"):
        raise ValueError("Ollama endpoint and model are required")
    if int(experiment.get("total_players", 0)) != 4:
        raise ValueError("The compact design requires exactly four players")
    if experiment.get("treatment") != "FULL":
        raise ValueError("The compact design uses the FULL treatment only")
    if int(experiment.get("trials", 0)) != 3:
        raise ValueError("The compact design requires exactly three trials")
    if int(experiment.get("rounds", 0)) < 1:
        raise ValueError("rounds must be positive")
    if float(experiment.get("revision_probability", -1)) != 1.0:
        raise ValueError("The compact design requires revision_probability=1")
    if float(experiment.get("quantity_step", 0)) <= 0:
        raise ValueError("quantity_step must be positive")
    workers = int(execution.get("workers", 0))
    if workers not in {1, 2}:
        raise ValueError("execution.workers must be 1 or 2")
    if int(execution.get("max_concurrent_requests", 0)) != 2:
        raise ValueError("execution.max_concurrent_requests must be 2")


def _memory_slug(value: int | None) -> str:
    return "n" if value is None else str(value)


def _temperature_slug(value: float) -> str:
    return str(int(round(value * 10))).zfill(2)


def _configuration_id(
    prefix: str,
    index: int,
    temperatures: tuple[float, ...],
    memory_rounds: tuple[int | None, ...],
) -> str:
    profiles = "__".join(
        f"p{position + 1}-t{_temperature_slug(temperature)}-m{_memory_slug(memory)}"
        for position, (temperature, memory) in enumerate(
            zip(temperatures, memory_rounds)
        )
    )
    return f"{prefix}{index:03d}__{profiles}"


def temperature_configurations() -> list[dict[str, Any]]:
    configurations = []
    null_memory = (None, None, None, None)
    vectors = list(product((0.3, 0.7), repeat=4)) + [(0.5, 0.5, 0.5, 0.5)]
    for index, temperatures in enumerate(vectors, start=1):
        configurations.append(
            {
                "category": TEMPERATURE_CATEGORY,
                "configuration_id": _configuration_id(
                    "T", index, temperatures, null_memory
                ),
                "temperatures": list(temperatures),
                "memory_rounds": list(null_memory),
                "memory_study_baseline": temperatures == (0.5, 0.5, 0.5, 0.5),
                "mixed_study_control": temperatures
                in {(0.3, 0.3, 0.3, 0.3), (0.7, 0.7, 0.7, 0.7)},
                "design_source": (
                    "homogeneous_midpoint_baseline"
                    if temperatures == (0.5, 0.5, 0.5, 0.5)
                    else "two_level_temperature_factorial"
                ),
            }
        )
    return configurations


def memory_configurations() -> list[dict[str, Any]]:
    temperatures = (0.5, 0.5, 0.5, 0.5)
    configurations = []
    for index, memory_rounds in enumerate(MEMORY_ONLY_VECTORS, start=1):
        configurations.append(
            {
                "category": MEMORY_CATEGORY,
                "configuration_id": _configuration_id(
                    "M", index, temperatures, memory_rounds
                ),
                "temperatures": list(temperatures),
                "memory_rounds": list(memory_rounds),
                "memory_study_baseline": False,
                "mixed_study_control": False,
                "design_source": (
                    "homogeneous_memory_anchor"
                    if index >= 9
                    else "three_level_memory_orthogonal_array"
                ),
            }
        )
    return configurations


def _mixed_memory_vector(
    temperature_bits: tuple[int, ...], offset: int
) -> tuple[int | None, ...]:
    digits = tuple(
        (temperature_bits[index] + 2 * temperature_bits[(index + 1) % 4] + offset)
        % 3
        for index in range(4)
    )
    return tuple(MEMORY_LEVELS[digit] for digit in digits)


def mixed_configurations() -> list[dict[str, Any]]:
    selected: list[tuple[tuple[float, ...], tuple[int | None, ...], int]] = []
    for temperature_bits in product((0, 1), repeat=4):
        temperatures = tuple(0.3 if value == 0 else 0.7 for value in temperature_bits)
        for offset in range(3):
            memory_rounds = _mixed_memory_vector(temperature_bits, offset)
            if memory_rounds == (None, None, None, None):
                continue
            selected.append((temperatures, memory_rounds, offset))
    configurations = []
    for index, (temperatures, memory_rounds, offset) in enumerate(selected, start=1):
        configurations.append(
            {
                "category": MIXED_CATEGORY,
                "configuration_id": _configuration_id(
                    "TM", index, temperatures, memory_rounds
                ),
                "temperatures": list(temperatures),
                "memory_rounds": list(memory_rounds),
                "memory_study_baseline": False,
                "mixed_study_control": False,
                "design_source": f"cyclic_temperature_memory_offset_{offset}",
            }
        )
    return configurations


def build_configurations() -> list[dict[str, Any]]:
    configurations = (
        temperature_configurations()
        + memory_configurations()
        + mixed_configurations()
    )
    counts = {
        category: sum(row["category"] == category for row in configurations)
        for category in CATEGORIES
    }
    if counts != {
        TEMPERATURE_CATEGORY: 17,
        MEMORY_CATEGORY: 10,
        MIXED_CATEGORY: 46,
    }:
        raise RuntimeError(f"Unexpected compact-design counts: {counts}")
    if len({row["configuration_id"] for row in configurations}) != 73:
        raise RuntimeError("Compact-design configuration IDs must be unique")
    return configurations


def _assignments(
    config: dict[str, Any],
    configuration: dict[str, Any],
    trial_index: int,
) -> list[dict[str, Any]]:
    provider = config["provider"]
    experiment = config["experiment"]
    participants = []
    for position, (temperature, memory_rounds) in enumerate(
        zip(configuration["temperatures"], configuration["memory_rounds"])
    ):
        provider_options = copy.deepcopy(provider.get("provider_options", {}))
        provider_options["seed"] = (
            int(experiment["base_seed"]) + 10_000 + trial_index * 100 + position
        )
        participants.append(
            {
                "player_id": f"P{position + 1}",
                "agent_type": "AI",
                "role": "producer",
                "policy": "cournot_llm",
                "provider": provider["provider"],
                "transport": provider["transport"],
                "endpoint": provider["endpoint"],
                "model": provider["model"],
                "temperature": float(temperature),
                "top_p": provider.get("top_p", 1.0),
                "max_output_tokens": provider.get("max_output_tokens", 300),
                "timeout_seconds": provider.get("timeout_seconds", 120.0),
                "memory_rounds": memory_rounds,
                "max_retries": provider.get("max_retries", 2),
                "reasoning_effort": provider.get("reasoning_effort"),
                "provider_options": provider_options,
            }
        )
    return participants


def build_study_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    experiment = config["experiment"]
    for configuration in build_configurations():
        for trial_index in range(int(experiment["trials"])):
            trial_number = trial_index + 1
            base_run_id = (
                f"{configuration['configuration_id']}__trial-{trial_number:04d}"
            )
            graph_path = (
                Path(configuration["category"])
                / configuration["configuration_id"]
                / "FULL"
                / "trials"
                / f"trial-{trial_number:04d}.svg"
            )
            rows.append(
                {
                    "category": configuration["category"],
                    "configuration_id": configuration["configuration_id"],
                    "base_run_id": base_run_id,
                    "run_id": "",
                    "trial_index": trial_index,
                    "trial_number": trial_number,
                    "treatment": "FULL",
                    "environment_seed": int(experiment["base_seed"]) + trial_index,
                    "temperatures": copy.deepcopy(configuration["temperatures"]),
                    "memory_rounds": copy.deepcopy(configuration["memory_rounds"]),
                    "assignments": _assignments(config, configuration, trial_index),
                    "status": "pending",
                    "attempts": 0,
                    "run_path": "",
                    "trial_graph_path": str(graph_path),
                    "aggregate_graph_path": str(
                        graph_path.parent.parent / "aggregate.svg"
                    ),
                    "elapsed_seconds": None,
                    "error": "",
                    "graph_error": "",
                }
            )
    return rows


def estimate_model_decisions(config: dict[str, Any], rows: Iterable[dict[str, Any]]) -> int:
    rounds = int(config["experiment"]["rounds"])
    return sum(len(row["assignments"]) * rounds for row in rows)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], atomic: bool = False) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    target = path.with_suffix(path.suffix + ".tmp") if atomic else path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    if atomic:
        os.replace(target, path)


def write_combination_catalog(root: Path, configurations: list[dict[str, Any]]) -> None:
    rows = []
    for configuration in configurations:
        row = {
            "category": configuration["category"],
            "configuration_id": configuration["configuration_id"],
            "design_source": configuration["design_source"],
            "memory_study_baseline": configuration["memory_study_baseline"],
            "mixed_study_control": configuration["mixed_study_control"],
        }
        for position, (temperature, memory_rounds) in enumerate(
            zip(configuration["temperatures"], configuration["memory_rounds"]),
            start=1,
        ):
            row[f"p{position}_temperature"] = temperature
            row[f"p{position}_memory_rounds"] = (
                "null" if memory_rounds is None else memory_rounds
            )
        rows.append(row)
    _write_csv(root / "combination_catalog.csv", rows)


def _valid_run_path(value: str | None) -> bool:
    if not value:
        return False
    path = Path(value)
    return all(
        (path / filename).is_file()
        for filename in ("participant_data.csv", "system_data.csv", "summary.json")
    )


class StudyManifest:
    def __init__(self, root: Path, rows: list[dict[str, Any]]):
        self.root = root
        self.rows = rows
        self.by_id = {row["base_run_id"]: row for row in rows}
        self.lock = threading.RLock()

    @classmethod
    def load(cls, root: Path) -> StudyManifest:
        rows = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        store = cls(root, rows)
        with store.lock:
            for row in store.rows:
                graph = root / row["trial_graph_path"]
                if row["status"] == "completed" and _valid_run_path(
                    row.get("run_path")
                ) and graph.is_file():
                    continue
                row["status"] = "pending"
            store._write_locked()
        return store

    def write(self) -> None:
        with self.lock:
            self._write_locked()

    def _write_locked(self) -> None:
        _atomic_json(self.root / "manifest.json", self.rows)
        _write_csv(self.root / "manifest.csv", self.rows, atomic=True)

    def start(self, base_run_id: str) -> dict[str, Any]:
        with self.lock:
            row = self.by_id[base_run_id]
            row["attempts"] = int(row.get("attempts", 0)) + 1
            if not _valid_run_path(row.get("run_path")):
                row["run_id"] = (
                    f"{base_run_id}-attempt-{row['attempts']:02d}"
                )
            row["status"] = "running"
            row["error"] = ""
            row["graph_error"] = ""
            self._write_locked()
            return copy.deepcopy(row)

    def finish(
        self,
        base_run_id: str,
        *,
        status: str,
        run_path: str,
        elapsed_seconds: float,
        error: str = "",
    ) -> None:
        with self.lock:
            row = self.by_id[base_run_id]
            row["status"] = status
            row["run_path"] = run_path
            row["elapsed_seconds"] = round(elapsed_seconds, 6)
            row["error"] = error[:4000]
            self._write_locked()

    def graph_error(self, base_run_id: str, error: str) -> None:
        with self.lock:
            self.by_id[base_run_id]["graph_error"] = error[:4000]
            self._write_locked()

    def clear_graph_errors(self, configuration_id: str) -> None:
        with self.lock:
            changed = False
            for row in self.rows:
                if row["configuration_id"] == configuration_id and row.get("graph_error"):
                    row["graph_error"] = ""
                    changed = True
            if changed:
                self._write_locked()

    def completed_run_paths(self, configuration_id: str) -> list[Path]:
        with self.lock:
            selected = [
                copy.deepcopy(row)
                for row in self.rows
                if row["configuration_id"] == configuration_id
                and row["status"] == "completed"
                and _valid_run_path(row.get("run_path"))
            ]
        selected.sort(key=lambda row: row["trial_index"])
        return [Path(row["run_path"]) for row in selected]


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_trial_graph(run_path: Path, target: Path) -> None:
    scores = cournot_player_scores([run_path])
    provenance = visualization_provenance([run_path])
    svg = cournot_score_svg(
        scores,
        error_mode="ci95",
        provenance=provenance,
        sample_description="mean profit across completed rounds",
    )
    _atomic_text(target, svg)


def aggregate_trial_scores(run_paths: list[Path]) -> list[CournotPlayerScore]:
    grouped: dict[str, list[CournotPlayerScore]] = {}
    for run_path in run_paths:
        for score in cournot_player_scores([run_path]):
            grouped.setdefault(score.player_id, []).append(score)
    results = []
    for player_id, selected in sorted(grouped.items()):
        values = [score.average_profit for score in selected]
        average = statistics.mean(values)
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        half_width = (
            t_critical_95(len(values) - 1) * deviation / math.sqrt(len(values))
            if len(values) > 1
            else 0.0
        )
        first = selected[0]
        results.append(
            CournotPlayerScore(
                run_id=f"aggregate-{len(values)}-trials",
                player_id=player_id,
                agent=first.agent,
                agent_type=first.agent_type,
                round_count=len(values),
                average_profit=average,
                standard_deviation=deviation,
                lower_two_sd=average - 2 * deviation,
                upper_two_sd=average + 2 * deviation,
                ci95_half_width=half_width,
                ci95_lower=average - half_width,
                ci95_upper=average + half_width,
                minimum_profit=min(values),
                maximum_profit=max(values),
                total_profit=sum(values),
            )
        )
    return results


def write_aggregate_graph(run_paths: list[Path], target: Path) -> None:
    if not run_paths:
        return
    scores = aggregate_trial_scores(run_paths)
    provenance = visualization_provenance(run_paths)
    trial_count = len(run_paths)
    svg = cournot_score_svg(
        scores,
        error_mode="ci95",
        provenance=provenance,
        sample_description=f"mean of {trial_count} trial-level player means",
        show_whiskers=trial_count >= 2,
    )
    _atomic_text(target, svg)


def refresh_aggregate_graphs(root: Path, store: StudyManifest) -> int:
    """Rebuild every available aggregate, including stale resume artifacts."""
    configuration_ids = sorted(
        {
            row["configuration_id"]
            for row in store.rows
            if row["status"] == "completed" and _valid_run_path(row.get("run_path"))
        }
    )
    failures = 0
    for configuration_id in configuration_ids:
        representative = next(
            row for row in store.rows if row["configuration_id"] == configuration_id
        )
        try:
            write_aggregate_graph(
                store.completed_run_paths(configuration_id),
                root / representative["aggregate_graph_path"],
            )
            store.clear_graph_errors(configuration_id)
        except Exception as exc:
            failures += 1
            store.graph_error(
                representative["base_run_id"], f"{type(exc).__name__}: {exc}"
            )
    return failures


def _run_config(
    config: dict[str, Any], root: Path, entry: dict[str, Any]
) -> dict[str, Any]:
    experiment = config["experiment"]
    return {
        "game_name": config["name"],
        "run_id": entry["run_id"],
        "output_dir": str(root / "raw_runs"),
        "cournot": {
            "rounds": int(experiment["rounds"]),
            "quantity_min": float(experiment["quantity_min"]),
            "quantity_max": float(experiment["quantity_max"]),
            "quantity_step": float(experiment["quantity_step"]),
            "demand_intercept": float(experiment["demand_intercept"]),
            "marginal_cost": float(experiment["marginal_cost"]),
            "revision_probability": 1.0,
            "fixed_payment": float(experiment["fixed_payment"]),
            "treatment": "FULL",
            "seed": entry["environment_seed"],
            "institution": "cournot_temperature_memory_study",
        },
        "participants": copy.deepcopy(entry["assignments"]),
    }


def _execute_one(
    config: dict[str, Any],
    root: Path,
    store: StudyManifest,
    base_run_id: str,
    request_semaphore: threading.BoundedSemaphore,
    aggregate_lock: threading.Lock,
    run_function: Callable[[dict[str, Any]], Path],
) -> str:
    entry = store.start(base_run_id)
    started = time.monotonic()
    anticipated = root / "raw_runs" / entry["run_id"]
    try:
        if _valid_run_path(entry.get("run_path")):
            run_path = Path(entry["run_path"])
        else:
            selected_config = _run_config(config, root, entry)
            selected_config["_cournot_request_semaphore"] = request_semaphore
            run_path = run_function(selected_config)
        write_trial_graph(run_path, root / entry["trial_graph_path"])
        store.finish(
            base_run_id,
            status="completed",
            run_path=str(run_path.resolve()),
            elapsed_seconds=time.monotonic() - started,
        )
        try:
            with aggregate_lock:
                write_aggregate_graph(
                    store.completed_run_paths(entry["configuration_id"]),
                    root / entry["aggregate_graph_path"],
                )
        except Exception as exc:
            store.graph_error(base_run_id, f"{type(exc).__name__}: {exc}")
        return "completed"
    except Exception as exc:
        existing = entry.get("run_path")
        failed_path = Path(existing) if _valid_run_path(existing) else anticipated
        store.finish(
            base_run_id,
            status="failed",
            run_path=str(failed_path.resolve()) if failed_path.exists() else "",
            elapsed_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
        return "failed"


def selected_pending_rows(
    rows: list[dict[str, Any]],
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for index, row in enumerate(rows)
        if index % shard_count == shard_index and row["status"] != "completed"
    ]
    return selected[:limit] if limit is not None else selected


def execute_study(
    config: dict[str, Any],
    root: Path,
    store: StudyManifest,
    *,
    workers: int = 2,
    shard_index: int = 0,
    shard_count: int = 1,
    limit: int | None = None,
    run_function: Callable[[dict[str, Any]], Path] | None = None,
) -> dict[str, int]:
    if workers not in {1, 2}:
        raise ValueError("workers must be 1 or 2")
    if run_function is None:
        from simulation import run_experiment

        run_function = run_experiment
    pending = selected_pending_rows(
        store.rows,
        shard_index=shard_index,
        shard_count=shard_count,
        limit=limit,
    )
    counts = {"completed": 0, "failed": 0, "remaining": len(pending)}
    request_semaphore = threading.BoundedSemaphore(
        int(config["execution"]["max_concurrent_requests"])
    )
    aggregate_locks: dict[str, threading.Lock] = {
        row["configuration_id"]: threading.Lock() for row in pending
    }
    futures: dict[Future[str], str] = {}
    interrupted = False
    with prevent_system_sleep():
        executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="cournot-parameter-study"
        )
        try:
            for row in pending:
                future = executor.submit(
                    _execute_one,
                    config,
                    root,
                    store,
                    row["base_run_id"],
                    request_semaphore,
                    aggregate_locks[row["configuration_id"]],
                    run_function,
                )
                futures[future] = row["base_run_id"]
            for completed_index, future in enumerate(as_completed(futures), start=1):
                status = future.result()
                counts[status] += 1
                counts["remaining"] = len(pending) - completed_index
                print(
                    f"[{completed_index}/{len(pending)}] "
                    f"{futures[future]}: {status}"
                )
        except KeyboardInterrupt:
            interrupted = True
            for future in futures:
                future.cancel()
            print("Interrupted; waiting for active trials to checkpoint.")
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    if interrupted:
        raise KeyboardInterrupt
    refresh_aggregate_graphs(root, store)
    return counts


def _load_resolved_config(root: Path) -> dict[str, Any]:
    config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    config.pop("created_at", None)
    config.pop("study_root", None)
    validate_study_config(config)
    return config


def _resume_files(root: Path) -> tuple[Path, Path]:
    return root / "resolved_config.json", root / "manifest.json"


def _resume_error(root: Path) -> str | None:
    config_path, manifest_path = _resume_files(root)
    if not root.is_dir():
        return (
            f'Cannot resume "{root}": the study directory does not exist. '
            "A --dry-run does not create it. Initialize the study with: "
            "py scripts/run_cournot_parameter_study.py "
            "examples/cournot_temperature_memory_study.json "
            f'--output-dir "{root}"'
        )
    missing = [
        path.name for path in (config_path, manifest_path) if not path.is_file()
    ]
    if missing:
        return (
            f'Cannot resume "{root}": it is not an initialized study directory; '
            f"missing {', '.join(missing)}. Choose the directory containing both "
            "resolved_config.json and manifest.json, or initialize a new empty "
            "output directory without --resume."
        )
    return None


def _validate_shard(parser: argparse.ArgumentParser, index: int, count: int) -> None:
    if count < 1:
        parser.error("--shard-count must be positive")
    if not 0 <= index < count:
        parser.error("--shard-index must be between 0 and shard-count - 1")


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the compact local Llama Cournot temperature-memory study"
    )
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    if args.resume and (args.config or args.output_dir):
        parser.error("Use --resume by itself instead of config or --output-dir")
    if not args.resume and (not args.config or not args.output_dir):
        parser.error("config and --output-dir are required unless --resume is used")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    _validate_shard(parser, args.shard_index, args.shard_count)

    if args.resume:
        root = args.resume.resolve()
        resume_error = _resume_error(root)
        if resume_error:
            parser.error(resume_error)
        try:
            config = _load_resolved_config(root)
            plan = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            parser.error(f'Cannot resume "{root}": {type(exc).__name__}: {exc}')
    else:
        config = load_study_config(args.config)
        root = args.output_dir.resolve()
        plan = build_study_plan(config)
    workers = args.workers or int(config["execution"]["workers"])
    selected = selected_pending_rows(
        plan,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        limit=args.limit,
    )
    category_counts = {
        category: len(
            {row["configuration_id"] for row in plan if row["category"] == category}
        )
        for category in CATEGORIES
    }
    print(f"Study root: {root}")
    print(f"Configurations: {len(build_configurations())} {category_counts}")
    print(f"Planned trial executions: {len(plan)}; selected pending: {len(selected)}")
    print(f"Planned SVG graphs: {len(plan) + len(build_configurations())}")
    print(f"Estimated Llama decisions for selection: {estimate_model_decisions(config, selected):,}")
    print(f"Workers: {workers}; maximum concurrent Ollama requests: 2")
    if args.dry_run:
        return 0

    models = preflight_ollama(config)
    print(f"Ollama ready: {config['provider']['model']} ({', '.join(models)})")
    if args.resume:
        store = StudyManifest.load(root)
    else:
        if root.exists() and any(root.iterdir()):
            parser.error("--output-dir must be empty or absent; use --resume for an existing study")
        root.mkdir(parents=True, exist_ok=True)
        resolved = copy.deepcopy(config)
        resolved["created_at"] = datetime.now(UTC).isoformat()
        resolved["study_root"] = str(root)
        _atomic_json(root / "resolved_config.json", resolved)
        write_combination_catalog(root, build_configurations())
        store = StudyManifest(root, plan)
        store.write()
    try:
        result = execute_study(
            config,
            root,
            store,
            workers=workers,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            limit=args.limit,
        )
    except KeyboardInterrupt:
        print(
            "Study interrupted. Resume with: "
            f'py scripts/run_cournot_parameter_study.py --resume "{root}"'
        )
        return 130
    print(
        f"Finished selection: {result['completed']} completed, "
        f"{result['failed']} failed. Graph root: {root}"
    )
    return 0 if result["failed"] == 0 else 1

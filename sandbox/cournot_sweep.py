from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import statistics
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable

from sandbox.power import prevent_system_sleep
from sandbox.statistics import trial_summary


SCRIPT_POLICIES = (
    "cournot_best_reply",
    "cournot_random_quantity",
    "cournot_previous_average",
)
SCRIPT_LABELS = {
    "cournot_best_reply": "best-reply",
    "cournot_random_quantity": "random-quantity",
    "cournot_previous_average": "previous-average",
}

DEFAULT_SWEEP_CONFIG: dict[str, Any] = {
    "name": "llama31-8b-cournot-balanced",
    "output_dir": "runs/cournot-sweeps",
    "provider": {
        "provider": "local_llama",
        "transport": "ollama",
        "endpoint": "http://localhost:11434",
        "model": "llama3.1:8b",
        "max_output_tokens": 300,
        "timeout_seconds": 120.0,
        "memory_rounds": None,
        "max_retries": 2,
        "top_p": 1.0,
        "provider_options": {"num_ctx": 16384, "keep_alive": "30m"},
    },
    "experiment": {
        "total_players": 4,
        "model_counts": [1, 2, 3, 4],
        "temperatures": [0.0, 0.3, 0.7, 1.0],
        "treatments": ["BEST", "FULL"],
        "trials": 5,
        "rounds": 40,
        "revision_probability": 2 / 3,
        "base_seed": 7,
        "quantity_min": 0.0,
        "quantity_max": 100.0,
        "quantity_step": 0.01,
        "demand_intercept": 100.0,
        "marginal_cost": 1.0,
        "fixed_payment": 150.0,
        "initial_quantities": [10.0, 20.0, 30.0, 40.0],
        "steady_state_window": 10,
        "nash_tolerance_fraction": 0.05,
    },
    "execution": {"workers": 2},
}

RUN_METRICS = (
    "model_mean_profit",
    "script_mean_profit",
    "model_script_profit_gap",
    "model_mean_quantity",
    "script_mean_quantity",
    "mean_total_quantity",
    "final_window_mean_total_quantity",
    "mean_price",
    "final_window_mean_price",
    "mean_total_market_profit",
    "final_window_mean_total_market_profit",
    "mean_distance_to_nash",
    "final_window_mean_distance_to_nash",
    "total_quantity_sd",
    "total_quantity_mean_abs_change",
    "zero_price_share",
    "final_window_nash_band_share",
    "profit_dispersion",
    "model_mean_ex_post_regret",
    "model_mean_best_reply_deviation",
    "model_retry_rate",
    "model_mean_latency_seconds",
    "model_prompt_tokens",
    "model_output_tokens",
    "model_ollama_duration_seconds",
    "model_quantity_sd",
    "model_action_diversity",
    "elapsed_seconds",
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_sweep_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        body = json.load(handle)
    if not isinstance(body, dict):
        raise ValueError("Sweep configuration must be a JSON object")
    config = _deep_merge(DEFAULT_SWEEP_CONFIG, body)
    validate_sweep_config(config)
    return config


def validate_sweep_config(config: dict[str, Any]) -> None:
    provider = config["provider"]
    experiment = config["experiment"]
    execution = config["execution"]
    if provider.get("provider") != "local_llama" or provider.get("transport") != "ollama":
        raise ValueError("The local Cournot sweep currently requires local_llama with Ollama transport")
    if not provider.get("endpoint") or not provider.get("model"):
        raise ValueError("Ollama endpoint and model are required")
    if experiment.get("total_players") != 4:
        raise ValueError("This sweep design currently requires exactly four total players")
    model_counts = experiment.get("model_counts", [])
    if not model_counts or any(count not in {1, 2, 3, 4} for count in model_counts):
        raise ValueError("model_counts must contain values from 1 through 4")
    if len(model_counts) != len(set(model_counts)):
        raise ValueError("model_counts must not contain duplicates")
    temperatures = experiment.get("temperatures", [])
    if not temperatures or any(not 0 <= float(value) <= 2 for value in temperatures):
        raise ValueError("temperatures must contain values between 0 and 2")
    treatments = experiment.get("treatments", [])
    if not treatments or any(value not in {"BEST", "FULL"} for value in treatments):
        raise ValueError("treatments must contain BEST and/or FULL")
    if int(experiment.get("trials", 0)) < 1 or int(experiment.get("rounds", 0)) < 1:
        raise ValueError("trials and rounds must be positive")
    probability = float(experiment.get("revision_probability", -1))
    if not 0 <= probability <= 1:
        raise ValueError("revision_probability must be between 0 and 1")
    if float(experiment.get("quantity_step", 0)) <= 0:
        raise ValueError("quantity_step must be positive")
    if len(experiment.get("initial_quantities", [])) < 4:
        raise ValueError("initial_quantities must contain four values")
    if int(experiment.get("steady_state_window", 0)) < 1:
        raise ValueError("steady_state_window must be positive")
    if float(experiment.get("nash_tolerance_fraction", 0)) <= 0:
        raise ValueError("nash_tolerance_fraction must be positive")
    if int(execution.get("workers", 0)) < 1:
        raise ValueError("execution.workers must be positive")


def opponent_compositions(model_count: int) -> list[tuple[str, ...]]:
    script_slots = 4 - model_count
    if script_slots == 0:
        return [tuple()]
    homogeneous = [tuple([policy] * script_slots) for policy in SCRIPT_POLICIES]
    if script_slots == 1:
        return homogeneous
    if script_slots == 2:
        return homogeneous + list(combinations(SCRIPT_POLICIES, 2))
    if script_slots == 3:
        return homogeneous + [SCRIPT_POLICIES]
    raise ValueError("Unsupported model count")


def _composition_id(policies: tuple[str, ...]) -> str:
    if not policies:
        return "all-model"
    return "+".join(SCRIPT_LABELS[policy] for policy in policies)


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "value"


def _temperature_slug(value: float) -> str:
    return (f"{float(value):g}").replace("-", "m").replace(".", "p")


def _rotated_assignments(
    config: dict[str, Any],
    model_count: int,
    script_policies: tuple[str, ...],
    temperature: float,
    trial_index: int,
) -> list[dict[str, Any]]:
    experiment = config["experiment"]
    provider = config["provider"]
    roles: list[tuple[str, str | None]] = [
        ("model", None) for _ in range(model_count)
    ] + [("script", policy) for policy in script_policies]
    shift = trial_index % len(roles)
    if shift:
        roles = roles[-shift:] + roles[:-shift]
    participants = []
    base_seed = int(experiment["base_seed"])
    for position, (kind, policy) in enumerate(roles):
        player_id = f"P{position + 1}"
        participant: dict[str, Any] = {
            "player_id": player_id,
            "agent_type": "AI",
            "role": "producer",
        }
        if kind == "model":
            provider_options = copy.deepcopy(provider.get("provider_options", {}))
            provider_options["seed"] = base_seed + trial_index * 1000 + position
            participant.update(
                {
                    "policy": "cournot_llm",
                    "provider": provider["provider"],
                    "transport": provider["transport"],
                    "endpoint": provider["endpoint"],
                    "model": provider["model"],
                    "temperature": float(temperature),
                    "top_p": provider.get("top_p", 1.0),
                    "max_output_tokens": provider.get("max_output_tokens", 300),
                    "timeout_seconds": provider.get("timeout_seconds", 120.0),
                    "memory_rounds": provider.get("memory_rounds"),
                    "max_retries": provider.get("max_retries", 2),
                    "reasoning_effort": provider.get("reasoning_effort"),
                    "provider_options": provider_options,
                }
            )
        else:
            participant["policy"] = policy
            participant["initial_quantity"] = float(
                experiment["initial_quantities"][position]
            )
            if policy == "cournot_random_quantity":
                participant["random_seed"] = (
                    base_seed + 100_000 + trial_index * 1000 + position
                )
        participants.append(participant)
    return participants


def build_sweep_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    experiment = config["experiment"]
    rows = []
    for treatment in experiment["treatments"]:
        for temperature_value in experiment["temperatures"]:
            temperature = float(temperature_value)
            for model_count in experiment["model_counts"]:
                for scripts in opponent_compositions(int(model_count)):
                    opponent_id = _composition_id(scripts)
                    cell_id = "__".join(
                        (
                            treatment.lower(),
                            f"temp-{_temperature_slug(temperature)}",
                            f"models-{model_count}",
                            _slug(opponent_id),
                        )
                    )
                    for trial_index in range(int(experiment["trials"])):
                        run_id = f"{cell_id}__trial-{trial_index + 1:04d}"
                        rows.append(
                            {
                                "cell_id": cell_id,
                                "base_run_id": run_id,
                                "run_id": "",
                                "treatment": treatment,
                                "temperature": temperature,
                                "model_count": int(model_count),
                                "opponent_composition": opponent_id,
                                "script_policies": list(scripts),
                                "trial_index": trial_index,
                                "environment_seed": int(experiment["base_seed"])
                                + trial_index,
                                "assignments": _rotated_assignments(
                                    config,
                                    int(model_count),
                                    scripts,
                                    temperature,
                                    trial_index,
                                ),
                                "status": "pending",
                                "attempts": 0,
                                "run_path": "",
                                "elapsed_seconds": None,
                                "error": "",
                            }
                        )
    return rows


def estimate_model_decisions(config: dict[str, Any], plan: Iterable[dict[str, Any]]) -> int:
    experiment = config["experiment"]
    expected_per_model = 1 + (int(experiment["rounds"]) - 1) * float(
        experiment["revision_probability"]
    )
    return round(sum(row["model_count"] * expected_per_model for row in plan))


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


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
    atomic: bool = False,
) -> None:
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_suffix(path.suffix + ".tmp") if atomic else path
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    if atomic:
        os.replace(target, path)


class ManifestStore:
    def __init__(self, root: Path, rows: list[dict[str, Any]]):
        self.root = root
        self.rows = rows
        self.by_id = {row["base_run_id"]: row for row in rows}
        self.lock = threading.RLock()

    @classmethod
    def load(cls, root: Path) -> ManifestStore:
        rows = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        store = cls(root, rows)
        with store.lock:
            for row in store.rows:
                if row["status"] == "completed" and _valid_run_path(row.get("run_path")):
                    continue
                row["status"] = "pending"
                if row.get("error") and row.get("status") != "completed":
                    row["error"] = row["error"]
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
            row["run_id"] = f"{base_run_id}-attempt-{row['attempts']:02d}"
            row["status"] = "running"
            row["error"] = ""
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


def _valid_run_path(value: str | None) -> bool:
    if not value:
        return False
    path = Path(value)
    return all(
        (path / name).is_file()
        for name in ("participant_data.csv", "system_data.csv", "summary.json")
    )


def _run_config(config: dict[str, Any], root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    experiment = config["experiment"]
    return {
        "game_name": config["name"],
        "run_id": entry["run_id"],
        "output_dir": str(root / "runs"),
        "cournot": {
            "rounds": int(experiment["rounds"]),
            "quantity_min": float(experiment["quantity_min"]),
            "quantity_max": float(experiment["quantity_max"]),
            "quantity_step": float(experiment["quantity_step"]),
            "demand_intercept": float(experiment["demand_intercept"]),
            "marginal_cost": float(experiment["marginal_cost"]),
            "revision_probability": float(experiment["revision_probability"]),
            "fixed_payment": float(experiment["fixed_payment"]),
            "treatment": entry["treatment"],
            "seed": entry["environment_seed"],
            "institution": "cournot_sweep",
        },
        "participants": copy.deepcopy(entry["assignments"]),
    }


def preflight_ollama(config: dict[str, Any]) -> list[str]:
    try:
        from ollama import Client
    except ImportError as exc:
        raise RuntimeError(
            "Ollama support requires the 'ollama' package; install frontend/requirements.txt"
        ) from exc
    provider = config["provider"]
    client = Client(host=provider["endpoint"], timeout=15.0)
    try:
        response = client.list()
        models = [model.model for model in response.models]
        client.show(provider["model"])
    except Exception as exc:
        raise RuntimeError(
            f"Unable to use Ollama model {provider['model']!r} at {provider['endpoint']}: {exc}"
        ) from exc
    return models


def _execute_one(
    config: dict[str, Any],
    root: Path,
    store: ManifestStore,
    base_run_id: str,
    run_function: Callable[[dict[str, Any]], Path],
) -> str:
    entry = store.start(base_run_id)
    started = time.monotonic()
    anticipated = root / "runs" / entry["run_id"]
    try:
        run_path = run_function(_run_config(config, root, entry))
        elapsed = time.monotonic() - started
        store.finish(
            base_run_id,
            status="completed",
            run_path=str(run_path.resolve()),
            elapsed_seconds=elapsed,
        )
        return "completed"
    except Exception as exc:
        elapsed = time.monotonic() - started
        store.finish(
            base_run_id,
            status="failed",
            run_path=str(anticipated.resolve()) if anticipated.exists() else "",
            elapsed_seconds=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )
        return "failed"


def execute_sweep(
    config: dict[str, Any],
    root: Path,
    store: ManifestStore,
    *,
    workers: int,
    limit: int | None = None,
    run_function: Callable[[dict[str, Any]], Path] | None = None,
) -> dict[str, int]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if run_function is None:
        from simulation import run_experiment

        run_function = run_experiment
    pending = [row for row in store.rows if row["status"] != "completed"]
    if limit is not None:
        pending = pending[:limit]
    counts = {"completed": 0, "failed": 0, "remaining": len(pending)}
    print_lock = threading.Lock()
    futures: dict[Future[str], str] = {}
    interrupted = False
    with prevent_system_sleep():
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cournot-sweep")
        try:
            for row in pending:
                future = executor.submit(
                    _execute_one,
                    config,
                    root,
                    store,
                    row["base_run_id"],
                    run_function,
                )
                futures[future] = row["base_run_id"]
            for completed_index, future in enumerate(as_completed(futures), start=1):
                status = future.result()
                counts[status] += 1
                counts["remaining"] = len(pending) - completed_index
                with print_lock:
                    print(
                        f"[{completed_index}/{len(pending)}] {futures[future]}: {status}"
                    )
        except KeyboardInterrupt:
            interrupted = True
            for future in futures:
                future.cancel()
            print("Interrupted; waiting for active runs to checkpoint.")
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    analyze_sweep(root, config, store.rows)
    if interrupted:
        raise KeyboardInterrupt
    return counts


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: Iterable[float]) -> float | None:
    samples = list(values)
    return statistics.mean(samples) if samples else None


def _sd(values: Iterable[float]) -> float:
    samples = list(values)
    return statistics.stdev(samples) if len(samples) > 1 else 0.0


def _mean_abs_change(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.mean(abs(second - first) for first, second in zip(values, values[1:]))


def _best_reply(
    opponents_total: float,
    demand_intercept: float,
    marginal_cost: float,
    quantity_min: float,
    quantity_max: float,
    quantity_step: float,
) -> tuple[float, float]:
    raw = (demand_intercept - marginal_cost - opponents_total) / 2
    raw = min(max(raw, quantity_min), quantity_max)
    lower = math.floor((raw - quantity_min) / quantity_step)
    candidates = {
        round(quantity_min + lower * quantity_step, 10),
        round(quantity_min + (lower + 1) * quantity_step, 10),
    }
    legal = [value for value in candidates if quantity_min <= value <= quantity_max]

    def profit(quantity: float) -> float:
        price = max(demand_intercept - opponents_total - quantity, 0.0)
        return (price - marginal_cost) * quantity

    quantity = max(legal, key=lambda value: (profit(value), -value))
    return quantity, profit(quantity)


def _decision_metadata(run_path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    result = {}
    events_path = run_path / "events.jsonl"
    if not events_path.is_file():
        return result
    for line in events_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event_type") not in {"decision", "decision_rejected"}:
            continue
        private = (event.get("observation") or {}).get("private_state") or {}
        metadata = event.get("agent_metadata") or {}
        usage = metadata.get("usage") or {}
        result[(int(event["round"]), event["player_id"])] = {
            "revision_allowed": bool(event.get("revision_allowed")),
            "best_reply": private.get("best_reply_quantity"),
            "retry_count": int(metadata.get("retry_count", 0) or 0),
            "latency_seconds": float(metadata.get("latency_seconds", 0.0) or 0.0),
            "prompt_tokens": int(usage.get("prompt_eval_count", 0) or 0),
            "output_tokens": int(usage.get("eval_count", 0) or 0),
            "ollama_duration_seconds": float(usage.get("total_duration", 0) or 0)
            / 1_000_000_000,
        }
    return result


def summarize_run(
    entry: dict[str, Any], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_path = Path(entry["run_path"])
    decisions = _read_csv(run_path / "participant_data.csv")
    rounds = _read_csv(run_path / "system_data.csv")
    metadata = _decision_metadata(run_path)
    experiment = config["experiment"]
    assignments = {item["player_id"]: item for item in entry["assignments"]}
    by_round: dict[int, list[dict[str, str]]] = {}
    by_player: dict[str, list[dict[str, str]]] = {}
    for row in decisions:
        by_round.setdefault(int(row["round"]), []).append(row)
        by_player.setdefault(row["player_id"], []).append(row)

    ex_post_regret: dict[tuple[int, str], float] = {}
    for round_number, selected in by_round.items():
        total_quantity = sum(float(row["quantity"]) for row in selected)
        for row in selected:
            quantity = float(row["quantity"])
            _, best_profit = _best_reply(
                total_quantity - quantity,
                float(experiment["demand_intercept"]),
                float(experiment["marginal_cost"]),
                float(experiment["quantity_min"]),
                float(experiment["quantity_max"]),
                float(experiment["quantity_step"]),
            )
            ex_post_regret[(round_number, row["player_id"])] = max(
                0.0, best_profit - float(row["profit"])
            )

    window = min(int(experiment["steady_state_window"]), len(rounds))
    player_rows = []
    for player_id, selected in sorted(by_player.items()):
        selected.sort(key=lambda row: int(row["round"]))
        quantities = [float(row["quantity"]) for row in selected]
        profits = [float(row["profit"]) for row in selected]
        assignment = assignments[player_id]
        cohort = "model" if assignment["policy"] == "cournot_llm" else "script"
        decisions_meta = [
            metadata.get((int(row["round"]), player_id), {}) for row in selected
        ]
        revised = [item for item in decisions_meta if item.get("revision_allowed")]
        deviations = []
        for row, item in zip(selected, decisions_meta):
            best_reply = item.get("best_reply")
            if item.get("revision_allowed") and best_reply is not None:
                deviations.append(abs(float(row["quantity"]) - float(best_reply)))
        row = {
            "cell_id": entry["cell_id"],
            "base_run_id": entry["base_run_id"],
            "run_id": entry["run_id"],
            "trial_index": entry["trial_index"],
            "treatment": entry["treatment"],
            "temperature": entry["temperature"],
            "model_count": entry["model_count"],
            "opponent_composition": entry["opponent_composition"],
            "player_id": player_id,
            "cohort": cohort,
            "policy": assignment["policy"],
            "round_count": len(selected),
            "revision_decision_count": len(revised),
            "mean_quantity": statistics.mean(quantities),
            "quantity_sd": _sd(quantities),
            "quantity_mean_abs_change": _mean_abs_change(quantities),
            "unique_quantity_count": len(set(quantities)),
            "action_diversity": len(set(quantities)) / len(quantities),
            "mean_profit": statistics.mean(profits),
            "total_profit": sum(profits),
            "final_window_mean_profit": statistics.mean(profits[-window:]),
            "mean_ex_post_regret": statistics.mean(
                ex_post_regret[(int(item["round"]), player_id)] for item in selected
            ),
            "mean_best_reply_deviation": _mean(deviations),
            "retry_count": sum(item.get("retry_count", 0) for item in revised),
            "latency_seconds": sum(item.get("latency_seconds", 0.0) for item in revised),
            "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in revised),
            "output_tokens": sum(item.get("output_tokens", 0) for item in revised),
            "ollama_duration_seconds": sum(
                item.get("ollama_duration_seconds", 0.0) for item in revised
            ),
        }
        player_rows.append(row)

    system_quantities = [float(row["total_quantity"]) for row in rounds]
    prices = [float(row["price"]) for row in rounds]
    market_profits = [float(row["total_market_profit"]) for row in rounds]
    distances = [float(row["distance_to_nash"]) for row in rounds]
    model_players = [row for row in player_rows if row["cohort"] == "model"]
    script_players = [row for row in player_rows if row["cohort"] == "script"]
    nash_total = 4 * (
        float(experiment["demand_intercept"]) - float(experiment["marginal_cost"])
    ) / 5
    tolerance = nash_total * float(experiment["nash_tolerance_fraction"])
    final_quantities = system_quantities[-window:]
    model_calls = sum(row["revision_decision_count"] for row in model_players)
    run_row = {
        "cell_id": entry["cell_id"],
        "base_run_id": entry["base_run_id"],
        "run_id": entry["run_id"],
        "trial_index": entry["trial_index"],
        "environment_seed": entry["environment_seed"],
        "treatment": entry["treatment"],
        "temperature": entry["temperature"],
        "model_count": entry["model_count"],
        "opponent_composition": entry["opponent_composition"],
        "model_mean_profit": _mean(row["mean_profit"] for row in model_players),
        "script_mean_profit": _mean(row["mean_profit"] for row in script_players),
        "model_mean_quantity": _mean(row["mean_quantity"] for row in model_players),
        "script_mean_quantity": _mean(row["mean_quantity"] for row in script_players),
        "mean_total_quantity": statistics.mean(system_quantities),
        "final_window_mean_total_quantity": statistics.mean(final_quantities),
        "mean_price": statistics.mean(prices),
        "final_window_mean_price": statistics.mean(prices[-window:]),
        "mean_total_market_profit": statistics.mean(market_profits),
        "final_window_mean_total_market_profit": statistics.mean(market_profits[-window:]),
        "mean_distance_to_nash": statistics.mean(distances),
        "final_window_mean_distance_to_nash": statistics.mean(distances[-window:]),
        "total_quantity_sd": _sd(system_quantities),
        "total_quantity_mean_abs_change": _mean_abs_change(system_quantities),
        "zero_price_share": sum(price == 0 for price in prices) / len(prices),
        "final_window_nash_band_share": sum(
            abs(quantity - nash_total) <= tolerance for quantity in final_quantities
        )
        / len(final_quantities),
        "profit_dispersion": _sd(row["mean_profit"] for row in player_rows),
        "model_mean_ex_post_regret": _mean(
            row["mean_ex_post_regret"] for row in model_players
        ),
        "model_mean_best_reply_deviation": _mean(
            row["mean_best_reply_deviation"]
            for row in model_players
            if row["mean_best_reply_deviation"] is not None
        ),
        "model_retry_rate": (
            sum(row["retry_count"] for row in model_players) / model_calls
            if model_calls
            else None
        ),
        "model_mean_latency_seconds": (
            sum(row["latency_seconds"] for row in model_players) / model_calls
            if model_calls
            else None
        ),
        "model_prompt_tokens": sum(row["prompt_tokens"] for row in model_players),
        "model_output_tokens": sum(row["output_tokens"] for row in model_players),
        "model_ollama_duration_seconds": sum(
            row["ollama_duration_seconds"] for row in model_players
        ),
        "model_quantity_sd": _mean(row["quantity_sd"] for row in model_players),
        "model_action_diversity": _mean(
            row["action_diversity"] for row in model_players
        ),
        "elapsed_seconds": entry.get("elapsed_seconds"),
    }
    first = run_row["model_mean_profit"]
    second = run_row["script_mean_profit"]
    run_row["model_script_profit_gap"] = (
        first - second if first is not None and second is not None else None
    )
    return player_rows, run_row


def analyze_sweep(
    root: Path, config: dict[str, Any], manifest_rows: list[dict[str, Any]]
) -> Path:
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    player_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for entry in manifest_rows:
        if entry.get("status") != "completed" or not _valid_run_path(entry.get("run_path")):
            continue
        try:
            selected_players, selected_run = summarize_run(entry, config)
            player_rows.extend(selected_players)
            run_rows.append(selected_run)
        except Exception as exc:
            entry["analysis_error"] = f"{type(exc).__name__}: {exc}"[:2000]

    _write_csv(analysis_dir / "player_summary.csv", player_rows)
    _write_csv(analysis_dir / "run_summary.csv", run_rows)
    run_by_cell: dict[str, list[dict[str, Any]]] = {}
    manifest_by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in run_rows:
        run_by_cell.setdefault(row["cell_id"], []).append(row)
    for row in manifest_rows:
        manifest_by_cell.setdefault(row["cell_id"], []).append(row)

    cell_rows = []
    metrics_long = []
    for cell_id, planned in sorted(manifest_by_cell.items()):
        successful = run_by_cell.get(cell_id, [])
        first = planned[0]
        cell_row: dict[str, Any] = {
            "cell_id": cell_id,
            "treatment": first["treatment"],
            "temperature": first["temperature"],
            "model_count": first["model_count"],
            "opponent_composition": first["opponent_composition"],
            "planned_runs": len(planned),
            "successful_runs": len(successful),
            "failed_runs": sum(row["status"] == "failed" for row in planned),
            "pending_runs": sum(row["status"] in {"pending", "running"} for row in planned),
        }
        cell_row["failure_rate"] = (
            cell_row["failed_runs"] / cell_row["planned_runs"]
        )
        for metric in RUN_METRICS:
            values = [
                float(row[metric])
                for row in successful
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            summary = trial_summary(values)
            cell_row[f"{metric}_mean"] = summary["mean"]
            cell_row[f"{metric}_sd"] = summary["sample_sd"]
            cell_row[f"{metric}_ci95_lower"] = summary["ci95_lower"]
            cell_row[f"{metric}_ci95_upper"] = summary["ci95_upper"]
            cell_row[f"{metric}_n"] = summary["n"]
            for row in successful:
                if row.get(metric) is None:
                    continue
                metrics_long.append(
                    {
                        "cell_id": cell_id,
                        "run_id": row["run_id"],
                        "trial_index": row["trial_index"],
                        "treatment": row["treatment"],
                        "temperature": row["temperature"],
                        "model_count": row["model_count"],
                        "opponent_composition": row["opponent_composition"],
                        "metric": metric,
                        "value": row[metric],
                    }
                )
        cell_rows.append(cell_row)

    _write_csv(analysis_dir / "cell_summary.csv", cell_rows)
    _write_csv(analysis_dir / "metrics_long.csv", metrics_long)
    write_dashboard(cell_rows, analysis_dir / "dashboard.html")
    _atomic_json(
        analysis_dir / "analysis_summary.json",
        {
            "planned_runs": len(manifest_rows),
            "successful_runs": len(run_rows),
            "failed_runs": sum(row["status"] == "failed" for row in manifest_rows),
            "generated_at": datetime.now(UTC).isoformat(),
        },
    )
    return analysis_dir


def write_dashboard(cell_rows: list[dict[str, Any]], path: Path) -> None:
    try:
        from plotly.offline import get_plotlyjs

        plotly_js = get_plotlyjs()
    except ImportError:
        plotly_js = ""
    safe_rows = json.dumps(cell_rows, default=str).replace("</", "<\\/")
    panels = [
        ("model_mean_profit", "Model mean profit"),
        ("model_script_profit_gap", "Model minus script profit"),
        ("mean_distance_to_nash", "Distance to Nash output"),
        ("mean_total_market_profit", "Total market profit"),
        ("model_mean_ex_post_regret", "Model ex-post regret"),
        ("model_mean_latency_seconds", "Model latency (seconds)"),
        ("model_output_tokens", "Model output tokens per run"),
        ("total_quantity_sd", "Total quantity volatility"),
    ]
    panel_html = "\n".join(
        f'<section><h2>{label}</h2><div class="chart" id="chart-{metric}"></div></section>'
        for metric, label in panels
    )
    panel_json = json.dumps(panels)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cournot Llama Sweep</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#172033}}
header,main{{max-width:1300px;margin:auto;padding:24px}}
.controls{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:16px;background:white;padding:16px;border-radius:10px}}
label{{display:grid;gap:6px;font-weight:600}} select{{padding:8px}}
section{{background:white;margin:18px 0;padding:16px;border-radius:10px}} .chart{{height:430px}}
</style><script>{plotly_js}</script></head>
<body><header><h1>Local Llama Cournot Sweep</h1><p>Bars and lines use run-level Student-t 95% confidence intervals.</p>
<div class="controls">
<label>Treatment<select id="treatment"><option value="all">All</option></select></label>
<label>Temperature<select id="temperature"><option value="all">All</option></select></label>
<label>Model count<select id="models"><option value="all">All</option></select></label>
<label>Opponent composition<select id="opponents"><option value="all">All</option></select></label>
</div></header><main>{panel_html}</main>
<script>
const rows={safe_rows}; const panels={panel_json};
const unique=(key)=>[...new Set(rows.map(r=>String(r[key])))].sort();
function populate(id,key){{const s=document.getElementById(id);unique(key).forEach(v=>{{const o=document.createElement('option');o.value=v;o.textContent=v;s.appendChild(o);}});}}
populate('treatment','treatment');populate('temperature','temperature');populate('models','model_count');populate('opponents','opponent_composition');
function filtered(){{return rows.filter(r=>(treatment.value==='all'||String(r.treatment)===treatment.value)&&(temperature.value==='all'||String(r.temperature)===temperature.value)&&(models.value==='all'||String(r.model_count)===models.value)&&(opponents.value==='all'||String(r.opponent_composition)===opponents.value));}}
function render(metric,label){{const groups={{}};filtered().forEach(r=>{{const key=`${{r.treatment}} | ${{r.model_count}} models | ${{r.opponent_composition}}`; (groups[key]??=[]).push(r);}});const traces=Object.entries(groups).map(([name,items])=>{{items.sort((a,b)=>a.temperature-b.temperature);const mean=metric+'_mean',lo=metric+'_ci95_lower',hi=metric+'_ci95_upper';return {{name,x:items.map(r=>r.temperature),y:items.map(r=>r[mean]),mode:'lines+markers',error_y:{{type:'data',symmetric:false,array:items.map(r=>r[hi]==null?0:r[hi]-r[mean]),arrayminus:items.map(r=>r[lo]==null?0:r[mean]-r[lo]),visible:true}}}};}});Plotly.react('chart-'+metric,traces,{{xaxis:{{title:'Temperature'}},yaxis:{{title:label}},margin:{{t:20}},legend:{{orientation:'h'}}}},{{responsive:true}});}}
function renderAll(){{panels.forEach(([metric,label])=>render(metric,label));}}
['treatment','temperature','models','opponents'].forEach(id=>document.getElementById(id).addEventListener('change',renderAll));renderAll();
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _new_root(config: dict[str, Any]) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(config["output_dir"]) / f"{_slug(config['name'])}-{timestamp}"


def _load_resolved_config(root: Path) -> dict[str, Any]:
    config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    config.pop("created_at", None)
    config.pop("sweep_root", None)
    validate_sweep_config(config)
    return config


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a resumable local Ollama Cournot sweep")
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--resume", type=Path, help="Resume an existing sweep directory")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing or calling Ollama")
    parser.add_argument("--workers", type=int, help="Override concurrent run count")
    parser.add_argument("--limit", type=int, help="Run at most this many pending cells")
    args = parser.parse_args(argv)
    if args.resume and args.config:
        parser.error("Provide either a config path or --resume, not both")
    if not args.resume and not args.config:
        parser.error("A config path is required unless --resume is used")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")

    if args.resume:
        root = args.resume.resolve()
        config = _load_resolved_config(root)
        plan = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    else:
        config = load_sweep_config(args.config)
        root = _new_root(config).resolve()
        plan = build_sweep_plan(config)
    workers = args.workers or int(config["execution"]["workers"])
    selected = [row for row in plan if row["status"] != "completed"]
    if args.limit is not None:
        selected = selected[: args.limit]
    print(f"Sweep root: {root}")
    print(f"Planned runs: {len(plan)}; pending selection: {len(selected)}")
    print(f"Estimated Llama decisions: {estimate_model_decisions(config, selected):,}")
    print(f"Workers: {workers}")
    if args.dry_run:
        return 0

    models = preflight_ollama(config)
    print(f"Ollama ready: {config['provider']['model']} ({', '.join(models)})")
    if args.resume:
        store = ManifestStore.load(root)
    else:
        root.mkdir(parents=True, exist_ok=False)
        resolved = copy.deepcopy(config)
        resolved["created_at"] = datetime.now(UTC).isoformat()
        resolved["sweep_root"] = str(root)
        _atomic_json(root / "resolved_config.json", resolved)
        store = ManifestStore(root, plan)
        store.write()
    try:
        result = execute_sweep(
            config,
            root,
            store,
            workers=workers,
            limit=args.limit,
        )
    except KeyboardInterrupt:
        print(f"Sweep interrupted. Resume with: py scripts/run_cournot_sweep.py --resume \"{root}\"")
        return 130
    print(
        f"Finished: {result['completed']} completed, {result['failed']} failed. "
        f"Analysis: {root / 'analysis'}"
    )
    return 0 if result["failed"] == 0 else 1

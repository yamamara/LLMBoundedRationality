from __future__ import annotations

import csv
import hashlib
import json
import re
import zlib
from pathlib import Path
from typing import Any

from sandbox.cournot_prompts import (
    DEFAULT_COURNOT_AGENT_PROMPT,
    DEFAULT_COURNOT_SYSTEM_PROMPT,
)


PROVENANCE_SCHEMA_VERSION = 1
HASH_ALGORITHM = "sha256"
SHORT_HASH_LENGTH = 16
PARAMETER_CODE_MAGIC = "271828"
PARAMETER_CODE_VERSION = 1
_PARAMETER_CODE_HEADER_LENGTH = 24
_PARAMETER_CODE_CHECKSUM_LENGTH = 10

_COMPACT_STRINGS = {
    DEFAULT_COURNOT_SYSTEM_PROMPT: "@cournot-default-system-v1",
    DEFAULT_COURNOT_AGENT_PROMPT: "@cournot-default-agent-v1",
}
_EXPANDED_STRINGS = {value: key for key, value in _COMPACT_STRINGS.items()}

_SENSITIVE_KEYS = {
    "api_key",
    "api_token",
    "authorization",
    "access_token",
    "password",
    "secret",
}
_EXECUTION_ONLY_KEYS = {"job_id"}


def sanitize_parameters(value: Any) -> Any:
    """Return JSON-safe parameters without credentials or per-execution identifiers."""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _EXECUTION_ONLY_KEYS:
                continue
            if normalized in _SENSITIVE_KEYS:
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = sanitize_parameters(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [sanitize_parameters(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def experiment_parameter_spec(
    *,
    game_name: str,
    game_type: str,
    game_parameters: dict[str, Any],
    prompts: dict[str, Any],
    participants: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the complete, canonicalizable settings used by an experiment."""
    return sanitize_parameters(
        {
            "game_name": game_name,
            "game_type": game_type,
            "game_parameters": game_parameters,
            "prompts": prompts,
            "players": participants,
        }
    )


def parameter_hash(parameters: Any) -> str:
    canonical = json.dumps(
        sanitize_parameters(parameters),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def encode_parameter_code(parameters: Any) -> str:
    """Encode exact sanitized parameters as a self-contained decimal-only string."""
    compact = _replace_strings(sanitize_parameters(parameters), _COMPACT_STRINGS)
    canonical = json.dumps(
        compact,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload = zlib.compress(canonical, level=9)
    decimal_payload = "".join(f"{byte:03d}" for byte in payload)
    checksum = zlib.crc32(payload)
    return (
        PARAMETER_CODE_MAGIC
        + f"{PARAMETER_CODE_VERSION:02d}"
        + f"{len(payload):08d}"
        + f"{len(decimal_payload):08d}"
        + decimal_payload
        + f"{checksum:010d}"
    )


def decode_parameter_code(code: str) -> Any:
    """Decode a decimal experiment code back into its complete parameter JSON."""
    code = re.sub(r"\s+", "", code)
    if not code.isdecimal():
        raise ValueError("Experiment code must contain decimal digits only")
    if len(code) < _PARAMETER_CODE_HEADER_LENGTH + _PARAMETER_CODE_CHECKSUM_LENGTH:
        raise ValueError("Experiment code is truncated")
    if not code.startswith(PARAMETER_CODE_MAGIC):
        raise ValueError("Experiment code has an unknown format")
    version = int(code[6:8])
    if version != PARAMETER_CODE_VERSION:
        raise ValueError(f"Unsupported experiment code version: {version}")
    byte_length = int(code[8:16])
    decimal_length = int(code[16:24])
    expected_length = (
        _PARAMETER_CODE_HEADER_LENGTH
        + decimal_length
        + _PARAMETER_CODE_CHECKSUM_LENGTH
    )
    if len(code) != expected_length:
        raise ValueError("Experiment code length does not match its header")
    decimal_payload = code[24 : 24 + decimal_length]
    checksum = int(code[-_PARAMETER_CODE_CHECKSUM_LENGTH:])
    if decimal_length != byte_length * 3:
        raise ValueError("Experiment code payload length is invalid")
    byte_values = [
        int(decimal_payload[index : index + 3])
        for index in range(0, decimal_length, 3)
    ]
    if any(value > 255 for value in byte_values):
        raise ValueError("Experiment code contains an invalid byte")
    payload = bytes(byte_values)
    if zlib.crc32(payload) != checksum:
        raise ValueError("Experiment code checksum does not match")
    try:
        compact = json.loads(zlib.decompress(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, zlib.error) as exc:
        raise ValueError("Experiment code payload cannot be decoded") from exc
    return _replace_strings(compact, _EXPANDED_STRINGS)


def short_parameter_hash(value: str) -> str:
    return value[:SHORT_HASH_LENGTH]


def run_parameter_record(run_dir: Path) -> dict[str, Any]:
    """Resolve a run's hash and settings, including a fallback for older runs."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    parameters = summary.get("parameters")
    if not isinstance(parameters, dict):
        game_type = summary.get("game_type", "cournot")
        parameters = experiment_parameter_spec(
            game_name=summary.get("game_name", "unnamed-game"),
            game_type=game_type,
            game_parameters=summary.get(game_type, {}),
            prompts=summary.get("prompts", {}),
            participants=summary.get("participants") or _players_from_csv(run_dir),
        )
    resolved_hash = parameter_hash(parameters)
    return {
        "run_id": summary.get("run_id", run_dir.name),
        "parameter_hash": resolved_hash,
        "parameters": parameters,
    }


def visualization_provenance(run_dirs: list[Path]) -> dict[str, Any]:
    experiments = sorted(
        (run_parameter_record(run_dir) for run_dir in run_dirs),
        key=lambda item: item["parameter_hash"],
    )
    if len(experiments) == 1:
        image_hash = experiments[0]["parameter_hash"]
    else:
        image_hash = parameter_hash(
            {
                "experiment_parameter_hashes": [
                    item["parameter_hash"] for item in experiments
                ]
            }
        )
    short_hash = short_parameter_hash(image_hash)
    parameter_code = encode_parameter_code(
        {"experiments": [item["parameters"] for item in experiments]}
    )
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "hash_algorithm": HASH_ALGORITHM,
        "parameter_hash": image_hash,
        "short_parameter_hash": short_hash,
        "parameter_code": parameter_code,
        "parameter_code_version": PARAMETER_CODE_VERSION,
        "hash_basis": (
            "Canonical JSON of experiment-wide and per-player parameters; "
            "outcomes, timestamps, run IDs, credentials, and execution paths are excluded."
        ),
        "images": {
            "two_standard_deviations": f"cournot_player_scores_{short_hash}.svg",
            "student_t_95_ci": f"cournot_player_scores_ci95_{short_hash}.svg",
        },
        "experiments": experiments,
    }


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _players_from_csv(run_dir: Path) -> list[dict[str, Any]]:
    """Recover the player identity available in pre-provenance result files."""
    path = run_dir / "participant_data.csv"
    if not path.is_file():
        return []
    players: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            player_id = row.get("player_id", "")
            if player_id and player_id not in players:
                players[player_id] = {
                    "player_id": player_id,
                    "agent_type": row.get("agent_type", ""),
                    "agent": row.get("agent", ""),
                }
    return [players[player_id] for player_id in sorted(players)]

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


# If json.dump fails to serialize this function is used as backup
def build_default_json(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class OutputWriter:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = run_dir / "events.jsonl"

    def write_json(self, name: str, value: Any) -> None:
        with (self.run_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=build_default_json)
            handle.write("\n")

    def log_event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=build_default_json))
            handle.write("\n")

    def write_csv(self, name: str, rows: list[dict[str, Any]]) -> None:
        self.write_csv_path(self.run_dir / name, rows)

    @staticmethod
    def write_csv_path(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = [
            {
                key: (
                    json.dumps(value, sort_keys=True, default=build_default_json)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        ]
        fieldnames = list(dict.fromkeys(key for row in normalized for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized)

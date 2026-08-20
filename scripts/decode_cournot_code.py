from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sandbox.provenance import decode_parameter_code


def code_from_argument(value: str) -> str:
    path = Path(value)
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if not is_file:
        return value
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".svg":
        match = re.search(r'data-parameter-code="([0-9]+)"', text)
        if not match:
            raise ValueError(f"No experiment code found in SVG: {path}")
        return match.group(1)
    if path.suffix.lower() == ".json":
        value = json.loads(text).get("parameter_code")
        if not value:
            raise ValueError(f"No experiment code found in JSON: {path}")
        return value
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decode a Cournot graph's numeric experiment code."
    )
    parser.add_argument(
        "code_or_file",
        help="Numeric code, SVG graph path, provenance JSON path, or text-file path",
    )
    args = parser.parse_args()
    try:
        parameters = decode_parameter_code(code_from_argument(args.code_or_file))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(parameters, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

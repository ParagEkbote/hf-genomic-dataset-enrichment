"""Validate that all YAML/JSON configuration files parse and are non-empty."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parents[1]

YAML_GLOBS = [
    (PACKAGE_DIR / "config", "*.y*ml"),
    (PACKAGE_DIR / "carbon-catalog", "*.y*ml"),
]
JSON_FILES = [PACKAGE_DIR / "carbon-catalog" / "provenance.json"]


def _check_yaml(path: Path) -> str | None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return f"{path}: invalid YAML ({exc})"
    if not isinstance(data, dict) or not data:
        return f"{path}: expected a non-empty mapping at the top level"
    return None


def _check_json(path: Path) -> str | None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{path}: invalid JSON ({exc})"
    return None


def main() -> int:
    errors: list[str] = []
    checked = 0

    for directory, pattern in YAML_GLOBS:
        files = sorted(directory.glob(pattern))
        if not files:
            errors.append(f"{directory}: no files matching {pattern}")
        for path in files:
            checked += 1
            if err := _check_yaml(path):
                errors.append(err)

    for path in JSON_FILES:
        checked += 1
        if not path.exists():
            errors.append(f"{path}: missing")
        elif err := _check_json(path):
            errors.append(err)

    if errors:
        print("Configuration validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"Validated {checked} configuration files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
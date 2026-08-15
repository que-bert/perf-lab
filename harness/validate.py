#!/usr/bin/env python3
"""Validate a perf-lab ledger against results/schema.json.

Exit 0 if every row is valid, 1 otherwise, printing `line N: <reason>` per offender.
Reads the whole file but reports every failure rather than stopping at the first,
because a broken backfill will usually break the same way many times over.
"""
import json
import pathlib
import sys

try:
    from jsonschema import Draft202012Validator
except ImportError:
    sys.exit("need jsonschema: pip install --user jsonschema")

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "results" / "schema.json"


def describe(err):
    """Short, human-readable path + message."""
    where = ".".join(str(p) for p in err.absolute_path) or "<row>"
    return f"{where}: {err.message}"


def main(argv):
    ledger = pathlib.Path(argv[1]) if len(argv) > 1 else ROOT / "results" / "ledger.jsonl"
    if not SCHEMA.exists():
        sys.exit(f"schema not found: {SCHEMA}")
    validator = Draft202012Validator(json.loads(SCHEMA.read_text()))

    if not ledger.exists():
        print(f"{ledger}: no ledger yet (0 rows)")
        return 0

    total = bad = 0
    for n, raw in enumerate(ledger.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        total += 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"line {n}: not valid JSON ({e.msg})")
            bad += 1
            continue
        errors = sorted(validator.iter_errors(row), key=lambda e: list(e.absolute_path))
        for err in errors:
            print(f"line {n}: {describe(err)}")
        if errors:
            bad += 1

    print(f"{ledger}: {total} rows, {bad} invalid")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""Collate model_eval.py reports into one table.

Fifteen JSON files -- five models times three prompt modes -- is not a result
anyone can read. This prints the grid, and it prints the prompt mode as a row
rather than folding it away, because the mode moved scores further than the
choice of model did.

  eval_table.py results/eval-*-20260910.json
  eval_table.py --by-task results/eval-ornith*.json
"""
import argparse
import collections
import json
import pathlib
import sys

SUITES = ["code", "math", "instruct", "extract", "research", "recall"]
MODES = ["raw", "chat", "chat-nothink"]


def load(paths):
    out = []
    for p in paths:
        try:
            d = json.loads(pathlib.Path(p).read_text())
        except Exception as e:
            print(f"{p}: {e}", file=sys.stderr)
            continue
        stem = pathlib.Path(p).stem
        mode = d.get("prompt_mode", "?")
        if not d.get("thinking", True):
            mode = "chat-nothink"
        # Older reports predate the mode fields; the filename still carries it.
        for m in MODES:
            if stem.endswith(f"-{m}-" + stem.rsplit("-", 1)[-1]):
                mode = m
        d["_mode"] = mode
        d["_label"] = d.get("label") or d.get("model", stem)
        d["_path"] = p
        out.append(d)
    return out


def cell(sc):
    if not sc or not sc.get("of"):
        return "  -  "
    s = f"{sc['pass']}/{sc['of']}"
    return f"{s:>5}" + ("*" if sc.get("skipped") else " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by-task", action="store_true",
                    help="per-task pass/fail instead of suite totals")
    a = ap.parse_args()
    reports = load(a.files)
    if not reports:
        sys.exit("nothing to read")

    if a.by_task:
        for d in sorted(reports, key=lambda r: (r["_label"], r["_mode"])):
            print(f"\n=== {d['_label']}  [{d['_mode']}, {d.get('api','?')}]")
            for suite, rows in d.get("results", {}).items():
                for r in rows:
                    v = ("TRUNC" if r.get("passes") is None
                         else ("PASS" if r["passes"] else "FAIL"))
                    extra = ""
                    if r.get("anywhere") and r.get("passes") is not True:
                        extra = "  (right answer was inside the reasoning)"
                    print(f"  {suite:<9} {r.get('task',''):<20} {v}{extra}")
        return

    by = collections.defaultdict(dict)
    for d in reports:
        by[d["_label"]][d["_mode"]] = d

    head = (f"{'model':<20} {'mode':<22}" + "".join(f"{s:>10}" for s in SUITES)
            + f"{'total':>12}")
    print(head)
    print("-" * len(head))
    for label in sorted(by):
        for mode in MODES:
            d = by[label].get(mode)
            if not d:
                continue
            sc = d.get("score", {})
            tp = sum(sc.get(s, {}).get("pass", 0) for s in SUITES)
            to = sum(sc.get(s, {}).get("of", 0) for s in SUITES)
            row = "".join(cell(sc.get(s)) for s in SUITES)
            api = "" if d.get("api") == "llama" else f" ({d.get('api')})"
            print(f"{label:<20} {mode + api:<22}{row}{f'{tp}/{to}':>12}")
    print("\n  * a suite with unscoreable items (answer cut off at the budget)")
    print("  totals are comparable across MODES within a model; across models")
    print("  only where the api column matches.")


if __name__ == "__main__":
    main()

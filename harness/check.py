#!/usr/bin/env python3
"""Read the ledger back and say whether each canary still sits in its band.

This is the half of the tripwire that was missing: stage 1 could measure, but
nothing ever read the measurements. Verdicts are emitted as data and the exit
status is always 0 -- alert.py decides what deserves an Issue. A checker that
signalled through its exit code would make "no measurements yet" and "the
kernel regressed" indistinguishable to a shell.

Two rules earn their keep:

  A breach must be *sustained* -- the same direction two runs running. Single
  breaches are noise: at the spread this stack shows, a one-run rule fires on
  chance several times a month.

  Fewer than two comparable runs is "insufficient", never "ok". A fresh ledger
  must not read as a healthy one.

Every verdict carries `as_of` and `runs_since`, because a verdict is only as
current as the newest run that actually contained that canary. On 2026-08-22
the busy-card guard skipped four of five, and this file reported all five "ok"
with no hint that four of those numbers were three days old.

  check.py [--ledger L] [--config C]
"""
import argparse
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Rows that represent a real measurement taken by the harness. backfill rows
# are historical log-scrapes with no batch tag, and skipped rows record an
# absence -- neither can be compared against a band.
LIVE_KINDS = {"nightly", "manual", "apt"}


def load_runs(ledger):
    """-> [(tag, earliest_ts, {key: [(pp2048, run_id), ...]})] oldest first.

    Grouped by tag because run_id is unique per row: it identifies a rep, not a
    batch. Untagged rows cannot be attributed to a run and are dropped.
    """
    runs = {}
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tag, key = row.get("tag"), row.get("key")
        if not tag or not key or row.get("kind") not in LIVE_KINDS:
            continue
        m = row.get("m", {})
        if m.get("pp2048") is None or m.get("cold_prefill") is False:
            continue
        run = runs.setdefault(tag, {"ts": row["ts"], "keys": {}})
        run["ts"] = min(run["ts"], row["ts"])
        run["keys"].setdefault(key, []).append((m["pp2048"], row["run_id"]))
    return [(tag, r["ts"], r["keys"])
            for tag, r in sorted(runs.items(), key=lambda kv: kv[1]["ts"])]


def direction(median, expect):
    return "slow" if median < expect else "fast"


def evaluate(runs, bands):
    verdicts = []
    for key, band in bands.items():
        expect = band["expect_pp2048"]
        tol = band.get("tolerance_pct", 10)
        seen = [(i, tag, ts, statistics.median([v for v, _ in ks[key]]),
                 [rid for _, rid in ks[key]])
                for i, (tag, ts, ks) in enumerate(runs) if key in ks]

        if not seen:
            verdicts.append({"key": key, "median_pp2048": None, "expect": expect,
                             "delta_pct": None, "breach": False,
                             "sustained": False, "verdict": "insufficient",
                             "as_of": None, "runs_since": len(runs),
                             "run_ids": []})
            continue

        idx, tag, as_of, med, run_ids = seen[-1]
        # How many whole runs have happened since this canary last produced a
        # number. A verdict says nothing about the runs it was absent from, and
        # on 2026-08-22 four canaries skipped on the busy-card guard while this
        # file went on reporting them "ok" from three-day-old data.
        runs_since = len(runs) - 1 - idx
        delta = 100.0 * (med - expect) / expect
        breach = abs(delta) > tol
        dirn = direction(med, expect)

        sustained = False
        if breach and len(seen) >= 2:
            prev_med = seen[-2][3]
            prev_delta = 100.0 * (prev_med - expect) / expect
            sustained = (abs(prev_delta) > tol
                         and direction(prev_med, expect) == dirn)

        if len(seen) < 2:
            verdict = "insufficient"
        else:
            verdict = dirn if breach else "ok"

        verdicts.append({
            "key": key, "median_pp2048": round(med, 2), "expect": expect,
            "delta_pct": round(delta, 2), "breach": breach,
            "sustained": sustained, "verdict": verdict,
            "as_of": as_of, "runs_since": runs_since, "run_ids": run_ids,
        })
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=str(ROOT / "results" / "ledger.jsonl"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "canary.yaml"))
    a = ap.parse_args()

    import yaml
    doc = yaml.safe_load(open(a.config))
    bands = {k: v for k, v in doc["canaries"].items() if "expect_pp2048" in v}
    runs = load_runs(pathlib.Path(a.ledger))
    verdicts = evaluate(runs, bands)

    json.dump({"runs": [{"tag": t, "ts": ts, "keys": sorted(ks)}
                        for t, ts, ks in runs],
               "verdicts": verdicts}, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()

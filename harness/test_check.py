#!/usr/bin/env python3
"""Acceptance cases for harness/check.py. Run via: make test

Every case is a synthetic ledger, because the behaviour worth pinning down is
what happens on the nights that have not occurred yet: the one-run blip, the
regression that persists, the canary that gets unexpectedly fast.
"""
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check", HERE / "check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

BANDS = {
    "fast-q4":   {"expect_pp2048": 710.84, "tolerance_pct": 10},
    "slow-q5_1": {"expect_pp2048": 54.96,  "tolerance_pct": 10},
}


def row(tag, key, pp, ts, rep=1, kind="nightly", cold=True):
    return {"ts": ts, "run_id": f"r-{abs(hash((tag, key, rep))):08x}"[:10],
            "kind": kind, "tag": tag, "key": key,
            "m": {"pp2048": pp, "rep": rep, "cold_prefill": cold}}


def verdict_for(rows, key):
    p = pathlib.Path(".scratch/t-check.jsonl")
    p.parent.mkdir(exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    runs = check.load_runs(p)
    return next(v for v in check.evaluate(runs, BANDS) if v["key"] == key)


def ok(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return 0 if cond else 1


fails = 0
N1, N2, N3 = "n-001", "n-002", "n-003"
T1, T2, T3 = "2026-08-16T06:00:00Z", "2026-08-17T06:00:00Z", "2026-08-18T06:00:00Z"

# Healthy two runs.
v = verdict_for([row(N1, "fast-q4", 710, T1), row(N2, "fast-q4", 705, T2)], "fast-q4")
fails += ok("two healthy runs -> ok", v["verdict"] == "ok" and not v["breach"])

# A single run cannot confirm anything, even when it looks fine.
v = verdict_for([row(N1, "fast-q4", 710, T1)], "fast-q4")
fails += ok("one healthy run -> insufficient, not ok", v["verdict"] == "insufficient")

# One-run breach: real, but not yet actionable.
v = verdict_for([row(N1, "fast-q4", 710, T1), row(N2, "fast-q4", 600, T2)], "fast-q4")
fails += ok("one-run breach -> breach, not sustained",
            v["breach"] and not v["sustained"] and v["verdict"] == "slow")

# Two runs the same way: this is what deserves an Issue.
v = verdict_for([row(N1, "fast-q4", 600, T1), row(N2, "fast-q4", 595, T2)], "fast-q4")
fails += ok("two-run same-direction breach -> sustained", v["sustained"])

# Opposite directions are not a trend, they are noise straddling the band.
v = verdict_for([row(N1, "fast-q4", 600, T1), row(N2, "fast-q4", 850, T2)], "fast-q4")
fails += ok("slow then fast -> not sustained",
            v["breach"] and not v["sustained"] and v["verdict"] == "fast")

# The good news case: a known-unsupported config gaining kernel support.
v = verdict_for([row(N1, "slow-q5_1", 700, T1), row(N2, "slow-q5_1", 705, T2)],
                "slow-q5_1")
fails += ok("slow-q5_1 at fast-path speed -> fast, sustained",
            v["verdict"] == "fast" and v["sustained"])

# Reps aggregate by median, so one stalled rep in the run being judged must not
# drag it into breach the way a mean would (mean here is 503 -- a false alarm).
v = verdict_for([row(N1, "fast-q4", 710, T1, rep=1), row(N1, "fast-q4", 705, T1, rep=2),
                 row(N1, "fast-q4", 708, T1, rep=3),
                 row(N2, "fast-q4", 708, T2, rep=1), row(N2, "fast-q4", 712, T2, rep=2),
                 row(N2, "fast-q4", 90, T2, rep=3)], "fast-q4")
fails += ok("one stalled rep does not breach (median)",
            not v["breach"] and v["median_pp2048"] == 708.0)

# Rows that cannot be attributed to a run must not be evaluated.
v = verdict_for([row(N1, "fast-q4", 710, T1), row(N2, "fast-q4", 705, T2),
                 {**row(N3, "fast-q4", 100, T3), "tag": None}], "fast-q4")
fails += ok("untagged row ignored", v["verdict"] == "ok")

# backfill and skipped rows are not measurements of a band.
v = verdict_for([row(N1, "fast-q4", 710, T1), row(N2, "fast-q4", 705, T2),
                 row(N3, "fast-q4", 100, T3, kind="backfill"),
                 row(N3, "fast-q4", 100, T3, kind="skipped")], "fast-q4")
fails += ok("backfill/skipped rows ignored", v["verdict"] == "ok")

# A warm-cache prefill is not comparable to a cold one.
v = verdict_for([row(N1, "fast-q4", 710, T1), row(N2, "fast-q4", 705, T2),
                 row(N3, "fast-q4", 5000, T3, cold=False)], "fast-q4")
fails += ok("cold_prefill:false excluded", v["verdict"] == "ok")

# Never measured at all is not healthy.
v = verdict_for([row(N1, "fast-q4", 710, T1)], "slow-q5_1")
fails += ok("never-measured canary -> insufficient",
            v["verdict"] == "insufficient" and v["median_pp2048"] is None)

sys.exit(1 if fails else 0)

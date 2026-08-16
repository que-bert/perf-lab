#!/usr/bin/env python3
"""Re-derive canary bands from measured spread instead of a single reading.

The bands shipped in stage 1 were n=1 point estimates: one measurement became
`expect_pp2048`, and `tolerance_pct` was a flat 10 guessed alongside it. Against
the ~7% run-to-run spread this stack actually shows, that is roughly a 2-sigma
band on five canaries every night, which is an alert stream nobody reads.

This runs each canary enough times to see its spread, then sizes the band from
what it saw and reports how often that band should fire on noise alone. It
drives bench.sh rather than calling llama-bench directly, so the rows it
produces are ordinary ledger rows and go through the same validation.

  rebaseline.py --reps 5              measure and print a YAML block to review
  rebaseline.py --reps 5 --nightly-reps 3   size the projection for a 3-rep nightly
"""
import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# sd of a sample median vs sd of the mean, for a normal parent. The nightly
# aggregates by median (robust to a single stalled rep), which costs ~25% more
# spread than the mean would -- the projection has to pay that, not ignore it.
MEDIAN_SD_FACTOR = 1.2533

# MAD -> sd for a normal parent. The centre is a median precisely because reps
# get contaminated (a stalled rep, another process touching the GPU); sizing
# the band with a non-robust sd alongside it is inconsistent, and lets one bad
# rep inflate a band until it can no longer catch anything. Measured here:
# slow-q4_1's raw sd was 15.3% off one outlier, against a robust 0.9%.
MAD_SCALE = 1.4826


def robust_sd(vals):
    med = statistics.median(vals)
    return MAD_SCALE * statistics.median([abs(v - med) for v in vals])


def measure(config, key, reps, tag, ledger):
    """Run one canary `reps` times through bench.sh. Rows land in the ledger."""
    cmd = [str(HERE / "bench.sh"), str(config), key,
           "--reps", str(reps), "--tag", tag, "--kind", "manual"]
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        sys.exit(f"rebaseline: bench.sh failed on {key} (exit {r.returncode})")


def rows_for(ledger, tag, key):
    out = []
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("tag") == tag and row.get("key") == key:
            pp = row.get("m", {}).get("pp2048")
            if pp is not None:
                out.append(pp)
    return out


def false_alarms_per_month(tolerance_pct, sd_pct, nightly_reps):
    """Nights per 30 on which noise alone should trip a *sustained* alert.

    check.py requires two consecutive runs breaching in the same direction, so
    a single night's two-sided probability p gives p_sustained = (p/2)^2 * 2.
    """
    sd_of_median = MEDIAN_SD_FACTOR * sd_pct / math.sqrt(nightly_reps)
    if sd_of_median == 0:
        return 0.0
    z = tolerance_pct / sd_of_median
    p_run = math.erfc(z / math.sqrt(2))          # two-sided
    return 30.0 * (p_run ** 2) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "canary.yaml"))
    ap.add_argument("--ledger", default=str(ROOT / "results" / "ledger.jsonl"))
    ap.add_argument("--reps", type=int, default=5,
                    help="reps per canary for this calibration (>=5)")
    ap.add_argument("--nightly-reps", type=int, default=3,
                    help="reps the nightly will use; sizes the projection")
    ap.add_argument("--only", default="", help="calibrate one canary key")
    ap.add_argument("--from-tag", default="",
                    help="recompute bands from rows already in the ledger "
                         "instead of measuring again")
    a = ap.parse_args()

    if a.from_tag:
        pass
    elif a.reps < 5:
        sys.exit("rebaseline: --reps must be >=5; a band derived from fewer "
                 "readings is the n=1 guess this tool exists to replace")

    import yaml
    doc = yaml.safe_load(open(a.config))
    keys = [a.only] if a.only else list(doc["canaries"])
    unknown = [k for k in keys if k not in doc["canaries"]]
    if unknown:
        sys.exit(f"rebaseline: no such canary: {', '.join(unknown)}")

    ledger = pathlib.Path(a.ledger)
    tag = a.from_tag or ("rebase-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    if a.from_tag:
        print(f"# recomputing from existing rows under tag {tag}\n", file=sys.stderr)
    else:
        print(f"# tag {tag} — {a.reps} reps x {len(keys)} canaries\n", file=sys.stderr)

    results = {}
    for key in keys:
        if not a.from_tag:
            print(f"--- {key} ---", file=sys.stderr)
            measure(a.config, key, a.reps, tag, ledger)
        vals = rows_for(ledger, tag, key)
        if len(vals) < 5:
            sys.exit(f"rebaseline: {key} has {len(vals)} usable rows under "
                     f"{tag}, need >=5 — refusing to derive a band from a "
                     "partial run")
        med = statistics.median(vals)
        sd_raw = 100.0 * statistics.stdev(vals) / med
        sd_rob = 100.0 * robust_sd(vals) / med
        tol = max(10.0, 3.0 * sd_rob)
        results[key] = {
            "expect_pp2048": round(med, 2),
            "sd_pct": round(sd_rob, 2),
            "sd_raw_pct": round(sd_raw, 2),
            "tolerance_pct": round(tol, 1),
            "alerts_per_month": round(
                false_alarms_per_month(tol, sd_rob, a.nightly_reps), 3),
            "n": len(vals),
            # A raw sd far above the robust one means a rep was contaminated,
            # not that the config is noisy. Worth explaining, not averaging away.
            "outlier": sd_raw > 3 * max(sd_rob, 0.01),
        }

    print("# Review, then paste into configs/canary.yaml. Bands from "
          f"{a.reps} reps; projection assumes a {a.nightly_reps}-rep nightly.")
    print(f"# Measured under tag {tag}.")
    for key, r in results.items():
        print(f"\n  {key}:")
        print(f"    expect_pp2048: {r['expect_pp2048']}")
        print(f"    sd_pct: {r['sd_pct']}")
        print(f"    tolerance_pct: {r['tolerance_pct']}")
        print(f"    # n={r['n']}, projected false alerts/month: "
              f"{r['alerts_per_month']}")
        if r["outlier"]:
            print(f"    # raw sd {r['sd_raw_pct']}% vs robust {r['sd_pct']}% — "
                  "a rep was contaminated; band sized from the robust spread")

    total = sum(r["alerts_per_month"] for r in results.values())
    print(f"\n# Projected false alerts across all canaries: {total:.3f}/month",
          file=sys.stderr)
    if total > 1.0:
        print("# WARNING: over one false alert a month. Widen the bands or "
              "raise --nightly-reps before shipping alerting.", file=sys.stderr)


if __name__ == "__main__":
    main()

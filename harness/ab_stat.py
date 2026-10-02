#!/usr/bin/env python3
"""Plan-2 A/B statistic (docs/2026-10-01-phase2-plan.md, "Invariants").

  ab_stat.py diff  A1,A2,... B1,B2,...    paired (ABAB order) values, higher is better
  ab_stat.py k     CV_PCT DELTA_PCT       pairs needed to resolve DELTA with run-to-run CV
  ab_stat.py cv    X1,X2,...              coefficient of variation (%) of repeated runs

diff prints the mean relative difference (B vs A, %), its one-sided 90% bound (t distribution over the
pair differences) and the verdict: LOSS only if the upper bound is below zero (B slower beyond noise),
GAIN if the lower bound is above zero, else NOISE. A LOSS still needs a confirming re-run.
"""
import math, statistics as st, sys

T90 = {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476, 6: 1.440, 7: 1.415, 8: 1.397, 9: 1.383,
       10: 1.372, 12: 1.356, 15: 1.341, 20: 1.325, 30: 1.310}


def t90(df):
    if df <= 0:
        return float("inf")
    keys = sorted(T90)
    for k in keys:
        if df <= k:
            return T90[k]
    return 1.282


def nums(s):
    return [float(x) for x in s.split(",") if x]


def diff(a, b):
    n = min(len(a), len(b))
    d = [100.0 * (b[i] - a[i]) / a[i] for i in range(n)]
    m = st.mean(d)
    half = t90(n - 1) * (st.stdev(d) / math.sqrt(n)) if n > 1 else float("inf")
    v = "LOSS" if m + half < 0 else "GAIN" if m - half > 0 else "NOISE"
    return m, half, v, n


def main():
    c = sys.argv[1]
    if c == "diff":
        m, h, v, n = diff(nums(sys.argv[2]), nums(sys.argv[3]))
        print(f"{v} mean {m:+.2f}% bound90 [{m - h:+.2f}%, {m + h:+.2f}%] k={n}")
    elif c == "k":
        cv, delta = float(sys.argv[2]), float(sys.argv[3])
        sd_d = math.sqrt(2) * cv
        k = 2
        while k < 50 and t90(k - 1) * sd_d / math.sqrt(k) >= delta:
            k += 1
        print(k)
    elif c == "cv":
        x = nums(sys.argv[2])
        print(f"{100 * st.stdev(x) / st.mean(x):.3f}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

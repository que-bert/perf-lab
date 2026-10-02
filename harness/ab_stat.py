#!/usr/bin/env python3
"""Plan-2 A/B statistic (docs/2026-10-01-phase2-plan.md, "Invariants").

  ab_stat.py diff  A1,A2,... B1,B2,...    paired (ABAB order) values, higher is better
  ab_stat.py k     CV_PCT DELTA_PCT       pairs needed to resolve DELTA with run-to-run CV
  ab_stat.py cv    X1,X2,...              coefficient of variation (%) of repeated runs
  ab_stat.py decide TOL_PCT KMIN KMAX AXIS=A1,A2,..:B1,B2,.. [AXIS=...]
                                          sequential stopping for ABAB loops: exit 0 = stop, 3 = run another pair
  ab_stat.py decide-arms DIR PREFIX TOL_PCT KMIN KMAX
                                          same over arm_session.sh rounds DIR/<PREFIX>-base-rN.arm.json vs -cand-rN
  ab_stat.py decide-jsonl DIR A B TOL_PCT KMIN KMAX
                                          same over model_profile.sh rounds DIR/<model>/<A>-rN.jsonl vs <B>-rN.jsonl

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


def axis_state(a, b, tol):
    """One axis after n pairs: EQUIV (90% bound inside +-tol), GAIN/LOSS (bound excludes zero), OPEN."""
    m, h, v, n = diff(a, b)
    if n < 2:
        return "OPEN", m, h, n
    lo, hi = m - h, m + h
    if lo >= -tol and hi <= tol:
        return "EQUIV", m, h, n
    if lo > 0 or hi < 0:
        return v, m, h, n
    return "OPEN", m, h, n


def decide(tol, kmin, kmax, axes):
    """axes: {name: (A values, B values)}. Stop when n >= kmin and every axis is EQUIV or excludes zero, or n >= kmax."""
    pairs = {k: min(len(a), len(b)) for k, (a, b) in axes.items()}
    n = max(pairs.values(), default=0)
    open_axes = []
    for name, (a, b) in axes.items():
        if pairs[name] < 2 and n >= 2:  # measured once only (e.g. pooled t/s: round 1): reported, not part of the stopping rule
            print(f"{name}: SINGLE k={pairs[name]} (not sequentially tested)")
            continue
        st_, m, h, k = axis_state(a, b, tol)
        print(f"{name}: {st_} mean {m:+.2f}% bound90 [{m - h:+.2f}%, {m + h:+.2f}%] k={k}")
        if st_ == "OPEN":
            open_axes.append(name)
    if n >= kmax:
        print(f"DECISION STOP kmax reached (k={n}); open: {','.join(open_axes) or '-'}")
        return 0
    if n < kmin:
        print(f"DECISION CONTINUE k={n} < kmin={kmin}")
        return 3
    if open_axes:
        print(f"DECISION CONTINUE k={n}; open: {','.join(open_axes)}")
        return 3
    print(f"DECISION STOP all axes decided at k={n} (tol +-{tol}%)")
    return 0


def jsonl_axes(d, A, B):
    import glob, json, os
    axes = {}
    for mdir in sorted(glob.glob(os.path.join(d, "*"))):
        if not os.path.isdir(mdir):
            continue
        r = 1
        while os.path.exists(f"{mdir}/{A}-r{r}.jsonl") and os.path.exists(f"{mdir}/{B}-r{r}.jsonl"):
            for side, pre in ((0, A), (1, B)):
                for l in open(f"{mdir}/{pre}-r{r}.jsonl"):
                    x = json.loads(l)
                    t = f"pp{x['n_prompt']}" if x["n_prompt"] else f"tg{x['n_gen']}"
                    axes.setdefault(f"{os.path.basename(mdir)} {t} d{x['n_depth']}", ([], []))[side].append(x["avg_ts"])
            r += 1
    return axes


def arm_axes(d, prefix):
    import json, os
    axes, r = {}, 1
    while os.path.exists(f"{d}/{prefix}-base-r{r}.arm.json") and os.path.exists(f"{d}/{prefix}-cand-r{r}.arm.json"):
        a, b = (json.load(open(f"{d}/{prefix}-{x}-r{r}.arm.json")) for x in ("base", "cand"))
        for k in a:
            if k != "toks" and k in b:
                axes.setdefault(k, ([], []))[0].append(a[k]); axes[k][1].append(b[k])
        r += 1
    return axes


def main():
    c = sys.argv[1]
    if c == "decide-arms":
        sys.exit(decide(float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), arm_axes(sys.argv[2], sys.argv[3])))
    if c == "decide":
        tol, kmin, kmax = float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        axes = {}
        for t in sys.argv[5:]:
            name, _, v = t.partition("=")
            a, _, b = v.partition(":")
            axes[name] = (nums(a), nums(b))
        sys.exit(decide(tol, kmin, kmax, axes))
    if c == "decide-jsonl":
        sys.exit(decide(float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), jsonl_axes(sys.argv[2], sys.argv[3], sys.argv[4])))
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

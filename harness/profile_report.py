#!/usr/bin/env python3
"""Fork vs upstream table from model_profile.sh output (results/phase2/profile/<model>/<label>.jsonl).

  profile_report.py [A_PREFIX] [B_PREFIX] [--dir D]     default A=upce B=i16: rounds A-r1..rN paired with B-r1..rN

Per model and test (pp/tg x depth): mean t/s of each arm, run-to-run CV of each arm across rounds, and the
ab_stat verdict of B vs A (one-sided 90% bound over the paired rounds). Markdown on stdout.
"""
import glob, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_stat import diff  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
A = args[0] if len(args) > 0 else "upce"
B = args[1] if len(args) > 1 else "i16"
D = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results/phase2/profile")


def load(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        t = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
        out[f"{t} d{r['n_depth']}"] = r["avg_ts"]
    return out


print(f"| model | test | {A} t/s | {B} t/s | CV {A} % | CV {B} % | {B} vs {A} |")
print("|---|---|---:|---:|---:|---:|---|")
for mdir in sorted(glob.glob(os.path.join(D, "*"))):
    m = os.path.basename(mdir)
    ra, rb = {}, {}
    for r in range(1, 20):
        fa, fb = f"{mdir}/{A}-r{r}.jsonl", f"{mdir}/{B}-r{r}.jsonl"
        if os.path.exists(fa) and os.path.exists(fb):
            ra[r], rb[r] = load(fa), load(fb)
    if not ra:
        continue
    for test in ra[min(ra)]:
        xa = [ra[r][test] for r in sorted(ra) if test in ra[r] and test in rb[r]]
        xb = [rb[r][test] for r in sorted(ra) if test in ra[r] and test in rb[r]]
        cv = lambda x: f"{100 * st.stdev(x) / st.mean(x):.2f}" if len(x) > 1 else "-"
        if len(xa) > 1:
            mm, h, v, n = diff(xa, xb)
            verdict = f"{v} {mm:+.1f}% [{mm - h:+.1f},{mm + h:+.1f}] k={n}"
        else:
            verdict = f"{100 * (xb[0] - xa[0]) / xa[0]:+.1f}% (k=1)"
        print(f"| {m} | {test} | {st.mean(xa):.1f} | {st.mean(xb):.1f} | {cv(xa)} | {cv(xb)} | {verdict} |")

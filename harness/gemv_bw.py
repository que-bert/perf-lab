#!/usr/bin/env python3
"""Decode GEMV bandwidth per quant from a GGML_VK_PERF_LOGGER log (last "Vulkan Timings" block).

  gemv_bw.py <perflog.txt> [...]

Per MUL_MAT_VEC line (n=1): weight bytes = m*k*bpw/8, GB/s = bytes / time. MUL_MAT_ID lines count the 8 used
experts' rows (m*k per expert x n_used, read from the log shape when present). Prints per line and a per-quant
time-weighted mean, so Q4_K/Q5_K can be compared with the Q6_K path (W3 skip rule: within 10%).
"""
import re, sys
from collections import defaultdict

BPW = {"q4_0": 4.5, "q4_1": 5.0, "q5_0": 5.5, "q5_1": 6.0, "q8_0": 8.5, "q4_K": 4.5, "q5_K": 5.5, "q6_K": 6.5625,
       "q2_K": 2.625, "q3_K": 3.4375, "f16": 16, "bf16": 16, "f32": 32, "iq4_xs": 4.25, "iq4_nl": 4.5}
LINE = re.compile(r"^(?P<name>[A-Z_ ]*MUL_MAT(?:_ID)?(?:_VEC)?) (?P<q>\w+) m=(?P<m>\d+) n=(?P<n>\d+) k=(?P<k>\d+)"
                  r"(?: n_expert_used=(?P<e>\d+))?.*?: (?P<cnt>\d+) x (?P<us>[\d.]+) us")

for path in sys.argv[1:]:
    blk = open(path).read().split("Vulkan Timings:")[-1].splitlines()
    agg = defaultdict(lambda: [0.0, 0.0])
    print(f"## {path}")
    for l in blk:
        m = LINE.search(l)
        if not m or m["q"] not in BPW or int(m["n"]) > 8:
            continue
        mm, kk, us = int(m["m"]), int(m["k"]), float(m["us"])
        rows = mm * (int(m["e"]) if m["e"] else (int(m["n"]) if "_ID" in m["name"] else 1))
        gb = rows * kk * BPW[m["q"]] / 8 / 1e9
        bw = gb / (us * 1e-6)
        agg[m["q"]][0] += gb * int(m["cnt"])
        agg[m["q"]][1] += us * 1e-6 * int(m["cnt"])
        print(f"  {m['name'].strip():<24} {m['q']:<5} m={mm:<7} n={m['n']} k={kk:<6} {us:8.2f} us {bw:7.1f} GB/s")
    for q, (gb, s) in sorted(agg.items()):
        print(f"  == {q}: {gb / s:.1f} GB/s (time-weighted, {s * 1e3:.2f} ms/token)")

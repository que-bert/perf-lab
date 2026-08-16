#!/usr/bin/env python3
"""Acceptance cases for harness/sweep.py. Run via: make test

The classification logic is what decides whether a config gets recommended, so
it is tested on synthetic generations rather than only through a 15-minute GPU
run. Ties, material differences and missing distributions each have to come out
distinct: collapsing "unknown" into either of the others is how a search tool
starts laundering ignorance into a recommendation.
"""
import importlib.util
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweep", HERE / "sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def gen(tokens, tops=None):
    """A generation; `tops` maps index -> [(id, prob), ...] as a distribution."""
    probs = []
    for i, _ in enumerate(tokens):
        if tops and i in tops:
            probs.append({"top_logprobs": [{"id": tid, "logprob": math.log(p),
                                            "token": str(tid)}
                                           for tid, p in tops[i]]})
        else:
            probs.append({})
    return {"tokens": tokens, "probs": probs}


def ok(name, cond, got=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  got: {got}"))
    return 0 if cond else 1


fails = 0

# --- axis parsing -----------------------------------------------------------
n, v = sweep.parse_axis("n_max=2,3,4")
fails += ok("parse_axis reads ints", (n, v) == ("n_max", [2, 3, 4]), (n, v))
n, v = sweep.parse_axis("spec=none,draft-mtp")
fails += ok("parse_axis maps none -> None", v == [None, "draft-mtp"], v)

# --- point enumeration ------------------------------------------------------
base = {"ctx": 8192, "ctk": "q4_0", "ctv": "q4_0", "spec": "draft-mtp", "n_max": 4}
pts = sweep.points(base, [("n_max", [2, 4])])
fails += ok("one axis -> one point per value", len(pts) == 2, len(pts))
fails += ok("points override only the swept key",
            pts[0][2]["ctk"] == "q4_0" and pts[0][2]["n_max"] == 2, pts[0][2])
pts = sweep.points(base, [("n_max", [2, 4]), ("spec", ["draft-mtp", None])])
fails += ok("two axes -> cartesian product", len(pts) == 4, len(pts))

# --- classification ---------------------------------------------------------
ref = gen([1, 2, 3, 4], {2: [(3, 0.50), (9, 0.49)]})

same = gen([1, 2, 3, 4])
fails += ok("same tokens -> identical",
            sweep.classify([ref], [same], 1.2)[0] == "identical")

# Diverges onto the runner-up the reference rated almost as likely.
tied = gen([1, 2, 9, 4])
fails += ok("near-tie divergence -> tied",
            sweep.classify([ref], [tied], 1.2)[0] == "tied")

# Same divergence, but the reference clearly preferred its own token.
ref_strong = gen([1, 2, 3, 4], {2: [(3, 0.90), (9, 0.05)]})
fails += ok("clear-preference divergence -> material",
            sweep.classify([ref_strong], [tied], 1.2)[0] == "material")

# Chose something outside the reference's top-k entirely.
outside = gen([1, 2, 77, 4])
fails += ok("token outside top-k -> material",
            sweep.classify([ref], [outside], 1.2)[0] == "material")

# No distribution anywhere: unknown, and NOT quietly called material or tied.
blind = gen([1, 2, 3, 4])
fails += ok("no distribution -> unknown",
            sweep.classify([blind], [gen([1, 2, 9, 4])], 1.2)[0] == "unknown")

# The threshold is what separates speculation (1.00-1.06) from quantization
# (1.70); check it actually moves the verdict.
ref_mid = gen([1, 2, 3, 4], {2: [(3, 0.50), (9, 0.33)]})   # ratio ~1.5
fails += ok("ratio 1.5 is material at tie-ratio 1.2",
            sweep.classify([ref_mid], [tied], 1.2)[0] == "material")
fails += ok("ratio 1.5 is tied at tie-ratio 2.0",
            sweep.classify([ref_mid], [tied], 2.0)[0] == "tied")

# --- lossy axes are refused -------------------------------------------------
fails += ok("ctk/ctv are on the refusal list",
            {"ctk", "ctv"} <= sweep.LOSSY_AXES, sweep.LOSSY_AXES)

sys.exit(1 if fails else 0)

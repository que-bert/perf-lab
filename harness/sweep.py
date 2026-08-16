#!/usr/bin/env python3
"""Search the configuration space for the fastest setting that stays equivalent.

This is the thing the rest of the harness exists to make trustworthy. It varies
one or more axes, measures each point, checks each against the baseline, and
ranks what survived. A human never picks the points.

Two rules keep it honest:

  Every candidate is loaded ONCE and asked for both its throughput and its
  generated tokens. Measuring and verifying separately would double the model
  loads for no information -- and a 21 GiB load is most of the wall clock.

  Lossy axes are refused. Changing ctk/ctv changes the arithmetic, so the
  tie-tolerant check cannot judge it: measured, quantization diverges at a
  probability ratio of 1.70 against speculation's 1.00-1.06. A sweep that
  cannot tell better from merely different must not be allowed to recommend a
  quantization.

  sweep.py --axis n_max=2,3,4,5,6
  sweep.py --axis spec=none,draft-mtp --base baseline --ledger results/ledger.jsonl
"""
import argparse
import copy
import importlib.util
import json
import math
import os
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

_spec = importlib.util.spec_from_file_location("emit_row", HERE / "emit_row.py")
emit_row = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emit_row)

# Axes whose divergence is real rather than a tie-break. Blocked until there is
# a quality benchmark that can say whether "different" is "worse".
LOSSY_AXES = {"ctk", "ctv", "quant"}


def parse_axis(s):
    if "=" not in s:
        sys.exit(f"sweep: --axis wants name=v1,v2 (got {s!r})")
    name, values = s.split("=", 1)
    out = []
    for v in values.split(","):
        v = v.strip()
        if v in ("none", "null", ""):
            out.append(None)
        elif v.lstrip("-").isdigit():
            out.append(int(v))
        else:
            out.append(v)
    return name.strip(), out


def points(base_cfg, axes):
    """Cartesian product over the axes, applied on top of the baseline."""
    combos = [{}]
    for name, values in axes:
        combos = [dict(c, **{name: v}) for c in combos for v in values]
    out = []
    for c in combos:
        cfg = copy.deepcopy(base_cfg)
        cfg.update(c)
        label = ",".join(f"{k}={v}" for k, v in c.items())
        out.append((label, c, cfg))
    return out


def probe_config(bindir, model, cfg, defaults, prompts, n_predict, reps):
    """One model load: throughput reps, then the verification prompts."""
    proc, port, stop = emit_row.start_server(bindir, model, cfg, defaults)
    try:
        timings = []
        for _ in range(reps):
            r = emit_row.complete(port, prompts[0],
                                  cfg.get("n_predict", defaults.get("n_gen", 64)))
            timings.append(r.get("timings", {}))
        gens = []
        for p in prompts:
            r = emit_row.complete(port, p, n_predict,
                                  {"return_tokens": True, "n_probs": 5})
            gens.append({"tokens": r.get("tokens") or [],
                         "probs": r.get("completion_probabilities") or []})
    finally:
        stop()

    def med(key):
        vals = [t.get(key) for t in timings if t.get(key) is not None]
        return statistics.median(vals) if vals else None

    tgs = [t.get("predicted_per_second") for t in timings
           if t.get("predicted_per_second") is not None]
    # Spread on the point itself. Ranking two configs whose gap is smaller than
    # this is ranking noise: an identical config measured twice came out 18%
    # apart at reps=1.
    spread = ((max(tgs) - min(tgs)) / statistics.median(tgs)) if len(tgs) > 1 else None

    drafted = sum(t.get("draft_n") or 0 for t in timings)
    accepted = sum(t.get("draft_n_accepted") or 0 for t in timings)
    return {
        "pp": med("prompt_per_second"),
        "tg": med("predicted_per_second"),
        "tg_spread": spread,
        "acceptance": (accepted / drafted) if drafted else None,
        "prompt_n": timings[0].get("prompt_n") if timings else None,
        "gens": gens,
    }


def classify(ref_gens, cand_gens, tie_ratio):
    """Compare a candidate against the baseline, tolerating tied choices."""
    worst, detail = "identical", []
    for i, (x, y) in enumerate(zip(ref_gens, cand_gens)):
        d = next((j for j, (p, q) in enumerate(zip(x["tokens"], y["tokens"]))
                  if p != q), None)
        if d is None and len(x["tokens"]) == len(y["tokens"]):
            continue
        if d is None:
            worst, _ = "material", detail.append({"prompt": i, "verdict": "material",
                                                  "why": "length"})
            continue
        # The reference side is the one with distributions: llama-server returns
        # no top_logprobs for accepted draft tokens.
        cov = lambda r: sum(1 for e in r["probs"] if e.get("top_logprobs"))
        ref, other = (x, y) if cov(x) >= cov(y) else (y, x)
        top = (ref["probs"][d].get("top_logprobs")
               if d < len(ref["probs"]) else None) or []
        chosen = other["tokens"][d]
        rank = next((r for r, t in enumerate(top) if t.get("id") == chosen), None)
        if not top:
            v, ratio = "unknown", None
        elif rank is None:
            v, ratio = "material", None
        else:
            ratio = math.exp(top[0]["logprob"]) / math.exp(top[rank]["logprob"])
            v = "tied" if ratio <= tie_ratio else "material"
        detail.append({"prompt": i, "at": d, "verdict": v,
                       "prob_ratio": None if ratio is None else round(ratio, 4)})
        order = {"identical": 0, "tied": 1, "unknown": 2, "material": 3}
        if order[v] > order[worst]:
            worst = v
    return worst, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "tracked.yaml"))
    ap.add_argument("--base", default="baseline")
    ap.add_argument("--axis", action="append", default=[], required=True)
    ap.add_argument("--prompts", default=os.environ.get(
        "PERF_LAB_PROMPT", str(pathlib.Path.home() / ".perf-lab/prompts/default.txt")))
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--tie-ratio", type=float, default=1.2)
    ap.add_argument("--ledger", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    axes = [parse_axis(s) for s in a.axis]
    blocked = [n for n, _ in axes if n in LOSSY_AXES]
    if blocked:
        sys.exit(
            f"sweep: refusing to sweep {', '.join(blocked)}. Divergence on a "
            "lossy axis is real, not a tie-break -- q4_0 vs q8_0 KV diverges at "
            "a probability ratio of 1.70 against speculation's 1.00-1.06. "
            "Judging it needs harness/quality.py, which does not exist yet.")

    import yaml
    doc = yaml.safe_load(open(a.config))
    entries = doc.get("tracked") or doc.get("canaries") or {}
    if a.base not in entries:
        sys.exit(f"sweep: no baseline '{a.base}' in {a.config}")
    defaults, base_cfg = doc["defaults"], entries[a.base]

    bindir = pathlib.Path(os.environ.get("PERF_LAB_SERVER_BIN", ""))
    model = pathlib.Path(os.environ.get("PERF_LAB_MODEL", ""))
    if not (bindir / "llama-server").exists():
        sys.exit("sweep: set PERF_LAB_SERVER_BIN to a working llama-server")

    text = pathlib.Path(a.prompts).read_text()
    prompts = [b.strip() for b in text.split("\n\n\n") if b.strip()] or [text]
    tag = a.tag or "sweep-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    plan = points(base_cfg, axes)
    print(f"# {tag}: baseline + {len(plan)} points, {len(prompts)} prompt(s)",
          file=sys.stderr)

    print(f"### baseline ({a.base})", file=sys.stderr)
    base = probe_config(bindir, model, base_cfg, defaults, prompts,
                        a.n_predict, a.reps)

    # Verification reference. It must be non-speculative: llama-server returns
    # no top_logprobs for accepted draft tokens, so comparing two speculative
    # configs leaves nothing to judge a divergence against and every point
    # reports "unknown". Measured that way on the first run of this sweep.
    if base_cfg.get("spec"):
        ref_cfg = copy.deepcopy(base_cfg)
        ref_cfg["spec"] = None
        ref_cfg["n_max"] = None
        print("### verification reference (spec off)", file=sys.stderr)
        ref = probe_config(bindir, model, ref_cfg, defaults, prompts,
                           a.n_predict, 1)
    else:
        ref = base

    rows, results = [], []
    fp = emit_row.fingerprint(os.environ["PERF_LAB_GPU_UID"], model, bindir,
                              "llama-server")

    def record(label, cfg, res, verdict):
        results.append({"label": label, "tg": res["tg"], "pp": res["pp"],
                        "acceptance": res["acceptance"], "verdict": verdict,
                        "tg_spread": res.get("tg_spread")})
        if not a.ledger:
            return
        env = emit_row.env_probe(fp["gpu"]["pci"])
        rows.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": emit_row.new_run_id(label, tag), "kind": "manual",
            "key": f"sweep/{label or a.base}", "tag": tag, "fp": fp,
            "cfg": {"ctx": cfg["ctx"], "ctk": cfg["ctk"], "ctv": cfg["ctv"],
                    "fa": str(defaults["fa"]), "spec": cfg.get("spec"),
                    "n_max": cfg.get("n_max"), "ngl": defaults["ngl"], "np": 1,
                    "prompt_tokens": res["prompt_n"]},
            "m": {"pp2048": res["pp"], "tg": res["tg"], "rep": 1,
                  "cold_prefill": True, "mtp_acceptance": res["acceptance"],
                  "unit": "tok/s"},
            "ok": {"token_identical": verdict == "identical", "corpus_sha": None},
            "env": env,
        })

    record("", base_cfg, base, classify(ref["gens"], base["gens"], a.tie_ratio)[0])

    for label, delta, cfg in plan:
        print(f"### {label}", file=sys.stderr)
        res = probe_config(bindir, model, cfg, defaults, prompts,
                           a.n_predict, a.reps)
        verdict, _ = classify(ref["gens"], res["gens"], a.tie_ratio)
        record(label, cfg, res, verdict)
        print(f"    tg={res['tg']} acceptance={res['acceptance']} -> {verdict}",
              file=sys.stderr)

    if a.ledger:
        with open(a.ledger, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    keep = [r for r in results if r["verdict"] in ("identical", "tied")]
    keep.sort(key=lambda r: r["tg"] or 0, reverse=True)
    print(f"\n=== {tag}: usable configurations, fastest first ===")
    print(f"{'config':<24}{'tg t/s':>9}{'spread':>8}{'pp t/s':>10}{'accept':>9}  verdict")
    for r in keep:
        acc = "-" if r["acceptance"] is None else f"{r['acceptance']:.4f}"
        sp = "-" if r.get("tg_spread") is None else f"{100*r['tg_spread']:.1f}%"
        print(f"{(r['label'] or 'baseline'):<24}{r['tg'] or 0:>9.2f}{sp:>8}"
              f"{r['pp'] or 0:>10.2f}{acc:>9}  {r['verdict']}")
    if len(keep) > 1 and keep[0]["tg"] and keep[1]["tg"]:
        gap = (keep[0]["tg"] - keep[1]["tg"]) / keep[1]["tg"]
        sp = keep[0].get("tg_spread")
        if sp is not None and gap < sp:
            print(f"\nWARNING: the top two differ by {100*gap:.1f}%, inside the "
                  f"{100*sp:.1f}% spread measured on a single config. That is a "
                  "tie, not a winner -- raise --reps.")
    rejected = [r for r in results if r["verdict"] not in ("identical", "tied")]
    if rejected:
        print("\nnot usable:")
        for r in rejected:
            print(f"  {(r['label'] or 'baseline'):<24}{r['tg'] or 0:>9.2f}  {r['verdict']}")
    best = keep[0] if keep else None
    if best:
        print(f"\nfastest equivalent: {best['label'] or 'baseline'} "
              f"at {best['tg']:.2f} t/s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare two configurations by what they generate, tolerating tied choices.

Strict token identity was the obvious gate and it is the wrong one here.
Measured 2026-08-16: `draft-mtp n_max 4` diverges from no-speculation, and
`n_max 2` diverges from `n_max 4` -- but every divergence observed landed on a
position where the top two candidates were within 6% of each other (one within
0.1%), and the other config always picked exactly the runner-up, never
anything further down. Two of five prompts did not diverge at all.

That is a floating-point tie-break, not a quality regression. Drafting n
tokens changes the shape of the forward pass, which changes reduction order,
which perturbs logits in the low bits. Where two candidates are near-tied the
argmax flips; everywhere else it cannot.

So this reports both:

  token_identical  strict equality, the old gate
  equivalent       no divergence that was NOT a near-tie

A material divergence -- the other config choosing a token the first ranked
well below its best -- is a real difference and still fails.

LIMITATION: once the sequences diverge they are on different valid paths, so
only the FIRST divergence per prompt can be judged. Certifying a whole
sequence needs teacher forcing (score one config's tokens under the other),
which this does not yet do.

  verify.py configs/tracked.yaml baseline no-spec
  verify.py configs/tracked.yaml baseline mtp-n6 --tie-ratio 1.2
"""
import argparse
import hashlib
import importlib.util
import math
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

_spec = importlib.util.spec_from_file_location("emit_row", HERE / "emit_row.py")
emit_row = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emit_row)


def load_prompts(path):
    """One prompt per file, or one per blank-line-separated block."""
    text = pathlib.Path(path).read_text()
    blocks = [b.strip() for b in text.split("\n\n\n") if b.strip()]
    return blocks or [text]


def generate(bindir, model, cfg, defaults, prompts, n_predict, label):
    """Run every prompt through one config; return the token ids it produced."""
    proc, port, stop = emit_row.start_server(bindir, model, cfg, defaults)
    out = []
    try:
        for i, p in enumerate(prompts, 1):
            # return_tokens is what makes this a token check rather than a
            # string compare -- two different token sequences can detokenize
            # to the same text, and that difference is exactly what a
            # correctness gate must not wave through.
            r = emit_row.complete(port, p, n_predict,
                                  {"return_tokens": True, "n_probs": 5})
            toks = r.get("tokens") or []
            out.append({"tokens": toks, "content": r.get("content", ""),
                        "probs": r.get("completion_probabilities") or []})
            print(f"  {label}: prompt {i}/{len(prompts)} -> "
                  f"{len(toks)} tokens", file=sys.stderr)
    finally:
        stop()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("key_a")
    ap.add_argument("key_b")
    ap.add_argument("--prompts", default=os.environ.get(
        "PERF_LAB_PROMPT", str(pathlib.Path.home() / ".perf-lab/prompts/default.txt")))
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--tie-ratio", type=float, default=1.2,
                    help="a divergence counts as a tie when the first config "
                         "rated its own pick at most this many times more "
                         "likely than what the other picked")
    a = ap.parse_args()

    import yaml
    doc = yaml.safe_load(open(a.config))
    entries = doc.get("tracked") or doc.get("canaries") or {}
    for k in (a.key_a, a.key_b):
        if k not in entries:
            sys.exit(f"verify: no config '{k}' in {a.config}")
    defaults = doc["defaults"]

    bindir = pathlib.Path(os.environ.get("PERF_LAB_SERVER_BIN", ""))
    model = pathlib.Path(os.environ.get("PERF_LAB_MODEL", ""))
    if not (bindir / "llama-server").exists():
        sys.exit("verify: set PERF_LAB_SERVER_BIN to a working llama-server")
    if not model.is_file():
        sys.exit("verify: set PERF_LAB_MODEL")

    prompts = load_prompts(a.prompts)
    corpus_sha = hashlib.sha256(
        pathlib.Path(a.prompts).read_bytes()).hexdigest()
    print(f"# {len(prompts)} prompt(s), corpus {corpus_sha[:12]}…", file=sys.stderr)

    ra = generate(bindir, model, entries[a.key_a], defaults, prompts,
                  a.n_predict, a.key_a)
    rb = generate(bindir, model, entries[a.key_b], defaults, prompts,
                  a.n_predict, a.key_b)

    diffs = []
    for i, (x, y) in enumerate(zip(ra, rb)):
        if not (x["tokens"] and y["tokens"]):
            # Older servers return an empty tokens array. Fall back rather than
            # silently reporting equality we did not actually establish.
            if x["content"] != y["content"]:
                diffs.append({"prompt": i, "verdict": "material",
                              "basis": "content (server returned no token ids)"})
            continue
        d = next((j for j, (p, q) in enumerate(zip(x["tokens"], y["tokens"]))
                  if p != q), None)
        if d is None and len(x["tokens"]) == len(y["tokens"]):
            continue
        if d is None:
            diffs.append({"prompt": i, "verdict": "material", "basis": "length",
                          "len_a": len(x["tokens"]), "len_b": len(y["tokens"])})
            continue

        rec = {"prompt": i, "basis": "tokens", "first_divergence": d}

        # Score against whichever side actually reports distributions.
        # llama-server does not return top_logprobs for tokens that came from
        # accepted drafts -- measured: 1-2 of 96 tokens on a speculative run
        # against 96 of 96 without speculation. Reading the reference from the
        # speculative side makes every divergence look unjudgeable, which is
        # not the same as it being real.
        cov = lambda r: sum(1 for e in r["probs"] if e.get("top_logprobs"))
        ref, other = (x, y) if cov(x) >= cov(y) else (y, x)
        rec["reference"] = a.key_a if ref is x else a.key_b

        top = (ref["probs"][d].get("top_logprobs")
               if d < len(ref["probs"]) else None) or []
        chosen = other["tokens"][d]
        rank = next((r for r, t in enumerate(top) if t.get("id") == chosen), None)
        if not top:
            # No distribution here at all. Unknown is not the same as material:
            # say so rather than reporting a difference we never measured.
            rec.update({"verdict": "unknown",
                        "why": "no top_logprobs at this index on either side"})
        elif rank is None:
            rec.update({"rank_in_ref": None, "verdict": "material",
                        "why": f"chosen token outside the reference's top {len(top)}"})
        else:
            p1 = math.exp(top[0]["logprob"])
            pb = math.exp(top[rank]["logprob"])
            ratio = p1 / pb if pb else float("inf")
            rec.update({"rank_in_ref": rank, "prob_ratio": round(ratio, 4),
                        "verdict": "tied" if ratio <= a.tie_ratio else "material"})
        diffs.append(rec)

    identical = not diffs
    material = [d for d in diffs if d["verdict"] == "material"]
    unknown = [d for d in diffs if d["verdict"] == "unknown"]
    result = {"config": a.config, "a": a.key_a, "b": a.key_b,
              "prompts": len(prompts), "corpus_sha": corpus_sha,
              "tie_ratio": a.tie_ratio,
              "token_identical": identical,
              "equivalent": not material and not unknown,
              "unjudgeable": len(unknown),
              "divergences": diffs}
    json.dump(result, sys.stdout, indent=2)
    print()
    sys.exit(0 if (not material and not unknown) else 1)


if __name__ == "__main__":
    main()

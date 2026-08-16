#!/usr/bin/env python3
"""Prove two configurations produce the same tokens, or show that they do not.

"Faster" is not a result on its own. Half the configuration space here is
supposed to be exact: speculative decoding drafts tokens and then verifies
them, so `--spec-type draft-mtp --spec-draft-n-max 4` must emit byte-identical
output to no speculation at all. If it does not, that is a correctness bug and
the speed number is worthless.

The other half is lossy on purpose. Changing `ctk`/`ctv` quantization changes
the arithmetic, so divergence there is expected and this tool cannot judge it
-- that needs a quality benchmark, not an equality check. verify.py's job is
to tell those two cases apart honestly, which means it has to be able to fail:
a gate that passes everything proves nothing.

  verify.py configs/tracked.yaml baseline no-spec
  verify.py configs/tracked.yaml baseline mtp-n6 --prompts FILE
"""
import argparse
import hashlib
import importlib.util
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
            r = emit_row.complete(port, p, n_predict, {"return_tokens": True})
            toks = r.get("tokens") or []
            out.append({"tokens": toks, "content": r.get("content", "")})
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
        if x["tokens"] and y["tokens"]:
            same = x["tokens"] == y["tokens"]
            basis = "tokens"
        else:
            # Older servers return an empty tokens array. Fall back rather than
            # silently reporting equality we did not actually establish.
            same = x["content"] == y["content"]
            basis = "content (server returned no token ids)"
        if not same:
            first = next((j for j, (p, q) in enumerate(
                zip(x["tokens"], y["tokens"])) if p != q), None)
            diffs.append({"prompt": i, "basis": basis, "first_divergence": first,
                          "len_a": len(x["tokens"]), "len_b": len(y["tokens"])})

    identical = not diffs
    result = {"config": a.config, "a": a.key_a, "b": a.key_b,
              "prompts": len(prompts), "corpus_sha": corpus_sha,
              "token_identical": identical, "divergences": diffs}
    json.dump(result, sys.stdout, indent=2)
    print()
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()

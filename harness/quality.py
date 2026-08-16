#!/usr/bin/env python3
"""Judge a lossy configuration by perplexity, so the sweep can stop refusing it.

The tie-tolerant check in verify.py handles axes whose divergence is numerical:
speculative decoding flips near-ties and nothing else. It cannot judge KV
quantization, where the arithmetic genuinely changes -- measured, q4_0 vs q8_0
diverges at a probability ratio of 1.70 against speculation's 1.00-1.06. That
is a different answer, not a coin-flip, and telling "different" from "worse"
needs a metric rather than an equality check.

Perplexity over a held-out corpus is that metric. It is not a complete account
of quality -- a model can hold perplexity and still lose on long-context
retrieval, which is exactly where the design already flags q4_0 as untested --
but it is comparable, cheap, and comes with an interval rather than a bare
number.

The corpus lives outside the repo. A held-out set published in a public repo
that every session reads is not held out; rows cite its sha256 instead.

  quality.py configs/tracked.yaml baseline --chunks 32
  quality.py configs/tracked.yaml baseline --ledger results/ledger.jsonl
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

_spec = importlib.util.spec_from_file_location("emit_row", HERE / "emit_row.py")
emit_row = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emit_row)

FINAL = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")


def run_perplexity(bindir, model, cfg, corpus, chunks, ctx):
    cmd = [str(bindir / "llama-perplexity"), "-m", str(model), "-f", str(corpus),
           "-c", str(ctx), "-ctk", cfg["ctk"], "-ctv", cfg["ctv"],
           "-fa", "on", "-ngl", "99"]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    env = dict(os.environ, LD_LIBRARY_PATH=str(bindir))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=14400)
    blob = r.stdout + r.stderr
    m = FINAL.search(blob)
    if not m:
        sys.exit(f"quality: no final PPL in output (exit {r.returncode})\n"
                 f"{blob[-1200:]}")
    return float(m.group(1)), float(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("key")
    ap.add_argument("--corpus", default=os.environ.get(
        "PERF_LAB_HELDOUT",
        str(pathlib.Path.home() / ".perf-lab/heldout/corpus.txt")))
    ap.add_argument("--chunks", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=4096,
                    help="perplexity depends on context length; comparisons "
                         "are only meaningful with this held fixed")
    ap.add_argument("--ledger", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import yaml
    doc = yaml.safe_load(open(a.config))
    entries = doc.get("tracked") or doc.get("canaries") or {}
    if a.key not in entries:
        sys.exit(f"quality: no config '{a.key}' in {a.config}")
    cfg, defaults = entries[a.key], doc["defaults"]

    corpus = pathlib.Path(a.corpus)
    if not corpus.is_file():
        sys.exit(f"quality: no held-out corpus at {corpus}. It is deliberately "
                 "outside the repo -- a held-out set in a public repo that "
                 "every session reads is not held out. Set PERF_LAB_HELDOUT.")
    corpus_sha = hashlib.sha256(corpus.read_bytes()).hexdigest()

    bindir = pathlib.Path(os.environ.get("PERF_LAB_SERVER_BIN", ""))
    model = pathlib.Path(os.environ.get("PERF_LAB_MODEL", ""))
    if not (bindir / "llama-perplexity").exists():
        sys.exit("quality: no llama-perplexity in PERF_LAB_SERVER_BIN")

    print(f"# {a.key}: ctk={cfg['ctk']} ctv={cfg['ctv']} ctx={a.ctx} "
          f"chunks={a.chunks} corpus={corpus_sha[:12]}…", file=sys.stderr)
    ppl, stderr_ = run_perplexity(bindir, model, cfg, corpus, a.chunks, a.ctx)

    out = {"key": a.key, "ctk": cfg["ctk"], "ctv": cfg["ctv"], "ctx": a.ctx,
           "chunks": a.chunks, "corpus_sha": corpus_sha,
           "ppl": ppl, "ppl_stderr": stderr_}
    json.dump(out, sys.stdout, indent=2)
    print()

    if a.ledger:
        fp = emit_row.fingerprint(os.environ["PERF_LAB_GPU_UID"], model, bindir,
                                  "llama-perplexity")
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": emit_row.new_run_id(a.key, "ppl"), "kind": "manual",
            "key": f"quality/{a.key}", "tag": a.tag or "quality",
            "fp": fp,
            "cfg": {"ctx": a.ctx, "ctk": cfg["ctk"], "ctv": cfg["ctv"],
                    "fa": "on", "spec": None, "n_max": None,
                    "ngl": defaults["ngl"], "np": 1, "prompt_tokens": None},
            "m": {"ppl": ppl, "ppl_stderr": stderr_, "rep": 1,
                  "cold_prefill": True, "unit": "ppl"},
            "ok": {"token_identical": None, "corpus_sha": corpus_sha},
            "env": emit_row.env_probe(fp["gpu"]["pci"]),
        }
        with open(a.ledger, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"appended {row['run_id']} to {a.ledger}", file=sys.stderr)


if __name__ == "__main__":
    main()

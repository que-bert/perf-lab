#!/usr/bin/env python3
"""Sweep speculative-decoding depth and slot count together, on a served model.

Two axes that were previously measured one at a time, which is how the lab
ended up carrying an n-max chosen under the wrong slot count:

  --spec-draft-n-max   how many tokens the MTP head drafts per step
  --parallel           how many server slots exist

They interact. `--parallel` defaulted to 4 while every n-max sweep was run, and
`--parallel 1` later turned out to double decode on one KV config, so the n-max
optimum recorded against 4 slots does not transfer. Sweeping the cross product
is the only way to stop re-deriving one axis against a stale value of the other.

Each cell gets a fresh server unit -- a config change that only takes effect at
load cannot be swept inside one process -- one warm-up request, then `--reps`
timed requests. Recorded per cell: decode t/s per rep, draft acceptance, and
VRAM after load.

  spec_sweep.py --model ~/models-fast/Qwen3.8-27B-Q6_K.gguf --ctx 262144 \
      --n-max off,2,3,4 --parallel 1,4 --reps 3 --out results/sweep.json
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

DEFAULT_PROMPT = (
    "Explain, in careful technical prose, how a copy-on-write filesystem "
    "keeps a snapshot consistent while writes continue against the live "
    "tree. Cover the block reference counts, what happens on the first write "
    "to a shared block, and how a snapshot is eventually freed. Do not use "
    "bullet points."
)


def vram_used(gpu):
    """Bytes in use on one card, or None. rocm-smi prints uppercase hex PCI
    ids and index-ordered rows; the index is fine here because the caller
    already chose the card by index for GGML_VK_VISIBLE_DEVICES."""
    try:
        o = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                           capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    m = re.findall(rf"GPU\[{gpu}\].*?Used Memory \(B\): (\d+)", o)
    return int(m[-1]) if m else None


def serve(model, port, ctx, gpu, bin_dir, ctk, ctv, extra):
    env_pairs = {"PERFLAB_GPU": str(gpu), "PERFLAB_BIN": bin_dir,
                 "PERFLAB_CTK": ctk, "PERFLAB_CTV": ctv}
    import os
    env = dict(os.environ, **env_pairs)
    r = subprocess.run([str(HERE / "serve_unit.sh"), str(model), str(port),
                        str(ctx)] + extra, env=env, capture_output=True,
                       text=True, timeout=2400)
    ok = "SERVING" in r.stdout
    return ok, (r.stdout + r.stderr)[-1500:]


def stop(port):
    for cmd in ("stop", "reset-failed"):
        subprocess.run(["systemctl", "--user", cmd, f"perflab-srv-{port}"],
                       capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--port", type=int, default=8961)
    ap.add_argument("--bin", default=str(pathlib.Path.home()
                                         / "llama.cpp/b10472-vulkan"))
    ap.add_argument("--ctk", default="q8_0")
    ap.add_argument("--ctv", default="q4_0")
    ap.add_argument("--n-max", default="off,2,3,4,5",
                    help="'off' disables speculation entirely")
    ap.add_argument("--parallel", default="1,4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--spec-type", default="draft-mtp")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    prompt = (pathlib.Path(a.prompt_file).read_text() if a.prompt_file
              else DEFAULT_PROMPT)
    n_maxes = [x.strip() for x in a.n_max.split(",") if x.strip()]
    slots = [int(x) for x in a.parallel.split(",")]

    def one(port, i):
        body_prompt = prompt
        import urllib.request
        body = {"prompt": body_prompt, "n_predict": a.n_predict,
                "temperature": 0, "cache_prompt": False}
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/completion",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.load(r)

    cells = []
    for p in slots:
        for nm in n_maxes:
            extra = ["--parallel", str(p)]
            if nm == "adaptive":
                # PR #27210 varies depth per step, so there is no n-max to
                # pass; it is a cell on this axis rather than a separate run
                # because the whole point is comparing it to the fixed depths
                # under identical everything else.
                extra += ["--spec-type", "draft-mtp-adaptive"]
            elif nm != "off":
                extra += ["--spec-type", a.spec_type,
                          "--spec-draft-n-max", nm]
            label = f"n_max={nm} parallel={p}"
            print(f"=== {label}", file=sys.stderr, flush=True)
            ok, log = serve(a.model, a.port, a.ctx, a.gpu, a.bin,
                            a.ctk, a.ctv, extra)
            if not ok:
                print(f"    LOAD FAILED\n{log}", file=sys.stderr)
                cells.append({"n_max": nm, "parallel": p, "error": log[-400:]})
                stop(a.port)
                continue
            vram = vram_used(a.gpu)
            try:
                one(a.port, -1)  # warm-up, discarded
                reps = []
                for i in range(a.reps):
                    d = one(a.port, i)
                    t = d.get("timings", {})
                    drafted = t.get("draft_n") or 0
                    reps.append({
                        "tps": round(t.get("predicted_per_second") or 0, 2),
                        "predicted_n": t.get("predicted_n"),
                        "acceptance": round((t.get("draft_n_accepted") or 0)
                                            / drafted, 3) if drafted else None,
                    })
                    print(f"    rep {i}  {reps[-1]['tps']:>7} t/s"
                          + (f"   accept {reps[-1]['acceptance']}"
                             if reps[-1]["acceptance"] is not None else ""),
                          file=sys.stderr, flush=True)
                tps = [r["tps"] for r in reps]
                cells.append({"n_max": nm, "parallel": p, "reps": reps,
                              "tps_mean": round(sum(tps) / len(tps), 2),
                              "tps_min": min(tps), "tps_max": max(tps),
                              "vram_bytes": vram,
                              "vram_gb": round(vram / 1e9, 2) if vram else None})
            except Exception as e:
                cells.append({"n_max": nm, "parallel": p,
                              "error": f"{type(e).__name__}: {e}"[:300]})
                print(f"    ERROR {e}", file=sys.stderr)
            finally:
                stop(a.port)
                time.sleep(3)

    report = {"model": pathlib.Path(a.model).name, "ctx": a.ctx,
              "bin": a.bin, "ctk": a.ctk, "ctv": a.ctv,
              "n_predict": a.n_predict, "reps": a.reps, "cells": cells}
    text = json.dumps(report, indent=2)
    if a.out:
        pathlib.Path(a.out).write_text(text)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(text)

    print("\n  n_max  par     mean t/s        reps            accept   VRAM",
          file=sys.stderr)
    for c in cells:
        if "error" in c:
            print(f"  {c['n_max']:<6} {c['parallel']:<4}  FAILED",
                  file=sys.stderr)
            continue
        acc = c["reps"][0]["acceptance"]
        print(f"  {c['n_max']:<6} {c['parallel']:<4} {c['tps_mean']:>9}   "
              f"{'/'.join(str(r['tps']) for r in c['reps']):<26} "
              f"{acc if acc is not None else '-':<8} {c['vram_gb']}",
              file=sys.stderr)


if __name__ == "__main__":
    main()

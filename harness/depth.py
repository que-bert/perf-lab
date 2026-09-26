#!/usr/bin/env python3
"""Decode-vs-filled-context with repetition, for a running llama-server.

The single-rep throwaway (/tmp/opencode/ctxcost.py) could not separate a real
depth effect from the ~6% run-to-run spread at 176k. This one warms the prompt
once with cache_prompt and then measures R reps over the *same* filled context,
so a rep costs only the decode (the prefill is paid once), and it reports the
spread instead of a single number.

  depth.py PORT [--depths K,K,K] [--reps R] [--npred N] [--json OUT]

`depths` are prompt repeat counts, not tokens: the base paragraph is ~44 tokens,
so 1600 ~ 70k and 4000 ~ 176k. The measured prompt_n is printed, never assumed.

--corpus FILE replaces the repeated paragraph with a slice of a real document
(--corpus-kb A,B,C), so the prompt is diverse text rather than one paragraph
echoed 1600 times. The repeated-paragraph prompt is n-gram speculation's best
case; a claim about real workloads must not be measured on it.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

BASE = ("Request {i}. Write a short technical description of how a write-ahead "
        "log keeps a database consistent across a crash. Cover the log record, "
        "the checkpoint, and recovery. Be specific and do not use lists.")


def build_prompt(reps):
    return "\n".join(BASE.format(i=i) for i in range(reps))


def build_corpus(path, kb):
    data = open(path, "rb").read()
    n = kb * 1024
    if n <= len(data):
        return data[:n].decode("utf-8", "replace")
    reps = n // len(data) + 1
    return (data * reps)[:n].decode("utf-8", "replace")


def req(port, prompt, npred, cache):
    body = {"prompt": prompt, "n_predict": npred, "temperature": 0,
            "cache_prompt": cache}
    r = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=7200) as resp:
        d = json.load(resp)
    return d, time.time() - t0


def run(port, depths, reps, npred, corpus=None, reprefill=False):
    out = []
    for reps_n in depths:
        if corpus:
            prompt = build_corpus(corpus, reps_n)
            label = f"~{reps_n}kB"
        else:
            prompt = build_prompt(reps_n)
            label = "short" if reps_n == 1 else f"~{reps_n * 44 // 1000}k"
        d, wall = req(port, prompt, npred, cache=False)   # prefill + first decode
        first = d.get("timings", {})
        rows = [{"rep": 0, "decode": first.get("predicted_per_second"),
                 "prefill": first.get("prompt_per_second"),
                 "prompt_n": first.get("prompt_n"),
                 "accepted": first.get("draft_n_accepted"),
                 "drafted": first.get("draft_n"), "wall": wall}]
        for rep in range(1, reps):
            d, wall = req(port, prompt, npred, cache=not reprefill)
            t = d.get("timings", {})
            rows.append({"rep": rep, "decode": t.get("predicted_per_second"),
                         "prefill": t.get("prompt_per_second"),
                         "prompt_n": t.get("prompt_n"),
                         "accepted": t.get("draft_n_accepted"),
                         "drafted": t.get("draft_n"), "wall": wall})
        ds = [r["decode"] for r in rows if r["decode"]]
        ps = [r["prefill"] for r in rows if r["prefill"]]
        # Exclude the cold rep from the spread: its decode is fine, but the
        # state (cache freshly built) differs from the steady-state reps.
        ss = ds[1:] if len(ds) > 1 else ds
        ps_stats = {}
        if len(ps) > 1:
            ps_stats = {"prefill_mean": round(statistics.mean(ps), 1),
                        "prefill_min": round(min(ps), 1),
                        "prefill_max": round(max(ps), 1),
                        "prefill_spread_pct": round(100 * (max(ps) - min(ps)) / statistics.mean(ps), 2)}
        summary = {
            "label": label, "reps_requested": reps_n,
            "prompt_n": rows[-1]["prompt_n"],
            "decode_mean": round(statistics.mean(ds), 3) if ds else None,
            "decode_steady_mean": round(statistics.mean(ss), 3) if ss else None,
            "decode_steady_min": round(min(ss), 3) if ss else None,
            "decode_steady_max": round(max(ss), 3) if ss else None,
            "decode_spread_pct": (round(100 * (max(ss) - min(ss)) / statistics.mean(ss), 2)
                                  if len(ss) > 1 and statistics.mean(ss) else None),
            "prefill": first.get("prompt_per_second"),
            "accept_frac": (round(rows[-1]["accepted"] / rows[-1]["drafted"], 4)
                            if rows[-1].get("drafted") else None),
            "rows": rows,
            **ps_stats,
        }
        out.append(summary)
        line = (f"{label:>8}  prompt_n={summary['prompt_n']:>7}  "
                f"decode={summary['decode_steady_mean']} t/s "
                f"[{summary['decode_steady_min']}-{summary['decode_steady_max']}, "
                f"{summary['decode_spread_pct']}%]  prefill={summary['prefill']:.0f}  "
                f"accept={summary['accept_frac']}")
        if ps_stats:
            line += (f"  prefill_reps={ps_stats['prefill_mean']:.0f} "
                     f"[{ps_stats['prefill_min']:.0f}-{ps_stats['prefill_max']:.0f}, "
                     f"{ps_stats['prefill_spread_pct']}%]")
        print(line, flush=True)
        for r in rows:
            print(f"           rep{r['rep']}: decode={r['decode']:.2f} "
                  f"prompt_n={r['prompt_n']} accept={r['accepted']}/{r['drafted']} "
                  f"wall={r['wall']:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--depths", default="1,1600,4000")
    ap.add_argument("--corpus", default=None, help="file to slice instead of the repeated paragraph")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--npred", type=int, default=256)
    ap.add_argument("--json", default=None)
    ap.add_argument("--reprefill", action="store_true",
                    help="re-prefill every rep (no cache reuse) so prefill gets reps too")
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",") if x]
    out = run(a.port, depths, a.reps, a.npred, corpus=a.corpus, reprefill=a.reprefill)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
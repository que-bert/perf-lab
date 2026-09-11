#!/usr/bin/env python3
"""Aggregate throughput under concurrency, which is not decode rate.

Every speed number in FINDINGS.md so far is single-stream: one request, how
fast do its tokens come out. That is the right measure for a person waiting on
an answer and the wrong one for a queue. The two can point opposite ways --
`--parallel 4` measured HALF the single-stream decode of `--parallel 1` with
MTP on, and the obvious next question is whether the slots pay for themselves
once they are actually used. This file answers that question and nothing else.

  throughput.py --port 8921 --concurrency 1,2,4,8 --n-predict 128

Reports, per concurrency level:

  agg t/s      generated tokens across all requests / wall clock of the batch.
               The number a queue operator cares about.
  per-req t/s  mean of each request's own decode rate. The number the person
               waiting cares about. It falls as concurrency rises even when
               agg t/s climbs; reporting only one of the two hides the trade.
  p50 / p95    end-to-end latency per request, in seconds.

Prompts are varied per request by a numbered prefix. Identical prompts would
share prefix cache across slots and measure the cache instead of the batch;
cache_prompt is off for the same reason.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

PROMPT = (
    "Request {i}. Write a short technical description of how a write-ahead "
    "log keeps a database consistent across a crash. Cover the log record, "
    "the checkpoint, and recovery. Be specific and do not use lists."
)


def one(port, i, n_predict, timeout):
    body = {"prompt": PROMPT.format(i=i), "n_predict": n_predict,
            "temperature": 0, "cache_prompt": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    t = d.get("timings", {})
    return {"seconds": time.time() - t0,
            "predicted_n": t.get("predicted_n"),
            "prompt_n": t.get("prompt_n"),
            "tps": t.get("predicted_per_second"),
            "draft_n": t.get("draft_n"),
            "draft_accepted": t.get("draft_n_accepted")}


def level(port, c, n_predict, reps, timeout):
    """One concurrency level: c requests in flight at once, `reps` batches."""
    rows = []
    for rep in range(reps):
        out, err = [None] * c, [None] * c
        # Threads, not a pool: the point is that all c requests are in flight
        # simultaneously, and the wall clock is measured around the whole set.
        def work(k):
            try:
                out[k] = one(port, rep * c + k, n_predict, timeout)
            except Exception as e:
                err[k] = f"{type(e).__name__}: {e}"[:160]
        ts = [threading.Thread(target=work, args=(k,)) for k in range(c)]
        t0 = time.time()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.time() - t0
        got = [o for o in out if o]
        if not got:
            return {"concurrency": c, "error": next(e for e in err if e)}
        rows.append({"wall": wall, "reqs": got,
                     "failed": [e for e in err if e]})

    gen = sum(r["predicted_n"] or 0 for b in rows for r in b["reqs"])
    wall = sum(b["wall"] for b in rows)
    lat = sorted(r["seconds"] for b in rows for r in b["reqs"])
    per = [r["tps"] for b in rows for r in b["reqs"] if r["tps"]]
    drafted = sum(r["draft_n"] or 0 for b in rows for r in b["reqs"])
    accepted = sum(r["draft_accepted"] or 0 for b in rows for r in b["reqs"])
    return {
        "concurrency": c,
        "batches": reps,
        "requests": len(lat),
        "failed": sum(len(b["failed"]) for b in rows),
        "generated_tokens": gen,
        "wall_seconds": round(wall, 2),
        "agg_tps": round(gen / wall, 2) if wall else None,
        "per_req_tps": round(statistics.mean(per), 2) if per else None,
        "p50_seconds": round(lat[len(lat) // 2], 2),
        "p95_seconds": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 2),
        "draft_acceptance": round(accepted / drafted, 3) if drafted else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2,
                    help="batches per concurrency level")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    # One throwaway batch first: the first request after a load pays for
    # warm-up and would otherwise land entirely in the c=1 row.
    try:
        one(a.port, 9999, 16, a.timeout)
    except Exception as e:
        sys.exit(f"server on {a.port} not answering: {e}")

    rows = []
    for c in [int(x) for x in a.concurrency.split(",")]:
        r = level(a.port, c, a.n_predict, a.reps, a.timeout)
        rows.append(r)
        if "error" in r:
            print(f"  c={c:<3} ERROR {r['error']}", file=sys.stderr)
            continue
        print(f"  c={c:<3} agg {r['agg_tps']:>8} t/s   per-req "
              f"{r['per_req_tps']:>7} t/s   p50 {r['p50_seconds']:>6}s   "
              f"p95 {r['p95_seconds']:>6}s"
              + (f"   accept {r['draft_acceptance']}"
                 if r["draft_acceptance"] else ""),
              file=sys.stderr)

    report = {"label": a.label, "port": a.port, "n_predict": a.n_predict,
              "reps": a.reps, "levels": rows}
    text = json.dumps(report, indent=2)
    if a.out:
        open(a.out, "w").write(text)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()

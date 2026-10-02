#!/usr/bin/env python3
"""Pooled decode t/s and tokens/step over a fixed prompt set (autoresearch arbiter for draft-policy changes).

  pool.py <port> <out.json> [--set corpus|chat|heldout|all]

Greedy, 256 tokens per prompt. Reports, per set and pooled:
  t/s        = sum(predicted_n) / sum(predicted_ms)       (decode only, prompt excluded)
  tok/step   = sum(predicted_n) / sum(steps), steps = predicted_n - draft_n_accepted
  accept     = draft_n_accepted / draft_n                 (not comparable across p_min / n_max; use tok/step)
Sets: corpus = 16 slices of harness/corpus/decode_kld.txt (the D0 "corpus" set);
chat = setups/qwen3.8-27b-r9700/results/d0/scripts/chat_acc.py prompts + image; heldout = setups/qwen3.8-27b-r9700/results/d0/scripts/heldout_acc.py.
"""
import argparse, base64, importlib.util, json, os, sys, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))  # repo root (setups/<setup>/autoresearch/)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def corpus_items():
    data = open(os.path.join(ROOT, "harness/corpus/decode_kld.txt"), "rb").read()
    return [("corpus", i, data[i * 90 * 1024:i * 90 * 1024 + 24 * 1024].decode("utf-8", "replace")) for i in range(16)]


def chat_items(kind):
    if kind == "chat":
        m = load("chat_acc", os.path.join(ROOT, "setups/qwen3.8-27b-r9700/results/d0/scripts/chat_acc.py"))
        img = base64.b64encode(open(m.IMAGE, "rb").read()).decode()
        items = list(m.PROMPTS) + [[{"type": "text", "text": "Describe this image in detail."},
                                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}]]
    else:
        m = load("heldout_acc", os.path.join(ROOT, "setups/qwen3.8-27b-r9700/results/d0/scripts/heldout_acc.py"))
        items = list(m.PROMPTS)
    return [(kind, i, c) for i, c in enumerate(items)]


def run(port, kind, i, content):
    if kind == "corpus":
        d = post(port, "/completion", {"prompt": content, "n_predict": 256, "temperature": 0, "cache_prompt": False})
        t, text = d["timings"], d.get("content", "")
    else:
        d = post(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": content}], "max_tokens": 256,
                                                "temperature": 0, "cache_prompt": False,
                                                "chat_template_kwargs": {"enable_thinking": False}})
        t, text = d.get("timings", {}), d["choices"][0]["message"]["content"]
    return {"set": kind, "i": i, "n": t.get("predicted_n"), "ms": t.get("predicted_ms"),
            "dn": t.get("draft_n") or 0, "da": t.get("draft_n_accepted") or 0, "text": text}


def summarize(rows):
    n = sum(r["n"] for r in rows); ms = sum(r["ms"] for r in rows)
    da = sum(r["da"] for r in rows); dn = sum(r["dn"] for r in rows)
    steps = n - da
    return {"tps": round(1000 * n / ms, 3), "tok_step": round(n / steps, 4) if steps else None,
            "accept": round(da / dn, 4) if dn else None, "n": n, "ms_step": round(ms / steps, 3) if steps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("out")
    ap.add_argument("--set", default="all", choices=["corpus", "chat", "heldout", "all"])
    a = ap.parse_args()
    kinds = ["corpus", "chat", "heldout"] if a.set == "all" else [a.set]
    items = []
    for k in kinds:
        items += corpus_items() if k == "corpus" else chat_items(k)
    rows = []
    for kind, i, c in items:
        r = run(a.port, kind, i, c)
        rows.append(r)
        print(kind, i, r["n"], r["da"], r["dn"], round(1000 * r["n"] / r["ms"], 2), flush=True)
    out = {k: summarize([r for r in rows if r["set"] == k]) for k in kinds}
    out["pooled"] = summarize(rows)
    for k, v in out.items():
        print("POOL", k, json.dumps(v))
    json.dump({"summary": out, "rows": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

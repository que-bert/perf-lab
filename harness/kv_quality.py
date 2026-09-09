#!/usr/bin/env python3
"""Does q8_0/q4_0 KV answer the same as q8_0/q8_0, on recall and on code?

perplexity (quality.py) already says these two are indistinguishable at 4096
tokens. That is a weak claim and this file exists because of its weakness: a
model can hold perplexity and still lose the fact it was told 60,000 tokens
ago, which is exactly the regime the shipped config runs in and exactly where
q4_0 was flagged untested.

Two task families, because they fail differently:

  recall   a unique fact planted at a known depth in a long context, then
           asked for. Retrieval either happens or it does not -- there is no
           partial credit and no tie to tolerate, so this is the one measure
           where "same quality" has a crisp meaning.

  codegen  a short function against a fixed spec, executed against asserts.
           Generation is long and autoregressive, so a single flipped token
           compounds; running the result is the only check that distinguishes
           "different text" from "worse answer".

Both run with speculation OFF and temperature 0. Speculative decoding changes
the shape of the forward pass and perturbs logits in the low bits (see
verify.py); leaving it on would mix that in with the KV effect under test.

What this can and cannot settle:

  It CAN say whether the two configs succeed at the same rate, and whether
  their text is byte-identical.
  It CANNOT certify that two differing texts are of equal quality in general.
  Where they differ, the task outcome is the arbiter, not the diff.

  kv_quality.py --bin ~/llama.cpp/b10472-vulkan --ctx 65536
"""
import argparse
import importlib.util
import json
import pathlib
import random
import re
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("emit_row", HERE / "emit_row.py")
emit_row = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emit_row)

# Distinct needles so a later probe cannot be answered from an earlier one, and
# so a model that pattern-matches "passphrase" without reading gets it wrong.
NEEDLES = [
    ("vault 7", "CORAL-MERIDIAN-48"),
    ("vault 12", "BASALT-LANTERN-91"),
    ("vault 33", "TUNDRA-VESPER-07"),
    ("vault 58", "QUARTZ-HALYARD-26"),
    ("vault 64", "AMBER-SEXTANT-73"),
    ("vault 81", "INDIGO-CAIRN-15"),
    ("vault 95", "PUMICE-ORACLE-62"),
    ("vault 99", "SABLE-KESTREL-30"),
    ("vault 21", "FLINT-AURORA-84"),
]

CODE_TASKS = [
    ("rle",
     "Write a Python function `encode(s: str) -> str` that run-length encodes a "
     "string: 'aaabbc' becomes 'a3b2c1'. Single characters still get their count. "
     "Return only the function in a ```python block, no explanation.",
     "assert encode('aaabbc')=='a3b2c1'\n"
     "assert encode('')==''\n"
     "assert encode('x')=='x1'\n"
     "assert encode('aabbaa')=='a2b2a2'\n"),
    ("roman",
     "Write a Python function `to_roman(n: int) -> str` converting 1..3999 to a "
     "Roman numeral. Return only the function in a ```python block, no explanation.",
     "assert to_roman(1)=='I'\nassert to_roman(4)=='IV'\n"
     "assert to_roman(1994)=='MCMXCIV'\nassert to_roman(3999)=='MMMCMXCIX'\n"),
    ("balanced",
     "Write a Python function `balanced(s: str) -> bool` returning True if the "
     "brackets ()[]{} in s are correctly nested and matched. Other characters are "
     "ignored. Return only the function in a ```python block, no explanation.",
     "assert balanced('([]{})')\nassert not balanced('([)]')\n"
     "assert balanced('')\nassert not balanced('(')\nassert balanced('a(b)c')\n"),
    ("merge",
     "Write a Python function `merge(a: list, b: list) -> list` merging two "
     "already-sorted integer lists into one sorted list, without using sorted(). "
     "Return only the function in a ```python block, no explanation.",
     "assert merge([1,3,5],[2,4])==[1,2,3,4,5]\nassert merge([],[1])==[1]\n"
     "assert merge([],[])==[]\nassert merge([1,1],[1])==[1,1,1]\n"),
    ("wordfreq",
     "Write a Python function `top_word(text: str) -> str` returning the most "
     "frequent whitespace-separated word, lowercased, ties broken by first "
     "appearance. Return only the function in a ```python block, no explanation.",
     "assert top_word('a b a')=='a'\nassert top_word('X x y')=='x'\n"
     "assert top_word('one')=='one'\n"),
    ("basen",
     "Write a Python function `to_base(n: int, b: int) -> str` rendering a "
     "non-negative int in base b (2..16) using lowercase digits. to_base(0,2) is "
     "'0'. Return only the function in a ```python block, no explanation.",
     "assert to_base(0,2)=='0'\nassert to_base(255,16)=='ff'\n"
     "assert to_base(10,2)=='1010'\nassert to_base(7,8)=='7'\n"),
]

CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
THINK = re.compile(r"<think>.*?</think>", re.S)


def haystack(corpus: str, target_chars: int, needle: str, depth: float, rng):
    """Filler with one planted fact at a fractional depth.

    The filler is the held-out corpus repeated -- it is prose the model has not
    been tuned on, and repetition is standard for this probe: what is under
    test is retrieval from a long span, not the interest of the span.
    """
    reps = max(1, target_chars // max(1, len(corpus)) + 1)
    text = (corpus * reps)[:target_chars]
    cut = int(len(text) * depth)
    # Land on a sentence boundary so the needle is not spliced mid-word.
    nl = text.find(". ", cut)
    cut = (nl + 2) if 0 <= nl < cut + 2000 else cut
    return text[:cut] + "\n\n" + needle + "\n\n" + text[cut:]


def run_recall(port, corpus, lengths, depths, log):
    out = []
    for i, tchars in enumerate(lengths):
        for j, depth in enumerate(depths):
            where, secret = NEEDLES[(i * len(depths) + j) % len(NEEDLES)]
            needle = (f"Remember this. The secret passphrase for {where} "
                      f"is {secret}. That is the passphrase for {where}.")
            body = haystack(corpus, tchars, needle, depth, None)
            prompt = (body + "\n\nQuestion: what is the secret passphrase for "
                      f"{where}? Reply with only the passphrase.\nAnswer:")
            t0 = time.time()
            # 384, not 32. This model opens with a <think> block whose length
            # grows with context, and a 32-token budget truncated the answer
            # before the passphrase at 58k tokens -- scoring a MISS that was
            # the harness running out of room, not the model failing to
            # retrieve. A recall probe that cannot tell those apart measures
            # nothing.
            r = emit_row.complete(port, prompt, 384)
            txt = (r.get("content") or "").strip()
            n = r.get("timings", {}).get("prompt_n")
            hit = secret.lower() in txt.lower()
            said = THINK.sub("", txt).strip() or txt
            out.append({"target_chars": tchars, "prompt_tokens": n,
                        "depth": depth, "needle": secret, "answer": txt[:400],
                        "after_think": said[:120], "hit": hit,
                        "truncated": r.get("stop_type") == "limit",
                        "seconds": round(time.time() - t0, 1)})
            log(f"    recall {n or '?':>7} tok  depth {depth:>4.0%}  "
                f"{'HIT ' if hit else 'MISS'}  {said[:44]!r}")
    return out


def run_codegen(port, log):
    out = []
    for name, prompt, asserts in CODE_TASKS:
        r = emit_row.complete(port, prompt, 512)
        txt = r.get("content") or ""
        m = CODE_FENCE.search(txt)
        code = m.group(1) if m else txt
        ok, err = exec_check(code, asserts)
        out.append({"task": name, "chars": len(txt), "text": txt,
                    "code": code, "passes": ok, "error": err})
        log(f"    codegen {name:<9} {'PASS' if ok else 'FAIL'}"
            f"{'' if ok else '  ' + (err or '')[:60]}")
    return out


def exec_check(code, asserts):
    """Run the generated function against its asserts, in a separate process."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code + "\n\n" + asserts + "\nprint('OK')\n")
        path = fh.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=30)
        if r.returncode == 0 and "OK" in r.stdout:
            return True, None
        return False, (r.stderr.strip().splitlines() or [""])[-1]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


def measure(bindir, model, ctx, ctk, ctv, corpus, lengths, depths, log):
    cfg = {"ctx": ctx, "ctk": ctk, "ctv": ctv, "spec": None, "n_max": None}
    defaults = {"fa": "on", "ngl": 99, "threads": 8}
    log(f"  starting llama-server  ctk={ctk} ctv={ctv} ctx={ctx}")
    proc, port, stop = emit_row.start_server(bindir, model, cfg, defaults)
    try:
        return {"recall": run_recall(port, corpus, lengths, depths, log),
                "codegen": run_codegen(port, log)}
    finally:
        stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--lengths", default="16000,120000,240000",
                    help="haystack sizes in CHARACTERS, not tokens")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import os
    model = pathlib.Path(a.model or os.environ["PERF_LAB_MODEL"])
    corpus = pathlib.Path(a.corpus or os.environ.get(
        "PERF_LAB_HELDOUT", str(pathlib.Path.home() / ".perf-lab/heldout/corpus.txt"))
    ).read_text()
    lengths = [int(x) for x in a.lengths.split(",")]
    depths = [float(x) for x in a.depths.split(",")]
    bindir = pathlib.Path(a.bin)

    def log(m):
        print(m, file=sys.stderr, flush=True)

    results = {}
    for ctk, ctv in (("q8_0", "q8_0"), ("q8_0", "q4_0")):
        key = f"{ctk}/{ctv}"
        log(f"=== {key} ===")
        results[key] = measure(bindir, model, a.ctx, ctk, ctv, corpus,
                               lengths, depths, log)

    report = {"bin": str(bindir), "ctx": a.ctx, "model": model.stem,
              "results": results, "comparison": compare(results)}
    text = json.dumps(report, indent=2)
    if a.out:
        pathlib.Path(a.out).write_text(text)
        log(f"wrote {a.out}")
    else:
        print(text)


def compare(results):
    a, b = results["q8_0/q8_0"], results["q8_0/q4_0"]
    rec = [{"prompt_tokens": x["prompt_tokens"], "depth": x["depth"],
            "q8q8": x["hit"], "q8q4": y["hit"],
            "truncated": bool(x.get("truncated") or y.get("truncated"))}
           for x, y in zip(a["recall"], b["recall"])]
    code = [{"task": x["task"], "q8q8": x["passes"], "q8q4": y["passes"],
             "identical": x["text"] == y["text"]}
            for x, y in zip(a["codegen"], b["codegen"])]
    return {
        "recall_hits_q8q8": sum(x["q8q8"] for x in rec),
        "recall_hits_q8q4": sum(x["q8q4"] for x in rec),
        "recall_probes": len(rec),
        "recall_disagreements": [x for x in rec if x["q8q8"] != x["q8q4"]],
        "codegen_pass_q8q8": sum(x["q8q8"] for x in code),
        "codegen_pass_q8q4": sum(x["q8q4"] for x in code),
        "codegen_tasks": len(code),
        "codegen_byte_identical": sum(x["identical"] for x in code),
        "codegen_detail": code,
    }


if __name__ == "__main__":
    main()

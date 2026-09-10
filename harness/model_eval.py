#!/usr/bin/env python3
"""What is this model actually good at? Scored per skill, not per vibe.

kv_quality.py answers "do two KV configs behave the same". This answers a
different question -- given a model file, which skills does it hold up on --
and it exists because "try them and see" produces opinions that cannot be
compared across models or re-checked next month.

Every task is scored programmatically. No task is graded by reading the output
and forming an impression:

  code       generate a function, then EXECUTE it against asserts. The only
             check that separates "different text" from "worse answer".
  math       multi-step word problems with a single numeric answer, matched
             after stripping the reasoning block. Arithmetic is unforgiving,
             which is what makes it a usable signal at small sizes.
  instruct   strict output-format compliance -- exact JSON keys, exact list
             lengths, exact casing. Scored by parsing, so "close enough"
             fails, which is the point: format compliance is the skill.
  extract    pull a named fact out of a passage that contains near-miss
             distractors. Tests reading, not recall of training data.
  recall     a planted fact at depth in a long context. Reuses the shape of
             kv_quality.py's probe; here it varies by model, not by KV type.

Deliberately NOT measured: prose quality, "helpfulness", tone, creativity.
Those need human or model judges, and a judge that is itself a small model
measures the judge. Where this file is silent, it is silent on purpose.

Both temperature 0. Speculation off -- this is about the weights, not decode.

  model_eval.py --bin ~/llama.cpp/b10472-vulkan \
      --model /path/to/Model-Q6_K.gguf --ctx 32768 --out results/eval-x.json
"""
import argparse
import importlib.util
import json
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("emit_row", HERE / "emit_row.py")
emit_row = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emit_row)

THINK = re.compile(r"<think>.*?</think>", re.S)
CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def strip_think(t):
    return THINK.sub("", t or "").strip()


# ---------------------------------------------------------------- code
CODE_TASKS = [
    ("rle",
     "Write a Python function rle(s) that run-length encodes a string: "
     "'aaabbc' becomes 'a3b2c1'. Single characters still get their count. "
     "Return only the function in a ```python block, no explanation.",
     "assert rle('aaabbc')=='a3b2c1'\nassert rle('')==''\n"
     "assert rle('x')=='x1'\nassert rle('aab')=='a2b1'\n"),
    ("roman",
     "Write a Python function to_roman(n) converting 1..3999 to Roman "
     "numerals. Return only the function in a ```python block.",
     "assert to_roman(1)=='I'\nassert to_roman(3999)=='MMMCMXCIX'\n"
     "assert to_roman(4)=='IV'\nassert to_roman(1994)=='MCMXCIV'\n"),
    ("balanced",
     "Write a Python function balanced(s) returning True if brackets "
     "()[]{} in s are correctly nested and closed, else False. Other "
     "characters are ignored. Return only the function in a ```python block.",
     "assert balanced('([]{})')\nassert not balanced('([)]')\n"
     "assert balanced('')\nassert not balanced('(')\n"),
    ("merge",
     "Write a Python function merge(a,b) merging two sorted lists into one "
     "sorted list, without using sorted() or .sort(). Return only the "
     "function in a ```python block.",
     "assert merge([1,3],[2,4])==[1,2,3,4]\nassert merge([],[1])==[1]\n"
     "assert merge([1,1],[1])==[1,1,1]\n"),
    ("wordfreq",
     "Write a Python function top_word(s) returning the most frequent "
     "lowercase word in s, breaking ties alphabetically. Words are "
     "sequences of letters. Return only the function in a ```python block.",
     "assert top_word('a b a')=='a'\nassert top_word('b a')=='a'\n"
     "assert top_word('The the cat')=='the'\n"),
    ("basen",
     "Write a Python function to_base(n,b) converting a non-negative int n "
     "to a string in base b (2..16), using lowercase digits. Zero is '0'. "
     "Return only the function in a ```python block.",
     "assert to_base(0,2)=='0'\nassert to_base(255,16)=='ff'\n"
     "assert to_base(10,2)=='1010'\nassert to_base(7,8)=='7'\n"),
]


def run_code(port, log, budget):
    out = []
    for name, prompt, asserts in CODE_TASKS:
        r = emit_row.complete(port, prompt, budget)
        txt = r.get("content") or ""
        m = CODE_FENCE.search(txt)
        body = m.group(1) if m else strip_think(txt)
        ok, err = False, None
        try:
            ns = {}
            exec(compile(body + "\n" + asserts, "<gen>", "exec"), ns)
            ok = True
        except Exception as e:  # a wrong answer is data, not a crash
            err = f"{type(e).__name__}: {e}"[:200]
        out.append({"task": name, "passes": ok, "error": err,
                    "text": body[:600]})
        log(f"    code     {name:<10} {'PASS' if ok else 'FAIL'}"
            f"{'' if ok else '  ' + str(err)[:60]}")
    return out


# ---------------------------------------------------------------- math
MATH_TASKS = [
    ("crates", "A warehouse has 17 crates. Each crate holds 24 bottles. "
     "38 bottles are broken and removed. How many bottles remain? "
     "Give only the final number.", 370),
    ("trip", "A car drives 240 km at 80 km/h, rests 30 minutes, then drives "
     "150 km at 60 km/h. How many minutes did the whole trip take? "
     "Give only the final number.", 360),
    ("discount", "A jacket costs 180 dollars. It is discounted 25 percent, "
     "then 8 percent sales tax is added. What is the final price in dollars, "
     "rounded to the nearest whole dollar? Give only the final number.", 146),
    ("ages", "Ana is three times as old as Ben. In 6 years she will be twice "
     "as old as Ben. How old is Ana now? Give only the final number.", 18),
    ("pipes", "One pipe fills a tank in 6 hours, another in 12 hours. "
     "Running together, how many hours to fill it? Give only the final "
     "number.", 4),
    ("seats", "A theatre has 22 rows. The first row has 14 seats and each "
     "later row has 2 more than the one before. How many seats total? "
     "Give only the final number.", 770),
    ("coins", "A jar has 3 times as many dimes as quarters, and 5 more "
     "nickels than dimes. The total value is 655 cents. How many quarters "
     "are there? Give only the final number.", 9),
    ("primes", "What is the sum of all prime numbers strictly between 20 "
     "and 50? Give only the final number.", 251),
]

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def run_math(port, log, budget):
    out = []
    for name, prompt, want in MATH_TASKS:
        r = emit_row.complete(port, prompt, budget)
        said = strip_think(r.get("content") or "")
        nums = [n.replace(",", "") for n in NUM.findall(said)]
        # last number in the post-reasoning text: models restate the answer
        got = None
        if nums:
            try:
                got = float(nums[-1])
            except ValueError:
                got = None
        ok = got is not None and abs(got - want) < 1e-6
        out.append({"task": name, "want": want, "got": got, "passes": ok,
                    "said": said[:200],
                    "truncated": r.get("stop_type") == "limit"})
        log(f"    math     {name:<10} {'PASS' if ok else 'FAIL'}"
            f"  want={want} got={got}")
    return out


# ---------------------------------------------------------------- instruct
def _json_obj(txt):
    t = strip_think(txt)
    m = re.search(r"```(?:json)?\s*\n(.*?)```", t, re.S)
    if m:
        t = m.group(1)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


def _chk_keys(txt):
    d = _json_obj(txt)
    return isinstance(d, dict) and set(d) == {"name", "count", "tags"} and \
        isinstance(d["tags"], list) and len(d["tags"]) == 3


def _chk_three_lines(txt):
    lines = [l for l in strip_think(txt).splitlines() if l.strip()]
    return len(lines) == 3 and all(l.strip().startswith("- ") for l in lines)


def _chk_upper(txt):
    t = strip_think(txt)
    return t.isupper() and "BANANA" in t


def _chk_nocomma(txt):
    t = strip_think(txt)
    return "," not in t and len(t.split()) >= 8


def _chk_exact_words(txt):
    return len(strip_think(txt).split()) == 5


def _chk_yesno(txt):
    return strip_think(txt).strip().rstrip(".").lower() in ("yes", "no")


INSTRUCT_TASKS = [
    ("json_keys",
     'Reply with ONLY a JSON object having exactly the keys "name", "count", '
     '"tags". "name" a string, "count" an integer, "tags" a list of exactly '
     "3 strings. Describe a fruit basket. No prose, no code fence.",
     _chk_keys),
    ("three_bullets",
     "List exactly three benefits of version control. Reply with exactly "
     "three lines, each beginning with '- '. No heading, no other text.",
     _chk_three_lines),
    ("shout",
     "Reply in ALL UPPERCASE with a single sentence about bananas. "
     "The word BANANA must appear.", _chk_upper),
    ("no_commas",
     "Write one sentence of at least 8 words about the sea, containing no "
     "comma characters at all.", _chk_nocomma),
    ("exact_five",
     "Reply with a phrase of exactly five words describing rain. "
     "No punctuation, no other text.", _chk_exact_words),
    ("yes_no",
     "Is 91 a prime number? Reply with exactly one word: Yes or No.",
     _chk_yesno),
]


def run_instruct(port, log, budget):
    out = []
    for name, prompt, check in INSTRUCT_TASKS:
        r = emit_row.complete(port, prompt, budget)
        txt = r.get("content") or ""
        try:
            ok = bool(check(txt))
        except Exception:
            ok = False
        out.append({"task": name, "passes": ok,
                    "said": strip_think(txt)[:200],
                    "truncated": r.get("stop_type") == "limit"})
        log(f"    instruct {name:<10} {'PASS' if ok else 'FAIL'}")
    return out


# ---------------------------------------------------------------- extract
PASSAGE = """\
The Kestrel Bridge was completed in 1971 by the Halloran Engineering Company.
An earlier crossing, the Vance Bridge, was built at the same site in 1934 by
Delmar Ironworks and demolished in 1969. The Kestrel Bridge spans 412 metres
and carries four lanes. The Vance Bridge spanned 268 metres and carried two
lanes. In 1998 the Kestrel Bridge was resurfaced by Halloran, and in 2004 its
southern approach was widened by Trent Civil. The Vance Bridge's original
design engineer was Ida Moreau; the Kestrel Bridge's was Paul Anselm. A third
crossing, the Rowe Viaduct, opened in 2011 downstream and spans 530 metres.
"""

EXTRACT_TASKS = [
    ("span_kestrel", "How many metres does the Kestrel Bridge span? "
     "Reply with only the number.", "412"),
    ("builder_vance", "Which company built the Vance Bridge? "
     "Reply with only the company name.", "Delmar Ironworks"),
    ("engineer_kestrel", "Who was the design engineer of the Kestrel Bridge? "
     "Reply with only the name.", "Paul Anselm"),
    ("year_widened", "In which year was the Kestrel Bridge's southern "
     "approach widened? Reply with only the year.", "2004"),
    ("span_rowe", "How many metres does the Rowe Viaduct span? "
     "Reply with only the number.", "530"),
]


def run_extract(port, log, budget):
    out = []
    for name, q, want in EXTRACT_TASKS:
        r = emit_row.complete(port, PASSAGE + "\n\nQuestion: " + q + "\nAnswer:",
                              budget)
        said = strip_think(r.get("content") or "")
        ok = want.lower() in said.lower()
        out.append({"task": name, "want": want, "passes": ok,
                    "said": said[:200],
                    "truncated": r.get("stop_type") == "limit"})
        log(f"    extract  {name:<16} {'PASS' if ok else 'FAIL'}  {said[:40]!r}")
    return out


# ---------------------------------------------------------------- recall
def run_recall(port, corpus, lengths, log, budget):
    out = []
    needles = [("the north vault", "CORAL-MERIDIAN-48"),
               ("the west annex", "BASALT-LANTERN-91"),
               ("the old signal box", "TUNDRA-VESPER-07")]
    for i, tchars in enumerate(lengths):
        where, secret = needles[i % len(needles)]
        needle = (f"Remember this. The secret passphrase for {where} is "
                  f"{secret}. That is the passphrase for {where}.")
        reps = max(1, tchars // max(1, len(corpus)) + 1)
        text = (corpus * reps)[:tchars]
        cut = int(len(text) * 0.5)
        nl = text.find(". ", cut)
        cut = (nl + 2) if 0 <= nl < cut + 2000 else cut
        body = text[:cut] + "\n\n" + needle + "\n\n" + text[cut:]
        prompt = (body + "\n\nQuestion: what is the secret passphrase for "
                  f"{where}? Reply with only the passphrase.\nAnswer:")
        t0 = time.time()
        try:
            r = emit_row.complete(port, prompt, budget)
        except Exception as e:
            out.append({"target_chars": tchars, "skipped": str(e)[:120],
                        "passes": None})
            log(f"    recall   {tchars:>8} ch  SKIP  {str(e)[:50]}")
            continue
        said = strip_think(r.get("content") or "")
        ok = secret.lower() in (r.get("content") or "").lower()
        out.append({"target_chars": tchars,
                    "prompt_tokens": r.get("timings", {}).get("prompt_n"),
                    "passes": ok, "said": said[:120],
                    "seconds": round(time.time() - t0, 1)})
        log(f"    recall   {r.get('timings',{}).get('prompt_n','?'):>8} tok  "
            f"{'PASS' if ok else 'FAIL'}")
    return out


# ---------------------------------------------------------------- driver
def score(rows):
    scored = [r for r in rows if r.get("passes") is not None]
    return {"pass": sum(1 for r in scored if r["passes"]),
            "of": len(scored),
            "skipped": len(rows) - len(scored)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--recall-lengths", default="",
                    help="CHARACTERS, comma separated; empty skips recall")
    ap.add_argument("--budget", type=int, default=768,
                    help="n_predict; reasoning models need room before the answer")
    ap.add_argument("--suites", default="code,math,instruct,extract,recall")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import os
    model = pathlib.Path(a.model)
    bindir = pathlib.Path(a.bin)
    suites = [s.strip() for s in a.suites.split(",") if s.strip()]

    def log(m):
        print(m, file=sys.stderr, flush=True)

    cfg = {"ctx": a.ctx, "ctk": "q8_0", "ctv": "q4_0",
           "spec": None, "n_max": None}
    defaults = {"fa": "on", "ngl": 99, "threads": 8}
    log(f"  starting llama-server  {model.name}  ctx={a.ctx}")
    proc, port, stop = emit_row.start_server(bindir, model, cfg, defaults)
    results = {}
    try:
        if "code" in suites:
            results["code"] = run_code(port, log, a.budget)
        if "math" in suites:
            results["math"] = run_math(port, log, a.budget)
        if "instruct" in suites:
            results["instruct"] = run_instruct(port, log, a.budget)
        if "extract" in suites:
            results["extract"] = run_extract(port, log, a.budget)
        if "recall" in suites and a.recall_lengths:
            corpus = pathlib.Path(a.corpus or os.environ.get(
                "PERF_LAB_HELDOUT",
                str(pathlib.Path.home() / ".perf-lab/heldout/corpus.txt"))
            ).read_text()
            lengths = [int(x) for x in a.recall_lengths.split(",")]
            results["recall"] = run_recall(port, corpus, lengths, log, a.budget)
    finally:
        stop()

    report = {"bin": str(bindir), "model": model.stem, "ctx": a.ctx,
              "budget": a.budget, "results": results,
              "score": {k: score(v) for k, v in results.items()}}
    text = json.dumps(report, indent=2)
    if a.out:
        pathlib.Path(a.out).write_text(text)
        log(f"wrote {a.out}")
    else:
        print(text)
    for k, s in report["score"].items():
        log(f"  {k:<9} {s['pass']}/{s['of']}"
            + (f"  ({s['skipped']} skipped)" if s["skipped"] else ""))


if __name__ == "__main__":
    main()

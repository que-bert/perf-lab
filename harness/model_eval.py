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
import textwrap
import time

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("emit_row", HERE / "emit_row.py")
emit_row = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emit_row)

THINK = re.compile(r"<think>.*?</think>", re.S)
# A reasoning model prompted RAW emits a closing </think> with no opening tag,
# because the opening tag is normally injected by the chat template rather than
# generated. Measured 2026-09-10: every reasoning block in raw mode arrived this
# way, so THINK alone stripped nothing and the whole scratchpad -- every
# intermediate number in it -- was scored as the answer. Anything before an
# unmatched </think> is reasoning.
ORPHAN_THINK = re.compile(r"^.*?</think>", re.S)
CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def strip_think(t):
    t = THINK.sub("", t or "")
    if "</think>" in t:
        t = ORPHAN_THINK.sub("", t)
    return t.strip()


# --------------------------------------------------------------- backends
# Everything below calls _complete(port, prompt, n_predict) and reads back
# {content, stop_type, timings{prompt_n, predicted_per_second}}. That is
# llama-server's own shape; the ollama backend adapts to it.
#
# The ollama path exists because some GGUFs only load there. ollama ships
# model files that upstream llama.cpp refuses -- measured 2026-09-10 on
# qwen3.5:9b (mrope sections 3 where llama.cpp wants 4) and gemma4:e4b/e2b
# (2131 tensors declared, 720 built). Capability is a property of the weights,
# so it is fair to measure those models under ollama; SPEED is not, because
# the runtime, KV types and speculation all differ. Never put an ollama-backed
# number in a speed table.
_API = "llama"
_OLLAMA_MODEL = None
_CTX = 32768


def _complete_llama(port, prompt, n_predict, extra=None):
    return emit_row.complete(port, prompt, n_predict, extra)


def _complete_ollama(port, prompt, n_predict, extra=None):
    import urllib.request
    body = {"model": _OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "raw": True,
            # num_ctx must be sent explicitly: ollama otherwise serves its own
            # default and the run silently measures a context the caller never
            # asked for -- observed at 32768 against a requested 65536.
            "options": {"temperature": 0, "num_predict": n_predict,
                        "num_ctx": _CTX}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    ns = d.get("eval_duration") or 0
    return {
        "content": d.get("response", ""),
        # ollama says "length" where llama-server says "limit".
        "stop_type": "limit" if d.get("done_reason") == "length" else "eos",
        "timings": {
            "prompt_n": d.get("prompt_eval_count"),
            "predicted_n": d.get("eval_count"),
            "predicted_per_second": (d.get("eval_count", 0) / (ns / 1e9))
            if ns else None,
        },
    }


def _chat_llama(port, prompt, n_predict, extra=None):
    """Same prompt, but through the model's own chat template.

    Worth a separate backend rather than a detail: an instruct-tuned model
    prompted raw is being tested outside the format it was tuned in, and the
    format suites are exactly where that shows. Which mode a table was taken
    in therefore has to be recorded next to the score.
    """
    import urllib.request
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict, "temperature": 0, "stream": False}
    body.update(extra or {})
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    ch = (d.get("choices") or [{}])[0]
    u = d.get("usage") or {}
    msg = ch.get("message") or {}
    # Some templates split the reasoning block into its own field instead of
    # leaving it inline; put it back so strip_think() sees a consistent shape.
    content = msg.get("content") or ""
    if msg.get("reasoning_content"):
        content = f"<think>{msg['reasoning_content']}</think>" + content
    return {"content": content,
            "stop_type": "limit" if ch.get("finish_reason") == "length"
            else "eos",
            "timings": {"prompt_n": u.get("prompt_tokens"),
                        "predicted_n": u.get("completion_tokens"),
                        "predicted_per_second": None}}


def _chat_ollama(port, prompt, n_predict, extra=None):
    import urllib.request
    body = {"model": _OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "num_predict": n_predict,
                        "num_ctx": _CTX}}
    body.update(extra or {})
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    ns = d.get("eval_duration") or 0
    content = d.get("response", "")
    if d.get("thinking"):
        content = f"<think>{d['thinking']}</think>" + content
    return {"content": content,
            "stop_type": "limit" if d.get("done_reason") == "length"
            else "eos",
            "timings": {"prompt_n": d.get("prompt_eval_count"),
                        "predicted_n": d.get("eval_count"),
                        "predicted_per_second": (
                            d.get("eval_count", 0) / (ns / 1e9))
                        if ns else None}}


_BACKENDS = {("llama", False): _complete_llama, ("llama", True): _chat_llama,
             ("ollama", False): _complete_ollama, ("ollama", True): _chat_ollama}
_CHAT = False
# Both families' chat templates take an enable_thinking switch. Turning it off
# is the single largest lever on time-to-answer for a reasoning model -- it
# removes the scratchpad rather than decoding it faster -- and it is the only
# thing measured here that makes Ornith's short-answer tasks terminate at all.
# Only reachable through the template, so it requires chat mode.
_NO_THINK = False


def _complete(port, prompt, n_predict, extra=None):
    if _NO_THINK and _CHAT:
        extra = dict(extra or {})
        if _API == "ollama":
            extra["think"] = False
        else:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
    return _BACKENDS[(_API, _CHAT)](port, prompt, n_predict, extra)


# ------------------------------------------------------------ tool calling
# Native tool calling, not prompt-engineered JSON. mimir builds a Request with
# a Tools list and reads tool_calls back off the response, so a suite that
# scores hand-rolled JSON in the message body would measure something mimir
# never does. Both backends therefore go through the endpoint that carries a
# tools array: /v1/chat/completions for llama-server, /api/chat for ollama --
# NOT /api/generate, which the other ollama paths here use and which has no
# tool support at all.
#
# Always chat-shaped: a tool catalog only reaches the model through the chat
# template. There is no raw-mode row for this suite, by construction.

def _toolcall_llama(port, prompt, tools, n_predict, extra=None):
    import urllib.request
    body = {"messages": [{"role": "user", "content": prompt}],
            "tools": tools, "tool_choice": "auto",
            "max_tokens": n_predict, "temperature": 0, "stream": False}
    body.update(extra or {})
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    calls = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
    return {"calls": calls, "content": msg.get("content") or "",
            "stop_type": "limit" if ch.get("finish_reason") == "length"
            else "eos"}


def _toolcall_ollama(port, prompt, tools, n_predict, extra=None):
    import urllib.request
    body = {"model": _OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "tools": tools, "stream": False,
            "options": {"temperature": 0, "num_predict": n_predict,
                        "num_ctx": _CTX}}
    body.update(extra or {})
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    msg = d.get("message") or {}
    calls = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        # ollama returns arguments already decoded; llama-server returns a
        # JSON string. Normalise to the string form so one scorer reads both.
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args)
        calls.append({"name": fn.get("name"), "arguments": args})
    return {"calls": calls, "content": msg.get("content") or "",
            "stop_type": "limit" if d.get("done_reason") == "length"
            else "eos"}


def _toolcall(port, prompt, tools, n_predict):
    extra = None
    if _NO_THINK:
        extra = {"think": False} if _API == "ollama" \
            else {"chat_template_kwargs": {"enable_thinking": False}}
    fn = _toolcall_ollama if _API == "ollama" else _toolcall_llama
    return fn(port, prompt, tools, n_predict, extra)


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
        r = _complete(port, prompt, budget)
        txt = r.get("content") or ""
        truncated = r.get("stop_type") == "limit"
        # Strip the scratchpad BEFORE looking for a fence. A reasoning model
        # drafts a ```python block inside <think> and then writes the finished
        # version after it; searching the raw text finds the draft first.
        # Measured 2026-09-11 on minicpm-q4km: `balanced` scored on the literal
        # stub "def balanced(s):\n    # code" from the scratchpad while the
        # real answer was never examined.
        said = strip_think(txt)
        m = CODE_FENCE.search(said)
        # A fence nested inside prose or a list arrives uniformly indented, and
        # exec() then raises IndentationError on line 1. Scored as a failure
        # that way, qwen3.5:9b read 1/6 on code it had in fact written
        # correctly -- the indentation was the reviewer's problem, not the
        # model's. dedent is a no-op on an unindented body.
        body = textwrap.dedent(m.group(1) if m else said)
        ok, err = False, None
        try:
            ns = {}
            exec(compile(body + "\n" + asserts, "<gen>", "exec"), ns)
            ok = True
        except Exception as e:  # a wrong answer is data, not a crash
            err = f"{type(e).__name__}: {e}"[:200]
        # A generation cut off inside its reasoning block emitted no answer at
        # all, and exec() then fails with NameError on the function that was
        # never defined. Counting that as wrong code measures the budget, not
        # the model -- the same artifact the math suite records as TRUNC.
        # Measured 2026-09-11: minicpm-q4km chat hit the limit on 5 of 6 items
        # at budget 6144 and read 1/6.
        if truncated and not ok:
            ok, err = None, None
        verdict = "TRUNC" if ok is None else ("PASS" if ok else "FAIL")
        out.append({"task": name, "passes": ok, "error": err,
                    "text": body[:600], "truncated": truncated})
        log(f"    code     {name:<10} {verdict:<5}"
            f"{'' if ok is not False else '  ' + str(err)[:60]}")
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


def run_math(port, log, budget, only=None):
    out = []
    for name, prompt, want in MATH_TASKS:
        if only and name not in only:
            continue
        r = _complete(port, prompt, budget)
        said = strip_think(r.get("content") or "")
        truncated = r.get("stop_type") == "limit"
        nums = [n.replace(",", "") for n in NUM.findall(said)]
        # Last number, because a finished answer ends on its result. This is
        # only sound if the answer FINISHED: these models lead with
        # "**Answer:** 408", then work through it and correct themselves, so a
        # response cut mid-explanation ends on an intermediate value. On
        # 2026-09-10 that scored MiniCPM-Q8 0/8 when it had in fact computed
        # the first answer correctly -- a scorer artifact, not a model failure.
        # A cut-off answer is unscoreable, so record it as such rather than
        # counting it wrong.
        got = None
        if nums:
            try:
                got = float(nums[-1])
            except ValueError:
                got = None
        if truncated:
            ok = None
            verdict = "TRUNC"
        else:
            ok = got is not None and abs(got - want) < 1e-6
            verdict = "PASS" if ok else "FAIL"
        out.append({"task": name, "want": want, "got": got, "passes": ok,
                    "said": said[:300], "truncated": truncated})
        log(f"    math     {name:<10} {verdict:<5} want={want} got={got}")
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
        r = _complete(port, prompt, budget)
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
        r = _complete(port, PASSAGE + "\n\nQuestion: " + q + "\nAnswer:",
                              budget)
        said = strip_think(r.get("content") or "")
        ok = want.lower() in said.lower()
        out.append({"task": name, "want": want, "passes": ok,
                    "said": said[:200],
                    "truncated": r.get("stop_type") == "limit"})
        log(f"    extract  {name:<16} {'PASS' if ok else 'FAIL'}  {said[:40]!r}")
    return out


# ---------------------------------------------------------------- research
# extract/ asks whether a model can find a fact that is written down. research/
# asks the harder thing a research assistant is actually for: combine facts
# that are written in different places, say where each came from, notice when
# two sources disagree, and refuse to answer what the sources do not contain.
#
# The last of those is the one that decides whether a small model is usable
# unsupervised. A model that answers every question confidently scores well on
# extract and is dangerous here, so `abstain` and `conflict` are weighted the
# same as the multi-hop items rather than treated as bonus tasks.
#
# All six are scored by parsing: a number, a name, or the presence of an
# absence marker. Nothing is graded by reading the prose.
RESEARCH_DOCS = """\
[D1] Ardent Freight operates three depots: Marrow, Kell, and Vance. The Marrow
depot opened in 2009 and is the oldest of the three.

[D2] Depot throughput for 2024. Marrow handled 41,200 containers. Kell handled
18,650 containers. Vance handled 27,150 containers.

[D3] The Kell depot was closed for retooling from March to June 2024, and ran
at reduced capacity for the remainder of the year.

[D4] Ardent's per-container handling fee was 14 dollars in 2023. It rose to 17
dollars in 2024.

[D5] The regional average depot throughput for 2024 was 31,000 containers
across all operators.

[D6] A later audit restated the Vance depot's 2024 throughput as 26,400
containers.
"""

# An absence marker. The model has to signal that the documents do not say,
# rather than produce a year. Deliberately broad -- any honest phrasing counts;
# what must not happen is a confident fabricated answer.
ABSENT = re.compile(
    r"\b(not (?:stated|given|specified|mentioned|provided|available|in the)"
    r"|no(?:t any)? (?:information|mention|record|data)"
    r"|does(?: not|n't) (?:say|state|mention|specify|provide|contain)"
    r"|do(?: not|n't) (?:say|state|mention|specify|provide|contain)"
    r"|cannot be determined|can(?:no|')t be determined"
    r"|unknown|unspecified|insufficient information|unable to determine)\b",
    re.I)

YEARISH = re.compile(r"\b(19|20)\d{2}\b")


def _nums(t):
    return [n.replace(",", "") for n in NUM.findall(t or "")]


def _has_num(t, want):
    for n in _nums(t):
        try:
            if abs(float(n) - want) < 1e-6:
                return True
        except ValueError:
            pass
    return False


def _first_num(t, want):
    """The FIRST number, for items whose prompt says "give only the number".

    Not _has_num: on 2026-09-10 a model answered 96,000 to the aggregate item,
    then talked itself through the arithmetic and mentioned 87,000 on the way.
    "Contains the right number somewhere" scored that a pass. What the caller
    receives is the first number, so that is what is scored.
    """
    ns = _nums(t)
    if not ns:
        return False
    try:
        return abs(float(ns[0]) - want) < 1e-6
    except ValueError:
        return False


def _chk_revenue(t):
    # 41,200 containers x 17 dollars. Requires D2 and D4 together.
    return _first_num(t, 700400)


def _chk_above_avg(t):
    # Only Marrow (41,200) clears the 31,000 regional average. Naming Kell or
    # Vance is a false positive and fails, which is the point of the item.
    low = t.lower()
    return "marrow" in low and "kell" not in low and "vance" not in low


def _chk_abstain(t):
    # D1 gives Marrow's opening year and no other. A model that answers 2009
    # has attached the one year present to the wrong depot.
    return bool(ABSENT.search(t)) and not YEARISH.search(t)


def _chk_conflict(t):
    # Naming the wrong depot fails even when both Vance figures also appear:
    # a model that answered "Kell" while quoting 27,150 and 26,400 passed the
    # first version of this check, which made the item measure nothing.
    low = t.lower()
    return ("vance" in low and "kell" not in low
            and _has_num(t, 27150) and _has_num(t, 26400))


def _chk_cite(t):
    return _has_num(t, 14) and re.search(r"\bd4\b", t, re.I) is not None


def _chk_aggregate(t):
    # 41,200 + 18,650 + 27,150
    return _first_num(t, 87000)


RESEARCH_TASKS = [
    ("multihop_revenue",
     "Using only the documents, what was Ardent's total 2024 handling-fee "
     "revenue from the Marrow depot, in dollars? Give only the final number.",
     _chk_revenue),
    ("multihop_above_avg",
     "Using only the documents, which Ardent depots handled more containers "
     "in 2024 than the regional average? Reply with only the depot names, "
     "comma separated.",
     _chk_above_avg),
    ("abstain_vance_year",
     "Using only the documents, in which year did the Vance depot open? "
     "If the documents do not say, reply exactly: NOT STATED.",
     _chk_abstain),
    ("conflict_vance",
     "The documents disagree about one depot's 2024 throughput. Name that "
     "depot and give both figures.",
     _chk_conflict),
    ("cite_fee_2023",
     "What was the per-container handling fee in 2023, and which document "
     "states it? Reply in the form <number>|<document id>, for example "
     "12|D9.",
     _chk_cite),
    ("aggregate_throughput",
     "Using the figures in document D2 only, what was the combined 2024 "
     "throughput of all three depots? Give only the final number.",
     _chk_aggregate),
]


def run_research(port, log, budget, complete=None):
    complete = complete or _complete
    out = []
    for name, q, check in RESEARCH_TASKS:
        prompt = (RESEARCH_DOCS + "\n\nAnswer using only the documents above. "
                  "If they do not contain the answer, say so.\n\nQuestion: "
                  + q + "\nAnswer:")
        r = complete(port, prompt, budget)
        raw = r.get("content") or ""
        said = strip_think(raw)
        truncated = r.get("stop_type") == "limit"

        def run(text):
            try:
                return bool(check(text))
            except Exception:
                return False

        # Two scores, the same split kv_quality.py draws for recall: `passes`
        # is what the caller actually receives, `anywhere` also counts an
        # answer that only ever appeared inside the reasoning block. A model
        # that reasons its way to the right answer and then prints something
        # else has not answered the question, but the gap between the two
        # columns says whether the failure is knowledge or presentation.
        ok, anywhere = run(said), run(raw)
        # A cut-off answer is unscoreable in both directions -- the same
        # reasoning as run_math. Record it rather than counting it wrong.
        if truncated and not ok:
            ok = None
        out.append({"task": name, "passes": ok, "anywhere": anywhere,
                    "said": said[:300], "raw_tail": raw[-200:],
                    "truncated": truncated})
        v = "TRUNC" if ok is None else ("PASS" if ok else "FAIL")
        flag = "  (in reasoning only)" if anywhere and ok is not True else ""
        log(f"    research {name:<20} {v:<5} {said[:46]!r}{flag}")
    return out


# ---------------------------------------------------------------- recall
# ---------------------------------------------------------------- tools
# A catalog shaped like the one an agent actually carries: several tools whose
# descriptions overlap, so picking correctly requires reading them rather than
# matching the first keyword. Two pairs are deliberately confusable --
# search_files against search_memory, and read_file against fetch_url.
TOOL_CATALOG = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the contents of a file from the local disk.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string",
                     "description": "Absolute path to the file."},
            "max_bytes": {"type": "integer",
                          "description": "Stop after this many bytes."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search_files",
        "description": "Search file CONTENTS on the local disk for a regex. "
                       "Use for source code and documents on this machine.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regex to match."},
            "directory": {"type": "string",
                          "description": "Directory to search under."}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Search the agent's own long-term MEMORY of past "
                       "conversations and decisions. Not files.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch a document over HTTP from a remote URL.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "The URL to fetch."}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "set_log_level",
        "description": "Change the daemon's logging verbosity.",
        "parameters": {"type": "object", "properties": {
            "level": {"type": "string", "enum": ["debug", "info", "warn",
                                                 "error"],
                      "description": "New level."}},
            "required": ["level"]}}},
]


def _args_of(call):
    """Decode a call's arguments, tolerating a model that emits junk."""
    try:
        v = json.loads(call.get("arguments") or "{}")
        return v if isinstance(v, dict) else {}
    except (ValueError, TypeError):
        return {}


def _one_call(calls, name):
    """Exactly one call, to `name`. Over-calling is a failure, not a pass."""
    return len(calls) == 1 and calls[0].get("name") == name


# Each task: (name, prompt, check(calls) -> bool). The checks are deliberately
# strict about COUNT as well as content: an agent that fires three tools when
# one was asked for is broken in a way that scoring "did it call the right one
# at least once" would hide.
TOOL_TASKS = [
    ("single_call",
     "What is in /etc/hostname? Use a tool.",
     lambda c: _one_call(c, "read_file")
     and _args_of(c[0]).get("path") == "/etc/hostname"),

    ("pick_files_not_memory",
     "Find every place the string TODO appears in the source files under "
     "/srv/app. Use a tool.",
     lambda c: _one_call(c, "search_files")
     and "TODO" in str(_args_of(c[0]).get("pattern", ""))),

    ("pick_memory_not_files",
     "What did we decide about the retry policy in an earlier conversation? "
     "Use a tool.",
     lambda c: _one_call(c, "search_memory")),

    ("pick_url_not_file",
     "Get the document at https://example.com/spec.txt. Use a tool.",
     lambda c: _one_call(c, "fetch_url")
     and _args_of(c[0]).get("url") == "https://example.com/spec.txt"),

    ("enum_constraint",
     "Turn the daemon's logging down so it only reports genuine failures, "
     "nothing lower. Use a tool.",
     lambda c: _one_call(c, "set_log_level")
     and _args_of(c[0]).get("level") == "error"),

    ("integer_type",
     "Read the first 500 bytes of /var/log/app.log. Use a tool.",
     lambda c: _one_call(c, "read_file")
     and _args_of(c[0]).get("path") == "/var/log/app.log"
     # The schema says integer. A model that sends "500" has produced a
     # payload the caller must coerce or reject, which is the bug this
     # catches -- mimir unmarshals into a typed struct.
     and _args_of(c[0]).get("max_bytes") == 500),

    ("abstain_no_tool_needed",
     "What is 17 multiplied by 24? Answer directly.",
     lambda c: len(c) == 0),

    ("abstain_no_such_tool",
     "Delete every file under /srv/app and confirm when done.",
     # Nothing in the catalog deletes anything. The failure mode being caught
     # is inventing a plausible name like delete_file, or bending an unrelated
     # tool to the task.
     lambda c: len(c) == 0),

    # ---- harder tier. The eight above saturate: MiniCPM5-2B-Q8_0 scored 8/8
    # first try on 2026-09-12. These four attack the failure modes that a
    # keyword match survives -- a lexical decoy, an invented argument, an
    # ordering dependency, and a plural that invites over-calling.
    ("lexical_decoy_memory",
     "Our design notes are checked into the repo. Search the files under "
     "/srv/app/docs for what they say about the memory allocator. Use a tool.",
     # Says "memory" twice and "notes" once, but the target is files on disk.
     # A model matching on the word picks search_memory.
     lambda c: _one_call(c, "search_files")),

    ("no_invented_argument",
     "Read the config file. Use a tool.",
     # The path is genuinely unspecified. Calling read_file with a guessed
     # path like /etc/config.json is worse than not calling: the caller acts
     # on a fabricated argument. Either abstain and ask, or do not call.
     lambda c: len(c) == 0),

    ("ordering_dependency",
     "The file /srv/app/target.txt contains a single URL. Fetch whatever "
     "document that URL points at. Use a tool.",
     # read_file must come first -- the URL is not knowable yet. A model that
     # calls fetch_url has hallucinated a URL it cannot have.
     lambda c: len(c) >= 1 and c[0].get("name") == "read_file"
     and _args_of(c[0]).get("path") == "/srv/app/target.txt"),

    ("plural_single_call",
     "Read /etc/hostname. Do not read anything else.",
     # Tests restraint: exactly one call, no speculative extras alongside it.
     lambda c: _one_call(c, "read_file")
     and _args_of(c[0]).get("path") == "/etc/hostname"),
]


def run_tools(port, log, budget):
    """Native tool calling: does the model call the right tool, once, with
    arguments that satisfy the declared schema -- and does it stay quiet when
    no tool applies.

    This is the suite closest to what an agent runtime needs, and the one the
    format suites were only ever a proxy for.
    """
    out = []
    for name, prompt, check in TOOL_TASKS:
        r = _toolcall(port, prompt, TOOL_CATALOG, budget)
        calls = r.get("calls") or []
        truncated = r.get("stop_type") == "limit"
        try:
            ok = bool(check(calls))
        except Exception:      # a malformed call is a fail, not a crash
            ok = False
        # Same rule the other suites use: a generation cut off before it could
        # emit a call has not been measured.
        if truncated and not ok:
            ok = None
        verdict = "TRUNC" if ok is None else ("PASS" if ok else "FAIL")
        names = [c.get("name") for c in calls]
        out.append({"task": name, "passes": ok, "truncated": truncated,
                    "calls": names,
                    "arguments": [c.get("arguments") for c in calls][:4],
                    "said": (r.get("content") or "")[:200]})
        log(f"    tools    {name:<24} {verdict:<5} calls={names}")
    return out


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
            r = _complete(port, prompt, budget)
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
    ap.add_argument("--suites",
                    default="code,math,instruct,extract,research,recall",
                    help="tools needs --chat: a tool catalog only reaches "
                         "the model through the chat template")
    ap.add_argument("--out", default="")
    ap.add_argument("--api", choices=("llama", "ollama"), default="llama",
                    help="ollama drives ollama's own runtime, for GGUFs "
                         "upstream llama.cpp refuses to load. Capability "
                         "only -- never speed.")
    ap.add_argument("--ollama-model", default="",
                    help="tag to send with --api ollama, e.g. gemma4:e4b")
    ap.add_argument("--chat", action="store_true",
                    help="prompt through the model's chat template instead "
                         "of raw completion")
    ap.add_argument("--no-think", action="store_true",
                    help="ask the template to suppress the reasoning block; "
                         "requires --chat")
    ap.add_argument("--label", default="",
                    help="name for this run in the report; defaults to the "
                         "model file stem")
    # Loading a model inside a supervised task gets that task killed on this box
    # while page cache fills, even with tens of GB free. Start the server as a
    # systemd --user unit and point this at its port; the client is cheap and
    # restartable, and the model stays warm if the client dies.
    # Long client runs get killed too while a model is resident, so the suite
    # has to be splittable into short calls. Names come from MATH_TASKS.
    ap.add_argument("--tasks", default="",
                    help="comma-separated math task names to run; empty = all")
    ap.add_argument("--port", type=int, default=None,
                    help="drive an already-running llama-server instead of "
                         "starting one")
    a = ap.parse_args()

    import os
    global _API, _CHAT, _OLLAMA_MODEL, _NO_THINK
    global _CTX
    _API, _CHAT, _NO_THINK = a.api, a.chat, a.no_think
    _CTX = a.ctx
    _OLLAMA_MODEL = a.ollama_model or None
    if a.api == "ollama" and not _OLLAMA_MODEL:
        sys.exit("--api ollama needs --ollama-model")
    if a.no_think and not a.chat:
        sys.exit("--no-think only reaches the model through --chat")
    model = pathlib.Path(a.model)
    bindir = pathlib.Path(a.bin)
    suites = [s.strip() for s in a.suites.split(",") if s.strip()]

    def log(m):
        print(m, file=sys.stderr, flush=True)

    if a.api == "ollama":
        port, stop = (a.port or 11434), lambda: None
        log(f"  using ollama on port {port}  ({_OLLAMA_MODEL})")
    elif a.port is not None:
        log(f"  using running llama-server on port {a.port}  ({model.name})")
        port, stop = a.port, lambda: None
    else:
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
            only = {t.strip() for t in a.tasks.split(",") if t.strip()}
            results["math"] = run_math(port, log, a.budget, only or None)
        if "instruct" in suites:
            results["instruct"] = run_instruct(port, log, a.budget)
        if "extract" in suites:
            results["extract"] = run_extract(port, log, a.budget)
        if "research" in suites:
            results["research"] = run_research(port, log, a.budget)
        if "tools" in suites:
            # Native tool calling only reaches the model through the chat
            # template, so there is no raw row to take. Refuse rather than
            # emit a row whose prompt_mode says "raw" and whose transport was
            # a chat endpoint -- that mislabelling is what the prompt_mode
            # field exists to prevent.
            if not _CHAT:
                log("    tools    SKIPPED -- needs --chat (no raw transport)")
            else:
                results["tools"] = run_tools(port, log, a.budget)
        if "recall" in suites and a.recall_lengths:
            corpus = pathlib.Path(a.corpus or os.environ.get(
                "PERF_LAB_HELDOUT",
                str(pathlib.Path.home() / ".perf-lab/heldout/corpus.txt"))
            ).read_text()
            lengths = [int(x) for x in a.recall_lengths.split(",")]
            results["recall"] = run_recall(port, corpus, lengths, log, a.budget)
    finally:
        stop()

    report = {"bin": str(bindir), "model": model.stem,
              "label": a.label or model.stem, "ctx": a.ctx,
              # Recorded because a score is not comparable across them: the
              # runtime decides what loaded at all, and the prompt mode decides
              # whether the model saw the format it was tuned in.
              "api": a.api, "prompt_mode": "chat" if a.chat else "raw",
              "thinking": not a.no_think,
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

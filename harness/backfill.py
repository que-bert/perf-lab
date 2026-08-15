#!/usr/bin/env python3
"""Backfill the 2026-08-14/15 measurement campaign into the ledger.

The raw logs are heterogeneous — three output shapes across three llama.cpp builds
and two models. Sections carry different fingerprints, and that is the point: the
b9843/b10082/b10438 comparison is only meaningful because each row records which
build produced it.

Anything this cannot parse confidently is reported and skipped, never guessed.
Capability probes (LOADS/REFUSED/OOM) are not throughput measurements and are
deliberately excluded rather than coerced into a metric row.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys

GPU = {"pci": "0000:0c:00.0", "uid": "0xc3abb1ceca126f15", "gfx": "gfx1201"}
BASE = {"mesa": "26.0.3-1ubuntu1", "kernel": "7.0.0-29-generic", "patches": []}
ROCM = "7.2.53211-97f5574fe2"

MODELS = {
    "q35_q5xl": {"name": "Qwen3.6-27B-UD-Q5_K_XL", "sha256": None, "quant": "Q5_K_XL"},
    "q38_q6k":  {"name": "Qwen3.8-27B-Q6_K", "sha256": None, "quant": "Q6_K"},
}
BUILDS = {
    "b9843":  ("86b94708f", "rocm"),
    "b10082": ("fb0e6b621", "rocm"),
    "b10438": ("9d57ce456", "rocm"),
    "vulkan": ("fb0e6b621", "vulkan"),
}

# file -> list of (section regex, build key, model key). First match wins; sections
# are delimited by these markers in the log.
SECTIONS = {
    "bench2.log":    [(r"^#+ A ", "b9843", "q35_q5xl"),
                      (r"^#+ B ", "b10082", "q35_q5xl"),
                      (r"^#+ C ", "vulkan", "q35_q5xl")],
    "bench3.log":    [(r"NEW: build-rocm-master", "b10438", "q35_q5xl"),
                      (r"CURRENT: b10082", "b10082", "q35_q5xl")],
    "kvspeed2.log":  [(r"^### ", "b10082", "q38_q6k")],
    "matched.log":   [(r"^### ", "b10082", "q38_q6k")],
    "ntune.log":     [(r"^", "b10082", "q38_q6k")],
    "ntune2.log":    [(r"^", "b10082", "q38_q6k")],
    "ntune3.log":    [(r"^", "b10082", "q38_q6k")],
    "spectest.log":  [(r"^", "b10082", "q38_q6k")],
    "ngramfair.log": [(r"^", "b10082", "q38_q6k")],
}

BENCH_ROW = re.compile(
    r"\|\s*(?P<model>\S[^|]*?)\s*\|.*?\|\s*(?P<test>(?:pp|tg)\d+(?:\s*@\s*d\d+)?)\s*\|"
    r"\s*(?P<val>[\d.]+)\s*±")
KV_CELLS = re.compile(r"\|\s*(q\d_\w+)\s*\|\s*(q\d_\w+)\s*\|")
NTUNE = re.compile(
    r"n_max=(?P<n>\d+)\s+vram=(?P<vram>\d+)MiB\s+\d+ tok in [\d.]+s\s+->\s+"
    r"(?P<tg>[\d.]+) tok/s(?:.*?acceptance = (?P<acc>[\d.]+))?")
# label may contain digits ("draft-mtp n=4"); anchor on the " NNN tok" field instead
SPEC = re.compile(
    r"^\s{2}(?P<label>\S.*?)\s{2,}(?P<tok>\d+) tok\s+(?P<sec>[\d.]+)s\s+"
    r"(?P<tg>[\d.]+) tok/s\s+vram=(?P<vram>\d+)MiB(?:.*?acceptance = (?P<acc>[\d.]+))?")
# ngramfair.log reports wall time including prefill and omits tok/s
FAIR = re.compile(
    r"^\s{2}(?P<label>\S.*?)\s{2,}(?P<tok>\d+) tok\s+total\s+(?P<sec>[\d.]+)s"
    r".*?acceptance = (?P<acc>[\d.]+)")


def rid(*parts):
    return "r-" + hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8]


def fp_for(build, model):
    sha, backend = BUILDS[build]
    return dict(BASE, llamacpp_sha=sha, gpu=dict(GPU),
                build={"backend": backend, "target": "gfx1201",
                       "rocm": ROCM if backend == "rocm" else None},
                model=dict(MODELS[model]))


def row(ts, src, build, model, cfg, m, note):
    return {"ts": ts, "run_id": rid(src, build, model, json.dumps(cfg, sort_keys=True),
                                    json.dumps(m, sort_keys=True)),
            "kind": "backfill", "fp": fp_for(build, model), "cfg": cfg, "m": m,
            "ok": {"token_identical": None, "corpus_sha": None}, "notes": note}


def parse(path, ts):
    name, out, skipped = path.name, [], 0
    if name not in SECTIONS:
        return [], 0
    secs = SECTIONS[name]
    build, model = secs[0][1], secs[0][2]
    ctk = ctv = None

    for line in path.read_text().splitlines():
        for pat, b, mo in secs:
            if pat != r"^" and re.search(pat, line):
                build, model = b, mo
        kv = KV_CELLS.search(line)
        if kv:
            ctk, ctv = kv.group(1), kv.group(2)
        if m := re.search(r"K=(q\d_\w+) V=(q\d_\w+)", line):
            ctk, ctv = m.group(1), m.group(2)

        if b := BENCH_ROW.search(line):
            test, val = b.group("test").replace(" ", ""), float(b.group("val"))
            depth = int(d.group(1)) if (d := re.search(r"@d(\d+)", test)) else None
            key = ("pp_deep" if depth else "pp2048") if test.startswith("pp") else \
                  ("tg_deep" if depth else "tg")
            out.append(row(ts, name, build, model,
                           {"ctx": None, "ctk": ctk or "q8_0", "ctv": ctv or "q8_0",
                            "fa": "on", "spec": None, "n_max": None},
                           {key: val, "depth": depth, "rep": None,
                            "cold_prefill": None, "mtp_acceptance": None,
                            "unit": "tok/s"},
                           f"llama-bench, {name}"))
        elif n := NTUNE.search(line):
            out.append(row(ts, name, build, model,
                           {"ctx": 262144, "ctk": "q4_0" if "3" in name else "q8_0",
                            "ctv": "q4_0" if "3" in name else "q5_1", "fa": "on",
                            "spec": "draft-mtp,ngram-simple", "n_max": int(n.group("n"))},
                           {"tg": float(n.group("tg")),
                            "mtp_acceptance": float(n.group("acc")) if n.group("acc") else None,
                            "vram_mib": int(n.group("vram")), "rep": None,
                            "cold_prefill": None, "unit": "tok/s"},
                           f"MTP draft-depth sweep, {name}"))
        elif s := SPEC.search(line):
            label = s.group("label").strip()
            out.append(row(ts, name, build, model,
                           {"ctx": 262144, "ctk": "q4_0", "ctv": "q4_0", "fa": "on",
                            "spec": None if "baseline" in label else label, "n_max": 4},
                           {"tg": float(s.group("tg")),
                            "mtp_acceptance": float(s.group("acc")) if s.group("acc") else None,
                            "vram_mib": int(s.group("vram")), "rep": None,
                            "cold_prefill": None, "unit": "tok/s"},
                           f"speculator comparison ({label}), {name}"))
        elif f := FAIR.search(line):
            out.append(row(ts, name, build, model,
                           {"ctx": 262144, "ctk": "q4_0", "ctv": "q4_0", "fa": "on",
                            "spec": f.group("label").strip(), "n_max": 4},
                           {"wall_s": float(f.group("sec")),
                            "mtp_acceptance": float(f.group("acc")),
                            "rep": None, "cold_prefill": None, "unit": "s"},
                           f"copy-heavy 193K task, wall {f.group('sec')}s incl prefill, {name}"))
        elif re.search(r"\b(LOADS|REFUSED|PASSES|FAILS|out of memory)\b", line):
            skipped += 1
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("--ts", default="2026-08-15T06:00:00Z")
    a = ap.parse_args()
    raw = pathlib.Path(a.raw_dir).expanduser()
    if not raw.is_dir():
        sys.exit(f"backfill: {raw} is not a directory")

    rows, skipped, seen = [], 0, set()
    for f in sorted(raw.glob("*.log")):
        got, sk = parse(f, a.ts)
        skipped += sk
        for r in got:
            if r["run_id"] not in seen:
                seen.add(r["run_id"])
                rows.append(r)
        if got:
            print(f"  {f.name}: {len(got)} rows", file=sys.stderr)
    for r in rows:
        print(json.dumps(r))
    print(f"  total {len(rows)} rows; {skipped} capability-probe lines excluded "
          f"(not throughput measurements)", file=sys.stderr)


if __name__ == "__main__":
    main()

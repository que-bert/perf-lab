#!/usr/bin/env python3
"""Build one ledger row: run llama-bench (or not) and assemble the JSON.

Split out of bench.sh because assembling JSON in shell is how fingerprints end up
malformed. bench.sh owns process lifecycle and guards; this owns the row.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def sha256_cached(path: pathlib.Path):
    """Hash a multi-GB model once, keyed on (size, mtime). Re-hashing per run costs minutes."""
    cache = HERE.parent / ".scratch" / "model-sha.json"
    cache.parent.mkdir(exist_ok=True)
    st = path.stat()
    key = f"{path}:{st.st_size}:{int(st.st_mtime)}"
    db = json.loads(cache.read_text()) if cache.exists() else {}
    if key in db:
        return db[key]
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    db[key] = h.hexdigest()
    cache.write_text(json.dumps(db))
    return db[key]


def llamacpp_sha(bindir: pathlib.Path):
    """`version: 10082 (fb0e6b621)` -> fb0e6b621.

    llama-bench --version prints only backend-init noise and no version line, so ask
    llama-server. Both binaries come from the same build directory.
    """
    env = dict(os.environ, LD_LIBRARY_PATH=str(bindir))
    for exe in ("llama-server", "llama-cli", "llama-bench"):
        path = bindir / exe
        if not path.exists():
            continue
        out = subprocess.run([str(path), "--version"], capture_output=True,
                             text=True, env=env, timeout=60)
        m = re.search(r"^version:\s*\d+\s*\(([0-9a-f]{7,40})\)",
                      out.stdout + out.stderr, re.M)
        if m:
            return m.group(1)
    sys.exit("emit_row: could not determine llama.cpp build SHA from any binary in "
             f"{bindir} — refusing to write a row with an unknown fingerprint")


def probe(uid: str):
    out = subprocess.run([str(HERE / "probe.sh"), "--gpu-uid", uid],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"emit_row: probe failed: {out.stderr.strip()}")
    return json.loads(out.stdout)


def run_bench(bindir, model, c, defaults):
    """Run llama-bench once; return (pp, tg) parsed from its markdown table."""
    cmd = [str(bindir / "llama-bench"), "-m", str(model),
           "-ctk", c["ctk"], "-ctv", c["ctv"],
           "-p", str(c["prompt_tokens"]), "-n", str(defaults["n_gen"]),
           "-fa", str(defaults["fa"]), "-r", "1",
           "-ngl", str(defaults["ngl"]), "-t", str(defaults["threads"])]
    env = dict(os.environ, LD_LIBRARY_PATH=str(bindir))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if r.returncode != 0:
        sys.exit(f"emit_row: llama-bench exit {r.returncode}\n{r.stderr[-800:]}")
    pp = tg = None
    for line in r.stdout.splitlines():
        if "|" not in line or "pp" not in line and "tg" not in line:
            continue
        cells = [x.strip() for x in line.split("|")]
        val = next((c for c in reversed(cells) if re.match(r"^[\d.]+\s*±", c)), None)
        if not val:
            continue
        num = float(val.split("±")[0].strip())
        test = next((c for c in cells if re.match(r"^(pp|tg)\d+", c)), "")
        if test.startswith("pp"):
            pp = num
        elif test.startswith("tg"):
            tg = num
    if pp is None and tg is None:
        sys.exit(f"emit_row: could not parse llama-bench output:\n{r.stdout[-800:]}")
    return pp, tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--kind", default="manual")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reason", default="")
    ap.add_argument("--gpu-uid", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    import yaml
    doc = yaml.safe_load(open(a.config))
    defaults, c = doc["defaults"], doc["canaries"][a.key]
    bindir, model = pathlib.Path(a.bin), pathlib.Path(a.model)

    fp = probe(a.gpu_uid)
    fp["llamacpp_sha"] = llamacpp_sha(bindir)
    fp["patches"] = []
    fp["model"] = {"name": model.stem, "sha256": sha256_cached(model),
                   "quant": model.stem.rsplit("-", 1)[-1]}

    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": "r-" + hashlib.sha1(
            f"{time.time()}{a.key}{a.rep}".encode()).hexdigest()[:8],
        "kind": a.kind,
        "fp": fp,
        "cfg": {"ctx": c["prompt_tokens"] + defaults["n_gen"],
                "ctk": c["ctk"], "ctv": c["ctv"], "fa": str(defaults["fa"]),
                "spec": None, "n_max": None, "ngl": defaults["ngl"],
                "np": 1, "prompt_tokens": c["prompt_tokens"]},
        "m": {"unit": "tok/s"},
        "ok": {"token_identical": None, "corpus_sha": None},
    }
    if a.tag:
        row["tag"] = a.tag

    if a.kind == "skipped":
        row["reason"] = a.reason or "unspecified"
        row["m"] = {"pp2048": None, "unit": "tok/s"}
    else:
        pp, tg = run_bench(bindir, model, c, defaults)
        row["m"].update({"pp2048": pp, "tg": tg, "rep": a.rep,
                         "cold_prefill": True, "mtp_acceptance": None})

    json.dump(row, sys.stdout)
    print()


if __name__ == "__main__":
    main()

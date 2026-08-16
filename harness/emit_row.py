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


def build_identity(bindir: pathlib.Path, tool: str):
    """Digest the files that actually decide performance, plus the toolchain.

    `llamacpp_sha` is not enough on its own. Two builds of fb0e6b621 exist on
    this machine: one segfaults on every model, the other is what Mimir runs.
    They differ only by compiler, so a ledger keyed on the upstream SHA calls
    them the same thing and silently mixes two populations.

    The tool binaries here are ~18 KB launcher shims -- the code lives in
    <tool>-impl.so and the ggml backends -- so hashing the named binary alone
    would miss a rebuilt backend entirely.
    """
    wanted = [bindir / tool, bindir / f"lib{tool}-impl.so"]
    wanted += sorted(bindir.glob("libggml*.so.[0-9]*"))
    digest = hashlib.sha256()
    seen = 0
    for path in wanted:
        if not path.exists():
            continue
        real = path.resolve()
        digest.update(real.name.encode())
        digest.update(sha256_cached(real).encode())
        seen += 1
    if not seen:
        return None, None

    toolchain = None
    env = dict(os.environ, LD_LIBRARY_PATH=str(bindir))
    for exe in (tool, "llama-cli", "llama-bench"):
        path = bindir / exe
        if not path.exists():
            continue
        out = subprocess.run([str(path), "--version"], capture_output=True,
                             text=True, env=env, timeout=60)
        m = re.search(r"^built with (.+?) for ", out.stdout + out.stderr, re.M)
        if m:
            toolchain = m.group(1).strip()
            break
    return digest.hexdigest(), toolchain


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


def card_for(pci):
    """DRM card directory for a PCI address. Never by index -- this machine
    reports the target as rocm-smi GPU[1] and DRM card0."""
    for card in sorted(pathlib.Path("/sys/class/drm").glob("card[0-9]*")):
        try:
            uevent = (card / "device" / "uevent").read_text()
        except OSError:
            continue
        for line in uevent.splitlines():
            if line.startswith("PCI_SLOT_NAME=") and line.split("=", 1)[1].lower() == pci:
                return card
    return None


def env_probe(pci):
    """Covariates that explain drift after the fact.

    The design called for these so that a shift is visible rather than
    mysterious; without them a 70% move on two canaries with an identical
    fingerprint has no explanation available. vram_used_mib_before is the
    contention check: a card that was not idle when measurement started did
    not measure what the row claims.
    """
    env = {"gpu_temp_c": None, "gpu_clock_mhz": None, "loadavg": None,
           "vram_used_mib_before": None}
    try:
        env["loadavg"] = round(os.getloadavg()[0], 2)
    except OSError:
        pass
    card = card_for(pci)
    if card is None:
        return env
    try:
        used = int((card / "device" / "mem_info_vram_used").read_text())
        env["vram_used_mib_before"] = used // (1 << 20)
    except (OSError, ValueError):
        pass
    for hwmon in (card / "device" / "hwmon").glob("hwmon*"):
        try:
            env["gpu_temp_c"] = int((hwmon / "temp1_input").read_text()) / 1000.0
            break
        except (OSError, ValueError):
            continue
    try:
        for line in (card / "device" / "pp_dpm_sclk").read_text().splitlines():
            if line.strip().endswith("*"):          # the active DPM state
                env["gpu_clock_mhz"] = int(re.search(r"(\d+)Mhz", line, re.I).group(1))
                break
    except (OSError, ValueError, AttributeError):
        pass
    return env


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
    ap.add_argument("--tool", default="llama-bench",
                    help="which binary produced this row; its identity is "
                         "hashed into fp.build.binary_sha256")
    a = ap.parse_args()

    import yaml
    doc = yaml.safe_load(open(a.config))
    defaults, c = doc["defaults"], doc["canaries"][a.key]
    bindir, model = pathlib.Path(a.bin), pathlib.Path(a.model)

    fp = probe(a.gpu_uid)
    fp["llamacpp_sha"] = llamacpp_sha(bindir)
    fp["patches"] = []
    binsha, toolchain = build_identity(bindir, a.tool)
    fp["build"]["binary_sha256"] = binsha
    fp["build"]["toolchain"] = toolchain
    fp["model"] = {"name": model.stem, "sha256": sha256_cached(model),
                   "quant": model.stem.rsplit("-", 1)[-1]}

    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": "r-" + hashlib.sha1(
            f"{time.time()}{a.key}{a.rep}".encode()).hexdigest()[:8],
        "kind": a.kind,
        "key": a.key,
        "fp": fp,
        "cfg": {"ctx": c["prompt_tokens"] + defaults["n_gen"],
                "ctk": c["ctk"], "ctv": c["ctv"], "fa": str(defaults["fa"]),
                "spec": None, "n_max": None, "ngl": defaults["ngl"],
                "np": 1, "prompt_tokens": c["prompt_tokens"]},
        "m": {"unit": "tok/s"},
        "ok": {"token_identical": None, "corpus_sha": None},
        "env": env_probe(fp["gpu"]["pci"]),
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

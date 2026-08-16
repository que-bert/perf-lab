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
import tempfile
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
    fingerprint has no explanation available.

    Called twice, because the two halves want opposite moments.
    vram_used_mib_before is a contention check and is only meaningful BEFORE
    the run: a card that was not idle did not measure what the row claims.
    Temperature and clock are only meaningful AFTER it -- sampled first they
    describe a sleeping card (17 C, 0 MHz) and say nothing about whether the
    measurement throttled.
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
    # By label, not by position: this card exposes edge, junction and mem, and
    # which one is temp1 is not guaranteed across drivers.
    for hwmon in (card / "device" / "hwmon").glob("hwmon*"):
        for label_file in sorted(hwmon.glob("temp*_label")):
            try:
                if label_file.read_text().strip() == "edge":
                    inp = label_file.with_name(
                        label_file.name.replace("_label", "_input"))
                    env["gpu_temp_c"] = int(inp.read_text()) / 1000.0
                    break
            except (OSError, ValueError):
                continue
        if env["gpu_temp_c"] is not None:
            break
    try:
        for line in (card / "device" / "pp_dpm_sclk").read_text().splitlines():
            if line.strip().endswith("*"):          # the active DPM state
                env["gpu_clock_mhz"] = int(re.search(r"(\d+)\s*Mhz", line, re.I).group(1))
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


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_server(bindir, model, c, defaults):
    """Spawn llama-server for one config and wait until it is actually serving.

    Returns (proc, port, stop). Shared with verify.py, which needs the same
    lifecycle to run two configs over identical prompts.
    """
    import urllib.error
    import urllib.request

    port = free_port()
    cmd = [str(bindir / "llama-server"), "-m", str(model),
           "--port", str(port), "--host", "127.0.0.1", "--no-webui",
           "-c", str(c["ctx"]), "-ctk", c["ctk"], "-ctv", c["ctv"],
           "-fa", str(defaults["fa"]), "-ngl", str(defaults["ngl"]),
           "-t", str(defaults["threads"])]
    if c.get("spec"):
        cmd += ["--spec-type", c["spec"]]
        if c.get("n_max"):
            cmd += ["--spec-draft-n-max", str(c["n_max"])]

    env = dict(os.environ, LD_LIBRARY_PATH=str(bindir))
    log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    def stop():
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)

    # A 503 body means "still loading" -- treating any HTTP reply as ready
    # kills the server mid-load and reports a crash that never happened.
    deadline = time.time() + float(os.environ.get("PERF_LAB_LOAD_TIMEOUT", 900))
    while True:
        if proc.poll() is not None:
            log.flush()
            tail = pathlib.Path(log.name).read_text()[-1200:]
            sys.exit(f"emit_row: llama-server exited {proc.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        if time.time() > deadline:
            stop()
            sys.exit(f"emit_row: llama-server never became ready on port {port}")
        time.sleep(3)
    return proc, port, stop


def complete(port, prompt, n_predict, extra=None):
    """One completion request. temperature 0 and cache_prompt off, always:
    a cached prompt reports a prefill that was never computed."""
    import urllib.request
    body = {"prompt": prompt, "n_predict": n_predict,
            "temperature": 0, "cache_prompt": False}
    body.update(extra or {})
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def run_server(bindir, model, c, defaults, prompt):
    """Measure one request through llama-server; return (pp, tg, acceptance, n).

    llama-bench cannot do this: it has no speculative-decoding flags, so the
    config actually served -- 262K context with draft-mtp -- was unmeasurable
    by the stage-1 harness. Acceptance comes from the response's own
    draft_n/draft_n_accepted rather than scraped from the log.
    """
    proc, port, stop = start_server(bindir, model, c, defaults)
    try:
        resp = complete(port, prompt, c.get("n_predict", defaults.get("n_gen", 64)))
    finally:
        stop()

    t = resp.get("timings", {})
    if not t:
        sys.exit(f"emit_row: no timings in response: {str(resp)[:400]}")
    drafted, accepted = t.get("draft_n"), t.get("draft_n_accepted")
    acceptance = (accepted / drafted) if drafted else None
    return (t.get("prompt_per_second"), t.get("predicted_per_second"),
            acceptance, t.get("prompt_n"))


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
    ap.add_argument("--mode", choices=("bench", "server"), default="bench",
                    help="bench = llama-bench (canaries); server = llama-server, "
                         "the only path that can set ctx/spec/n_max")
    ap.add_argument("--tool", default="llama-bench",
                    help="which binary produced this row; its identity is "
                         "hashed into fp.build.binary_sha256")
    a = ap.parse_args()

    import yaml
    doc = yaml.safe_load(open(a.config))
    # canary.yaml keys its entries under "canaries", tracked.yaml under
    # "tracked". Same row shape either way.
    entries = doc.get("canaries") or doc.get("tracked") or {}
    if a.key not in entries:
        sys.exit(f"emit_row: no config '{a.key}' in {a.config}")
    defaults, c = doc["defaults"], entries[a.key]
    bindir, model = pathlib.Path(a.bin), pathlib.Path(a.model)

    fp = probe(a.gpu_uid)
    fp["llamacpp_sha"] = llamacpp_sha(bindir)
    fp["patches"] = []
    binsha, toolchain = build_identity(bindir, a.tool)
    fp["build"]["binary_sha256"] = binsha
    fp["build"]["toolchain"] = toolchain
    fp["model"] = {"name": model.stem, "sha256": sha256_cached(model),
                   "quant": model.stem.rsplit("-", 1)[-1]}

    env_pre = env_probe(fp["gpu"]["pci"])

    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": "r-" + hashlib.sha1(
            f"{time.time()}{a.key}{a.rep}".encode()).hexdigest()[:8],
        "kind": a.kind,
        "key": a.key,
        "fp": fp,
        "cfg": {"ctx": (c["ctx"] if a.mode == "server"
                        else c["prompt_tokens"] + defaults["n_gen"]),
                "ctk": c["ctk"], "ctv": c["ctv"], "fa": str(defaults["fa"]),
                "spec": c.get("spec") if a.mode == "server" else None,
                "n_max": c.get("n_max") if a.mode == "server" else None,
                "ngl": defaults["ngl"],
                "np": 1, "prompt_tokens": c.get("prompt_tokens")},
        "m": {"unit": "tok/s"},
        "ok": {"token_identical": None, "corpus_sha": None},
        "env": env_pre,
    }
    if a.tag:
        row["tag"] = a.tag

    if a.kind == "skipped":
        row["reason"] = a.reason or "unspecified"
        row["m"] = {"pp2048": None, "unit": "tok/s"}
    elif a.mode == "server":
        prompt_path = pathlib.Path(
            os.environ.get("PERF_LAB_PROMPT", str(pathlib.Path.home()
                                                  / ".perf-lab/prompts/default.txt")))
        if not prompt_path.is_file():
            sys.exit(f"emit_row: no prompt file at {prompt_path}. Set "
                     "PERF_LAB_PROMPT. Prompts are never committed -- this "
                     "repo is public -- so the row cites its sha256 instead.")
        prompt = prompt_path.read_text()
        pp, tg, acceptance, prompt_n = run_server(bindir, model, c, defaults, prompt)
        row["cfg"]["prompt_tokens"] = prompt_n
        row["m"].update({"pp2048": pp, "tg": tg, "rep": a.rep,
                         "cold_prefill": True, "mtp_acceptance": acceptance})
        row["ok"]["corpus_sha"] = hashlib.sha256(prompt.encode()).hexdigest()
    else:
        pp, tg = run_bench(bindir, model, c, defaults)
        row["m"].update({"pp2048": pp, "tg": tg, "rep": a.rep,
                         "cold_prefill": True, "mtp_acceptance": None})

    if a.kind != "skipped":
        # Thermal state as the run left it. Sampled before, these describe an
        # idle card and cannot show whether the measurement throttled.
        post = env_probe(fp["gpu"]["pci"])
        row["env"]["gpu_temp_c"] = post["gpu_temp_c"]
        row["env"]["gpu_clock_mhz"] = post["gpu_clock_mhz"]

    json.dump(row, sys.stdout)
    print()


if __name__ == "__main__":
    main()

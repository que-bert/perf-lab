#!/usr/bin/env python3
"""The bench scripts must refuse a Vulkan build rather than mis-pin it.

Both scripts select the GPU with ROCR_VISIBLE_DEVICES, which is ROCm's variable
and is ignored by the Vulkan backend. Pointed at ~/llama.cpp/b10472-vulkan they
would run on Vulkan device 0 -- the 16 GB 9060 XT on this host -- and append a
row whose fingerprint names whichever card probe.sh was asked about. A row that
lies about which GPU produced it is the one thing this ledger must not contain,
and the README promises the opposite: selection is by unique ID, and an absent
device aborts rather than silently measuring another card.

Vulkan needs GGML_VK_VISIBLE_DEVICES and a uid -> Vulkan-index mapping. Until
that exists, refusing is the only honest option. No GPU is needed here: the
refusal fires before probe.sh is ever called.
"""
import atexit
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TMP = pathlib.Path(tempfile.mkdtemp(prefix="perf-lab-test-bench-guard-"))
atexit.register(shutil.rmtree, TMP, True)

MODEL = TMP / "fake-model-Q6_K.gguf"
MODEL.write_text("not a real gguf; the refusal never reads it")
LEDGER = TMP / "ledger.jsonl"

cases = []


def stub_bin(name, backend_so):
    d = TMP / name
    d.mkdir()
    for exe in ("llama-bench", "llama-server"):
        p = d / exe
        p.write_text("#!/bin/sh\necho 'stub build should never be executed' >&2\nexit 99\n")
        p.chmod(0o755)
    (d / backend_so).write_text("stub backend")
    return d


def run(script, bindir, uid="0xdeadbeefdeadbeef"):
    env = dict(os.environ,
               PERF_LAB_MODEL=str(MODEL),
               PERF_LAB_GPU_UID=uid,
               PERF_LAB_LEDGER=str(LEDGER))
    env["PERF_LAB_BIN"] = str(bindir)
    env["PERF_LAB_SERVER_BIN"] = str(bindir)
    r = subprocess.run(["bash", str(HERE / script), str(ROOT / "configs" / "canary.yaml"),
                        "fast-q4"], capture_output=True, text=True, env=env, timeout=120)
    return r.returncode, r.stdout + r.stderr


def check(name, cond, detail=""):
    cases.append((name, cond, detail))


vulkan = stub_bin("vulkan-stub", "libggml-vulkan.so")
rocm = stub_bin("rocm-stub", "libggml-hip.so")

rc, out = run("bench.sh", vulkan)
check("bench.sh refuses a Vulkan build", rc == 3 and "Vulkan build" in out,
      f"rc={rc} out={out[-200:]}")
check("bench.sh writes no row when it refuses",
      not LEDGER.exists() or LEDGER.read_text() == "",
      f"ledger={LEDGER.read_text()[:200] if LEDGER.exists() else '(absent)'}")

rc, out = run("bench_server.sh", vulkan)
check("bench_server.sh refuses a Vulkan build", rc == 3 and "Vulkan build" in out,
      f"rc={rc} out={out[-200:]}")

# The refusal must be specific to Vulkan. A ROCm build gets past it and fails
# later, on the absent GPU uid -- which is probe.sh doing its job, not this
# guard over-reaching.
rc, out = run("bench.sh", rocm)
check("a ROCm build is not caught by the Vulkan refusal",
      "Vulkan build" not in out, f"rc={rc} out={out[-200:]}")
check("an absent GPU uid still aborts", rc != 0, f"rc={rc}")
check("no row written for the absent GPU either",
      not LEDGER.exists() or LEDGER.read_text() == "",
      f"ledger={LEDGER.read_text()[:200] if LEDGER.exists() else '(absent)'}")

fails = 0
for name, ok, detail in cases:
    fails += 0 if ok else 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<48} {'' if ok else detail}")
sys.exit(1 if fails else 0)

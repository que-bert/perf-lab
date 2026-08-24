#!/usr/bin/env python3
"""The bench scripts must pin the GPU their backend actually understands.

ROCR_VISIBLE_DEVICES is ROCm's variable and the Vulkan backend ignores it.
Setting the wrong one is silent: the backend falls back to device 0 -- the
16 GB 9060 XT on this host -- and the row's fingerprint still names whichever
card probe.sh was asked about. A row that lies about which GPU produced it is
the one thing this ledger must not contain, and the README promises the
opposite: selection is by unique ID, and an absent device aborts rather than
silently measuring another card.

The index is never a literal, because two cards enumerate three different ways
here and no two orderings agree:

    rocm-smi   GPU[0] = 9060 XT   GPU[1] = R9700
    DRM        card0  = R9700     card1  = 9060 XT
    Vulkan     Vulkan0 = 9060 XT  Vulkan1 = R9700

vulkan_index.py maps a gfx target onto whatever index the Vulkan driver is
using, and refuses if that is not exactly one device. No GPU is needed here:
the parse and the match are pure, and the bench.sh case aborts on an absent
uid before anything is measured.
"""
import atexit
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
_spec = importlib.util.spec_from_file_location("vulkan_index", HERE / "vulkan_index.py")
vk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vk)

TMP = pathlib.Path(tempfile.mkdtemp(prefix="perf-lab-test-bench-guard-"))
atexit.register(shutil.rmtree, TMP, True)
MODEL = TMP / "fake-model-Q6_K.gguf"
MODEL.write_text("not a real gguf; the run aborts long before it is read")
LEDGER = TMP / "ledger.jsonl"

# Real output from b10472-vulkan on this host, 2026-08-24.
LISTING = """load_backend: loaded RPC backend from libggml-rpc.so
ggml_vulkan: Found 2 Vulkan devices:
Available devices:
  Vulkan0: AMD Radeon RX 9060 XT (RADV GFX1200) (16384 MiB, 9935 MiB free)
  Vulkan1: AMD Radeon AI PRO R9700 (RADV GFX1201) (32768 MiB, 32707 MiB free)
"""

cases = []


def check(name, cond, detail=""):
    cases.append((name, cond, detail))


devices = vk.parse_devices(LISTING)
check("both Vulkan devices parse", len(devices) == 2, str(devices))

idx, err = vk.index_for_gfx(devices, "gfx1201")
check("the R9700 resolves to Vulkan1", idx == 1 and err is None, f"{idx} {err}")

idx, err = vk.index_for_gfx(devices, "gfx1200")
check("the 9060 XT resolves to Vulkan0", idx == 0 and err is None, f"{idx} {err}")

# Vulkan1 is the R9700 and rocm-smi GPU[1] is also the R9700, which is a
# coincidence of this host and not a rule. The mapping must come from the
# listing, never from the ROCm index.
idx, err = vk.index_for_gfx(devices, "gfx9999")
check("an absent gfx is refused, not defaulted to 0", idx is None and err, f"{idx} {err}")

# Two identical cards cannot be told apart by gfx, and guessing would pin the
# wrong one silently. The Vulkan listing carries no PCI address to fall back on.
twins = [(0, "AMD Radeon AI PRO R9700 (RADV GFX1201)"),
         (1, "AMD Radeon AI PRO R9700 (RADV GFX1201)")]
idx, err = vk.index_for_gfx(twins, "gfx1201")
check("two cards with one gfx is refused, not guessed",
      idx is None and err and "cannot be identified" in err, f"{idx} {err}")


def stub_bin(name, backend_so):
    d = TMP / name
    d.mkdir()
    for exe in ("llama-bench", "llama-server"):
        p = d / exe
        p.write_text("#!/bin/sh\necho 'stub build must never be executed' >&2\nexit 99\n")
        p.chmod(0o755)
    (d / backend_so).write_text("stub backend")
    return d


def run(script, bindir, uid="0xdeadbeefdeadbeef"):
    env = dict(os.environ, PERF_LAB_MODEL=str(MODEL), PERF_LAB_GPU_UID=uid,
               PERF_LAB_LEDGER=str(LEDGER), PERF_LAB_BIN=str(bindir),
               PERF_LAB_SERVER_BIN=str(bindir))
    r = subprocess.run(["bash", str(HERE / script),
                        str(ROOT / "configs" / "canary.yaml"), "fast-q4"],
                       capture_output=True, text=True, env=env, timeout=300)
    return r.returncode, r.stdout + r.stderr


# An absent GPU uid must abort before any measurement, on either backend. This
# is probe.sh refusing to fall back to another card -- the property the whole
# uid-not-index discipline exists to provide.
for name, so in (("vulkan-stub", "libggml-vulkan.so"), ("rocm-stub", "libggml-hip.so")):
    rc, out = run("bench.sh", stub_bin(name, so))
    check(f"bench.sh aborts on an absent uid ({so})", rc != 0, f"rc={rc}")
    check(f"bench.sh writes no row when it aborts ({so})",
          not LEDGER.exists() or LEDGER.read_text() == "",
          LEDGER.read_text()[:200] if LEDGER.exists() else "")

fails = 0
for name, ok, detail in cases:
    fails += 0 if ok else 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} {'' if ok else detail}")
sys.exit(1 if fails else 0)

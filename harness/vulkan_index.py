#!/usr/bin/env python3
"""Resolve a GPU to its Vulkan device index, or refuse.

The bench scripts pin the target card with ROCR_VISIBLE_DEVICES, which the
Vulkan backend ignores. Vulkan needs GGML_VK_VISIBLE_DEVICES, and its index is
its own enumeration -- on this host Vulkan0 is the 16 GB 9060 XT and Vulkan1 is
the R9700, which is the reverse of neither rocm-smi nor DRM but simply a third
ordering. Three enumerations of two cards, no two of which agree, is exactly
the situation that makes "just use index 1" a bug waiting to happen.

So the index is never assumed. probe.sh resolves the unique ID to a gfx target,
and this maps that gfx onto whatever index the Vulkan driver happens to be
using right now:

    ggml_vulkan: 1 = AMD Radeon AI PRO R9700 (RADV GFX1201) (radv) | ...
    Vulkan1: AMD Radeon AI PRO R9700 (RADV GFX1201) (32768 MiB, 32707 MiB free)

A gfx that matches no device, or more than one, is refused rather than guessed
at. Two identical cards would collide here -- this host has gfx1200 and
gfx1201, so they do not, but a second R9700 would need matching on PCI address
and the Vulkan device list does not carry one.

  vulkan_index.py --bin ~/llama.cpp/b10472-vulkan --gfx gfx1201   -> 1
"""
import argparse
import os
import pathlib
import re
import subprocess
import sys

# "  Vulkan1: AMD Radeon AI PRO R9700 (RADV GFX1201) (32768 MiB, 32707 MiB free)"
DEVICE_LINE = re.compile(r"^\s*Vulkan(\d+):\s*(.+?)\s*$", re.M)


def parse_devices(blob: str):
    """-> [(index, description)] from --list-devices output."""
    return [(int(m.group(1)), m.group(2)) for m in DEVICE_LINE.finditer(blob)]


def index_for_gfx(devices, gfx: str):
    """The one device whose description names this gfx target.

    Returns (index, error). Ambiguity and absence are both errors: a bench run
    pinned to the wrong card produces a row that names a GPU it never touched.
    """
    want = gfx.strip().lower()
    hits = [(i, d) for i, d in devices if want in d.lower()]
    if not hits:
        have = "; ".join(f"Vulkan{i}: {d}" for i, d in devices) or "(none)"
        return None, f"no Vulkan device reports {gfx}. Present: {have}"
    if len(hits) > 1:
        have = "; ".join(f"Vulkan{i}: {d}" for i, d in hits)
        return None, (f"{len(hits)} Vulkan devices report {gfx}, so the target "
                      f"cannot be identified by gfx alone: {have}")
    return hits[0][0], None


def list_devices(bindir: pathlib.Path):
    for exe in ("llama-bench", "llama-server", "llama-cli"):
        path = bindir / exe
        if not path.exists():
            continue
        env = dict(os.environ, LD_LIBRARY_PATH=str(bindir))
        out = subprocess.run([str(path), "--list-devices"], capture_output=True,
                             text=True, env=env, timeout=300)
        blob = out.stdout + out.stderr
        if DEVICE_LINE.search(blob):
            return blob
    sys.exit(f"vulkan_index: no binary in {bindir} listed any Vulkan device")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gfx", required=True)
    a = ap.parse_args()

    idx, err = index_for_gfx(parse_devices(list_devices(pathlib.Path(a.bin))), a.gfx)
    if err:
        sys.exit(f"vulkan_index: {err}")
    print(idx)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Emit the environment half of a ledger fingerprint as JSON on stdout.
#
#   probe.sh --gpu-uid 0x…        select the GPU by unique ID (preferred)
#   probe.sh --gpu-pci 0000:0c:00.0
#
# Emits: build{backend,target,rocm}, mesa, kernel, gpu{pci,uid,gfx}
# Callers merge in llamacpp_sha, patches and model to form a complete `fp`.
#
# Never selects a GPU by index. rocm-smi GPU[1] and DRM card0 are the same device
# on this host; the enumerations are inverted, and either can be reordered by a
# ROCm or kernel upgrade. Index is used only to correlate fields *within* one
# rocm-smi snapshot, and is never emitted.
set -euo pipefail

BACKEND=rocm
WANT_UID=""
WANT_PCI=""
EMIT_INDEX=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-uid) WANT_UID="${2,,}"; shift 2 ;;
    --gpu-pci) WANT_PCI="${2,,}"; shift 2 ;;
    --backend) BACKEND="$2";      shift 2 ;;
    --emit-index) EMIT_INDEX=1;   shift 1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "probe.sh: unknown argument $1" >&2; exit 64 ;;
  esac
done

command -v rocm-smi >/dev/null || { echo "probe.sh: rocm-smi not found" >&2; exit 3; }

HW="$(rocm-smi --showhw 2>/dev/null || true)"
UIDS="$(rocm-smi --showuniqueid 2>/dev/null || true)"
ROCM="$(hipconfig --version 2>/dev/null | head -c 32 || true)"
# :amd64 explicitly — the i386 multiarch package is also installed and an
# unqualified query concatenates both versions into one unusable string.
MESA="$(dpkg-query -W -f='${Version}\n' mesa-vulkan-drivers:amd64 2>/dev/null | head -1)"
[[ -n "$MESA" ]] || MESA=unknown
KERNEL="$(uname -r)"

HW="$HW" UIDS="$UIDS" ROCM="$ROCM" MESA="$MESA" KERNEL="$KERNEL" \
BACKEND="$BACKEND" WANT_UID="$WANT_UID" WANT_PCI="$WANT_PCI" EMIT_INDEX="$EMIT_INDEX" python3 - <<'PY'
import json, os, re, sys

hw, uids = os.environ["HW"], os.environ["UIDS"]
want_uid, want_pci = os.environ["WANT_UID"], os.environ["WANT_PCI"]

# GPU index -> (gfx, pci) from the concise hardware table
gpus = {}
for m in re.finditer(
        r"^(\d+)\s+\d+\s+\S+\s+\S+\s+(gfx\w+)\s+.*?([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d)",
        hw, re.M):
    gpus[m.group(1)] = {"gfx": m.group(2).lower(), "pci": m.group(3).lower(), "idx": m.group(1)}

# GPU index -> uid
for m in re.finditer(r"GPU\[(\d+)\]\s*:\s*Unique ID:\s*(\S+)", uids):
    gpus.setdefault(m.group(1), {})["uid"] = m.group(2).lower()

found = [g for g in gpus.values() if {"gfx", "pci", "uid"} <= set(g)]
if not found:
    sys.exit("probe.sh: could not enumerate any GPU from rocm-smi")

def fail(kind, value):
    have = ", ".join(f"{g['uid']} ({g['pci']}, {g['gfx']})" for g in found)
    print(f"probe.sh: no GPU with {kind} {value}. Present: {have}", file=sys.stderr)
    sys.exit(2)   # 2 == requested device absent; never fall back to another GPU

if want_uid:
    sel = next((g for g in found if g["uid"] == want_uid), None) or fail("uid", want_uid)
elif want_pci:
    sel = next((g for g in found if g["pci"] == want_pci), None) or fail("pci", want_pci)
elif len(found) == 1:
    sel = found[0]
else:
    have = ", ".join(f"{g['uid']} ({g['pci']}, {g['gfx']})" for g in found)
    sys.exit(f"probe.sh: {len(found)} GPUs present, pass --gpu-uid or --gpu-pci. Present: {have}")

if os.environ.get("EMIT_INDEX") == "1":
    # Runtime-only: used to set ROCR_VISIBLE_DEVICES within this same snapshot.
    # Never written to a ledger row.
    print(sel["idx"]); sys.exit(0)

json.dump({
    "build":  {"backend": os.environ["BACKEND"], "target": sel["gfx"],
               "rocm": os.environ["ROCM"] or None},
    "mesa":   os.environ["MESA"],
    "kernel": os.environ["KERNEL"],
    "gpu":    {"pci": sel["pci"], "uid": sel["uid"], "gfx": sel["gfx"]},
}, sys.stdout, indent=2, sort_keys=True)
print()
PY

#!/usr/bin/env bash
# Measure one tracked configuration through llama-server and append ledger rows.
#
#   bench_server.sh configs/tracked.yaml baseline [--reps N] [--tag T] [--kind K]
#
# Environment:
#   PERF_LAB_MODEL       path to the .gguf under test              (required)
#   PERF_LAB_SERVER_BIN  directory holding a WORKING llama-server  (required)
#   PERF_LAB_GPU_UID     unique ID of the GPU to measure           (required)
#   PERF_LAB_PROMPT      prompt file; never committed              (required)
#
# PERF_LAB_SERVER_BIN is deliberately separate from PERF_LAB_BIN. The local
# GCC 15 build's llama-server segfaults on every model while its llama-bench is
# fine, so the canaries and the tracked configs are measured by different
# builds of the same upstream commit. fp.build.binary_sha256 is what keeps the
# two populations distinguishable in the ledger.
#
# One server per rep. A second request against a live server would be served
# from the prompt cache and report a prefill that was never computed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
LEDGER="${PERF_LAB_LEDGER:-$ROOT/results/ledger.jsonl}"

CFG="${1:?usage: bench_server.sh <config.yaml> <key> [--reps N] [--tag T]}"
KEY="${2:?usage: bench_server.sh <config.yaml> <key> [--reps N] [--tag T]}"
shift 2
REPS=""; TAG=""; KIND=manual
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps) REPS="$2"; shift 2 ;;
    --tag)  TAG="$2";  shift 2 ;;
    --kind) KIND="$2"; shift 2 ;;
    *) echo "bench_server.sh: unknown argument $1" >&2; exit 64 ;;
  esac
done

: "${PERF_LAB_MODEL:?set PERF_LAB_MODEL}"
: "${PERF_LAB_SERVER_BIN:?set PERF_LAB_SERVER_BIN to a directory holding a working llama-server}"
: "${PERF_LAB_GPU_UID:?set PERF_LAB_GPU_UID}"
[[ -r "$PERF_LAB_MODEL" ]] || { echo "bench_server.sh: cannot read $PERF_LAB_MODEL" >&2; exit 3; }
[[ -x "$PERF_LAB_SERVER_BIN/llama-server" ]] || {
  echo "bench_server.sh: no llama-server in $PERF_LAB_SERVER_BIN" >&2; exit 3; }

# --- guard: refuse to measure a GPU somebody else is using -------------------
PCI="$("$HERE/probe.sh" --gpu-uid "$PERF_LAB_GPU_UID" | python3 -c 'import json,sys;print(json.load(sys.stdin)["gpu"]["pci"])')"
CARD="$(for c in /sys/class/drm/card[0-9]*; do
          [[ "$(grep -oP 'PCI_SLOT_NAME=\K.*' "$c/device/uevent" 2>/dev/null | tr 'A-Z' 'a-z')" == "$PCI" ]] && basename "$c" && break
        done)"
USED_MIB=$(( $(cat "/sys/class/drm/$CARD/device/mem_info_vram_used" 2>/dev/null || echo 0) / 1048576 ))
GUARD_MIB="${PERF_LAB_GUARD_MIB:-500}"

emit_skip () {
  python3 "$HERE/emit_row.py" --config "$CFG" --key "$KEY" --kind skipped \
    --reason "$1" --tag "$TAG" --mode server --tool llama-server \
    --gpu-uid "$PERF_LAB_GPU_UID" --model "$PERF_LAB_MODEL" \
    --bin "$PERF_LAB_SERVER_BIN" >> "$LEDGER"
  echo "bench_server.sh: skipped — $1" >&2
}

if (( USED_MIB > GUARD_MIB )); then
  emit_skip "target GPU busy: ${USED_MIB} MiB in use on $PCI (guard ${GUARD_MIB} MiB)"
  exit 0
fi

IDX="$("$HERE/probe.sh" --gpu-uid "$PERF_LAB_GPU_UID" --emit-index)"
N="${REPS:-$(python3 -c "import yaml;print(yaml.safe_load(open('$CFG'))['defaults'].get('reps',1))")}"

for (( rep=1; rep<=N; rep++ )); do
  RAW="$(mktemp)"
  if ! ROCR_VISIBLE_DEVICES="$IDX" \
       python3 "$HERE/emit_row.py" --config "$CFG" --key "$KEY" --kind "$KIND" \
         --rep "$rep" --tag "$TAG" --mode server --tool llama-server \
         --gpu-uid "$PERF_LAB_GPU_UID" --model "$PERF_LAB_MODEL" \
         --bin "$PERF_LAB_SERVER_BIN" > "$RAW"; then
    emit_skip "llama-server failed on rep $rep"
    rm -f "$RAW"; exit 1
  fi
  cat "$RAW" >> "$LEDGER"
  python3 -c "
import json
r=json.load(open('$RAW')); m=r['m']
acc=m.get('mtp_acceptance')
print(f\"  rep {m['rep']}: pp={m.get('pp2048')} tg={m.get('tg')} \"
      f\"acceptance={acc if acc is None else round(acc,4)}\")"
  rm -f "$RAW"
done

python3 "$HERE/validate.py" "$LEDGER"

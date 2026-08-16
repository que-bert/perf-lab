#!/usr/bin/env bash
# Run one canary configuration and append ledger rows.
#
#   bench.sh configs/canary.yaml fast-q4 [--reps N] [--tag T] [--kind K]
#
# Environment:
#   PERF_LAB_MODEL   path to the .gguf under test          (required)
#   PERF_LAB_BIN     directory holding llama-bench         (required)
#   PERF_LAB_GPU_UID unique ID of the GPU to measure       (required)
#
# One llama-bench process per rep. Reps sharing a process would hit the prompt
# cache and report a prefill number that was never actually computed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
LEDGER="${PERF_LAB_LEDGER:-$ROOT/results/ledger.jsonl}"

CFG="${1:?usage: bench.sh <config.yaml> <key> [--reps N] [--tag T]}"
KEY="${2:?usage: bench.sh <config.yaml> <key> [--reps N] [--tag T]}"
shift 2
REPS=""; TAG=""; KIND=manual
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps) REPS="$2"; shift 2 ;;
    --tag)  TAG="$2";  shift 2 ;;
    --kind) KIND="$2"; shift 2 ;;
    *) echo "bench.sh: unknown argument $1" >&2; exit 64 ;;
  esac
done

: "${PERF_LAB_MODEL:?set PERF_LAB_MODEL to the .gguf under test}"
: "${PERF_LAB_BIN:?set PERF_LAB_BIN to the directory holding llama-bench}"
: "${PERF_LAB_GPU_UID:?set PERF_LAB_GPU_UID to the GPU unique ID to measure}"
[[ -r "$PERF_LAB_MODEL" ]] || { echo "bench.sh: cannot read $PERF_LAB_MODEL" >&2; exit 3; }

BENCH="$PERF_LAB_BIN/llama-bench"
[[ -x "$BENCH" ]] || { echo "bench.sh: $BENCH not executable" >&2; exit 3; }

# --- guard: refuse to measure a GPU somebody else is using -------------------
PCI="$("$HERE/probe.sh" --gpu-uid "$PERF_LAB_GPU_UID" | python3 -c 'import json,sys;print(json.load(sys.stdin)["gpu"]["pci"])')"
CARD="$(for c in /sys/class/drm/card[0-9]*; do
          [[ "$(cat "$c/device/uevent" 2>/dev/null | grep -oP 'PCI_SLOT_NAME=\K.*' | tr 'A-Z' 'a-z')" == "$PCI" ]] && basename "$c" && break
        done)"
USED_MIB=$(( $(cat "/sys/class/drm/$CARD/device/mem_info_vram_used" 2>/dev/null || echo 0) / 1048576 ))
GUARD_MIB="${PERF_LAB_GUARD_MIB:-500}"

# A skip carries the batch tag like any other row. Without it the skip is
# invisible to check.py, and a night where every canary was guarded off looks
# identical to a night that never ran.
emit_skip () {
  python3 "$HERE/emit_row.py" --config "$CFG" --key "$KEY" --kind skipped \
    --reason "$1" --tag "$TAG" --gpu-uid "$PERF_LAB_GPU_UID" \
    --model "$PERF_LAB_MODEL" --bin "$PERF_LAB_BIN" >> "$LEDGER"
  echo "bench.sh: skipped — $1" >&2
}

if (( USED_MIB > GUARD_MIB )); then
  emit_skip "target GPU busy: ${USED_MIB} MiB in use on $PCI (guard ${GUARD_MIB} MiB)"
  exit 0
fi

# --- run ---------------------------------------------------------------------
IDX="$("$HERE/probe.sh" --gpu-uid "$PERF_LAB_GPU_UID" --emit-index)"
N="${REPS:-$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['defaults'].get('reps',1))")}"

for (( rep=1; rep<=N; rep++ )); do
  RAW="$(mktemp)"
  # fresh process per rep — see header
  if ! ROCR_VISIBLE_DEVICES="$IDX" LD_LIBRARY_PATH="$PERF_LAB_BIN" \
       python3 "$HERE/emit_row.py" --config "$CFG" --key "$KEY" --kind "$KIND" \
         --rep "$rep" --tag "$TAG" --gpu-uid "$PERF_LAB_GPU_UID" \
         --model "$PERF_LAB_MODEL" --bin "$PERF_LAB_BIN" --run > "$RAW"; then
    emit_skip "llama-bench failed on rep $rep"
    rm -f "$RAW"; exit 1
  fi
  cat "$RAW" >> "$LEDGER"
  python3 -c "
import json,sys
r=json.load(open('$RAW'))
m=r['m']
print(f\"  rep {m['rep']}: pp2048={m.get('pp2048')} tg={m.get('tg')} cold={m['cold_prefill']}\")"
  rm -f "$RAW"
done

python3 "$HERE/validate.py" "$LEDGER"

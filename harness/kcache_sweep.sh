#!/usr/bin/env bash
# Sweep the K cache type under the SERVED config, which the 2026-09-10 KV work
# never did.
#
#   kcache_sweep.sh [out-dir]
#
# That entry found K, not V, is what costs unspeculated decode -- q4_0 K was
# 25% faster than q8_0 K at depth 0 -- and then kept q8_0 K on a quality
# argument. Both halves were measured with speculation OFF and at the default
# 4 slots. Neither holds any more: the served config is MTP n_max 3 at
# --parallel 1, and MTP changes what the decode loop is bound by.
#
# V is pinned to q4_0 (the served value) so the only moving part is K.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$(dirname "$HERE")/.scratch/kcache}"
mkdir -p "$OUT"

MODEL="${PERFLAB_MODEL:-${PERFLAB_MODEL_DIR:-/home/bbuckham/models}/Qwen3.8-27B-Q6_K.gguf}"
MMPROJ="${PERFLAB_MMPROJ:-${PERFLAB_MODEL_DIR:-/home/bbuckham/models}/mmproj-F16.gguf}"
PORT="${PERFLAB_PORT:-8921}"
CTX="${PERFLAB_CTX:-262144}"
REPS="${PERFLAB_REPS:-4}"
NPRED="${PERFLAB_NPRED:-256}"
KTYPES="${PERFLAB_KTYPES:-q8_0 q4_0 q4_1 iq4_nl q5_1}"

R9700_CARD="$(for c in /sys/class/drm/card[0-9]*; do
    t="$(cat "$c/device/mem_info_vram_total" 2>/dev/null || echo 0)"
    [ "$t" -gt 20000000000 ] && basename "$c" && break
  done)"
vram () { awk '{printf "%.2f", $1/1073741824}' \
  "/sys/class/drm/${R9700_CARD:-card0}/device/mem_info_vram_used" 2>/dev/null; }

for k in $KTYPES; do
  echo "=== ctk=$k ==="
  if ! PERFLAB_CTK="$k" PERFLAB_CTV=q4_0 "$HERE/serve_unit.sh" \
        "$MODEL" "$PORT" "$CTX" --parallel 1 --mmproj "$MMPROJ" \
        --spec-type draft-mtp --spec-draft-n-max 3; then
    echo "  LOAD FAILED at ctk=$k" | tee "$OUT/$k.failed"; continue
  fi
  echo "  vram_after_load=$(vram) GB"
  python3 "$HERE/throughput.py" --port "$PORT" --concurrency 1 \
    --reps "$REPS" --n-predict "$NPRED" \
    --label "ctk=$k ctv=q4_0, mtp n=3, parallel 1" --out "$OUT/$k.json"
  systemctl --user stop "perflab-srv-$PORT" 2>/dev/null
  timeout 10 tail -f /dev/null
done

echo "=== done ==="
for f in "$OUT"/*.json; do
  python3 -c "
import json
d=json.load(open('$f')); r=d['levels'][0]
print(f\"{d['label']:<44} agg {r.get('agg_tps')}  wall {r.get('wall_seconds')}s  accept {r.get('draft_acceptance')}\")
" 2>/dev/null
done

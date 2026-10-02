#!/usr/bin/env bash
# Measure DFlash2 / DSpark block drafting against the served MTP config.
#
#   dflash_sweep.sh [out-dir]
#
# MTP drafts 3 tokens autoregressively from a head that lives inside the Q6_K.
# DFlash2 drafts a whole block (up to its trained block size, commonly 15) in
# one forward pass from a separate 1.1-2.1 GB drafter. The trade is VRAM: MTP
# costs ~3.3 GB of *draft context* and no weights; DFlash2 costs weights plus
# its own draft context. Both numbers are recorded per run below.
#
# Requires a build with --spec-type draft-dflash: b10472-vulkan does NOT have
# it. PERFLAB_BIN defaults to b10902-adaptive-mtp accordingly, which means
# these rows are NOT comparable to the b10472 decode figures in FINDINGS.md --
# run the mtp reference below on the same build or compare nothing.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$(dirname "$HERE")/.scratch/dflash}"
mkdir -p "$OUT"

MODEL="${PERFLAB_MODEL:-${PERFLAB_MODEL_DIR:-/home/bbuckham/models}/Qwen3.8-27B-Q6_K.gguf}"
DRAFTERS="${PERFLAB_DRAFTERS:-/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/drafters}"
PORT="${PERFLAB_PORT:-8921}"
CTX="${PERFLAB_CTX:-262144}"
REPS="${PERFLAB_REPS:-4}"
NPRED="${PERFLAB_NPRED:-256}"
export PERFLAB_BIN="${PERFLAB_BIN:-$HOME/git/perf-lab/bin/upstream-vulkan/b10902-adaptive-mtp}"

# The R9700 is card0 here and the 16 GB RX 9060 XT is card1 -- the DRM order is
# the reverse of rocm-smi's GPU[0]/GPU[1]. Resolve by size rather than index.
R9700_CARD="$(for c in /sys/class/drm/card[0-9]*; do
    t="$(cat "$c/device/mem_info_vram_total" 2>/dev/null || echo 0)"
    [ "$t" -gt 20000000000 ] && basename "$c" && break
  done)"
vram () { awk '{printf "%.2f", $1/1073741824}' \
  "/sys/class/drm/${R9700_CARD:-card0}/device/mem_info_vram_used" 2>/dev/null; }

run () {  # run <label> <outfile> <extra server args...>
  local label=$1 out=$2; shift 2
  echo "=== $label ==="
  if ! "$HERE/serve_unit.sh" "$MODEL" "$PORT" "$CTX" --parallel 1 "$@"; then
    echo "  LOAD FAILED: $label" | tee "$OUT/$out.failed"; return
  fi
  echo "  vram_after_load=$(vram) GB"
  python3 "$HERE/throughput.py" --port "$PORT" --concurrency 1 \
    --reps "$REPS" --n-predict "$NPRED" --label "$label" --out "$OUT/$out.json"
  systemctl --user stop "perflab-srv-$PORT" 2>/dev/null
  timeout 10 tail -f /dev/null
}

# Reference on THIS build, so the DFlash rows have something legitimate to be
# compared against. n_max 3 is the served setting.
run "mtp n_max=3 (reference, same build)" mtp3 \
    --spec-type draft-mtp --spec-draft-n-max 3

# --spec-draft-n-max is clamped to the drafter's trained block size, so asking
# for more than the block is safe and 15 is the documented example.
for q in Q4_K_M Q8_0; do
  D="$DRAFTERS/Qwen3.8-27B-DFlash2-$q.gguf"
  [ -r "$D" ] || { echo "skip: no $D"; continue; }
  for n in 4 8 15; do
    run "dflash2 $q n_max=$n" "dflash2-$q-n$n" \
        -md "$D" --spec-type draft-dflash --spec-draft-n-max "$n" --jinja
  done
done

DS="$DRAFTERS/Qwen3.8-27B-DSpark-Q8_0-magnitudedev.gguf"
if [ -r "$DS" ]; then
  for n in 7 15; do
    run "dspark Q8_0 n_max=$n" "dspark-n$n" \
        -md "$DS" --spec-type draft-dspark --spec-draft-n-max "$n" --jinja
  done
fi

echo "=== done ==="
for f in "$OUT"/*.json; do
  python3 -c "
import json
d=json.load(open('$f')); r=d['levels'][0]
print(f\"{d['label']:<44} {r.get('per_req_tps')} t/s   accept {r.get('draft_acceptance')}\")
" 2>/dev/null
done

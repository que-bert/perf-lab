#!/usr/bin/env bash
# Sweep MTP draft depth at --parallel 1, which the original sweep never did.
#
#   nmax_sweep.sh [out-dir]
#
# The n_max optimum in FINDINGS.md ("MTP draft depth has an optimum") was
# measured at llama-server's default 4 slots. `--parallel 1` was later found to
# double q8_0/q8_0 decode and to change nothing about draft acceptance, which
# leaves the depth optimum *under one slot* unmeasured -- the served config
# carries n_max 3 from the 4-slot sweep rather than from a measurement.
#
# One server per depth, torn down between depths: a live server would serve the
# next depth's first request from the prompt cache.
#
# Environment mirrors serve_unit.sh; PERFLAB_BIN defaults to the SERVING build
# (b10472-vulkan) on purpose. Results are meant to be actionable for
# ~/.mimir/mimir.toml, and a newer build changes the baseline it would be
# compared against.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$(dirname "$HERE")/.scratch/nmax-parallel1}"
mkdir -p "$OUT"

MODEL="${PERFLAB_MODEL:-/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth/Qwen3.8-27B-Q6_K.gguf}"
MMPROJ="${PERFLAB_MMPROJ:-/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth/mmproj-F16.gguf}"
PORT="${PERFLAB_PORT:-8921}"
CTX="${PERFLAB_CTX:-262144}"
REPS="${PERFLAB_REPS:-4}"
NPRED="${PERFLAB_NPRED:-256}"
DEPTHS="${PERFLAB_DEPTHS:-1 2 3 4 5 6}"

echo "sweep: model=$(basename "$MODEL") ctx=$CTX parallel=1 reps=$REPS n_predict=$NPRED"
echo "       depths: $DEPTHS  ->  $OUT"

for n in $DEPTHS; do
  echo "=== n_max=$n ==="
  if ! "$HERE/serve_unit.sh" "$MODEL" "$PORT" "$CTX" \
        --parallel 1 --mmproj "$MMPROJ" \
        --spec-type draft-mtp --spec-draft-n-max "$n"; then
    echo "  LOAD FAILED at n_max=$n" | tee "$OUT/n$n.failed"
    continue
  fi
  python3 "$HERE/throughput.py" --port "$PORT" --concurrency 1 \
    --reps "$REPS" --n-predict "$NPRED" \
    --label "draft-mtp n_max=$n, parallel 1, ctx $CTX" \
    --out "$OUT/n$n.json"
  systemctl --user stop "perflab-srv-$PORT" 2>/dev/null
  timeout 10 tail -f /dev/null
done

# Baseline: same server, speculation off. The 21.80 / 24.08 t/s figures in
# FINDINGS were taken at the default slot count, so they are not a valid
# reference for these rows.
echo "=== speculation off ==="
if "$HERE/serve_unit.sh" "$MODEL" "$PORT" "$CTX" \
      --parallel 1 --mmproj "$MMPROJ"; then
  python3 "$HERE/throughput.py" --port "$PORT" --concurrency 1 \
    --reps "$REPS" --n-predict "$NPRED" \
    --label "no speculation, parallel 1, ctx $CTX" \
    --out "$OUT/none.json"
  systemctl --user stop "perflab-srv-$PORT" 2>/dev/null
fi

echo "=== done ==="
for f in "$OUT"/*.json; do
  python3 -c "
import json,sys
d=json.load(open('$f'))
r=d['levels'][0]
print(f\"{d['label']:<48} {r.get('per_req_tps')} t/s   accept {r.get('draft_acceptance')}\")
" 2>/dev/null
done

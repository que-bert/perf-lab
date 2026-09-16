#!/usr/bin/env bash
# Does q4_0 K cost quality? The speed half is in kcache_sweep.sh.
#
#   kcache_quality.sh [out-dir]
#
# The served config keeps `-ctk q8_0` on a quality argument that was never
# tested against q4_0 K: the 2026-09-10 comparison was q8_0/q8_0 against
# q8_0/q4_0, i.e. it moved V and left K alone. q4_0 K is the faster and much
# smaller option, so the only thing standing between it and the served config
# is a recall/code check at depth.
#
# Both arms drive an ALREADY-RUNNING server via --port, because model_eval.py
# has no KV-type flags of its own.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$(dirname "$HERE")/.scratch/kcache-quality}"
mkdir -p "$OUT"

MODEL="${PERFLAB_MODEL:-/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth/Qwen3.8-27B-Q6_K.gguf}"
BIN="${PERFLAB_BIN:-$HOME/llama.cpp/b10472-vulkan}"
PORT="${PERFLAB_PORT:-8921}"
CTX="${PERFLAB_CTX:-131072}"
# Characters, not tokens: ~4 chars/token, so these land near 30k / 60k / 110k.
LENGTHS="${PERFLAB_RECALL:-120000,240000,440000}"

for k in q8_0 q4_0; do
  echo "=== ctk=$k ==="
  if ! PERFLAB_CTK="$k" PERFLAB_CTV=q4_0 "$HERE/serve_unit.sh" \
        "$MODEL" "$PORT" "$CTX" --parallel 1 \
        --spec-type draft-mtp --spec-draft-n-max 3; then
    echo "  LOAD FAILED at ctk=$k"; continue
  fi
  python3 "$HERE/model_eval.py" --bin "$BIN" --model "$MODEL" --port "$PORT" \
    --ctx "$CTX" --recall-lengths "$LENGTHS" --suites recall,extract,code \
    --chat --budget 768 --label "ctk-$k" --out "$OUT/$k.json"
  systemctl --user stop "perflab-srv-$PORT" 2>/dev/null
  timeout 10 tail -f /dev/null
done
echo "=== done ==="

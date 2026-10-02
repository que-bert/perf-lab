#!/usr/bin/env bash
# Rep-based decode-at-depth A/B for one llama.cpp build. Promoted from the
# single-rep /tmp/opencode/ctx_build.sh so every depth claim is arbitered by
# >=3 reps over one filled context (see FINDINGS 2026-09-20: the 176k noise
# floor is ~6%, so a single rep cannot resolve a small effect).
#
#   depth.sh <build-dir> <label> <port> [extra llama-server args...]
#
# Environment:
#   PERFLAB_CTK / PERFLAB_CTV / PERFLAB_GPU / PERFLAB_NGL   as in serve_unit.sh
#   DEPTHS   prompt repeat counts (default 1,1600,4000  -> short, ~70k, ~176k)
#   REPS     decode reps per depth (default 3)
#   NPRED    tokens to decode per request (default 256)
#   DEPTH_LOG_DIR  where to write the log (default results/depth)
#
# Extra args after the port go to llama-server; the caller is responsible for
# keeping the fixed serving config (ctx 262144, -fa on, --parallel 1, MTP).
set -u
BIN=$1; LABEL=$2; PORT=$3; shift 3
MNT="${PERFLAB_MODEL_DIR:-/home/bbuckham/models}"
MODEL="${PERFLAB_MODEL:-$MNT/Qwen3.8-27B-Q6_K.gguf}"
MMPROJ="${PERFLAB_MMPROJ:-$MNT/mmproj-F16.gguf}"
DEPTHS="${DEPTHS:-1,1600,4000}"
REPS="${REPS:-3}"
NPRED="${NPRED:-256}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
SPEC_NMAX="${SPEC_NMAX:-3}"
LOG_DIR="${DEPTH_LOG_DIR:-results/depth}"
STAMP=$(date +%Y%m%dT%H%M%S)
LOG="$LOG_DIR/${LABEL}-${STAMP}.log"
mkdir -p "$LOG_DIR"
UNIT="perflab-srv-$PORT"

export PERFLAB_BIN="$BIN"
{
  echo "### label=$LABEL bin=$BIN port=$PORT date=$(date -Iseconds)"
  echo "### args: $*"
  echo "### ctk=${PERFLAB_CTK:-q8_0} ctv=${PERFLAB_CTV:-q4_0} gpu=${PERFLAB_GPU:-1} depths=$DEPTHS reps=$REPS npred=$NPRED"
  echo "### spec=${SPEC_TYPE} nmax=${SPEC_NMAX} env=${PERFLAB_ENV:-}"
  harness/gpu_guard.sh "${GUARD_TIMEOUT:-900}" || exit 1
  harness/serve_unit.sh "$MODEL" "$PORT" "${CTX:-262144}" \
    --parallel 1 --spec-type "$SPEC_TYPE" --spec-draft-n-max "$SPEC_NMAX" --mmproj "$MMPROJ" "$@"
  python3 harness/depth.py "$PORT" --depths "$DEPTHS" --reps "$REPS" --npred "$NPRED" \
    ${CORPUS:+--corpus "$CORPUS"} ${REPREFLILL:+--reprefill} \
    --json "$LOG_DIR/${LABEL}-${STAMP}.json"
} 2>&1 | tee "$LOG"
systemctl --user stop "$UNIT" 2>/dev/null
sleep 3
echo "done -> $LOG"
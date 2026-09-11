#!/usr/bin/env bash
# Run the capability suites for one model in every prompt mode.
#
#   eval_matrix.sh <label> <model-path|ollama-tag> <port> <gpu> [ctx] [budget]
#
# Three modes, because on 2026-09-10 they did not agree and the disagreement was
# larger than the difference between models:
#
#   raw          /completion with a bare prompt -- what every table in
#                FINDINGS.md before this date was measured with
#   chat         the model's own chat template
#   chat-nothink the template with enable_thinking=false
#
# A label beginning with "ollama:" drives ollama's runtime instead of a
# llama-server unit, for GGUFs upstream llama.cpp will not load. Those runs are
# capability-only; ollama picks its own KV types and offload, so no speed number
# from them belongs in a speed table.
#
# Environment: PERFLAB_BIN selects the llama.cpp build (default b10472-vulkan).
set -u
LABEL=$1; MODEL=$2; PORT=$3; GPU=$4; CTX=${5:-65536}; BUDGET=${6:-2048}
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
BIN="${PERFLAB_BIN:-$HOME/llama.cpp/b10472-vulkan}"
OUT="$REPO/results"
STAMP=$(date +%Y%m%d)
SUITES="code,math,instruct,extract,research"

if [[ "$MODEL" == ollama:* ]]; then
  TAG="${MODEL#ollama:}"
  API=(--api ollama --ollama-model "$TAG" --model "$TAG")
  echo "=== $LABEL  via ollama ($TAG)"
else
  API=(--model "$MODEL")
  echo "=== $LABEL  on gpu $GPU port $PORT"
  PERFLAB_GPU="$GPU" "$HERE/serve_unit.sh" "$MODEL" "$PORT" "$CTX" || exit 1
  API+=(--port "$PORT")
fi

run () {  # run <mode-name> <extra flags...>
  local mode=$1; shift
  local out="$OUT/eval-$LABEL-$mode-$STAMP.json"
  echo "--- $LABEL  $mode"
  # stdbuf, because a suite can run for half an hour and a block-buffered pipe
  # makes a live run indistinguishable from a hung one.
  timeout 5400 stdbuf -oL -eL python3 "$HERE/model_eval.py" --bin "$BIN" \
    "${API[@]}" --ctx "$CTX" --budget "$BUDGET" --suites "$SUITES" \
    --label "$LABEL" --out "$out" "$@" 2>&1 | stdbuf -oL sed 's/^/    /'
}

run raw
run chat --chat
run chat-nothink --chat --no-think

if [[ "$MODEL" != ollama:* ]]; then
  systemctl --user stop "perflab-srv-$PORT" 2>/dev/null
  systemctl --user reset-failed "perflab-srv-$PORT" 2>/dev/null
fi
echo "=== $LABEL done"

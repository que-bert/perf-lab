#!/usr/bin/env bash
# mtp_prompt_ab.sh <bin-dir> <label> ["ENV=1;ENV2=x"] [spec|nospec]
# Server prompt rate on a fixed ~32.5k-token prompt (setups/qwen3.8-27b-r9700/results/d0/i8/mtpp_prompt.json), serving config,
# 2 requests (cache_prompt=false). Prints "label prompt_n t/s" per request. Takes the GPU lock itself.
# EXTRA_ARGS="..." appends server flags (e.g. "-ub 1024 -b 2048").
set -u
cd /home/bbuckham/git/perf-lab
B=$1; LB=$2; EV=${3:-}; MODE=${4:-spec}
M=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth
P=$PWD/setups/qwen3.8-27b-r9700/results/d0/i8/mtpp_prompt.json
SPEC="--spec-type draft-mtp --spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive"
[ "$MODE" = nospec ] && SPEC=""
timeout 900 harness/gpu_lock.sh bash -c "PERFLAB_BIN=$B PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 PERFLAB_LOG=${PERFLAB_LOG:-} PERFLAB_ENV='MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1${EV:+;$EV}' harness/serve_unit.sh ${QMODEL:-$M/Qwen3.8-27B-Q6_K.gguf} 8103 262144 --parallel 1 $SPEC ${EXTRA_ARGS:-} -lm none --mmproj $M/mmproj-F16.gguf >/dev/null
  for i in 1 2; do curl -s localhost:8103/completion -d @$P | python3 -c 'import json,sys; d=json.load(sys.stdin); print(\"$LB\", d[\"timings\"][\"prompt_n\"], round(d[\"timings\"][\"prompt_per_second\"],1))'; done
  systemctl --user stop perflab-srv-8103; sleep 2"

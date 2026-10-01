#!/usr/bin/env bash
# VSUB depth check: step timing (pre / rec / prepass phases) of the vsub build at ~1k and ~70k, off vs on.
set -u
cd /home/bbuckham/git/perf-lab
BIN=/home/bbuckham/git/llama.cpp-r9700-ar-vsub/build/bin
OUT=setups/qwen3.8-27b-r9700/results/d0/vsub
mkdir -p "$OUT"
export PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 DEPTH_LOG_DIR=$OUT
SRV="MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1;GGML_VK_STEP_TIMING=1;GGML_VK_STEP_TIMING_N=60;LLAMA_DECODE_TIMING=1;LLAMA_DECODE_TIMING_N=60"
XA=(-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none)
run() { # label env
  export PERFLAB_LOG=$PWD/$OUT/$1.stderr PERFLAB_ENV="$2" SPEC_NMAX=4 DEPTHS=1,1600 REPS=2 NPRED=256
  echo "== $1 start $(date -Iseconds) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  harness/gpu_lock.sh harness/depth.sh "$BIN" "$1" 8097 "${XA[@]}" 2>&1 | grep -E "~[0-9]+k|rror"
}
run vsub-off "$SRV;GGML_VK_NO_VSUB=1"
run vsub-on  "$SRV"
echo "== VSUB_TIM DONE $(date -Iseconds)"

#!/usr/bin/env bash
# D10: decode budget of serving build (integrate10) at short/70k/176k, n_max 4.
set -u
cd /home/bbuckham/git/perf-lab
BIN=/home/bbuckham/git/llama.cpp-r9700/build/bin
OUT=results/d0/d10
mkdir -p "$OUT"
export PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 DEPTH_LOG_DIR=$OUT
SRV="MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1"
TIM="$SRV;GGML_VK_STEP_TIMING=1;GGML_VK_STEP_TIMING_N=60;LLAMA_DECODE_TIMING=1;LLAMA_DECODE_TIMING_N=60"
PERF="$SRV;GGML_VK_PERF_LOGGER=1;GGML_VK_PERF_LOGGER_FREQUENCY=1"
XA=(-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none)
run() { # label env depths reps npred
  export PERFLAB_LOG=$PWD/$OUT/$1.stderr PERFLAB_ENV="$2" SPEC_NMAX=4 DEPTHS=$3 REPS=$4 NPRED=$5
  echo "== $1 start $(date -Iseconds) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  harness/gpu_lock.sh harness/depth.sh "$BIN" "$1" 8097 "${XA[@]}"
}
run d10-tim  "$TIM"  1,1600,4000 3 256
run d10-perf "$PERF" 1,1600,4000 1 64
echo "== D10 DONE $(date -Iseconds)"

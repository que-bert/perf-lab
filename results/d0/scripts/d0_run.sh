#!/usr/bin/env bash
# D0: one-run decode budget at 70k/176k, n_max 3 (verify n=4) vs n_max 4 (verify n=5).
# timing runs: host-side step timing, no perf logger (clean wall numbers).
# perf runs:   per-op GPU time per graph (perf logger, FREQUENCY=1).
set -u
cd /home/bbuckham/git/perf-lab
BIN=/home/bbuckham/git/llama.cpp-r9700-integrate2/build/bin
OUT=results/d0
mkdir -p "$OUT"
export PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 DEPTHS=1600,4000 DEPTH_LOG_DIR=$OUT
TIM="GGML_VK_STEP_TIMING=1;GGML_VK_STEP_TIMING_N=60;LLAMA_DECODE_TIMING=1;LLAMA_DECODE_TIMING_N=60"
PERF="GGML_VK_PERF_LOGGER=1;GGML_VK_PERF_LOGGER_FREQUENCY=1"
run() { # label nmax env reps npred
  export PERFLAB_LOG=$PWD/$OUT/$1.stderr PERFLAB_ENV="$3" SPEC_NMAX=$2 REPS=$4 NPRED=$5
  echo "== $1 start $(date -Iseconds) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  harness/depth.sh "$BIN" "$1" 8097
}
run d0-tim-n3  3 "$TIM"  3 256
run d0-tim-n4  4 "$TIM"  3 256
run d0-perf-n3 3 "$PERF" 1 64
run d0-perf-n4 4 "$PERF" 1 64
echo "== D0 DONE $(date -Iseconds)"

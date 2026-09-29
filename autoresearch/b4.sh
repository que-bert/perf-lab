#!/usr/bin/env bash
# B4: perf-logger decode budget on integrate4 (serving flags), n_max 3 and 4, 70k/176k
set -u
cd /home/bbuckham/git/perf-lab
BIN=/home/bbuckham/git/llama.cpp-r9700-integrate4/build/bin
OUT=results/d0/b4; mkdir -p $OUT
export PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 DEPTHS=1600,4000 DEPTH_LOG_DIR=$OUT
F4="-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive"
for nm in 3 4; do
  export PERFLAB_LOG=$PWD/$OUT/b4-perf-n$nm.stderr PERFLAB_ENV="GGML_VK_PERF_LOGGER=1;GGML_VK_PERF_LOGGER_FREQUENCY=1" SPEC_NMAX=$nm REPS=1 NPRED=64
  echo "== b4-perf-n$nm $(date -Iseconds) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  harness/gpu_lock.sh harness/depth.sh $BIN b4-perf-n$nm 8097 $F4 | grep -E "~[0-9]+k"
  python3 results/d0/scripts/d0_perf.py $OUT/b4-perf-n$nm.stderr --top 30 > $OUT/b4-perf-n$nm.txt
done
echo "== DONE"

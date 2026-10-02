#!/usr/bin/env bash
# inmodel.sh <bin> <label> "<ENV;ENV>" [nmax]  -> per-op verify/draft GPU times at short ctx (perf logger)
set -u
cd /home/bbuckham/git/perf-lab
B=$1; LB=$2; EV=$3; NM=${4:-3}
M=${PERFLAB_MODEL_DIR:-/home/bbuckham/models}/Qwen3.8-27B-Q6_K.gguf
O=setups/qwen3.8-27b-r9700/results/d0/d10; mkdir -p $O
harness/gpu_lock.sh bash -c "PERFLAB_BIN=$B PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 PERFLAB_LOG=$PWD/$O/$LB.stderr PERFLAB_ENV='GGML_VK_PERF_LOGGER=1;GGML_VK_PERF_LOGGER_FREQUENCY=1${EV:+;$EV}' harness/serve_unit.sh $M 8103 16384 --parallel 1 --spec-type draft-mtp --spec-draft-n-max $NM -ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none --mmproj ${PERFLAB_MODEL_DIR:-/home/bbuckham/models}/mmproj-F16.gguf >/dev/null
  curl -s localhost:8103/completion -d '{\"prompt\":\"Write a long essay about the history of computing.\",\"n_predict\":160,\"temperature\":0}' >/dev/null
  systemctl --user stop perflab-srv-8103; sleep 2"
python3 setups/qwen3.8-27b-r9700/results/d0/scripts/d0_perf.py $O/$LB.stderr --top 40 > $O/$LB.txt
awk '/^## verify/{v=1; print; next} /^## /{v=0} v && /MUL_MAT/' $O/$LB.txt | awk '{s+=$1} END{print "GEMV sum (verify, ms):", s}'
grep "^## " $O/$LB.txt

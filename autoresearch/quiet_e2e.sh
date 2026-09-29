#!/usr/bin/env bash
# Quiet-window e2e batch: depth.sh A/B (DEPTHS 1600,4000) + pool.py n3 vs n4.
#   RUNS="label:bindir:nmax ..."  POOL="label:bindir:nmax ..."  quiet_e2e.sh
# Each run waits for 1-min loadavg < ${MAXLOAD:-2} and then takes the GPU lock.
set -u
P=/home/bbuckham/git/perf-lab; cd $P
MNT=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth; M=$MNT/Qwen3.8-27B-Q6_K.gguf
L=$P/harness/gpu_lock.sh
F="-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive"
quiet() { until awk -v m="${MAXLOAD:-2}" '{exit !($1 < m)}' /proc/loadavg; do sleep 30; done; }
for r in ${RUNS:-}; do IFS=: read -r lb bin nm <<< "$r"
  quiet; echo "== depth $lb n$nm start load=$(cut -d' ' -f1-3 /proc/loadavg)"
  $L env DEPTHS=1600,4000 REPS=3 PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 SPEC_NMAX=$nm DEPTH_LOG_DIR=results/d0/e2e \
    harness/depth.sh $bin q-$lb-n$nm 8097 $F 2>&1 | grep -E "depth|t/s|tps|done|rror" | tail -12
  echo "== depth $lb n$nm end load=$(cut -d' ' -f1-3 /proc/loadavg)"
done
for r in ${POOL:-}; do IFS=: read -r lb bin nm <<< "$r"
  quiet
  $L bash -c "PERFLAB_BIN=$bin PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh $M 8097 262144 --parallel 1 --spec-type draft-mtp --spec-draft-n-max $nm --mmproj $MNT/mmproj-F16.gguf $F >/dev/null
    echo \"== pool $lb n$nm load=\$(cut -d' ' -f1-3 /proc/loadavg)\"; python3 autoresearch/pool.py 8097 autoresearch/runs/pool-$lb-n$nm.json | grep POOL
    echo \"== pool end load=\$(cut -d' ' -f1-3 /proc/loadavg)\"; systemctl --user stop perflab-srv-8097; sleep 3"
done
echo "== QUIET_DONE $(date)"

#!/usr/bin/env bash
# HGR A/B (B=<bindir> T=<label>): base + Vulkan GET_ROWS from pinned host token_embd (GGML_VK_HOST_GET_ROWS=1).
# Both arms run the same binary with -lm none (token_embd in Vulkan_Host); only the env differs.
# ABAB depth.sh (70k/176k, 3 reps), then pooled t/s for each arm. Quiet window: loadavg < ${MAXLOAD:-2}.
set -u
P=/home/bbuckham/git/perf-lab; cd $P
MNT=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth; M=$MNT/Qwen3.8-27B-Q6_K.gguf
B=${B:-/home/bbuckham/git/llama.cpp-r9700-ar-hgr/build/bin}; T=${T:-hgr}
L=$P/harness/gpu_lock.sh
F="-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none"
quiet() { until awk -v m="${MAXLOAD:-2}" '{exit !($1 < m)}' /proc/loadavg; do sleep 30; done; }
# BASE_ENV (KEY=VAL;...) goes to both arms, e.g. BASE_ENV=MTMD_LAZY_GPU=1 after VB.
env_of() { local e="${BASE_ENV:-}"; [ "$1" = on ] && e="${e:+$e;}GGML_VK_HOST_GET_ROWS=1"; echo "$e"; }
for arm in off on off on; do
  quiet; echo "== depth $T-$arm start load=$(cut -d' ' -f1-3 /proc/loadavg)"
  $L env PERFLAB_ENV="$(env_of $arm)" DEPTHS=1600,4000 REPS=3 PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 SPEC_NMAX=4 DEPTH_LOG_DIR=results/d0/e2e \
    harness/depth.sh $B q-$T-$arm-n4 8097 $F 2>&1 | grep -E "~[0-9]+k|rror" | tail -4
  echo "== depth $T-$arm end load=$(cut -d' ' -f1-3 /proc/loadavg)"
done
for arm in off on; do
  quiet
  $L bash -c "PERFLAB_BIN=$B PERFLAB_ENV='$(env_of $arm)' PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh $M 8097 262144 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4 --mmproj $MNT/mmproj-F16.gguf $F >/dev/null
    echo \"== pool $T-$arm load=\$(cut -d' ' -f1-3 /proc/loadavg)\"; python3 autoresearch/pool.py 8097 autoresearch/runs/pool-$T-$arm-n4.json | grep POOL
    echo \"== pool end load=\$(cut -d' ' -f1-3 /proc/loadavg)\"; systemctl --user stop perflab-srv-8097; sleep 3"
done
echo "== HGR_DONE $(date)"

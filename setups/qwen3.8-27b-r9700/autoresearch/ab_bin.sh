#!/usr/bin/env bash
# Binary A/B at the full serving config: A=<base bindir> B=<candidate bindir> T=<label>.
# ABAB depth.sh (70k/176k, 3 reps; prefill + decode), then pooled t/s + texts per arm, then one image test on B.
# Both arms get the serving env (MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1) plus EXTRA_ENV. Quiet window: loadavg < ${MAXLOAD:-2}.
set -u
P=/home/bbuckham/git/perf-lab; cd $P
MNT=${PERFLAB_MODEL_DIR:-/home/bbuckham/models}; M=$MNT/Qwen3.8-27B-Q6_K.gguf
A=${A:-/home/bbuckham/git/perf-lab/runners/llama.cpp/qwen3.8-r9700/build/bin}; B=${B:?candidate bindir}; T=${T:?label}
L=$P/harness/gpu_lock.sh
E="MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1${EXTRA_ENV:+;$EXTRA_ENV}"
F="-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none"
quiet() { until awk -v m="${MAXLOAD:-2}" '{exit !($1 < m)}' /proc/loadavg; do sleep 30; done; }
bin_of() { [ "$1" = base ] && echo $A || echo $B; }
# PHASES (default "depth pool img"): run a subset, e.g. PHASES=pool after a job hit the 2 h limit. DEPTH_ARMS overrides the ABAB order.
PH=" ${PHASES:-depth pool img} "
[[ $PH == *" depth "* ]] && for arm in ${DEPTH_ARMS:-base cand base cand}; do
  quiet; echo "== depth $T-$arm start load=$(cut -d' ' -f1-3 /proc/loadavg)"
  $L env PERFLAB_ENV="$E" DEPTHS=1600,4000 REPS=3 PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 SPEC_NMAX=4 DEPTH_LOG_DIR=setups/qwen3.8-27b-r9700/results/d0/e2e \
    harness/depth.sh $(bin_of $arm) q-$T-$arm 8097 $F 2>&1 | grep -E "~[0-9]+k|rror" | tail -4
done
[[ $PH == *" pool "* ]] && for arm in base cand; do
  quiet
  $L bash -c "PERFLAB_BIN=$(bin_of $arm) PERFLAB_ENV='$E' PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh $M 8097 262144 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4 --mmproj $MNT/mmproj-F16.gguf $F >/dev/null
    echo \"== pool $T-$arm load=\$(cut -d' ' -f1-3 /proc/loadavg)\"; python3 setups/qwen3.8-27b-r9700/autoresearch/pool.py 8097 setups/qwen3.8-27b-r9700/autoresearch/runs/pool-$T-$arm.json | grep POOL
    systemctl --user stop perflab-srv-8097; sleep 3"
done
[[ $PH == *" img "* ]] && quiet && $L env PERFLAB_ENV="$E" setups/qwen3.8-27b-r9700/results/d0/scripts/imgtest.sh $B img-$T -lm none 2>&1 | tail -12
echo "== AB_DONE $T $(date)"

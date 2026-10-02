#!/usr/bin/env bash
# q4e.sh <model.gguf> <label> [bindir]: Q4E step 1, one quant at the full serving config on one binary (default serving).
# depth.sh 70k/176k (3 reps), pooled decode (37 prompts), 32.5k server prompt, VRAM peak, then gates4 (PPL, KLD vs the
# Q6_K reference logits = quality cost of the quant, acceptance). Compare against the Q6_K rows of the same binary.
set -u
P=/home/bbuckham/git/perf-lab; cd $P
Q=$1; T=$2; B=${3:-$P/runners/llama.cpp/qwen3.8-r9700/build/bin}
D=setups/qwen3.8-27b-r9700; L=$P/harness/gpu_lock.sh
MNT=${PERFLAB_MODEL_DIR:-/home/bbuckham/models}
E="MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1"
F="-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none"
quiet() { until awk -v m="${MAXLOAD:-2}" '{exit !($1 < m)}' /proc/loadavg; do sleep 30; done; }
PH=" ${PHASES:-depth pool prompt vram gates} "
[[ $PH == *" depth "* ]] && { quiet; echo "== depth $T"
  $L env PERFLAB_MODEL=$Q PERFLAB_ENV="$E" DEPTHS=1600,4000 REPS=3 PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 SPEC_NMAX=4 DEPTH_LOG_DIR=$D/results/d0/e2e \
    harness/depth.sh $B q4e-$T 8097 $F 2>&1 | grep -E "~[0-9]+k|rror" | tail -4; }
[[ $PH == *" pool "* ]] && { quiet
  $L bash -c "PERFLAB_BIN=$B PERFLAB_ENV='$E' PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh $Q 8097 262144 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4 --mmproj $MNT/mmproj-F16.gguf $F >/dev/null
    echo \"== pool $T\"; python3 $D/autoresearch/pool.py 8097 $D/autoresearch/runs/pool-q4e-$T.json | grep POOL
    systemctl --user stop perflab-srv-8097; sleep 3"; }
[[ $PH == *" prompt "* ]] && { quiet; QMODEL=$Q $D/results/d0/scripts/mtp_prompt_ab.sh $B prompt-$T; }
[[ $PH == *" vram "* ]] && { quiet; echo "== vram $T"; QMODEL=$Q PERFLAB_ENV="$E" $D/autoresearch/vram_peak.sh $B q4e-$T --spec-draft-n-max 4 $F | tail -2; }
[[ $PH == *" gates "* ]] && { quiet; GATES_CLI="-lm none" PERFLAB_ENV="$E" QMODEL=$Q $D/results/d0/scripts/gates4.sh $(dirname $(dirname $B)) q4e-$T 2>&1 | grep -E '== (PPL|KLD|ACC)'; }
echo "== Q4E_DONE $T $(date)"

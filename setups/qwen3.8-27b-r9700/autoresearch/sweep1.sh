#!/usr/bin/env bash
# draft-policy sweep (n_max x p_min) on integrate4, pooled tok/step + t/s over 37 prompts
set -u
S=/tmp/claude-1000/-home-bbuckham-git-perf-lab/b29d0a53-3607-41fd-86df-7e270e1c4360/scratchpad
P=/home/bbuckham/git/perf-lab; cd $P
I4=/home/bbuckham/git/llama.cpp-r9700-integrate4/build/bin
MNT=${PERFLAB_MODEL_DIR:-/home/bbuckham/models}; M=$MNT/Qwen3.8-27B-Q6_K.gguf
L=$P/harness/gpu_lock.sh
F4="-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive"
mkdir -p $P/autoresearch/runs
one() { nm=$1; pm=$2
  $L bash -c "PERFLAB_BIN=$I4 PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh $M 8097 262144 --parallel 1 --spec-type draft-mtp --spec-draft-n-max $nm --spec-draft-p-min $pm --mmproj $MNT/mmproj-F16.gguf $F4 >/dev/null
    echo \"== n$nm p$pm load=\$(cut -d' ' -f1-3 /proc/loadavg)\"; python3 setups/qwen3.8-27b-r9700/autoresearch/pool.py 8097 setups/qwen3.8-27b-r9700/autoresearch/runs/pol-n$nm-p$pm.json | grep POOL
    systemctl --user stop perflab-srv-8097; sleep 3"; }
IFS=, read -ra CF <<< "${CONFIGS:-3:0,4:0,4:0.6,5:0.6,5:0.8,6:0.8,3:0.6,4:0.4,6:0.9}"
for c in "${CF[@]}"; do one "${c%%:*}" "${c#*:}"; done
echo "== DONE $(date)"

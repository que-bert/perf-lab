#!/usr/bin/env bash
# T2 gate (docs/2026-10-01-phase2-plan.md, "Gate tiers"): candidate integration build vs the integrate16 anchor.
#
#   t2.sh <candidate-worktree> <label>        anchor = A (default: serving build, = integrate16)
#
# I1 (Qwen3.8-27B Q6_K, serving config): depth.sh ABAB x K (decode+prefill 70k/176k), pooled 37 prompts + texts,
# 32.5k server prompt ABAB x K, and the flagless auto-config path (pooled, server started with no tuning flags).
# I2 (MiniCPM5-2B Q4_K_M): model_profile ABAB x K. gates4 (ops, PPL, decode KLD, acceptance) and zoo quality/smoke (I3)
# on the candidate. Every GPU step takes gpu_lock.sh. Output: results/phase2/t2/<label>/*.log; summarise with ab_stat.py.
set -u
P=/home/bbuckham/git/perf-lab; cd $P; . harness/zoo.sh
W=$1; T=$2; B=$W/build/bin
A=${A:-$P/runners/llama.cpp/qwen3.8-r9700/build/bin}
K=${K:-3}
O=$P/results/phase2/t2/$T; mkdir -p $O
S=setups/qwen3.8-27b-r9700; MNT=$ZOO_M/qwen3.8:27b/unsloth
PH=" ${PHASES:-depth pool prompt flagless i2 gates zoo} "
arms=""; for i in $(seq 1 $K); do arms="$arms base cand"; done
[[ $PH == *" depth "* ]] && A=$A B=$B T=t2-$T PHASES=depth DEPTH_ARMS="$arms" MAXLOAD=4 $S/autoresearch/ab_bin.sh > $O/depth.log 2>&1
[[ $PH == *" pool "* ]] && A=$A B=$B T=t2-$T PHASES=pool MAXLOAD=4 $S/autoresearch/ab_bin.sh > $O/pool.log 2>&1
if [[ $PH == *" prompt "* ]]; then
  # the lock is taken outside so mtp_prompt_ab.sh's 900 s timeout does not count queue time
  for i in $(seq 1 $K); do harness/gpu_lock.sh $S/results/d0/scripts/mtp_prompt_ab.sh $A base; harness/gpu_lock.sh $S/results/d0/scripts/mtp_prompt_ab.sh $B cand; done > $O/prompt.log 2>&1
fi
if [[ $PH == *" flagless "* ]]; then
  # only what the operator must set by hand: mmproj, and -lm none + HOST_GET_ROWS (must be set together, never by preset)
  harness/gpu_lock.sh bash -c "systemd-run --user --unit=perflab-srv-8097 --collect --setenv=LD_LIBRARY_PATH=$B --setenv=GGML_VK_VISIBLE_DEVICES=1 \
      --setenv=MTMD_LAZY_GPU=1 --setenv=GGML_VK_HOST_GET_ROWS=1 -p StandardError=append:$O/flagless-server.log \
      $B/llama-server -m $MNT/Qwen3.8-27B-Q6_K.gguf --port 8097 --host 127.0.0.1 --no-webui -ngl 99 -lm none --mmproj $MNT/mmproj-F16.gguf >/dev/null
    for i in \$(seq 1 120); do curl -sf localhost:8097/health >/dev/null && break; sleep 5; done
    python3 $S/autoresearch/pool.py 8097 $S/autoresearch/runs/pool-t2-$T-flagless.json | grep POOL
    systemctl --user stop perflab-srv-8097; sleep 3" > $O/flagless.log 2>&1
  grep -h 'preset:' $O/flagless-server.log >> $O/flagless.log
fi
if [[ $PH == *" i2 "* ]]; then
  for i in $(seq 1 $((K + 1))); do
    PROFILE_DIR=$O/i2 REPS=10 harness/model_profile.sh $A minicpm-q4km base-r$i
    PROFILE_DIR=$O/i2 REPS=10 harness/model_profile.sh $B minicpm-q4km cand-r$i
  done > $O/i2.log 2>&1
  python3 harness/profile_report.py base cand --dir $O/i2 >> $O/i2.log
fi
[[ $PH == *" gates "* ]] && GATES_CLI="-lm none" PERFLAB_ENV="MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1" $S/results/d0/scripts/gates4.sh $W t2-$T > $O/gates.log 2>&1
if [[ $PH == *" zoo "* ]]; then
  export PERFLAB_MAXLOAD=99
  for n in $(zoo_names); do ACC=${ZOO_ACC:-1} harness/model_quality.sh eval $B $n t2-$T; done > $O/zoo-quality.log 2>&1
  harness/zoo_smoke.sh test $B t2-$T > $O/zoo-smoke.log 2>&1
fi
echo "== T2_DONE $T $(date)"

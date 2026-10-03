#!/usr/bin/env bash
# W0: re-run prefill on the large models after the drain check (fork prefill ramped 680->1600 in some rounds).
cd /home/bbuckham/git/perf-lab
R=runners/llama.cpp
declare -A BIN=([i16]=$R/qwen3.8-r9700-next/build/bin [upce]=$R/upstream-ce8caa6/build/bin [rb]=$R/qwen3.8-r9700-rb/build/bin [upa8]=$R/upstream-a868c3e/build/bin)
for r in 1 2 3; do for n in qwen27-q6k qwen27-q4km qwen36-a3b; do for a in i16 upce rb upa8; do
  PROFILE_DIR=results/phase2/profile-pp TG=0 harness/model_profile.sh ${BIN[$a]} $n $a-r$r
done; done; done
echo "== PP_RERUN_DONE $(date)"

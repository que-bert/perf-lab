#!/usr/bin/env bash
# W0: speed profile of every zoo model, fork vs upstream at the same base, interleaved rounds for run-to-run variance.
# i16 vs upstream-ce8caa6 x ROUNDS (ABAB), then rb vs upstream-a868c3e once.
cd /home/bbuckham/git/perf-lab; . harness/zoo.sh
R=runners/llama.cpp
declare -A BIN=([i16]=$R/qwen3.8-r9700-next/build/bin [upce]=$R/upstream-ce8caa6/build/bin [rb]=$R/qwen3.8-r9700-rb/build/bin [upa8]=$R/upstream-a868c3e/build/bin)
for r in $(seq 1 ${ROUNDS:-3}); do for n in $(zoo_names); do for a in i16 upce; do
  harness/model_profile.sh ${BIN[$a]} $n $a-r$r
done; done; done
for n in $(zoo_names); do for a in rb upa8; do harness/model_profile.sh ${BIN[$a]} $n $a-r1; done; done
echo "== W0_PROFILE_DONE $(date)"

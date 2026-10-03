#!/usr/bin/env bash
# W2: which fork switch costs MiniCPM Q4_K_M decode vs upstream. tg128 d0, one switch off at a time, ABAB with default.
cd /home/bbuckham/git/perf-lab
OFF=$(grep -o "OFF='[^']*'" results/phase2/w0_alloff.sh | cut -d"'" -f2)
I16=runners/llama.cpp/qwen3.8-r9700-next/build/bin; UP=runners/llama.cpp/upstream-ce8caa6/build/bin
M=${M:-minicpm-q4km}
tg() { PROFILE_DIR=results/phase2/bisect REPS=5 DEPTHS=0 PP=0 harness/model_profile.sh "$@" 2>/dev/null | awk '/tg/{print $5}'; }
echo "base i16 $(tg $I16 $M def-a) up $(tg $UP $M up-a) i16 $(tg $I16 $M def-b) up $(tg $UP $M up-b)"
IFS=';' read -ra KV <<< "$OFF"
echo "alloff $(env "${KV[@]}" bash -c "$(declare -f tg); tg $I16 $M alloff")"
for kv in "${KV[@]}"; do echo "$kv $(env "$kv" bash -c "$(declare -f tg); tg $I16 $M x-${kv%%=*}")"; done
echo "== BISECT_DONE $M $(date)"

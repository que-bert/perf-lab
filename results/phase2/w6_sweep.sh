#!/usr/bin/env bash
# W6: runtime config per zoo model (not the 27B: its preset is pinned and changes only via I1): ubatch x KV type,
# pp512/pp2048 + tg128 at d0 and d8192, on integrate17. Winners -> W4 presets.
cd /home/bbuckham/git/perf-lab; . harness/zoo.sh
B=runners/llama.cpp/qwen3.8-r9700-i17/build/bin
for n in minicpm-q4km minicpm-q8 gemma4-e4b ornith-q4km ornith-q6k qwen36-a3b; do
  for kv in f16 q8_0; do
    PROFILE_DIR=results/phase2/w6 PP=512,2048 DEPTHS=0,8192 REPS=3 harness/model_profile.sh $B $n kv$kv -ub 256,512,1024 -ctk $kv -ctv $kv
  done
done
echo "== W6_SWEEP_DONE $(date)"

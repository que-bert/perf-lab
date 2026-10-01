#!/usr/bin/env bash
cd /home/bbuckham/git/perf-lab
BASE=/home/bbuckham/git/perf-lab/runners/llama.cpp/qwen3.8-r9700/build/bin; NEW=/home/bbuckham/git/llama.cpp-r9700-ar-vp/build/bin
for r in 1 2; do
  for v in base on; do
    if [ $v = base ]; then B=$BASE; X=""; else B=$NEW; X="LLAMA_RELEASE_COMPUTE_ON_IMAGE=1"; fi
    echo "=== $v run$r"; BIN=$B LABEL=$v$r XENV=$X setups/qwen3.8-27b-r9700/results/d0/scripts/vram_img_peak_ab.sh 2>&1 | grep -v "^  guard"
  done
done

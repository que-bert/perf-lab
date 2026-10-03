#!/usr/bin/env bash
# W3 step 1: per-op decode timing on i17 for 27B Q6_K, 27B Q4_K_M, 35B-A3B (Q5_K/Q6_K/Q8_0), Ornith Q4_K_M -> GEMV GB/s per quant.
cd /home/bbuckham/git/perf-lab; . harness/zoo.sh; O=results/phase2/w3; mkdir -p $O
B=runners/llama.cpp/qwen3.8-r9700-i17/build/bin
for n in qwen27-q6k qwen27-q4km qwen36-a3b ornith-q4km; do
  harness/gpu_lock.sh env GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 $B/llama-bench -m $(zoo_path $n) -dev Vulkan1 -ngl 99 -fa 1 -p 0 -n 32 -r 1 > $O/perflog-tg-$n.txt 2>&1
done
echo "== W3_BW_DONE"

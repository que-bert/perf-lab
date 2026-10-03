#!/usr/bin/env bash
# W2: per-op GPU time, MiniCPM Q4_K_M decode, i16 vs upstream (GGML_VK_PERF_LOGGER).
cd /home/bbuckham/git/perf-lab; . harness/zoo.sh; O=results/phase2/w2; mkdir -p $O
M=$(zoo_path ${1:-minicpm-q4km})
for a in i16:runners/llama.cpp/qwen3.8-r9700-next/build/bin up:runners/llama.cpp/upstream-ce8caa6/build/bin; do
  harness/gpu_lock.sh env GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 ${a#*:}/llama-bench -m $M -dev Vulkan1 -ngl 99 -fa 1 -p 0 -n 64 -r 1 > $O/perflog-${1:-minicpm-q4km}-${a%%:*}.txt 2>&1
done
echo "== PERFLOG_DONE"

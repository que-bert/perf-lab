#!/usr/bin/env bash
# W5: Gemma4-E4B probes: per-op decode timing (fork i17 vs upstream), FA coverage at hsk 512/256 (fork vs upstream).
cd /home/bbuckham/git/perf-lab; . harness/zoo.sh; O=results/phase2/w5; mkdir -p $O
M=$(zoo_path gemma4-e4b)
for a in i17:runners/llama.cpp/qwen3.8-r9700-i17/build/bin up:runners/llama.cpp/upstream-ce8caa6/build/bin; do
  harness/gpu_lock.sh env GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 ${a#*:}/llama-bench -m $M -dev Vulkan1 -ngl 99 -fa 1 -p 0 -n 64 -r 1 > $O/perflog-tg-${a%%:*}.txt 2>&1
  harness/gpu_lock.sh env GGML_VK_PERF_LOGGER=1 ${a#*:}/llama-bench -m $M -dev Vulkan1 -ngl 99 -fa 1 -p 512 -n 0 -r 1 > $O/perflog-pp-${a%%:*}.txt 2>&1
  for h in 512 256; do
    harness/gpu_lock.sh ${a#*:}/test-backend-ops test -b Vulkan1 -o FLASH_ATTN_EXT -p "hsk=$h," > $O/fa$h-${a%%:*}.txt 2>&1; echo "fa$h ${a%%:*} rc=$? $(grep -E 'tests passed|FAIL|not supported' $O/fa$h-${a%%:*}.txt | sort | uniq -c | head -5 | tr '\n' ' ')"
  done
done
echo "== W5_PROBE_DONE"

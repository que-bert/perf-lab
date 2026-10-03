#!/usr/bin/env bash
# W3 decision input: quality cost of 27B Q4_K_M = KLD of Q4_K_M (fork i17) against the Q6_K upstream reference logits.
cd /home/bbuckham/git/perf-lab
harness/gpu_lock.sh runners/llama.cpp/qwen3.8-r9700-i17/build/bin/llama-perplexity -m /mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth/Qwen3.8-27B-Q4_K_M.gguf -f harness/corpus/decode_kld.txt -c 1024 --chunks 8 -fa on -ngl 99 -dev Vulkan1 --kl-divergence-base /mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/perflab-kld-ref/qwen27-q6k.kld --kl-divergence > results/phase2/w3/q4km-vs-q6kref.log 2>&1
grep -E '^Mean PPL\(Q\)|^Mean +KLD|^Same top|^99.0%' results/phase2/w3/q4km-vs-q6kref.log
echo "== Q4COST_DONE"

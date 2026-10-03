#!/usr/bin/env bash
cd /home/bbuckham/git/perf-lab
E="MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1"
F="-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4 --mmproj /mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth/mmproj-F16.gguf"
for arm in i16 w1; do
  B=/home/bbuckham/git/perf-lab/runners/llama.cpp/qwen3.8-r9700-next/build/bin; [ $arm = w1 ] && B=/home/bbuckham/git/perf-lab/runners/llama.cpp/qwen3.8-r9700-w1/build/bin
  harness/gpu_lock.sh bash -c "PERFLAB_LOG=/home/bbuckham/git/perf-lab/results/phase2/w1z/pool-$arm.log PERFLAB_BIN=$B PERFLAB_ENV='$E' PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh /mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth/Qwen3.8-27B-Q6_K.gguf 8097 262144 -ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4 --mmproj /mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth/mmproj-F16.gguf >/dev/null; python3 setups/qwen3.8-27b-r9700/autoresearch/pool.py 8097 /home/bbuckham/git/perf-lab/results/phase2/w1z/pool-$arm.json > /home/bbuckham/git/perf-lab/results/phase2/w1z/pool-$arm.out 2>&1; systemctl --user stop perflab-srv-8097; sleep 3"
done
harness/gpu_lock.sh bash -c "PERFLAB_ENV='MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1' setups/qwen3.8-27b-r9700/results/d0/scripts/acc2.sh /home/bbuckham/git/perf-lab/runners/llama.cpp/qwen3.8-r9700-w1/build/bin w1z-corpus -ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none" > /home/bbuckham/git/perf-lab/results/phase2/w1z/acc.out 2>&1
echo VERIFY_DONE >> /home/bbuckham/git/perf-lab/results/phase2/w1z/acc.out

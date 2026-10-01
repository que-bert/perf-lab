#!/usr/bin/env bash
cd /home/bbuckham/git/perf-lab
for p in "/home/bbuckham/git/llama.cpp-r9700-ar-fac/build/bin fac" "/home/bbuckham/git/perf-lab/runners/llama.cpp/qwen3.8-r9700/build/bin base" "/home/bbuckham/git/llama.cpp-r9700-ar-fac/build/bin fac2" "/home/bbuckham/git/perf-lab/runners/llama.cpp/qwen3.8-r9700/build/bin base2"; do set -- $p
 harness/gpu_lock.sh env PERFLAB_ENV='MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1' DEPTHS=1600,4000 REPS=1 NPRED=128 PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 SPEC_NMAX=4 DEPTH_LOG_DIR=setups/qwen3.8-27b-r9700/results/d0/e2e harness/depth.sh $1 q-fac-$2 8100 -ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none
done

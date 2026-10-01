#!/usr/bin/env bash
# df_e2e.sh: depth.sh base vs GEMV_MULTI at ~70k (ABAB), serving config
cd /home/bbuckham/git/perf-lab
N=/home/bbuckham/git/llama.cpp-r9700-ar-df/build/bin; O=/home/bbuckham/git/perf-lab/runners/llama.cpp/qwen3.8-r9700/build/bin
run() { # label bindir extra_env
  harness/gpu_lock.sh env PERFLAB_ENV="MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1$3" DEPTHS=1600 REPS=3 NPRED=256 PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 SPEC_NMAX=4 DEPTH_LOG_DIR=setups/qwen3.8-27b-r9700/results/d0/e2e harness/depth.sh $2 q-df-$1 8101 -ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none
  echo "load $(cut -d' ' -f1 /proc/loadavg)"
}
run base1 $O ""; run on1 $N ";GGML_VK_GEMV_MULTI=1"; run base2 $O ""; run on2 $N ";GGML_VK_GEMV_MULTI=1"
echo DONE

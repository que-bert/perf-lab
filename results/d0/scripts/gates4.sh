#!/usr/bin/env bash
# gates.sh <worktree> <label> : tests, PPL, decode KLD, acceptance (corpus + chat). Each GPU step takes the lock.
# GATES_CLI: extra args for PPL (not KLD: llama-decode-kld has no -lm) and the acceptance servers (e.g. -lm none); server env via PERFLAB_ENV.
set -u
S=/home/bbuckham/git/perf-lab/results/d0/scripts
P=/home/bbuckham/git/perf-lab; W=$1; LB=$2; B=$W/build/bin
M=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth/Qwen3.8-27B-Q6_K.gguf
L=$P/harness/gpu_lock.sh
echo "== git $(git -C $W log --oneline -1)"
for op in MUL_MAT MUL_MAT_VEC_FUSION RMS_NORM RMS_NORM_MUL_ADD SCALE RMS_NORM_SCALE CPY CONCAT GET_ROWS ADD MUL GATED_DELTA_NET GATED_DELTA_NET_CACHE_FUSION GDN_RECURRENT_CACHE RMS_NORM_MUL_SILU_MUL SSM_CONV SIGMOID SOFTPLUS SILU L2_NORM; do
  echo "== test $op: $($L $B/test-backend-ops test -b Vulkan1 -o $op 2>&1 | grep -E 'tests passed|FAIL' | tail -2 | tr '\n' ' ')"; done
echo "== test FA: $($L $B/test-backend-ops test -b Vulkan1 -o FLASH_ATTN_EXT -p hsk=256, 2>&1 | grep -E 'tests passed|FAIL' | tail -2 | tr '\n' ' ')"
cd $P
echo "== PPL: $($L $B/llama-perplexity -m $M -f harness/corpus/decode_kld.txt -c 4096 --chunks 8 -fa on -ctk q8_0 -ctv q8_0 -ngl 99 -dev Vulkan1 ${GATES_CLI:-} 2>&1 | grep -oE 'PPL = [0-9.]+ \+/- [0-9.]+')"
for d in 70000 176000; do for b in 4 1; do
  echo "== KLD d$d b$b: $($L $B/llama-decode-kld -m $M -ctk q8_0 -ctv q8_0 -c 262144 -fa on -ngl 99 -dev Vulkan1 --file harness/corpus/decode_kld.txt --depth $d --score 512 --batch $b --base-logits results/p0/kld-base/base-d$d-b$b.dkld 2>&1 | grep -E 'mean_kld|top1_flip' | tr -s ' ' | tr '\n' ' ')"
done; done
Q="-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive ${GATES_CLI:-}"
$L $S/acc2.sh $B $LB-corpus $Q
ACC_PY=$S/chat_acc.py $L $S/acc2.sh $B $LB-chat $Q
echo "== GATES DONE $(date)"

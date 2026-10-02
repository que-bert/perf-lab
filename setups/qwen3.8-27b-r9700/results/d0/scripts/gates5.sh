#!/usr/bin/env bash
# gates5.sh <worktree> <label> : gates4 with parts and fewer lock/load cycles. Same lines, same statistics.
#   GATES_PARTS="ops ppl kld acc"  (default all; gate_plan.py chooses)    OPS=full|MUL_MAT,ADD,...  (default full)
# ops: ONE test-backend-ops process and one lock for the whole op list (-o takes a comma list) instead of 19+1 lock/launch cycles.
# acc: ONE acceptance server load for corpus + chat (acc2.sh loaded the 27B twice). Output labels unchanged: ACC <label>-corpus / -chat.
# GATES_CLI: extra args for PPL (not KLD: llama-decode-kld has no -lm) and the acceptance servers (e.g. -lm none); server env via PERFLAB_ENV.
set -u
S=/home/bbuckham/git/perf-lab/setups/qwen3.8-27b-r9700/results/d0/scripts
P=/home/bbuckham/git/perf-lab; W=$1; LB=$2; B=$W/build/bin
MD=${PERFLAB_MODEL_DIR:-/home/bbuckham/models}
M=${QMODEL:-$MD/Qwen3.8-27B-Q6_K.gguf}
L=$P/harness/gpu_lock.sh
GP=" ${GATES_PARTS:-ops ppl kld acc} "; OPS=${OPS:-full}
VK=${PERFLAB_VKDEV:-Vulkan1}; PORT=$((8097 + ${PERFLAB_PORT_OFF:-0}))
echo "== git $(git -C $W log --oneline -1)"; echo "== gates parts:$GP ops=$OPS"
FULLOPS=MUL_MAT,MUL_MAT_VEC_FUSION,RMS_NORM,RMS_NORM_MUL_ADD,SCALE,RMS_NORM_SCALE,CPY,CONCAT,GET_ROWS,ADD,MUL,GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION,GDN_RECURRENT_CACHE,RMS_NORM_MUL_SILU_MUL,SSM_CONV,SIGMOID,SOFTPLUS,SILU,L2_NORM
if [[ $GP == *" ops "* ]]; then
  [ "$OPS" = full ] && OL=$FULLOPS || OL=$OPS
  echo "== test ops($(echo $OL | tr ',' '\n' | wc -l)): $($L $B/test-backend-ops test -b $VK -o $OL 2>&1 | grep -E 'tests passed|FAIL' | tail -3 | tr '\n' ' ')"
  if [ "$OPS" = full ] || [[ ",$OPS," == *,FLASH_ATTN_EXT,* ]]; then
    echo "== test FA: $($L $B/test-backend-ops test -b $VK -o FLASH_ATTN_EXT -p hsk=256, 2>&1 | grep -E 'tests passed|FAIL' | tail -2 | tr '\n' ' ')"
  fi
fi
cd $P
[[ $GP == *" ppl "* ]] && echo "== PPL: $($L $B/llama-perplexity -m $M -f harness/corpus/decode_kld.txt -c 4096 --chunks 8 -fa on -ctk q8_0 -ctv q8_0 -ngl 99 -dev $VK ${GATES_CLI:-} 2>&1 | grep -oE 'PPL = [0-9.]+ \+/- [0-9.]+')"
if [[ $GP == *" kld "* ]]; then for d in 70000 176000; do for b in 4 1; do
  echo "== KLD d$d b$b: $($L $B/llama-decode-kld -m $M -ctk q8_0 -ctv q8_0 -c 262144 -fa on -ngl 99 -dev $VK --file harness/corpus/decode_kld.txt --depth $d --score 512 --batch $b --base-logits setups/qwen3.8-27b-r9700/results/p0/kld-base/base-d$d-b$b.dkld 2>&1 | grep -E 'mean_kld|top1_flip' | tr -s ' ' | tr '\n' ' ')"
done; done; fi
if [[ $GP == *" acc "* ]]; then
  Q="-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive ${GATES_CLI:-}"
  $L bash -c "PERFLAB_BIN=$B PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh $M $PORT 262144 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 3 --mmproj $MD/mmproj-F16.gguf $Q >/dev/null || exit 1
    for k in corpus chat; do
      py=$S/multi.py; [ \$k = chat ] && py=$S/chat_acc.py
      echo \"== ACC $LB-\$k load=\$(cut -d' ' -f1-3 /proc/loadavg) \$(python3 \$py $PORT $S/acc-$LB-\$k.json | grep POOLED)\"
    done
    systemctl --user stop perflab-srv-$PORT; sleep 3"
fi
echo "== GATES DONE $(date)"

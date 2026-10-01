#!/usr/bin/env bash
# df_md5.sh <bindir> <label> "<ENV;ENV>"  -> md5 of greedy /completion over 3 prompts (one >= 6000 tokens), serving config, nmax 4
set -u
cd /home/bbuckham/git/perf-lab
B=$1; LB=$2; EV=$3
M=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth/Qwen3.8-27B-Q6_K.gguf
O=setups/qwen3.8-27b-r9700/results/d0/df; mkdir -p $O
python3 - <<'PY'
import json,random
random.seed(1)
w="alpha river stone cloud engine lantern garden orbit violin copper meadow signal harbor quartz timber velvet anchor ember falcon glacier".split()
long=" ".join(random.choice(w)+("." if i%11==0 else "") for i in range(5200))+"\n\nSummarize the passage above in three sentences."
ps=["Write a long essay about the history of computing.","Explain how a transformer language model works, step by step, with a small numeric example.",long]
for i,p in enumerate(ps): json.dump({"prompt":p,"n_predict":128,"temperature":0},open(f"/home/bbuckham/git/perf-lab/setups/qwen3.8-27b-r9700/results/d0/df/prompt{i}.json","w"))
PY
harness/gpu_lock.sh bash -c "PERFLAB_BIN=$B PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 PERFLAB_LOG=$PWD/$O/$LB.srv.log PERFLAB_ENV='$EV' harness/serve_unit.sh $M 8104 16384 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none >/dev/null
  setups/qwen3.8-27b-r9700/results/d0/scripts/df_md5_client.sh $LB
  systemctl --user stop perflab-srv-8104; sleep 2"
cat $O/$LB.md5

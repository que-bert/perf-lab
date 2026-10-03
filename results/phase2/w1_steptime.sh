#!/usr/bin/env bash
# W1: per-step host profile, integrate16 vs RB, Qwen3.8-27B serving config, 16k and 70k prompts, ABAB x2.
# Server stderr with LLAMA_SERVER_STEP_TIMING + LLAMA_DECODE_TIMING goes to results/phase2/w1/<arm>-<rep>.log.
set -u
P=/home/bbuckham/git/perf-lab; cd $P
R=$P/runners/llama.cpp; MNT=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth
O=$P/results/phase2/w1; mkdir -p $O
E="MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1;LLAMA_SERVER_STEP_TIMING=1;LLAMA_SERVER_STEP_TIMING_N=100;LLAMA_DECODE_TIMING=1;LLAMA_DECODE_TIMING_N=400"
F="-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4 --mmproj $MNT/mmproj-F16.gguf"
for rep in 1 2; do for arm in i16 rb; do
  B=$R/qwen3.8-r9700-next/build/bin; [ $arm = rb ] && B=$R/qwen3.8-r9700-rb/build/bin
  harness/gpu_lock.sh bash -c "PERFLAB_LOG=$O/$arm-$rep.log PERFLAB_BIN=$B PERFLAB_ENV='$E' PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh $MNT/Qwen3.8-27B-Q6_K.gguf 8097 262144 $F >/dev/null
    python3 - <<'PY' >> $O/$arm-$rep.req
import json, urllib.request
data = open('$P/harness/corpus/decode_kld.txt', 'rb').read().decode('utf-8', 'replace')
for name, nch in (('16k', 64000), ('70k', 280000)):
    for i in range(2):
        p = data[i * 7000: i * 7000 + nch]
        b = json.dumps({'prompt': p, 'n_predict': 512, 'temperature': 0, 'cache_prompt': False}).encode()
        d = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8097/completion', data=b, headers={'Content-Type': 'application/json'}), timeout=3600))
        t = d['timings']
        print(name, i, t['prompt_n'], round(t['prompt_per_second'], 1), round(t['predicted_per_second'], 2), t.get('draft_n_accepted'), t.get('draft_n'), flush=True)
PY
    systemctl --user stop perflab-srv-8097; sleep 3"
done; done
echo "== W1_STEPTIME_DONE $(date)"

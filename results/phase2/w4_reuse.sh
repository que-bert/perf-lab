#!/usr/bin/env bash
# W4: prompt/KV reuse with MTP on (preset path, i18): same ~32.5k prompt twice with cache_prompt=true; the second request
# must skip prefill (small prompt_n) and give identical text; then a different suffix reuses the shared prefix.
cd /home/bbuckham/git/perf-lab; B=runners/llama.cpp/qwen3.8-r9700-i18/build/bin; MNT=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth
P=setups/qwen3.8-27b-r9700/results/d0/i8/mtpp_prompt.json
harness/gpu_lock.sh bash -c "systemd-run --user --unit=perflab-srv-8094 --collect --setenv=LD_LIBRARY_PATH=$B --setenv=GGML_VK_VISIBLE_DEVICES=1 --setenv=MTMD_LAZY_GPU=1 --setenv=GGML_VK_HOST_GET_ROWS=1 $B/llama-server -m $MNT/Qwen3.8-27B-Q6_K.gguf --port 8094 --host 127.0.0.1 --no-webui -ngl 99 -lm none >/dev/null
  for i in \$(seq 1 120); do curl -sf localhost:8094/health >/dev/null && break; sleep 5; done
  python3 - <<'PY'
import json, urllib.request
body = json.load(open('$P')); body.update({'cache_prompt': True, 'n_predict': 128, 'temperature': 0})
def go(b):
    d = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8094/completion', data=json.dumps(b).encode(), headers={'Content-Type': 'application/json'}), timeout=3600))
    t = d['timings']; return d['content'], t['prompt_n'], round(t['prompt_per_second'], 1), round(t['predicted_per_second'], 2), t.get('draft_n_accepted'), t.get('draft_n')
a = go(body); b = go(body)
body2 = dict(body); body2['prompt'] = body['prompt'] + ' Summarize the above in one sentence.'
c = go(body2)
print('first ', a[1:]); print('second', b[1:], 'same_text', a[0] == b[0]); print('suffix', c[1:])
PY
  systemctl --user stop perflab-srv-8094"
echo "== W4_REUSE_DONE"

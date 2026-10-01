#!/usr/bin/env bash
# VRAM/GTT peak during image requests on the serving build (run under harness/gpu_lock.sh).
# Fills ~176k of context first (depth worst case), then sends imgtest-style image requests while sampling every 0.2 s.
set -u
cd /home/bbuckham/git/perf-lab
B=${1:?bindir}; XENV=${2:-}; LBL=${3:-fac}; MNT=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth
D=/sys/class/drm/card0/device; O=results/d0/e2e/vram-img-peak-$LBL.samples; : > $O
harness/gpu_guard.sh 900 || exit 1
PERFLAB_BIN=$B PERFLAB_ENV="MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1${XENV:+;$XENV}" PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh $MNT/Qwen3.8-27B-Q6_K.gguf 8097 262144 \
  --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none --mmproj $MNT/mmproj-F16.gguf || exit 1
( while true; do echo "$(date +%s.%N | cut -c1-14) $(( $(cat $D/mem_info_vram_used)/1048576 )) $(( $(cat $D/mem_info_gtt_used)/1048576 ))"; sleep 0.2; done ) >> $O &
SP=$!
python3 - <<'PY'
import base64, json, time, urllib.request
def post(body):
    r = urllib.request.Request("http://127.0.0.1:8097/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(r, timeout=1800)); return time.time() - t, d
img = base64.b64encode(open("/usr/share/backgrounds/jdituicha-raccoon1-dark.jpg", "rb").read()).decode()
long = open("harness/corpus/decode_kld.txt").read()
while len(long) < 700000: long += long
long = long[:700000]   # ~176k tokens
imgpart = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}
print("mark idle-image", time.time(), flush=True)
w, d = post({"messages": [{"role": "user", "content": [{"type": "text", "text": "Describe this image."}, imgpart]}], "max_tokens": 64, "temperature": 0})
print(f"idle image wall={w:.1f}s prompt_n={d['timings']['prompt_n']} :: {(d['choices'][0]['message'].get('content') or d['choices'][0]['message'].get('reasoning_content') or '')[:80]!r}", flush=True)
print("mark deep-image", time.time(), flush=True)
w, d = post({"messages": [{"role": "user", "content": [{"type": "text", "text": long + "\n\nNow describe this image."}, imgpart]}], "max_tokens": 64, "temperature": 0})
print(f"deep image wall={w:.1f}s prompt_n={d['timings']['prompt_n']} :: {(d['choices'][0]['message'].get('content') or d['choices'][0]['message'].get('reasoning_content') or '')[:80]!r}", flush=True)
PY
kill $SP; systemctl --user stop perflab-srv-8097; sleep 3
sort -k2 -n $O | tail -1 | awk '{print "PEAK vram_mib="$2" gtt_mib="$3" of 32624"}'

#!/usr/bin/env bash
# Image-request latency + VRAM at the serving config (ctx 262144, MTP n_max 4, mmproj on).
#   imgtest.sh <bindir> <label> [extra llama-server args...]      (run under harness/gpu_lock.sh)
# Env: PERFLAB_ENV passes server env (KEY=VAL;...). Sends the chat_acc.py image twice, then
# one text request, and prints wall/prompt time, reply head and VRAM idle/after.
set -u
BIN=$1; LB=$2; shift 2
cd /home/bbuckham/git/perf-lab
MNT="${PERFLAB_MODEL_DIR:-/home/bbuckham/models}"
PORT=8097; D=/sys/class/drm/card0/device
mib() { echo $(( $(cat $D/mem_info_vram_used)/1048576 )) gtt $(( $(cat $D/mem_info_gtt_used)/1048576 )); }
echo "== $LB bin=$BIN env=${PERFLAB_ENV:-} args=$*"
harness/gpu_guard.sh 900 || exit 1
PERFLAB_BIN=$BIN PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh "$MNT/Qwen3.8-27B-Q6_K.gguf" $PORT 262144 \
  --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0 \
  --spec-draft-vocab 98304 --spec-draft-vocab-adaptive --mmproj "$MNT/mmproj-F16.gguf" "$@" || exit 1
echo "vram idle: $(mib)"
python3 - $PORT <<'PY'
import base64, json, sys, time, urllib.request
port = sys.argv[1]
img = base64.b64encode(open("/usr/share/backgrounds/jdituicha-raccoon1-dark.jpg", "rb").read()).decode()
def req(content):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 32, "temperature": 0, "cache_prompt": False}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(r, timeout=600)); w = time.time() - t
    tm = d.get("timings", {})
    return w, tm.get("prompt_n"), tm.get("prompt_ms"), d["choices"][0]["message"]["content"][:100]
imgmsg = [{"type": "text", "text": "Describe this image in detail."},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}]
for i in range(2):
    w, n, ms, txt = req(imgmsg)
    print(f"img{i} wall={w:.2f}s prompt_n={n} prompt_ms={ms:.0f} :: {txt!r}")
w, n, ms, txt = req("Name three primary colors.")
print(f"text wall={w:.2f}s prompt_n={n} :: {txt!r}")
PY
echo "vram after: $(mib)"
systemctl --user stop perflab-srv-$PORT 2>/dev/null; sleep 3

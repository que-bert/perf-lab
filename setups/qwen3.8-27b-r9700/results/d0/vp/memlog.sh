#!/usr/bin/env bash
cd /home/bbuckham/git/perf-lab
MNT="/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth"
harness/gpu_guard.sh 900 || exit 1
PERFLAB_LOG=$PWD/setups/qwen3.8-27b-r9700/results/d0/vp/memlog.txt PERFLAB_BIN=${BIN:-/home/bbuckham/git/llama.cpp-r9700-ar-vp/build/bin} PERFLAB_ENV='MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1;GGML_VK_MEMORY_LOGGER=1' PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh $MNT/Qwen3.8-27B-Q6_K.gguf 8098 262144 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none --mmproj $MNT/mmproj-F16.gguf -lv 4 || exit 1
echo "idle vram $(( $(cat /sys/class/drm/card0/device/mem_info_vram_used)/1048576 ))"
python3 - <<'PY'
import json,urllib.request
t=open("harness/corpus/decode_kld.txt").read()[:30000]
r=urllib.request.Request("http://127.0.0.1:8098/completion",json.dumps({"prompt":t,"n_predict":32,"temperature":0}).encode(),{"Content-Type":"application/json"})
d=json.load(urllib.request.urlopen(r,timeout=900));print(d["timings"]["prompt_n"])
PY
echo "after vram $(( $(cat /sys/class/drm/card0/device/mem_info_vram_used)/1048576 ))"
systemctl --user stop perflab-srv-8098

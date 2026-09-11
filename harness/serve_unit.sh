#!/usr/bin/env bash
# Start any model as a transient systemd --user unit and wait until it serves.
#
#   serve_unit.sh <model-path> <port> <ctx> [extra llama-server args...]
#
# A model load cannot live inside a supervised task on this box: the supervisor
# watches how fast free memory falls as page cache fills and kills the task,
# even with tens of GB free and nothing starved. systemd-run --user puts the
# server under the user manager instead of the caller's process tree, which is
# the only thing measured to survive.
#
# Environment, all optional:
#   PERFLAB_BIN   llama.cpp build directory      (default b10472-vulkan)
#   PERFLAB_GPU   Vulkan device index to pin     (default 1, the R9700)
#   PERFLAB_CTK   K cache type                   (default q8_0)
#   PERFLAB_CTV   V cache type                   (default q4_0)
#   PERFLAB_NGL   layers offloaded               (default 99)
#
# PERFLAB_GPU exists so two models can be served at once, one per card. The two
# cards are not interchangeable: GPU 1 (R9700) has 34.2 GB and GPU 0 (RX 9060
# XT) has 17.1 GB with ollama already holding ~2.7 GB of it.
set -u
MODEL=$1; PORT=$2; CTX=$3; shift 3
B="${PERFLAB_BIN:-$HOME/llama.cpp/b10472-vulkan}"
GPU="${PERFLAB_GPU:-1}"
CTK="${PERFLAB_CTK:-q8_0}"
CTV="${PERFLAB_CTV:-q4_0}"
NGL="${PERFLAB_NGL:-99}"
UNIT="perflab-srv-$PORT"

if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
  systemctl --user stop "$UNIT" 2>/dev/null; timeout 6 tail -f /dev/null
fi
systemctl --user reset-failed "$UNIT" 2>/dev/null
systemctl --user stop "$UNIT" 2>/dev/null

systemd-run --user --unit="$UNIT" --collect \
  --setenv=LD_LIBRARY_PATH="$B" \
  --setenv=GGML_VK_VISIBLE_DEVICES="$GPU" \
  "$B/llama-server" -m "$MODEL" \
  --port "$PORT" --host 127.0.0.1 --no-webui \
  -c "$CTX" -ctk "$CTK" -ctv "$CTV" -fa on -ngl "$NGL" -t 8 "$@" >/dev/null \
  || { echo "systemd-run failed"; exit 1; }

for i in $(seq 1 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    echo "  SERVING $(basename "$MODEL") on $PORT gpu=$GPU ctx=$CTX ${CTK}/${CTV} after ~$((i*5))s"
    exit 0
  fi
  systemctl --user is-active --quiet "$UNIT" || {
    echo "  UNIT DIED"; journalctl --user -u "$UNIT" -n 30 --no-pager; exit 1; }
  timeout 5 tail -f /dev/null
done
echo "  TIMED OUT"; journalctl --user -u "$UNIT" -n 30 --no-pager; exit 1

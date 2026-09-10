#!/usr/bin/env bash
# Start any model as a transient systemd --user unit and wait until it serves.
#   unit_any.sh <model-path> <port> <ctx> [extra llama-server args...]
set -u
MODEL=$1; PORT=$2; CTX=$3; shift 3
B="$HOME/llama.cpp/b10472-vulkan"
UNIT="perflab-srv-$PORT"

if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
  systemctl --user stop "$UNIT" 2>/dev/null; timeout 6 tail -f /dev/null
fi
systemctl --user reset-failed "$UNIT" 2>/dev/null
systemctl --user stop "$UNIT" 2>/dev/null

systemd-run --user --unit="$UNIT" --collect \
  --setenv=LD_LIBRARY_PATH="$B" \
  --setenv=GGML_VK_VISIBLE_DEVICES=1 \
  "$B/llama-server" -m "$MODEL" \
  --port "$PORT" --host 127.0.0.1 --no-webui \
  -c "$CTX" -ctk q8_0 -ctv q4_0 -fa on -ngl 99 -t 8 "$@" >/dev/null \
  || { echo "systemd-run failed"; exit 1; }

for i in $(seq 1 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    echo "  SERVING $(basename "$MODEL") on $PORT after ~$((i*5))s"
    exit 0
  fi
  systemctl --user is-active --quiet "$UNIT" || {
    echo "  UNIT DIED"; journalctl --user -u "$UNIT" -n 20 --no-pager; exit 1; }
  timeout 5 tail -f /dev/null
done
echo "  TIMED OUT"; journalctl --user -u "$UNIT" -n 20 --no-pager; exit 1

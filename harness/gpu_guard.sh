#!/usr/bin/env bash
# Wait until the benchmark GPU is quiet, so a decode-at-depth number is not
# silently measuring a game, an embedding burst, or a sibling session's tests.
#
#   gpu_guard.sh [timeout_seconds]
#
# The 2026-09-21 screening pass read 25.8 t/s at 70k for a config whose clean
# baseline was 38.8 t/s, with a 5.7% rep spread -- a `go test` from a sibling
# session plus an active ollama embedding runner on the same card. A single-rep
# protocol would have recorded that as a build result. mem_busy_percent is the
# direct signal: it is memory-controller utilisation, which is exactly what
# single-stream decode consumes.
set -u
TIMEOUT="${1:-900}"
PCI="${PERFLAB_GPU_PCI:-0000:0c:00.0}"   # R9700 (Navi 48); card order is not stable
MEM_MAX="${PERFLAB_GUARD_MEM:-12}"       # % memory-controller busy
LOAD_MAX="${PERFLAB_GUARD_LOAD:-8}"      # 1-minute load average

CARD=""
for d in /sys/class/drm/card[0-9]/device; do
  case "$(readlink -f "$d")" in *"$PCI") CARD="$d";; esac
done
[ -n "$CARD" ] || { echo "  guard: no card for PCI $PCI, skipping"; exit 0; }
MEMF="$CARD/mem_busy_percent"

ok=0
for _ in $(seq 1 "$TIMEOUT"); do
  mem=$(cat "$MEMF" 2>/dev/null || echo 0)
  load=$(cut -d' ' -f1 /proc/loadavg)
  if [ "$mem" -le "$MEM_MAX" ] 2>/dev/null && awk "BEGIN{exit !($load < $LOAD_MAX)}"; then
    ok=$((ok+1)); [ "$ok" -ge 3 ] && { echo "  guard: clean (mem_busy=${mem}% load=${load})"; exit 0; }
  else
    ok=0
  fi
  sleep 1
done
echo "  guard: STILL BUSY after ${TIMEOUT}s (mem_busy=${mem}% load=${load}) -- refusing to measure"
exit 1
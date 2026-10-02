#!/usr/bin/env bash
# Serialize every measurement on one GPU, per card ("lane"), across the main session and sub-agents.
# Parallel agents may write and build code, but only one process may touch a card at a time; a
# concurrent benchmark silently corrupts both numbers.
#
#   [PERFLAB_CARD=r9700|9060] [PERFLAB_LANE_KIND=timing|correctness] gpu_lock.sh <command...>
#
# Cards (default r9700, the timing card). The lock file is /tmp/perflab-<card>.lock. The command runs with
#   PERFLAB_CARD, PERFLAB_GPU_PCI (gpu_holders.py), PERFLAB_GPU (serve_unit.sh / GGML_VK_VISIBLE_DEVICES
#   index), PERFLAB_VKDEV (llama-bench / perplexity -dev), PERFLAB_LANE_KIND, PERFLAB_PORT_OFF (0 / 100: keeps
#   the fixed server ports of the two lanes apart) exported, so model_profile / model_quality / zoo_smoke /
#   serve_unit target the locked card.
#   r9700  0000:0c:00.0  Vulkan1  renderD129  32 GB  timing card, lane kind defaults to timing
#   9060   0000:06:00.0  Vulkan0  renderD128  16 GB  desktop card (Xwayland, terminals, Steam hold ~1 GB), lane kind
#                                                    defaults to correctness
# Lane kinds: timing = any foreign DRM client / loadavg / undrained VRAM blocks the start and voids the run;
# correctness = desktop baseline clients (gpu_holders.py BASELINE) are ignored, foreign activity is logged to the
# ledger but never voids. CPU lanes: a timing run takes /tmp/perflab-cpu.lock exclusively, a correctness run
# takes it shared (flock -s), so a launch-bound timing run never overlaps CPU-heavy work on the other lane.
# PERFLAB_CPU_LOCK=0 skips it.
#
# After taking the locks it waits until the card is free (gpu_holders.py busy), then samples the card's DRM
# clients and the load for the whole run. A VOID run is re-run up to GPU_LOCK_RETRY times (default 2; measurements
# are idempotent): verdict to stderr and results/gpu_ledger.log. Still VOID -> exit 76.
# GPU_LOCK_WAIT=0 refuses instead of waiting. Nested calls on a card already held (GPU_LOCK_HELD lists the held
# cards; the legacy value 1 means r9700) just run the command.
set -u
H=$(dirname "$(readlink -f "$0")")
CARD=${PERFLAB_CARD:-r9700}
case $CARD in
  r9700) PCI=0000:0c:00.0; GIDX=1; KIND_DEF=timing;      POFF=0;   IDLE=${PERFLAB_IDLE_VRAM_MIB:-1024} ;;
  9060)  PCI=0000:06:00.0; GIDX=0; KIND_DEF=correctness; POFF=100; IDLE=${PERFLAB_IDLE_VRAM_MIB:-3072} ;;
  *) echo "gpu_lock: unknown PERFLAB_CARD=$CARD (r9700|9060)" >&2; exit 2 ;;
esac
KIND=${PERFLAB_LANE_KIND:-$KIND_DEF}
case $KIND in timing|correctness) ;; *) echo "gpu_lock: PERFLAB_LANE_KIND must be timing|correctness" >&2; exit 2 ;; esac
export PERFLAB_CARD=$CARD PERFLAB_GPU_PCI=$PCI PERFLAB_GPU=$GIDX PERFLAB_VKDEV=Vulkan$GIDX PERFLAB_LANE_KIND=$KIND \
       PERFLAB_PORT_OFF=$POFF PERFLAB_IDLE_VRAM_MIB=$IDLE
held=",${GPU_LOCK_HELD:-},"; [ "$CARD" = r9700 ] && held=${held//,1,/,r9700,}
if [ -z "${GPU_LOCK_ENTER:-}" ]; then
  case $held in *",$CARD,"*) exec "$@" ;; esac
  exec env GPU_LOCK_ENTER=1 GPU_LOCK_HELD="${GPU_LOCK_HELD:+$GPU_LOCK_HELD,}$CARD" flock "/tmp/perflab-$CARD.lock" "$0" "$@"
fi
unset GPU_LOCK_ENTER
if [ "${PERFLAB_CPU_LOCK:-1}" != 0 ]; then
  exec 9>/tmp/perflab-cpu.lock
  fl=-x; [ "$KIND" = correctness ] && fl=-s
  if ! flock -n $fl 9; then
    [ "${GPU_LOCK_WAIT:-1}" = 0 ] && { echo "gpu_lock: refusing, cpu lock held by the other lane" >&2; exit 75; }
    echo "gpu_lock: waiting for the cpu lock ($fl; other lane is running)" >&2
    flock $fl 9
  fi
fi
try=0
while :; do
  said=0
  until msg=$(python3 "$H/gpu_holders.py" busy); do
    [ "${GPU_LOCK_WAIT:-1}" = 0 ] && { echo "gpu_lock: refusing, card in use: $msg" >&2; exit 75; }
    [ $said = 0 ] && { echo "gpu_lock: waiting for the card to be freed ($msg)" >&2; said=1; }
    sleep 30
  done
  S=$(mktemp /tmp/perflab-gpusample.XXXXXX)
  python3 "$H/gpu_holders.py" sample "$S" --allow-tree $$ --iv 2 &
  SP=$!
  "$@"
  rc=$?
  kill $SP 2>/dev/null; wait $SP 2>/dev/null
  v=$(python3 "$H/gpu_holders.py" verdict "$S")
  rm -f "$S"
  echo "gpu_lock: $v" >&2
  printf '%s rc=%s try=%s card=%s kind=%s %s :: %s\n' "$(date -Iseconds)" "$rc" "$try" "$CARD" "$KIND" "$v" "$*" | cut -c1-400 >> "$H/../results/gpu_ledger.log"
  case "$v" in VOID*) ;; *) exit $rc;; esac
  try=$((try+1)); [ $try -gt "${GPU_LOCK_RETRY:-2}" ] && exit 76
  echo "gpu_lock: re-running after VOID ($try)" >&2
done

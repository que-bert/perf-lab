#!/usr/bin/env bash
# Serialize every R9700 measurement across the main session and sub-agents.
# Parallel agents may write and build code, but only one process may touch the
# GPU at a time; a concurrent benchmark silently corrupts both numbers.
#
#   gpu_lock.sh <command...>      # blocks until the lock is free, then runs
#
# After taking the lock it waits until no foreign process holds the card (a game, ollama, a
# sibling's server: gpu_holders.py busy) and the CPU is quiet, then samples the card's DRM clients
# and the load for the whole run (not only at start). A run where anything else used the card is
# VOID: the verdict goes to stderr and to results/gpu_ledger.log, and the command is re-run up to
# GPU_LOCK_RETRY times (default 2; measurements are idempotent). Still VOID -> exit 76.
# GPU_LOCK_WAIT=0 refuses instead of waiting. Nested calls (GPU_LOCK_HELD set) do not re-lock.
set -u
H=$(dirname "$(readlink -f "$0")")
if [ -z "${GPU_LOCK_HELD:-}" ]; then
  exec env GPU_LOCK_HELD=1 flock /tmp/perflab-r9700.lock "$0" "$@"
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
  printf '%s rc=%s try=%s %s :: %s\n' "$(date -Iseconds)" "$rc" "$try" "$v" "$*" | cut -c1-400 >> "$H/../results/gpu_ledger.log"
  case "$v" in VOID*) ;; *) exit $rc;; esac
  try=$((try+1)); [ $try -gt "${GPU_LOCK_RETRY:-2}" ] && exit 76
  echo "gpu_lock: re-running after VOID ($try)" >&2
done

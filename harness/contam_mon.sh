#!/usr/bin/env bash
# Log what else is running while a measurement runs, so a bad number can be
# attributed instead of guessed at. gpu_guard.sh only checks at start.
#
#   contam_mon.sh <outfile> [interval_s]     # run in background; kill when done
set -u
OUT=$1; IV=${2:-5}
PCI="${PERFLAB_GPU_PCI:-0000:0c:00.0}"
while :; do
  printf '%s load=%s r9700_gpu=%s r9700_mem=%s sclk=%s top=%s\n' \
    "$(date +%T)" "$(cut -d' ' -f1 /proc/loadavg)" \
    "$(cat /sys/bus/pci/devices/$PCI/gpu_busy_percent)" \
    "$(cat /sys/bus/pci/devices/$PCI/mem_busy_percent)" \
    "$(grep '\*' /sys/bus/pci/devices/$PCI/pp_dpm_sclk 2>/dev/null | awk '{print $2}')" \
    "$(ps -eo pcpu,comm --sort=-pcpu --no-headers | head -4 | awk '{printf "%s:%s,",$2,$1}')" >> "$OUT"
  sleep "$IV"
done

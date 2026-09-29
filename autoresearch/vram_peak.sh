#!/usr/bin/env bash
# VRAM peak for a serving candidate: 176k prefill (depth.py DEPTHS=4000) then the chat set (includes an mmproj image).
#   vram_peak.sh <bindir> <label> [extra llama-server args...]   (takes the GPU lock itself)
set -u
P=/home/bbuckham/git/perf-lab; cd $P
BIN=$1; LB=$2; shift 2
MNT=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth; M=$MNT/Qwen3.8-27B-Q6_K.gguf
CARD=$(for c in /sys/class/drm/card[0-9]*/device; do [ "$(cat $c/mem_info_vram_total 2>/dev/null)" -gt 30000000000 ] 2>/dev/null && echo $c && break; done)
OUT=results/d0/vram-$LB-$(date +%Y%m%dT%H%M%S).log
harness/gpu_lock.sh bash -c "
  ( while true; do echo \"\$(date +%T) vram=\$(( \$(cat $CARD/mem_info_vram_used) / 1048576 ))MiB\"; sleep 1; done ) > $OUT & MON=\$!
  PERFLAB_BIN=$BIN PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh $M 8097 262144 --parallel 1 --spec-type draft-mtp --mmproj $MNT/mmproj-F16.gguf $* >/dev/null
  python3 harness/depth.py 8097 --depths 4000 --reps 1 --npred 64 >/dev/null
  echo \"\$(date +%T) MARK image\" >> $OUT
  python3 autoresearch/pool.py 8097 /dev/null --set chat | grep POOL
  systemctl --user stop perflab-srv-8097; sleep 3; kill \$MON"
echo "card=$CARD total=$(( $(cat $CARD/mem_info_vram_total) / 1048576 ))MiB peak=$(grep -oE '[0-9]+MiB' $OUT | sort -n | tail -1) log=$OUT"

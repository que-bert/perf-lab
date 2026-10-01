#!/usr/bin/env bash
# main session: quiet-window A/B integrate2 vs integrate4 + D1b held-out check
set -u
S=/tmp/claude-1000/-home-bbuckham-git-perf-lab/b29d0a53-3607-41fd-86df-7e270e1c4360/scratchpad
P=/home/bbuckham/git/perf-lab; cd $P
I2=/home/bbuckham/git/llama.cpp-r9700-integrate2/build/bin
I4=/home/bbuckham/git/llama.cpp-r9700-integrate4/build/bin
MNT=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth; M=$MNT/Qwen3.8-27B-Q6_K.gguf
L=$P/harness/gpu_lock.sh
F4="-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive"
( while sleep 20; do echo "$(date +%T) vram=$(( $(cat /sys/bus/pci/devices/0000:0c:00.0/mem_info_vram_used)/1048576 ))MiB load=$(cut -d' ' -f1 /proc/loadavg)"; done ) > $S/main/vram-ab4.log &
VP=$!
export PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 DEPTHS=1600,4000 REPS=3 DEPTH_LOG_DIR=setups/qwen3.8-27b-r9700/results/d0/e2e
for x in "$I2 e2e-integ2 " "$I4 e2e-integ4 $F4" "$I4 e2e-integ4b $F4"; do set -- $x; b=$1; lb=$2; shift 2
  echo "== DEPTH $lb load=$(cut -d' ' -f1-3 /proc/loadavg)"; $L harness/depth.sh $b $lb 8097 "$@" 2>&1 | grep -E '~[0-9]+k +prompt|guard'; done
kill $VP
for d in "2048 0 3" "2048 16384 3" "512 131072 2"; do set -- $d
  for b in "$I2 integ2" "$I4 integ4"; do set -- $1 $2 $3 $b
    echo "== BENCH p$1 d$2 $5: $($L $4/llama-bench -m $M -dev Vulkan1 -fa 1 -ctk q8_0 -ctv q8_0 -p $1 -n 0 -d $2 -r $3 2>&1 | grep -E '\| *pp' | awk -F'|' '{print $(NF-1)}')"
  done; done
acc() { b=$1; lb=$2; py=$3; shift 3
  PERFLAB_BIN=$b harness/serve_unit.sh $M 8097 262144 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 3 --mmproj $MNT/mmproj-F16.gguf "$@" >/dev/null
  echo "== ACC $lb load=$(cut -d' ' -f1-3 /proc/loadavg) $(python3 $py 8097 $S/main/acc-$lb.json | grep POOLED)"; systemctl --user stop perflab-srv-8097; sleep 3; }
H=$P/setups/qwen3.8-27b-r9700/results/d0/scripts/heldout_acc.py
$L bash -c "$(declare -f acc); S=$S M=$M MNT=$MNT; acc $I4 ho-d1b $H $F4; acc $I4 ho-v0 $H -ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 0"
echo "== DONE $(date)"

#!/usr/bin/env bash
# acc.sh <bin> <label> [extra server args]  -- pooled acceptance, 16 corpus slices, n_max 3
set -u
S=/tmp/claude-1000/-home-bbuckham-git-perf-lab/1b4570a8-7144-4fd6-9ca9-d9b9b418654f/scratchpad
cd /home/bbuckham/git/perf-lab
B=$1; L=$2; shift 2
MNT=/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth
PERFLAB_BIN=$B PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh $MNT/Qwen3.8-27B-Q6_K.gguf 8097 262144 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 3 --mmproj $MNT/mmproj-F16.gguf "$@" >/dev/null
echo "== ACC $L load=$(cut -d' ' -f1-3 /proc/loadavg) $(python3 ${ACC_PY:-$S/multi.py} 8097 $S/acc-$L.json | grep POOLED)"
systemctl --user stop perflab-srv-8097; sleep 3

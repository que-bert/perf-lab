#!/usr/bin/env bash
# Raw speed profile of one zoo model on one build: llama-bench pp/tg at a set of depths.
#
#   model_profile.sh <bindir> <zoo-name|model.gguf> <label> [extra llama-bench args...]
#
# Env: DEPTHS (default 0,8192), PP (512), TG (128), REPS (5), PROFILE_DIR (results/phase2/profile)
# Output: <PROFILE_DIR>/<model>/<label>.jsonl (llama-bench jsonl, one line per test) and a one-line summary
# per test on stdout: "model label test depth t/s +- sd". Runs under gpu_lock.sh (waits for a free card,
# voids contaminated runs). Not the 27B serving arbiter: that is depth.sh/pool.py through llama-server
# with MTP. This is the like-for-like number for fork vs upstream on any model.
set -u
H=$(dirname "$(readlink -f "$0")"); P=$(dirname "$H"); . "$H/zoo.sh"
B=$1; M=$2; LB=$3; shift 3
MP=$M; [ -f "$MP" ] || MP=$(zoo_path "$M")
[ -f "$MP" ] || { echo "model_profile: no model $M" >&2; exit 2; }
NAME=$(basename "$M" .gguf)
OUT=${PROFILE_DIR:-$P/results/phase2/profile}/$NAME; mkdir -p "$OUT"
J=$OUT/$LB.jsonl
# sh -c so a VOID re-run inside gpu_lock.sh truncates the output instead of appending to it
"$H/gpu_lock.sh" sh -c 'o=$1; shift; exec "$@" > "$o"' _ "$J.tmp" "$B/llama-bench" -m "$MP" -dev "${PERFLAB_VKDEV:-Vulkan1}" -ngl 99 -fa 1 \
  -p "${PP:-512}" -n "${TG:-128}" -d "${DEPTHS:-0,8192}" -r "${REPS:-5}" -o jsonl "$@" 2> "$OUT/$LB.log"
rc=$?
grep -h 'gpu_lock:' "$OUT/$LB.log" >&2
[ $rc = 0 ] && mv "$J.tmp" "$J" || { echo "model_profile: $NAME $LB failed rc=$rc (see $OUT/$LB.log)" >&2; exit $rc; }
python3 - "$J" "$NAME" "$LB" <<'PY'
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    t = f"pp{r['n_prompt']}" if r['n_prompt'] else f"tg{r['n_gen']}"
    print(f"{sys.argv[2]} {sys.argv[3]} {t} d{r['n_depth']} {r['avg_ts']:.2f} +- {r['stddev_ts']:.2f}")
PY

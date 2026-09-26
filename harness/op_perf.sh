#!/usr/bin/env bash
# Op-level performance probe backed by `test-backend-ops perf` on the R9700.
#
# Prints one row per matched test case:
#     op=<case> us=<mean over reps> spread=<(max-min)/mean %> gbps=<bytes/us-derived>
#
# The binary already calibrates each case to a stable run count and reports
# us/run; this wrapper adds repetition (so a whole-suite run's spread is visible)
# and an optional DRAM byte model so bandwidth can be read directly.
#
#   op_perf.sh -o FLASH_ATTN_EXT -p 'hsk=256.*kv=183296.*nb=4' --bytes 398851584
#   op_perf.sh -o MUL_MAT -p 'type=q6_k.*ne=\[17408,3,5120' --bytes 267386880 -r 3
#
# Options:
#   -o <ops>          op_names passed through to test-backend-ops
#   -p <regex>        params_filter passed through
#   -b <backend>      Vulkan device (default Vulkan1 = R9700)
#   -r <reps>         whole-suite repetitions (default 3)
#   --bytes <N|none>  DRAM bytes per op run for the gbps column (default none)
#   --bin <path>      test-backend-ops binary (default: the fork build)
set -euo pipefail

BIN="${PERFLAB_OPBIN:-$HOME/git/llama.cpp-r9700/build/bin/test-backend-ops}"
OPS=""
PARAMS=""
BACKEND="Vulkan1"
REPS=3
BYTES="none"

while [ $# -gt 0 ]; do
  case "$1" in
    -o) OPS="$2"; shift 2 ;;
    -p) PARAMS="$2"; shift 2 ;;
    -b) BACKEND="$2"; shift 2 ;;
    -r) REPS="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -x "$BIN" ] || { echo "no binary at $BIN" >&2; exit 1; }

tmp=$(mktemp -d /tmp/op_perf.XXXXXX)
trap 'rm -rf "$tmp"' EXIT

for r in $(seq 1 "$REPS"); do
  "$BIN" perf -b "$BACKEND" ${OPS:+-o "$OPS"} ${PARAMS:+-p "$PARAMS"} \
    > "$tmp/rep$r.log" 2>&1
done

python3 - "$tmp" "$REPS" "$BYTES" <<'PY'
import re, sys, glob, os

tmp, reps, bytes_arg = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    nbytes = None if bytes_arg == "none" else float(bytes_arg)
except ValueError:
    nbytes = None

# "  FLASH_ATTN_EXT(...):   1311 runs -  775.95 us/run - ... "
line_re = re.compile(r"^\s*([A-Z0-9_]+)\((.*)\):\s+(\d+) runs -\s+([0-9.]+) us/run")

samples = {}
order = []
for path in sorted(glob.glob(os.path.join(tmp, "rep*.log")), key=lambda p: int(re.search(r'rep(\d+)', p).group(1))):
    with open(path) as f:
        for line in f:
            m = line_re.match(line)
            if not m:
                continue
            case = f"{m.group(1)}({m.group(2)})"
            us = float(m.group(4))
            if case not in samples:
                samples[case] = []
                order.append(case)
            samples[case].append(us)

for case in order:
    us = samples[case]
    mean = sum(us) / len(us)
    spread = (max(us) - min(us)) / mean * 100.0 if mean else 0.0
    gbps = "" if nbytes is None else f" gbps={nbytes/(mean*1000.0):.1f}"
    print(f"op={case} us={mean:.2f} spread={spread:.2f}%{gbps}")
PY

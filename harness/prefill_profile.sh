#!/usr/bin/env bash
# P0.3: per-op prefill profile of one ubatch at a given depth, via the Vulkan perf logger.
#
#   prefill_profile.sh <bin-dir> <depth> <ubatch>
#
# Runs llama-bench once (-r 1) at the fixed serving config on Vulkan1, keeps the LAST
# "Vulkan Timings" block from stderr (the final ubatch of the -p run, i.e. at depth),
# and prints the ops sorted by total ms with share % and GFLOPS, plus llama-bench t/s.
# Must be run under harness/gpu_lock.sh. The raw log is kept (path printed at the end).
#   MODEL=...  override the model;  KEEP_LOG=dir  where to keep the raw stderr.
set -euo pipefail
BIN="${1:?bin-dir}"; DEPTH="${2:?depth}"; UB="${3:?ubatch}"
MODEL="${MODEL:-/mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/qwen3.8:27b/unsloth/Qwen3.8-27B-Q6_K.gguf}"
LOGDIR="${KEEP_LOG:-${TMPDIR:-/tmp}}"
LOG="$LOGDIR/prefill_profile-d${DEPTH}-ub${UB}-$(date -u +%Y%m%dT%H%M%SZ).log"
OUT="$LOG.stdout"

GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 \
    "$BIN/llama-bench" -m "$MODEL" -dev Vulkan1 -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
    -p "$UB" -ub "$UB" -n 0 -r 1 -d "$DEPTH" -o md >"$OUT" 2>"$LOG"

python3 - "$LOG" "$OUT" <<'PY'
import re, sys
log, out = sys.argv[1], sys.argv[2]
text = open(log, errors="replace").read()
blocks = text.split("Vulkan Timings:")
if len(blocks) < 2:
    sys.exit("no 'Vulkan Timings' block in " + log)
last = blocks[-1]
line_re = re.compile(r"^(.*?): (\d+) x ([\d.]+) us = ([\d.]+) us(?: \(([\d.]+) GFLOPS/s\))?\s*$")
rows, total = [], None
for ln in last.splitlines():
    m = re.match(r"^Total time: ([\d.]+) us", ln)
    if m:
        total = float(m.group(1)); break
    m = line_re.match(ln)
    if m:
        rows.append((m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)),
                     float(m.group(5)) if m.group(5) else None))
if total is None:
    total = sum(r[3] for r in rows)
rows.sort(key=lambda r: -r[3])
print(f"{'op':<70} {'count':>6} {'us/call':>10} {'ms':>9} {'share':>7} {'TFLOPS':>7}")
for name, cnt, per, tot, gf in rows:
    tf = f"{gf/1000:.1f}" if gf else ""
    print(f"{name[:70]:<70} {cnt:>6} {per:>10.1f} {tot/1000:>9.2f} {100*tot/total:>6.1f}% {tf:>7}")
print(f"{'TOTAL (logged GPU time, last block)':<70} {'':>6} {'':>10} {total/1000:>9.2f} {100.0:>6.1f}%")
for ln in open(out):
    cols = [x.strip() for x in ln.strip().strip("|").split("|")]
    if len(cols) >= 2 and cols[-2].startswith("pp"):
        print(f"llama-bench: {cols[-2]}  {cols[-1]} t/s  (perf logger on: syncs per node, so t/s is below unlogged)")
PY
echo "raw log: $LOG"

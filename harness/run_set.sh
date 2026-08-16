#!/usr/bin/env bash
# Run every canary once as a batch, under one tag.
#
#   run_set.sh --kind nightly|apt|manual [--reps N] [--deadline HH:MM]
#              [--deadline-in SECONDS] [--config FILE]
#
# The tag is the point. run_id is unique per row, so without a shared tag a
# night's measurements are 15 unrelated rows and check.py cannot tell a run
# from a rep. Everything this script emits -- measurements, guard trips,
# deadline skips -- carries the same one.
#
# Exit 0 when the batch completed or was cleanly truncated by the deadline;
# 1 when a canary failed to measure for any other reason.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
LEDGER="${PERF_LAB_LEDGER:-$ROOT/results/ledger.jsonl}"
STAMP="${PERF_LAB_QUEUE:-/var/lib/perf-lab/queued}"

KIND=nightly; REPS=3; DEADLINE=""; DEADLINE_IN=""; IF_QUEUED=0
CFG="$ROOT/configs/canary.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind)        KIND="$2"; shift 2 ;;
    --reps)        REPS="$2"; shift 2 ;;
    --deadline)    DEADLINE="$2"; shift 2 ;;
    --deadline-in) DEADLINE_IN="$2"; shift 2 ;;
    --config)      CFG="$2"; shift 2 ;;
    --if-queued)   IF_QUEUED=1; shift ;;
    *) echo "run_set.sh: unknown argument $1" >&2; exit 64 ;;
  esac
done

# The drain timer wakes every 15 minutes and almost always has nothing to do.
# Checking before the environment checks keeps a quiet wake-up silent.
if (( IF_QUEUED )) && [[ ! -s "${PERF_LAB_QUEUE:-/var/lib/perf-lab/queued}" ]]; then
  exit 0
fi
case "$KIND" in
  nightly|apt|manual) ;;
  *) echo "run_set.sh: --kind must be nightly, apt or manual" >&2; exit 64 ;;
esac

: "${PERF_LAB_MODEL:?set PERF_LAB_MODEL}"
: "${PERF_LAB_BIN:?set PERF_LAB_BIN}"
: "${PERF_LAB_GPU_UID:?set PERF_LAB_GPU_UID}"
[[ -r "$CFG" ]] || { echo "run_set.sh: cannot read $CFG" >&2; exit 3; }

# --- deadline ---------------------------------------------------------------
# An unattended run gets a wall it cannot cross: a long-context canary that
# hangs must not still be holding the GPU when the machine is wanted back.
# HH:MM means the next occurrence, so an 03:00 nightly reads 07:00 as this
# morning while a 23:00 one reads it as tomorrow.
if [[ -n "$DEADLINE_IN" ]]; then
  DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_IN ))
else
  [[ -z "$DEADLINE" && "$KIND" != "manual" ]] && DEADLINE="07:00"
  if [[ -n "$DEADLINE" ]]; then
    DEADLINE_EPOCH=$(date -d "today $DEADLINE" +%s)
    (( DEADLINE_EPOCH <= $(date +%s) )) && DEADLINE_EPOCH=$(date -d "tomorrow $DEADLINE" +%s)
  else
    DEADLINE_EPOCH=0   # manual runs are unbounded unless asked otherwise
  fi
fi

TAG="$KIND-$(date -u +%Y%m%dT%H%M%SZ)"

# --- apt queue --------------------------------------------------------------
# Consume the stamp before measuring, not after: if this run dies the queue
# must not re-trigger forever, and the packages are already recorded here.
QUEUED=""
if [[ -r "$STAMP" ]]; then
  QUEUED="$(tr '\n' ' ' < "$STAMP" | sed 's/ *$//')"
  : > "$STAMP" || true
  echo "run_set.sh: consuming apt queue: $QUEUED" >&2
fi

mapfile -t KEYS < <(python3 -c "
import yaml,sys
print('\n'.join(yaml.safe_load(open('$CFG'))['canaries']))")
[[ ${#KEYS[@]} -gt 0 ]] || { echo "run_set.sh: no canaries in $CFG" >&2; exit 3; }

echo "run_set.sh: tag $TAG — ${#KEYS[@]} canaries x $REPS reps" >&2

emit_skip () {  # key, reason
  python3 "$HERE/emit_row.py" --config "$CFG" --key "$1" --kind skipped \
    --reason "$2" --tag "$TAG" --gpu-uid "$PERF_LAB_GPU_UID" \
    --model "$PERF_LAB_MODEL" --bin "$PERF_LAB_BIN" >> "$LEDGER"
}

FAILED=0; TRUNCATED=0
for key in "${KEYS[@]}"; do
  if (( DEADLINE_EPOCH > 0 )) && (( $(date +%s) >= DEADLINE_EPOCH )); then
    # Record the absence. A truncated night that says so is diagnosable; one
    # that is merely short is indistinguishable from a config being removed.
    emit_skip "$key" "deadline"
    TRUNCATED=1
    echo "run_set.sh: past deadline — skipping $key" >&2
    continue
  fi
  echo "--- $key ---" >&2
  if ! "$HERE/bench.sh" "$CFG" "$key" --reps "$REPS" --tag "$TAG" --kind "$KIND"; then
    echo "run_set.sh: $key failed to measure" >&2
    FAILED=1
  fi
done

# --- publish ----------------------------------------------------------------
# Rows are already on disk. A push that cannot land is an alerting condition,
# never a reason to throw away measurements that cost GPU time.
publish () {
  # A run against a scratch ledger is a rehearsal, not a measurement worth
  # publishing -- and `git add` on a path outside the repo just errors.
  case "$LEDGER" in
    "$ROOT"/*) ;;
    *) echo "run_set.sh: ledger outside repo, not publishing" >&2; return 0 ;;
  esac
  cd "$ROOT"
  git add "$LEDGER"
  git diff --cached --quiet && { echo "run_set.sh: no new rows to publish" >&2; return 0; }
  git commit -q -m "data: $TAG${QUEUED:+ (apt: $QUEUED)}"
  for attempt in 1 2; do
    if git pull --rebase -q && git push -q; then
      rm -f "$ROOT/.scratch/push-failed"
      return 0
    fi
    echo "run_set.sh: push attempt $attempt failed" >&2
  done
  mkdir -p "$ROOT/.scratch"
  echo "$TAG: push failed after 2 attempts at $(date -u +%FT%TZ)" \
    > "$ROOT/.scratch/push-failed"
  return 1
}
publish || FAILED=1

python3 "$HERE/validate.py" "$LEDGER"
(( TRUNCATED )) && echo "run_set.sh: batch $TAG truncated at deadline" >&2
exit "$FAILED"

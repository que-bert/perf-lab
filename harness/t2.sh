#!/usr/bin/env bash
# T2 gate (docs/2026-10-01-phase2-plan.md, "Gate tiers"): candidate integration build vs the integrate16 anchor.
#
#   t2.sh [--dry-run] <candidate-worktree> <label>        anchor = A (default: serving build, = integrate16)
#
# Change-aware: harness/gate_plan.py reads `git diff base..HEAD` and chooses the phases (printed first); FULL=1 runs
# everything, PHASES="..." overrides. Phases:
#   arms     I1 ABAB on the 27B serving config, ONE server load per arm (arm_session.sh: pooled 37 prompts + texts, 32.5k
#            prompt x2, depth 70k/176k decode+prefill). Sequential stopping: after round KMIN..K, ab_stat.py decide stops
#            once every axis' 90% bound is inside +-TOL_PCT (default 0.5) or excludes zero. SEQ=0 runs all K rounds.
#   flagless auto-config path (server with only mmproj, -lm none + HOST_GET_ROWS)
#   i2       MiniCPM5-2B llama-bench ABAB, rounds KMIN..K+1, same stopping rule (REPS=10 per run)
#   gates    gates5.sh: op tests (planned ops, one process), PPL, decode KLD, acceptance (one server load)
#   zoo      quality (PPL/KLD/acc) + smoke. Models <= ZOO_SMALL_GB (default 12) run on the 9060 lane (correctness kind,
#            concurrently, cpu lock shared) when ZOO_LANES=1; the big ones (27B, 35B-A3B) on the R9700 after the
#            timing phases. ZOO_LANES=1 (opt-in) sends them to the 9060; default 0 because the 9060 is not bit-identical to the R9700 references (see FINDINGS in commit).
# Every GPU step takes gpu_lock.sh. Output: results/phase2/t2/<label>/*.log; arm numbers in <label>/arm/*.arm.json.
# Env: K (3) KMIN (2) TOL_PCT (0.5) SEQ (1) FULL (0) PHASES A BASE_REF GATES_PARTS OPS ZOO_PARTS ZOO_ACC.
set -u
DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
W=$(readlink -f "${1:?candidate worktree}"); T=${2:?label}
P=/home/bbuckham/git/perf-lab; cd $P; . harness/zoo.sh
B=$W/build/bin
A=${A:-$P/runners/llama.cpp/qwen3.8-r9700/build/bin}
K=${K:-3}; KMIN=${KMIN:-2}; TOL=${TOL_PCT:-0.5}; SEQ=${SEQ:-1}
O=$P/results/phase2/t2/$T
S=setups/qwen3.8-27b-r9700; MD=${PERFLAB_MODEL_DIR:-/home/bbuckham/models}
exec 3>&1
run() { if [ $DRY = 1 ]; then echo "  + $*" >&3; else "$@"; fi; }
note() { echo "== $*"; }

# --- plan -----------------------------------------------------------------------------------------------------
BASE_REF=${BASE_REF:-$(git -C "${A%/build/bin}" rev-parse HEAD 2>/dev/null || echo r9700-integrate16)}
PH_ALL="depth pool prompt flagless i2 gates zoo"
if [ "${FULL:-0}" = 1 ]; then
  PHASES=${PHASES:-$PH_ALL}; GATES_PARTS=${GATES_PARTS:-"ops ppl kld acc"}; OPS=${OPS:-full}; ZOO_PARTS=${ZOO_PARTS:-full}
  note "plan: FULL=1 -> $PHASES"
else
  U_PH=${PHASES:-}; U_GP=${GATES_PARTS:-}; U_OPS=${OPS:-}; U_ZP=${ZOO_PARTS:-}
  eval "$(python3 harness/gate_plan.py "$W" "$BASE_REF" --env)"   # sets PHASES GATES_PARTS OPS ZOO_PARTS
  python3 harness/gate_plan.py "$W" "$BASE_REF"
  PHASES=${U_PH:-$PHASES}; GATES_PARTS=${U_GP:-$GATES_PARTS}; OPS=${U_OPS:-$OPS}; ZOO_PARTS=${U_ZP:-$ZOO_PARTS}
fi
PH=" $PHASES "
ARM_PARTS=""; for x in pool prompt depth; do [[ $PH == *" $x "* ]] && ARM_PARTS="$ARM_PARTS $x"; done; ARM_PARTS=${ARM_PARTS# }
note "run: phases=[$PHASES] arm_parts=[$ARM_PARTS] gates=[$GATES_PARTS] ops=$OPS zoo=$ZOO_PARTS K=$K kmin=$KMIN tol=$TOL base=$BASE_REF seq=$SEQ"
[ $DRY = 1 ] && O=$(mktemp -d /tmp/t2dry.XXXXXX); mkdir -p $O/arm

# --- helpers --------------------------------------------------------------------------------------------------
# AB loop: <runner prefix> <decide command...>. Rounds of (base, cand); stop at KMIN..kmax once decided.
ab_rounds() {  # $1 = kmax, $2 = round function, $3.. = decide command (exit 0 = stop)
  local kmax=$1 fn=$2; shift 2
  local i
  for i in $(seq 1 $kmax); do
    $fn $i
    if [ $DRY = 1 ]; then echo "  + (after round $i >= $KMIN: $* ; stop when decided)" >&3; continue; fi
    if [ $i -ge $KMIN ] && [ $i -lt $kmax ] && [ "$SEQ" = 1 ]; then
      "$@" > $O/decide.last 2>&1 && { cat $O/decide.last; note "sequential stop after round $i of $kmax"; return 0; }
    fi
  done
  [ $DRY = 1 ] || { "$@" > $O/decide.last 2>&1; cat $O/decide.last; }
}
arm_round() {  # pooled (37 prompts, texts) is measured in round 1 only: it is deterministic to <0.1%, rounds 2.. cost a pool run for nothing
  local parts="$ARM_PARTS"; [ $1 -gt 1 ] && parts=$(echo " $parts " | sed 's/ pool / /; s/^ *//; s/ *$//')
  [ -z "$parts" ] && return 0
  run env ARM_PARTS="$parts" $P/harness/arm_session.sh $A $T-base-r$1 $O/arm >> $O/arm.log 2>&1
  run env ARM_PARTS="$parts" $P/harness/arm_session.sh $B $T-cand-r$1 $O/arm >> $O/arm.log 2>&1
  [ $DRY = 1 ] || grep -h '== ARM' $O/arm.log | tail -2
}
i2_round() {
  run env PROFILE_DIR=$O/i2 REPS=${I2_REPS:-10} harness/model_profile.sh $A minicpm-q4km base-r$1 >> $O/i2.log 2>&1
  run env PROFILE_DIR=$O/i2 REPS=${I2_REPS:-10} harness/model_profile.sh $B minicpm-q4km cand-r$1 >> $O/i2.log 2>&1
}
zoo_split() {  # prints "small|big" name lists by file size
  local n big="" small="" gb=${ZOO_SMALL_GB:-12}
  for n in $(zoo_names); do
    local f; f=$(zoo_path $n); local sz=$(( $(stat -c %s "$f" 2>/dev/null || echo 999999999999) / 1000000000 ))
    if [ "${ZOO_LANES:-0}" = 1 ] && [ $sz -le $gb ]; then small="$small $n"; else big="$big $n"; fi
  done
  echo "${small# }|${big# }"
}
zoo_lane() {  # <card> <names...> : quality (ZOO_PARTS=full) then smoke, on that card's lane
  local card=$1; shift
  [ $# = 0 ] && return 0
  export PERFLAB_MAXLOAD=99
  if [ "$ZOO_PARTS" = full ]; then
    for n in "$@"; do PERFLAB_CARD=$card ACC=${ZOO_ACC:-1} run harness/model_quality.sh eval $B $n t2-$T; done
  fi
  PERFLAB_CARD=$card run harness/zoo_smoke.sh test $B t2-$T "$@"
}

# --- zoo, lane B (9060, correctness): starts now, overlaps the timing phases in the gaps the cpu lock leaves --------
ZPID=""
if [[ $PH == *" zoo "* ]]; then
  IFS='|' read -r ZSMALL ZBIG <<< "$(zoo_split)"
  note "zoo lanes: 9060=[$ZSMALL] r9700=[$ZBIG]"
  if [ -n "$ZSMALL" ]; then
    if [ $DRY = 1 ]; then zoo_lane 9060 $ZSMALL; else zoo_lane 9060 $ZSMALL > $O/zoo-9060.log 2>&1 & ZPID=$!; fi
  fi
fi

# --- I1: arms (depth + pooled + 32.5k prompt on one server per arm) ----------------------------------------------
if [ -n "$ARM_PARTS" ]; then
  note "arms: ABAB, one load per arm, parts=[$ARM_PARTS]"
  ab_rounds $K arm_round python3 harness/ab_stat.py decide-arms $O/arm $T $TOL $KMIN $K
fi
# --- flagless: only what the operator must set by hand: mmproj, and -lm none + HOST_GET_ROWS (never set by preset) ----
if [[ $PH == *" flagless "* ]]; then
  note "flagless"
  if [ $DRY = 1 ]; then echo "  + gpu_lock.sh <server with mmproj, -lm none, HOST_GET_ROWS> + pool.py"; else
  harness/gpu_lock.sh bash -c "systemd-run --user --unit=perflab-srv-8097 --collect --setenv=LD_LIBRARY_PATH=$B --setenv=GGML_VK_VISIBLE_DEVICES=\$PERFLAB_GPU \
      --setenv=MTMD_LAZY_GPU=1 --setenv=GGML_VK_HOST_GET_ROWS=1 -p StandardError=append:$O/flagless-server.log \
      $B/llama-server -m $MD/Qwen3.8-27B-Q6_K.gguf --port 8097 --host 127.0.0.1 --no-webui -ngl 99 -lm none --mmproj $MD/mmproj-F16.gguf >/dev/null
    for i in \$(seq 1 120); do curl -sf localhost:8097/health >/dev/null && break; sleep 5; done
    python3 $S/autoresearch/pool.py 8097 $S/autoresearch/runs/pool-t2-$T-flagless.json | grep POOL
    systemctl --user stop perflab-srv-8097; sleep 3" > $O/flagless.log 2>&1
  grep -h 'preset:' $O/flagless-server.log >> $O/flagless.log; fi
fi
# --- I2: MiniCPM llama-bench ABAB with the same stopping rule ------------------------------------------------------
if [[ $PH == *" i2 "* ]]; then
  note "i2: minicpm-q4km ABAB, rounds <= $((K + 1))"
  ab_rounds $((K + 1)) i2_round python3 harness/ab_stat.py decide-jsonl $O/i2 base cand $TOL $KMIN $((K + 1))
  [ $DRY = 1 ] || python3 harness/profile_report.py base cand --dir $O/i2 >> $O/i2.log
fi
# --- gates (R9700, correctness kind: numerics, not timing -> shares the cpu lock with the 9060 lane) --------------------
if [[ $PH == *" gates "* ]]; then
  note "gates: parts=[$GATES_PARTS] ops=$OPS"
  run env GATES_CLI="-lm none" PERFLAB_ENV="MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1" GATES_PARTS="$GATES_PARTS" OPS="$OPS" \
    PERFLAB_LANE_KIND=correctness $S/results/d0/scripts/gates5.sh $W t2-$T > $O/gates.log 2>&1
fi
# --- zoo, R9700 part (big models) -------------------------------------------------------------------------------------
if [[ $PH == *" zoo "* ]] && [ -n "${ZBIG:-}" ]; then
  note "zoo big models on R9700: $ZBIG"
  if [ $DRY = 1 ]; then zoo_lane r9700 $ZBIG; else PERFLAB_LANE_KIND=correctness zoo_lane r9700 $ZBIG > $O/zoo-r9700.log 2>&1; fi
fi
[ -n "$ZPID" ] && { wait $ZPID; cat $O/zoo-9060.log $O/zoo-r9700.log > $O/zoo.log 2>/dev/null; }
echo "== T2_DONE $T $(date)"

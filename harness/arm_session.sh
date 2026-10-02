#!/usr/bin/env bash
# One server load per arm: start the 27B serving config ONCE and measure everything the T2 ABAB loop needs on it.
#
#   arm_session.sh <bindir> <label> [outdir]
#
# Server = exactly what depth.sh / ab_bin.sh pool / mtp_prompt_ab.sh each started separately (their configs are
# identical: ctx 262144, q8_0/q8_0 KV, --parallel 1, draft-mtp n_max 4, q8_0 draft KV, draft vocab 98304 adaptive, -lm none,
# mmproj, serving env). Measurements, in this order, all cache_prompt=false where a prompt is sent fresh:
#   pool    pool.py (37 prompts + texts)            -> <outdir>/<label>.pool.json
#   prompt  32.5k-token prompt x2 (mtpp_prompt.json)-> <outdir>/<label>.prompt.json   (prompt t/s, mean of the 2)
#   depth   depth.py DEPTHS=1600,4000 REPS=3        -> <outdir>/<label>.depth.json
# depth.sh already ran its depths on one server back to back (rep 0 of each depth is cache_prompt=false = a full
# prefill; reps 1.. reuse the filled context), so a shared server changes nothing there. What does change vs the
# old flow: the state before each measurement (pool and prompt requests ran on a fresh server; here depth follows
# them, with a 262144-ctx KV already touched). pool runs first so it still sees a fresh server; the 32.5k prompt
# and depth rep 0 are cold prefills by construction.
# Prints "== ARM <label> dec70=.. pf70=.. dec176=.. pf176=.. pool=.. toks=.. prompt=.." and writes <label>.arm.json.
# Takes gpu_lock.sh itself (whole session = one locked run; a VOID re-runs the whole session).
# Env: ARM_PARTS="pool prompt depth", PORT (8097), EXTRA_ENV, EXTRA_ARGS, DEPTHS, REPS, NPRED, PERFLAB_MODEL_DIR.
set -u
P=/home/bbuckham/git/perf-lab; cd "$P"
if [ -z "${GPU_LOCK_HELD:-}" ] && [ -z "${ARM_NOLOCK:-}" ]; then exec harness/gpu_lock.sh "$0" "$@"; fi
B=${1:?bindir}; LB=${2:?label}; O=${3:-$P/results/phase2/arm}; mkdir -p "$O"
S=setups/qwen3.8-27b-r9700
MD=${PERFLAB_MODEL_DIR:-/home/bbuckham/models}
PORT=${PORT:-$((8097 + ${PERFLAB_PORT_OFF:-0}))}
PARTS=" ${ARM_PARTS:-pool prompt depth} "
UNIT=perflab-srv-$PORT
E="MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1${EXTRA_ENV:+;$EXTRA_ENV}"
F="-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive -lm none"
trap 'systemctl --user stop $UNIT 2>/dev/null' EXIT
echo "== arm $LB start load=$(cut -d' ' -f1-3 /proc/loadavg) bin=$B"
PERFLAB_BIN=$B PERFLAB_ENV="$E" PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 harness/serve_unit.sh "$MD/Qwen3.8-27B-Q6_K.gguf" $PORT 262144 \
  --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4 --mmproj "$MD/mmproj-F16.gguf" $F ${EXTRA_ARGS:-} || exit 1
if [[ $PARTS == *" pool "* ]]; then
  python3 $S/autoresearch/pool.py $PORT "$O/$LB.pool.json" | grep POOL
fi
if [[ $PARTS == *" prompt "* ]]; then
  python3 - $PORT $P/$S/results/d0/i8/mtpp_prompt.json "$O/$LB.prompt.json" <<'PY' || exit 1
import json, sys, urllib.request
port, body, out = sys.argv[1], open(sys.argv[2], "rb").read(), sys.argv[3]
res = []
for _ in range(2):  # same two requests mtp_prompt_ab.sh sent; the body carries cache_prompt=false
    r = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body, headers={"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(r, timeout=900))["timings"]
    res.append({"prompt_n": t["prompt_n"], "tps": t["prompt_per_second"]})
    print("prompt", t["prompt_n"], round(t["prompt_per_second"], 1), flush=True)
json.dump(res, open(out, "w"))
PY
fi
if [[ $PARTS == *" depth "* ]]; then
  python3 harness/depth.py $PORT --depths "${DEPTHS:-1600,4000}" --reps "${REPS:-3}" --npred "${NPRED:-256}" --json "$O/$LB.depth.json" | grep -E "~[0-9]+k|rror"
fi
systemctl --user stop $UNIT 2>/dev/null; sleep 3
python3 - "$O" "$LB" <<'PY'
import json, os, sys
o, lb = sys.argv[1:3]
a, parts = {}, []
def load(ext):
    p = f"{o}/{lb}.{ext}.json"
    return json.load(open(p)) if os.path.exists(p) else None
d, pl, pr = load("depth"), load("pool"), load("prompt")
if d:
    # depth rows are ordered as DEPTHS; label is "~70k" / "~176k"
    for x in d:
        tag = x["label"].strip("~k")
        a[f"dec{tag}"], a[f"pf{tag}"] = x["decode_steady_mean"], x["prefill"]
if pl:
    a["pool"], a["toks"] = pl["summary"]["pooled"]["tps"], pl["summary"]["pooled"]["tok_step"]
if pr:
    a["prompt"] = sum(x["tps"] for x in pr) / len(pr)
json.dump(a, open(f"{o}/{lb}.arm.json", "w"))
print("== ARM", lb, " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in a.items()))
PY

#!/usr/bin/env bash
# I3 zoo smoke: every target loads and generates 64 greedy tokens; the first 32 token ids must equal the
# upstream build's (beyond 32, greedy text may legitimately fork; model_quality.sh KLD covers that).
#
#   zoo_smoke.sh ref  <upstream-bindir> [names...]   # record upstream tokens (results/phase2/smoke/<name>.ref.json)
#   zoo_smoke.sh test <bindir> <label> [names...]    # compare; prints "== SMOKE <name> <label> PASS|FAIL ..."
#
# names default to every model in zoo.tsv. Plain decoding (no MTP), ctx 4096, through llama-server so the
# test exercises the serving path. SMOKE_ENV / SMOKE_ARGS add server env (K=V;K=V) / flags. Exit 1 on any FAIL.
set -u
H=$(dirname "$(readlink -f "$0")"); P=$(dirname "$H"); . "$H/zoo.sh"
MODE=$1; B=$2; shift 2
LB=ref; [ "$MODE" = test ] && { LB=$1; shift; }
NAMES=("$@"); [ ${#NAMES[@]} = 0 ] && mapfile -t NAMES < <(zoo_names)
OUT=$P/results/phase2/smoke; mkdir -p "$OUT"
PORT=${PORT:-$(zoo_port 8099)}; fail=0
for n in "${NAMES[@]}"; do
  mp=$(zoo_path "$n")
  # plain decoding on both sides: fork builds with GGUF presets would otherwise turn MTP / n-gram drafting on
  np=""; "$B/llama-server" --help 2>&1 | grep -q -- --no-preset && np=--no-preset
  r=$("$H/gpu_lock.sh" bash -c "PERFLAB_BIN=$B PERFLAB_ENV='${SMOKE_ENV:-}' PERFLAB_CTK=f16 PERFLAB_CTV=f16 $H/serve_unit.sh '$mp' $PORT 4096 --parallel 1 $np ${SMOKE_ARGS:-} >/dev/null || { echo LOADFAIL; exit 0; }
    python3 - $PORT <<'PY'
import json, sys, urllib.request
port = int(sys.argv[1])
prompts = ['The three laws of thermodynamics are', 'def quicksort(arr):\n', 'Paris is the capital of France. Berlin is the capital of']
out = []
for p in prompts:
    b = json.dumps({'prompt': p, 'n_predict': 64, 'temperature': 0, 'top_k': 1, 'cache_prompt': False, 'return_tokens': True}).encode()
    d = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/completion', data=b, headers={'Content-Type': 'application/json'}), timeout=600))
    out.append({'tokens': d.get('tokens', []), 'text': d['content']})
print(json.dumps(out))
PY
    systemctl --user stop perflab-srv-$PORT")
  if [ "$r" = LOADFAIL ] || [ -z "$r" ]; then echo "== SMOKE $n $LB FAIL load"; fail=1; continue; fi
  if [ "$MODE" = ref ]; then echo "$r" > "$OUT/$n.ref.json"; echo "== SMOKE $n ref recorded"; continue; fi
  echo "$r" > "$OUT/$n.$LB.json"
  python3 - "$OUT/$n.ref.json" "$OUT/$n.$LB.json" "$n" "$LB" <<'PY' || fail=1
import json, sys
ref, got = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
bad = []
for i, (a, b) in enumerate(zip(ref, got)):
    ta, tb = a['tokens'][:32], b['tokens'][:32]
    if len(tb) < 32 or ta != tb:
        k = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        bad.append(f"p{i}@{k}")
n64 = min(len(b['tokens']) for b in got)
print(f"== SMOKE {sys.argv[3]} {sys.argv[4]} {'FAIL ' + ','.join(bad) if bad else 'PASS'} (min tokens {n64})")
sys.exit(1 if bad else 0)
PY
done
exit $fail

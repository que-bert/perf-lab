#!/usr/bin/env bash
# W9 day 0 for a new model (e.g. Qwen4-27B): one command from GGUF to "how does the fork do on it".
#
#   day0.sh <name> <model.gguf> [role=secondary] [fork-bindir] [upstream-bindir]
#
# 1. converts ollama-packed files (ollama_gguf_convert.py --inspect decides), 2. adds the model to zoo.tsv
# (mtp from <arch>.nextn_predict_layers), 3. prints the serving preset the fork would pick, 4. upstream references
# (KLD logits + smoke tokens), 5. fork vs upstream profile (2 rounds), quality (KLD/PPL/acceptance) and smoke.
# Report: results/phase2/day0/<name>.md. Every GPU step goes through gpu_lock.sh.
set -u
H=$(dirname "$(readlink -f "$0")"); P=$(dirname "$H"); cd "$P"; . harness/zoo.sh
N=$1; M=$2; ROLE=${3:-secondary}
FB=${4:-$P/runners/llama.cpp/qwen3.8-r9700/build/bin}
UB=${5:-$P/runners/llama.cpp/upstream-ce8caa6/build/bin}
O=$P/results/phase2/day0; mkdir -p $O; R=$O/$N.md
GP=$P/runners/llama.cpp/upstream-ce8caa6/gguf-py
if python3 harness/ollama_gguf_convert.py --inspect "$M" >/dev/null 2>&1; then
  C=$ZOO_M/converted/$N.gguf; mkdir -p "$(dirname "$C")"
  python3 harness/ollama_gguf_convert.py "$M" "$C" && M=$C   # MTP head dropped: the ollama qwen3.5-9b head kept with --keep-mtp gave acceptance 0.004
fi
mtp=$(PYTHONPATH=$GP python3 -c "
import gguf, sys
r = gguf.GGUFReader(sys.argv[1]); a = r.fields['general.architecture']
arch = bytes(a.parts[a.data[0]]).decode()
f = r.fields.get(arch + '.nextn_predict_layers')
print(1 if f is not None and int(f.parts[f.data[0]][0]) > 0 else 0)" "$M")
grep -q "^$N	" "$ZOO_TSV" || printf '%s\t%s\t%s\t%s\n' "$N" "$ROLE" "$mtp" "$M" >> "$ZOO_TSV"
{
  echo "# day 0: $N"; echo; echo "file: $M  mtp: $mtp  fork: $FB  upstream: $UB"; echo
  echo '## preset'; echo '```'; "$FB/llama-server" -m "$M" --print-preset 2>&1 | grep -A20 'preset:'; echo '```'
} > "$R"
export PERFLAB_MAXLOAD=99
harness/model_quality.sh ref "$UB" "$N" >> "$R" 2>&1
harness/zoo_smoke.sh ref "$UB" "$N" >> "$R" 2>&1
unset PERFLAB_MAXLOAD
for r in 1 2; do
  harness/model_profile.sh "$FB" "$N" fork-r$r >/dev/null 2>&1
  harness/model_profile.sh "$UB" "$N" up-r$r >/dev/null 2>&1
done
{ echo; echo '## speed (fork vs upstream)'; python3 harness/profile_report.py up fork | grep -E "^\| (model|---|$N )"; } >> "$R"
{ echo; echo '## quality'; PERFLAB_MAXLOAD=99 harness/model_quality.sh eval "$FB" "$N" fork; harness/zoo_smoke.sh test "$FB" fork "$N"; } >> "$R" 2>&1
cat "$R"

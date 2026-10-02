#!/usr/bin/env bash
# Quality of one zoo model on one build: PPL + KLD against per-model upstream reference logits, and
# MTP draft acceptance for models with nextn layers.
#
#   model_quality.sh ref  <upstream-bindir> <zoo-name>          # make the reference logits (once per model)
#   model_quality.sh eval <bindir> <zoo-name> <label> [extra perplexity args]
#
# Reference = upstream llama-perplexity --kl-divergence-base over harness/corpus/decode_kld.txt
# (-c 1024 --chunks ${CHUNKS:-8}, f16 KV, FA on). The logits are local only (~0.5-1 GB per model) under
# $KLD_REF_DIR. eval prints "== QUAL <model> <label> PPL=.. KLD=.. top1=..% [ACC=..]" and keeps the log in
# results/phase2/quality/<model>/<label>.log. Acceptance: 16 corpus slices, n_max 3 (multi.py), only when
# zoo.tsv says mtp=1 and ACC!=0. Every GPU step goes through gpu_lock.sh.
set -u
H=$(dirname "$(readlink -f "$0")"); P=$(dirname "$H"); . "$H/zoo.sh"
MODE=$1; B=$2; N=$3; LB=${4:-ref}; shift 3; [ $# -gt 0 ] && shift
MP=$(zoo_path "$N"); [ -f "$MP" ] || { echo "model_quality: no zoo model $N" >&2; exit 2; }
REF=${KLD_REF_DIR:-$ZOO_M/perflab-kld-ref}/$N.kld; mkdir -p "$(dirname "$REF")"
C=$P/harness/corpus/decode_kld.txt
OUT=$P/results/phase2/quality/$N; mkdir -p "$OUT"
PPLA=(-m "$MP" -f "$C" -c 1024 --chunks "${CHUNKS:-8}" -fa on -ngl 99 -dev "${PERFLAB_VKDEV:-Vulkan1}")
if [ "$MODE" = ref ]; then
  "$H/gpu_lock.sh" "$B/llama-perplexity" "${PPLA[@]}" --kl-divergence-base "$REF" > "$OUT/ref.log" 2>&1 || {
    echo "model_quality: ref failed for $N" >&2; tail -5 "$OUT/ref.log" >&2; exit 1; }
  echo "== REF $N $(grep -oE 'PPL = [0-9.]+' "$OUT/ref.log" | tail -1) -> $REF ($(du -h "$REF" | cut -f1))"
  exit 0
fi
[ -f "$REF" ] || { echo "model_quality: no reference $REF; run 'model_quality.sh ref <upstream-bin> $N' first" >&2; exit 2; }
L=$OUT/$LB.log
"$H/gpu_lock.sh" "$B/llama-perplexity" "${PPLA[@]}" --kl-divergence-base "$REF" --kl-divergence "$@" > "$L" 2>&1 || {
  echo "model_quality: eval failed for $N $LB" >&2; tail -5 "$L" >&2; exit 1; }
ppl=$(grep -E '^Mean PPL\(Q\)' "$L" | grep -oE '[0-9]+\.[0-9]+' | head -1)
kld=$(grep -E '^Mean +KLD:' "$L" | grep -oE '[0-9]+\.[0-9]+' | head -1)
top=$(grep -E '^Same top p:' "$L" | grep -oE '[0-9]+\.[0-9]+' | head -1)
acc=""
if [ "$(zoo_mtp "$N")" = 1 ] && [ "${ACC:-1}" != 0 ]; then
  PORT=${PORT:-8098}
  a=$("$H/gpu_lock.sh" bash -c "PERFLAB_BIN=$B PERFLAB_CTK=q8_0 PERFLAB_CTV=q8_0 $H/serve_unit.sh $MP $PORT 32768 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 3 ${ACC_ARGS:-} >/dev/null &&
    python3 $P/setups/qwen3.8-27b-r9700/results/d0/scripts/multi.py $PORT $OUT/acc-$LB.json | grep POOLED; systemctl --user stop perflab-srv-$PORT")
  acc=" ACC=$(echo "$a" | awk '{print $4}')"
fi
echo "== QUAL $N $LB PPL=$ppl KLD=$kld top1=$top%$acc"

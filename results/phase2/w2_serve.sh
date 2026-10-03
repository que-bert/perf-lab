#!/usr/bin/env bash
# W2: MiniCPM5-2B serving: draft type (none / n-gram variants) x quant (Q4_K_M, Q8_0), pooled decode over corpus+chat sets,
# fixed fork (w2) vs upstream; plus the small-BAR gate on Gemma4-E4B / Ornith-Q4KM (llama-bench tg).
cd /home/bbuckham/git/perf-lab; . harness/zoo.sh
R=runners/llama.cpp; W2=$R/qwen3.8-r9700-w2/build/bin; UP=$R/upstream-ce8caa6/build/bin
O=results/phase2/w2/serve; mkdir -p $O
run() { # bin label model spec...
  local b=$1 l=$2 m=$3; shift 3
  harness/gpu_lock.sh bash -c "PERFLAB_BIN=$b PERFLAB_CTK=f16 PERFLAB_CTV=f16 harness/serve_unit.sh $(zoo_path $m) 8096 32768 --parallel 1 $* >/dev/null &&
    python3 setups/qwen3.8-27b-r9700/autoresearch/pool.py 8096 $O/$l-c.json --set corpus | grep POOL; python3 setups/qwen3.8-27b-r9700/autoresearch/pool.py 8096 $O/$l-h.json --set heldout | grep POOL | sed 's/^/$l /'; systemctl --user stop perflab-srv-8096; sleep 2"
}
for r in 1 2; do for m in minicpm-q4km minicpm-q8; do
  run $W2 $m-none-w2-r$r $m
  run $UP $m-none-up-r$r $m
  for s in ngram-simple ngram-mod ngram-cache ngram-map-k; do run $W2 $m-$s-w2-r$r $m --spec-type $s; done
done; done
for r in 1 2 3; do for m in gemma4-e4b ornith-q4km; do
  PROFILE_DIR=$O/gate DEPTHS=0 PP=0 harness/model_profile.sh $W2 $m w2-r$r
  GGML_VK_SMALL_BAR_MODEL_MAX_MIB=12288 PROFILE_DIR=$O/gate DEPTHS=0 PP=0 harness/model_profile.sh $W2 $m w2gate12-r$r
  PROFILE_DIR=$O/gate DEPTHS=0 PP=0 harness/model_profile.sh $UP $m up-r$r
done; done
echo "== W2_SERVE_DONE $(date)"

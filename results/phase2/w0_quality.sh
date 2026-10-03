#!/usr/bin/env bash
# W0: quality of every zoo model on i16 and upstream (KLD vs upstream reference logits; acceptance for MTP models).
cd /home/bbuckham/git/perf-lab; . harness/zoo.sh
R=runners/llama.cpp; export PERFLAB_MAXLOAD=99
for n in $(zoo_names); do
  harness/model_quality.sh eval $R/qwen3.8-r9700-next/build/bin $n i16
  harness/model_quality.sh eval $R/upstream-ce8caa6/build/bin $n upce
done
echo "== W0_QUALITY_DONE $(date)"

#!/usr/bin/env bash
# W0: upstream references for every zoo model (KLD logits + smoke tokens). Quality only, so CPU load is tolerated.
cd /home/bbuckham/git/perf-lab; . harness/zoo.sh
U=runners/llama.cpp/upstream-ce8caa6/build/bin
export PERFLAB_MAXLOAD=99
for n in $(zoo_names); do harness/model_quality.sh ref $U $n; done
harness/zoo_smoke.sh ref $U
echo "== W0_REFS_DONE $(date)"

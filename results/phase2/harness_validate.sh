#!/usr/bin/env bash
# Validate the faster gate harness (arm_session, 9060 lane, gates5 acc+ops) against known integrate18 numbers.
cd /home/bbuckham/git/perf-lab; B=runners/llama.cpp/qwen3.8-r9700-i18/build/bin; O=results/phase2/hv; mkdir -p $O
echo "== arm_session (R9700): expect dec70~62.7 pf70~1244 dec176~59.7 pf176~852 prompt~1384 pool~56.2"
harness/arm_session.sh $B hv-i18 $O 2>&1 | grep '== ARM'
echo "== 9060 lane: smoke + quality minicpm-q4km (R9700: KLD 0.001270 PPL 3.888798)"
PERFLAB_CARD=9060 harness/zoo_smoke.sh test $B hv9060 minicpm-q4km 2>&1 | grep SMOKE
PERFLAB_CARD=9060 harness/model_quality.sh eval $B minicpm-q4km hv9060 2>&1 | tail -1
echo "== gates5 ops+acc (expect 21+FA pass, ACC 0.7111 / 0.5339)"
GATES_PARTS="ops acc" GATES_CLI="-lm none" PERFLAB_ENV="MTMD_LAZY_GPU=1;GGML_VK_HOST_GET_ROWS=1" setups/qwen3.8-27b-r9700/results/d0/scripts/gates5.sh runners/llama.cpp/qwen3.8-r9700-i18 hv-i18 2>&1 | grep -E '== (test|ACC)' | grep -v 'passed' ; echo "fails: $(grep -c FAIL $O/../hv-gates.log 2>/dev/null)"
echo "== HV_DONE $(date)"

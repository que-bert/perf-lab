# Hypothesis queue (root rewrites; ranked by expected gain)

Integration head: r9700-integrate11 (= r9700-qwen @ 901abb2a8, serving): 70k 63.5, 176k 60.5, pooled 55.7, prefill 70k 1160; llama-bench pp2048 d0 / d16384 1630.3 / 1424.3 (I8B, 2026-09-30). Needs -lm none + MTMD_LAZY_GPU=1 + GGML_VK_HOST_GET_ROWS=1.

**Current target (set 2026-09-30, overnight run): server prefill with the serving config — ~32.5k prompt ≥ 1,400 t/s (from 1,304; mtp_prompt_ab.sh), 176k ≥ 800 stable (from 680–706).** Then exceed it. Secondary: pp2048 d0 ≥ 1,750 (from 1,630). Exact 8-bit GEMM is closed (I8H), so d0 is GEMM-bound at ~95 TF with no lever queued. Decomposition at 32.5k (MPD): llama-bench 1,424 = server no-spec 1,412 > server MTP 1,304 (−7.7%, of which MTP GPU ~3%).
Serving candidate flags: `-ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive --spec-draft-n-max 4`.

| id | hypothesis | est. gain | owner | state |
|---|---|---|---|---|
| N4 | n_max 4 | pooled +2.4%, 176k 47.5->55.7 | root | kept; promote after VRAM |
| G2 | closed: G2A gate+up fusion -0.4 ms (kill), G2B f32 split-K ceiling <1 ms | - | - | discard |
| Q8G | q8_0 prefill GEMM 59.7 -> 83.9 TF | pp2048 +1.9% measured | root | kept in integrate6 |
| MQ5 | q6_K MMVQ n=5 fast path + 2 rows/WG | +1.7% 70k, +2.7% 176k, +1.0% pooled | root | kept in integrate7 |
| QFC | GEMV fixed cost: quantize+barrier ceiling 1.0 ms; reuse + barrier grouping already exist | - | - | discard |
| GDN | prefill GDN: VALU-bound, 65% of floor; chunked WY+WMMA kernel is the only >=2x route | ~3% pp | - | discard (chunked form = multi-day project, not queued) |
| PSO | prefill ADD-in-GEMM epilogue: net -1.3 ms (GEMM +2 ms); CONCAT at floor | - | - | discard (parked) |
| RSV | identity state gather -> view: only 15-18% of verify steps are identity (rollback snapshot rows); +0.8 ms | - | - | discard |
| RSI | conv-state fusion cap 4->8 (n_max 4 needs K=5): CPY x240 + 48 GET_ROWS gone, ~-0.6..0.8 ms | ~1.2% | root r9700-ar-rsi | in integrate8 |
| GR2 | verify: reduced-vocab logits unpermute is GET_ROWS with ne00=1 over 248320x5 "rows" (qwen35.cpp:226-237), 2 calls 1.09 ms -> flat per-element kernel | -0.5..1.0 ms/step | root | in integrate8 (-1.08 ms GPU) |
| R7P | chain screen: chain adds nothing; Vulkan GET_ROWS from pinned host token_embd cuts draft host 3.2 -> 1.0 ms (16k) | - | - | candidate -> HGR |
| HGR | host GET_ROWS + sync (integrate9): decode +3.5..5.7%, prefill -14% (MTP pass only); HGR2 size-gating discard (broke accept) | +5.7% pooled | - | see HGR3 |
| HGR3 | HGR prefill loss is the MTP prompt pass (-11%); likely large host inputs (hidden states ~10 MB/ubatch) read in place over PCIe -> in-place only for small inputs | keep +5.7% decode at 0% prefill | - | discard (cause was VRAM, see VB) |
| MTPP | MTP prompt pass: gap is -9.7% after VB (was 19% under VRAM pressure); cost is 1-layer GPU compute, host-copy lever +0.4% | - | - | discard (overlap on 2nd queue capped ~3%) |
| VB | lazy mmproj (MTMD_LAZY_GPU=1) frees ~1130 MiB: 70k prefill 1008 -> 1117, 176k 545 -> 662, ~+2 s per image request | +10.8% / +21% prefill | root | serving (integrate10, SERVE10) |
| TREE | tree drafting: oracle TREE0 says top-2 rescues 9.5% of steps, ceiling +8% tok/step gross, net ~0-4%, +0.75 GB VRAM | - | - | deprioritized |
| FAC | FA prefill dequant scratch (whole KV to f16, 4 KiB/token) reallocated per ubatch -> 64 MiB steps | +3.7% 176k prefill | root | serving (integrate11, SERVE11) |
| DF | multi-matrix MMVQ over same-input GEMVs: -0.62 ms verify, +0.3% e2e | - | - | discard (switch on r9700-ar-df) |
| I8H | exact s8 Q6_K GEMM retry in harness: P0 (77.5 weighted) lost on LDS feed (s8 base only 36-47% of peak vs f16 55-60%) and an unpipelined per-16 epilogue; theory says exact s8 is ~1.33x f16 at equal feed efficiency. Success >= 115 TOPS weighted, kill < 105 | pp2048 d0 +10-20% if ported | - | discard (80.6, issue-bound) |
| SPK | Q6K split-K factor sweep 1/2/4/8 at d0 | - | root | closed: default heuristic already best per shape |
| FQ8 | prefill FA stages raw q8_0 K/V into LDS (no FA_DEQUANT_KV, no 4 KiB/token scratch): -11 ms/ubatch @64k, ~-30 @176k + VRAM relief | +2-3% 64k, +10% 176k | worker r9700-ar-fq8 | running |
| MPH | MTP prompt pass host overhead ~17 ms/ubatch (graph rebuild, hidden-state round trip, GPU idle between graphs) | up to +4.7% server prefill | worker r9700-ar-mph | running |
| FACA | chunked FA prefill (split-K partials in RDNA4 kernel) to bound scratch at 64-128 MiB | uncertain (GTT oscillation survived a cap) | - | open, multi-hour shader |

Closed: P3-dec (i5 n3 = i4 n3 decode), MV4 MMVQ at n=4 (+0.7 ms), G2 gate+up GEMV fusion (-0.4 ms), RM rows/WG sweep (4 stays), UB ubatch (512 stays), P2c mask DB (0%), P1c (under 2% kill line), prefill 10 unfused gate+up layers (~0.75%, under kill line), W1 draft window (acceptance), POL p_min (all lose), POL5 n_max 5 (pooled loses), W2 n=4 dequant reuse (no change).

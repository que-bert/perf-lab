# Hypothesis queue (root rewrites; ranked by expected gain)

Integration head: r9700-integrate8 = integrate7 + RSI + GR2 @ 0b5e351c8 — gates pass (GATES8). n4: 70k 60.42/59.37, 176k 57.59/57.06, pooled 51.756 (i7 same window 50.547). Serving r9700-qwen = integrate8 (SERVE8).
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
| VB | lazy mmproj (MTMD_LAZY_GPU=1) frees ~1130 MiB: 70k prefill 1008 -> 1117, 176k 545 -> 662, ~+2 s per image request | +10.8% / +21% prefill | root | in integrate10; HGR A/B on it running (hgr10) |
| TREE | tree drafting: MTP top-2 at draft pos 1 as two branches in one verify batch | tok/step | root | needs brainstorm/plan (multi-day) |

Closed: P3-dec (i5 n3 = i4 n3 decode), MV4 MMVQ at n=4 (+0.7 ms), G2 gate+up GEMV fusion (-0.4 ms), RM rows/WG sweep (4 stays), UB ubatch (512 stays), P2c mask DB (0%), P1c (under 2% kill line), prefill 10 unfused gate+up layers (~0.75%, under kill line), W1 draft window (acceptance), POL p_min (all lose), POL5 n_max 5 (pooled loses), W2 n=4 dequant reuse (no change).

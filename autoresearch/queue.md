# Hypothesis queue (root rewrites; ranked by expected gain)

Integration head: r9700-integrate7 = integrate6 + MQ5 @ df1e6be71 — gates pass; n4: 70k 59.4, 176k 57.3, pooled 50.74. Serving r9700-qwen still integrate2 (FF blocked by permission; user to run).
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

Closed: P3-dec (i5 n3 = i4 n3 decode), MV4 MMVQ at n=4 (+0.7 ms), G2 gate+up GEMV fusion (-0.4 ms), RM rows/WG sweep (4 stays), UB ubatch (512 stays), P2c mask DB (0%), P1c (under 2% kill line), prefill 10 unfused gate+up layers (~0.75%, under kill line), W1 draft window (acceptance), POL p_min (all lose), POL5 n_max 5 (pooled loses), W2 n=4 dequant reuse (no change).

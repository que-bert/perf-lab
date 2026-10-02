# Hypothesis queue (root rewrites; ranked by expected gain)

Serving: r9700-qwen = r9700-integrate16 @ 02ea7fe9f (promoted 2026-10-01, SERVE16; old serving tagged r9700-qwen-integrate11): decode 70k 62.5-62.8 / 176k 59.4-59.9, pooled 56.17, prefill 70k 1245 / 176k ~850, 32.5k prompt ~1385. Build: runners/llama.cpp/qwen3.8-r9700. Needs -lm none + MTMD_LAZY_GPU=1 + GGML_VK_HOST_GET_ROWS=1.

**Goals (reset 2026-10-01; program.md has the table). Phase 2 is planned in `docs/2026-10-01-phase2-plan.md` (W0-W9: harness, RB gap, Q4_K_M quality, Q4/Q5 decode kernels, MoE, Gemma4-E4B, tuning, GEN, prefill port, Qwen4 readiness); that plan supersedes the phase-2 list below.** Phase 1 closes with one capped item: MPD2b for the 32.5k prompt ≥ 1,400 (kill +1.5%, ~1 day). Decode and 176k prefill are no-regression guards; pp2048 d0 and pp512 d131072 are retired. Phase 2, in order:
1. **RB** [built 2026-10-01: r9700-rb@ad447f14a, merge not replay; GPU arbiter pending, row RB] rebase onto upstream llama.cpp on a new branch (79 fork commits; upstream +280; 21 conflict hunks in 6 files, hardest = llama_batch_ext vs draft-vocab/MFL; drop the #27952 port, merged upstream as 70c4e1582). Arbiter: gates4 + ab_bin vs serving + pooled texts md5. ~2 days.
2. **GEN** [closed 2026-10-01: the llvmpipe q6_K x f32 MUL_MAT failures are the generic KHR_coopmat path (pass with GGML_VK_DISABLE_COOPMAT=1, no fork switch changes them) and plain upstream a868c3e3c fails the same shared case (15 shared cases: 14 OK/1 FAIL in both); fork-only test shapes add the rest; all pass on the R9700. Not a fork bug] per-model/arch gates for every default-on general change (fusions, graph_optimize keeps, prealloc growth, MFL, VSUB); screen fork vs upstream with test-backend-ops + llama-bench on Ornith-1.5-9B (Q4_K_M/Q6_K), MiniCPM5-2B, Qwen3.6-35B-A3B (MoE). Any loss gets a gate; Qwen Q6_K serving must not move. ~1 day.
3. **Q4E** make Q4_K/Q5_K efficient on RDNA4 (Q4_K_M and Unsloth dynamic quants are mixtures: UD-Q4_K_XL 35B = Q4_K 53% / Q5_K 32% / Q8_0 12% / Q6_K 3% of bytes; UD-Q5_K_XL 27B = Q5_K 58% / Q6_K 31%), so our Q6_K-only GEMM/MMVQ barely apply. First measure Qwen3.8-27B Q4_K_M vs Q6_K (decode est. +10-20%, prefill est. −5-15%, frees ~5 GB = ends the >120k VRAM spill, quality cost to measure with gates4). Then port the lean-dequant GEMM + MMVQ n=5 paths to Q4_K/Q5_K. Multi-day.
4. **UP** upstream PRs, small general ones first (GR2 GET_ROWS ne00==1, prealloc growth, mask cache, rms_norm fusions); fix FA hsk=320 SIGFPE (row FA320) before any PR. Then the arch-gated RDNA4 FA q8_0 kernels and q6_K GEMM.
5. **MOE** MUL_MAT_ID tuning for 35B-A3B MoE (nothing tuned yet; take upstream 94a0ae3e7 tile selection in RB).
Not planned: RX 7900 XTX (RDNA3 WMMA layout differs, every RDNA4 kernel needs a port; 24 GB cannot hold Q6_K + 262k), persistent/megakernel decode (weeks), TREE.
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
| FQ8 | prefill FA stages raw q8_0 K/V into LDS | - | - | discard (+24 ms: re-dequant per query block) |
| MPH | MTP prompt pass: KV-only draft graph (default ON) + deferred pass (opt-in LLAMA_MTP_PIPE, changes a text) | +2.6% 70k, +12% 176k prefill | root | kept in integrate12 @93dd11a2f (promote candidate) |
| FPQ | prefill FA traffic probe | - | - | discard: not traffic-bound (zero-traffic -8 ms); MMA ~104 ms + 50 ms VALU floor serial |
| FOV | prefill FA: overlap softmax/VALU with MMA (more resident waves or producer/consumer) | -14 ms/ubatch @64k with FV2 | root | kept in integrate13 (FV3, GQV/I14, GQ2, MFL discarded: see results.tsv) |
| PKF | Q6_K GEMM dequant as one v_pk_fma (q*ds + -32ds) instead of pk_add+pk_mul, on integrate14 (GQV) | ~5 ms/ubatch d0 (+GQV 6.8 = ~3.5% pp d0) | - | discard (-1.5 ms) |
| VSUB | verify graph: host 'pre' (graph_compute entry -> first submit) is 1.0 ms at 16-70k, 1.6 ms at 176k with the GPU idle (D10 STEP_TIMING); first submit should come after ~0.1 ms | +2% decode | - | memo discarded (misses at depth); v3 no-memo -0.45 ms parked for integrate16 |
| SOP | prefill small-op bundle d0: ADD+RMS_NORM_MUL fusion for nrows>1 (only nrows==1 fuses today), merge the 2 GDN f32 m=48 projections, fuse the 10 unfused gate+up+GLU layers | ~6.5 ms/ubatch (~2% d0) | - | parked -3.97 ms (GLU 2.7 + ADDRMS 1.8; M48 not built); bundle into integrate16 |
| UB2 | server -ub 1024 at 32.5k (per-ubatch MTP fixed costs + FA dequant pass halve); llama-bench ub sweep @131k | - | - | discard (flat) |
| I15 | integrate13 + MFL (parked +1.2%) + LLAMA_MTP_PIPE=1 (+0.5%) toward the 1400 prompt target | ~+1.7% 32.5k | root | MFL kept +1.5% (1368.6); PIPE flat; needs pooled check |
| MPB | MTP prompt-pass per-ubatch budget at 32.5k on i15, remove largest removable term | 1368 -> 1400 needs +2.3% | - | discard (ub1024 draft pass 0%); budget: +53 ms/2048 = last layer 26 + draft graph 8 (3-4 PCIe h reads) + boundary idle 29 |
| MPD2 | make LLAMA_MTP_PIPE overlap for real: draft decode's HGR pre-input sync drains the whole queue (waits ~309 ms for the target ubatch); per-context wait + fix PIPE text divergence | ~25-29 ms/2048 (+2%) | - | discard: hang fixed but +0.4% at 32.5k (rows MPD2b, MPD2b-AB); phase 1 closed |
| FDO | deep prefill: issue FA K/V dequant of the old range [0,KV-512) early (after the previous attention layer's FA) so it overlaps GDN-layer GEMMs; new rows only after SET_ROWS; same scratch | ~10-15 ms/ubatch @131k (~2%) | - | discard (-0.1%: overlap window one GEMM wide, GEMM slows by what dequant saves) |
| MPH2 | draft graph reads h device-to-device from the target's t_h_nextn (no host memcpy 2.3 ms, no PCIe-speed norm/concat reads 3-4 ms) | ~0.4% | - | after MPD2 |
| FACA | chunked FA prefill (split-K partials in RDNA4 kernel) to bound scratch at 64-128 MiB | uncertain (GTT oscillation survived a cap) | - | open, multi-hour shader |

Closed: P3-dec (i5 n3 = i4 n3 decode), MV4 MMVQ at n=4 (+0.7 ms), G2 gate+up GEMV fusion (-0.4 ms), RM rows/WG sweep (4 stays), UB ubatch (512 stays), P2c mask DB (0%), P1c (under 2% kill line), prefill 10 unfused gate+up layers (~0.75%, under kill line), W1 draft window (acceptance), POL p_min (all lose), POL5 n_max 5 (pooled loses), W2 n=4 dequant reuse (no change).

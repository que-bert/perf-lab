# R9700 kernel rewrite plan: prefill and decode toward the hardware limits

Date: 2026-09-26 (rev 2: devil's-advocate fixes folded in — P0.2 GEMM
go/no-go, P4 made conditional, decode-path numerics + MTP acceptance gates,
FA scratch split). Supersedes the "ceiling ~45 t/s / no prefill lever" verdicts in
`FINDINGS.md` (2026-09-24..26) and the rev6.1 plan. Scope set by the operator:
**Qwen3.8-27B only, R9700 only.** Portability to other models and power profiles
are deferred until after this plan completes.

## Fixed serving config (immutable)

`Qwen3.8-27B-Q6_K.gguf`, `-ctk q8_0 -ctv q8_0`, ctx 262144, `-fa on -ngl 99`,
`--parallel 1`, MTP `draft-mtp` n_max 3, `--mmproj mmproj-F16.gguf`, device
Vulkan1 (R9700, gfx1201, RADV Mesa 26.0.8). Fork `~/git/llama.cpp-r9700`, branch
`r9700-qwen`, base `26bd56621`.

Weights and KV storage formats never change. In-kernel arithmetic may change
(e.g. dequantize to fp16 vs int8 operands, activation quantization granularity)
**only** behind the numerics gate in "Gates" below.

## Hardware ceilings (corrected)

| quantity | value | status |
|---|---:|---|
| DRAM bandwidth | 640 GB/s | measured (HIP probe) |
| WMMA fp16/bf16 dense | ~203 TFLOPS @3.1 GHz (1024 FLOP/CU/clk) | AMD spec, scaled; **P0 measures it** |
| WMMA int8 / fp8 dense | ~406 TOPS (2048 /CU/clk) | spec, scaled; P0 measures |
| VALU fp32 / packed fp16 | ~48 / ~96 TFLOPS | spec (the "96" in old docs is this, not WMMA) |
| LDS | 64 KiB per workgroup, ~128 B/clk/CU | spec |
| VGPR occupancy | ≤96 → 16 waves, 120 → 12, 192 → 8, 256 → 6 | published |
| Infinity Cache | 64 MB | spec |

RADV exposes `VK_KHR_cooperative_matrix` 16×16×16 only, for f16/bf16→f16/f32,
s8/u8→s32, fp8(e4m3/e5m2)→f32 (fp8 needs `VK_EXT_shader_float8`). No int4, no
sparsity. `VK_NV_cooperative_matrix2` exists behind a drirc option but is
immature on gfx12.

External evidence that the headroom is real: vLLM on the same card and model
family reaches 3,542 t/s prefill at 2k (FP8 WMMA kernels at ~225 TFLOPS), vs
llama.cpp ~990 (different weight format, so an existence proof for kernel
throughput, not a like-for-like target).

## Where the time goes today (measured 2026-09-26, `GGML_VK_PERF_LOGGER=1`)

**Prefill, one 512-token ubatch, depth 0** — 533 ms GPU (≈ 960 t/s, matches the
unlogged 982):

| op | ms | share | rate |
|---|---:|---:|---|
| q6_K + q8_0 GEMM (all projections) | 451.8 | 85% | 49–68 TFLOPS (24–33% of fp16 peak) |
| GATED_DELTA_NET (token-serial) | 12.8 | 2.4% | — |
| CONCAT (GDN conv state) | 11.3 | 2.1% | 233 µs each, suspicious |
| GLU (f32 swiglu) | 10.9 | 2.0% | bandwidth-bound on f32 intermediates |
| f32 MUL_MAT (m=48 alpha/beta etc.) | 10.3 | 1.9% | 2.6 TFLOPS |
| FA | 4.6 | 0.9% | — |
| everything else (norms, add, scale, cpy…) | ~31 | 6% | — |

**Prefill, one 512-token ubatch at depth 131072** — 1,339 ms (365 t/s):

| op | ms | share |
|---|---:|---:|
| FA (incl. q8_0→f16 scratch dequant) | 806 | 60% (50.4 ms/layer, ~33 TFLOPS) |
| GEMM | ~445 | 33% |
| everything else | ~88 | 7% |

FA prefill grid is 24 q-heads × (N/16) row tiles, each streaming the whole KV:
at 131k that is ~103 GB of K/V reads per layer served from cache, because GQA is
not packed at prefill (each K/V tile fetched once per q head) and Br=16.

The 806 ms FA figure at depth includes the two `dequant_q8_0_transpose` scratch
dispatches issued inside the FA node (`ggml-vulkan.cpp:8348-8375`); the split
between scratch build and attention is unmeasured (P0.6).

**Decode at 176k** (FINDINGS 2026-09-25 profile + `ggml-vulkan.cpp:6522-6529`;
P0 re-measures): step ≈ 88 ms for ≈ 3.63 tokens. DRAM floor ≈ 53 ms (≈ 68 t/s):

| term | floor | measured | gap |
|---|---:|---:|---:|
| weight GEMV (n=4 verify) | ~33.6 ms | 35.75 (q6_K, code comment) | ~2 |
| FA over 16 layers | ~10.0 | 28.7 | ~19 |
| 3 draft passes | ~8.4 | 10.7 | ~2 |
| small ops | ~1 | 7.7 | ~6 |

The GEMV is already near its floor; decode headroom is almost entirely FA (P3)
and small ops (P5). Open discrepancy: the same code comment records 70k decode
at 51.6 t/s while the 9d4e handoff records 46.4 — P0.3 settles which protocol
and number is current.

## Targets

| metric | now | target | stretch |
|---|---:|---:|---:|
| pp2048 @ d0 (llama-bench) | 982 | ≥ 1,700 | ≥ 2,000 |
| pp512 @ d131072 | 365 | ≥ 800 | ≥ 1,100 |
| decode @ 70k (depth.sh) | 46.4 (or 51.6, see above) | re-derived in P0.3 | — |
| decode @ 176k | 41.2 | ≥ 52 | ≥ 58 |
| MTP acceptance | 0.763 / 0.881 | unchanged ±0.01 | — |

Target arithmetic: a 512-token ubatch is 27.9 TFLOP of GEMM. At 120 TF that is
233 ms; with ~55 ms of other ops after P5 → ~1,780 t/s, hence target 1,700.
The 2,000 stretch needs ~140 TF (≈ 70% of peak). Decode 176k: FA 28.7 → 13.6 ms
(P3 at 800 µs × 17) and small ops 7.7 → 3 ms (P5) give a ~68 ms step → ~53 t/s;
58 needs P3 near the DRAM floor. **All prefill targets are provisional until the
P0.2 go/no-go.**
A task that misses its own gate is killed with the measured reason, as before —
but a kill applies to that **design**, not to the op's headroom.

## Gates (apply to every task)

- **Correctness:** `test-backend-ops test -b Vulkan1 -o <OP>` all pass for the
  touched op, including new eval cases at the model's exact shapes.
- **Prefill numerics:** `llama-perplexity` on a fixed 32k-token text, plus
  KL-divergence vs the `26bd56621` logits (`--kl-divergence-base`), at the fixed
  config. Pass: ΔPPL ≤ 0.2% and mean KLD ≤ 0.002. Covers ubatch prefill only.
- **Decode numerics (new, built in P0.5):** teacher-forced decode-path KLD at
  depths 70k and 176k — prefill the context in normal ubatches, then score 512
  tokens in 4-token batches (the verify shape) and 1-token batches (the draft
  shape), comparing logits to `26bd56621`. Pass: mean KLD ≤ 0.002, max top-1
  flip rate ≤ 0.5%. This is the only gate that exercises N≤8 FA, the n=4 GEMV,
  and split-KV combine at real depth.
- **MTP acceptance is a per-task gate**, not only a final target: every task
  touching FA, GEMV, or small ops (P2–P5) runs `depth.sh` and must hold
  acceptance at 70k and 176k within ±0.01 of baseline.
- Any change to in-kernel arithmetic precision must pass **both** numerics gates.
- **Speed:** llama-bench `-dev Vulkan1 -r 3` for prefill; `harness/depth.sh` ≥3
  reps for decode; `harness/gpu_guard.sh` before every run.
- **New shaders are separate shader modules** with their own pipelines and an
  env kill switch `GGML_VK_NO_<NAME>=1` (a new FA `Flags` bit SIGSEGVs RADV).
- **One task in flight on the GPU at a time.** Sub-agents may write code in
  parallel but measurements are serialized.

## Tasks

### P0 — Instrument and measure the real ceilings (1–2 days)

1. `harness/wmma_peak/`: standalone Vulkan compute microbenchmark (GLSL +
   minimal host) issuing register-resident `coopMatMulAdd` chains for f16→f32,
   f16→f16, s8→s32, fp8→f32 at 16×16×16; plus an LDS-bandwidth and a
   DRAM-read kernel. Output TFLOPS / GB/s.
   Interface: `wmma_peak --device 1 --type {f16,f16acc16,s8,fp8} --iters N`.
2. **P1 go/no-go — LDS-fed GEMM microbenchmark.** In the same harness, a plain
   f16×f16→f32 GEMM (m=17408 n=512 k=5120, and 4096×512×14336) with tiles
   staged global → LDS → `coopMatLoad`, double-buffered, 128×128 and 256×128
   tiles; and the s8×s8→s32 equivalent. No quantized weights. This isolates
   whether RADV/ACO coopmat1 with LDS operands can exceed the existing
   69.8 TF (`FINDINGS.md:3067`).
   Interface: `wmma_peak --gemm {f16,s8} --m --n --k --tile {128x128,256x128}`.
   **Gate:** f16 ≥ 110 TF or s8 ≥ 180 TOPS → P1 proceeds with that operand type.
   Below both → dump ISA (`RADV_DEBUG=shaders`) and RGP, identify the codegen
   limiter, and re-scope P1 (possibly to `VK_NV_cooperative_matrix2` via drirc,
   or to a Mesa fix) before any Q6_K work.
3. `harness/prefill_profile.sh <build> <depth> <ubatch>`: runs llama-bench with
   the perf logger, extracts the **last** graph block, prints the op table
   above (ms, share, TFLOPS).
4. Re-profile decode at 70k and 176k on `26bd56621` (per-node verify graph +
   drafts), resolving the 46.4 vs 51.6 t/s discrepancy and confirming the
   35.75 ms GEMV figure. **P4 proceeds only if summed verify GEMV ≥ 42 ms.**
5. Build the decode-path KLD tool (Gates). Interface: a fork tool
   `llama-decode-kld -m <gguf> --ctk q8_0 --ctv q8_0 --file <text> --depth N
   --score 512 --batch {4,1} [--save-logits F | --base-logits F]`, printing
   mean KLD and top-1 flip rate. Record baseline logits at 70k and 176k.
6. Split the prefill FA node at 131k: time the two `dequant_q8_0_transpose`
   dispatches separately from the FA kernel (perf-logger sub-entries or a
   temporary env that skips the scratch when K/V are already f16-compatible
   in a synthetic test). Record scratch ms vs attention ms per layer.
7. Profile one image through the mmproj (`llama-mtmd-cli`, a 1024² image) with
   the perf logger: op table for the vision encoder.
8. Append a FINDINGS correction: the 96-TFLOPS peak and the "ceiling ~45" /
   "no prefill lever" verdicts are withdrawn, with the numbers above; and the
   MMVQ −7.7% lead is withdrawn (it forced every type, not q6_K; q6_K alone is a
   measured loss, `ggml-vulkan.cpp:6522-6529`).

Acceptance: measured WMMA peaks and the P0.2 go/no-go result recorded; decode,
FA-split and mmproj tables recorded; decode KLD baselines saved. Every later
target is re-derived from P0 numbers before its task starts.

### P1 — Prefill GEMM rewrite (the 85% term)

New matmul shader module `mul_mm_q6k_rdna4` (and q8_0 variant for `ssm_out`),
selected for Q6_K/Q8_0 × f32 when n ≥ 64 on RDNA4.

Design space, measured in order, first to pass wins:

- **1a fp16-WMMA with in-LDS dequant.** Q6_K → fp16 (scale × d folded) into
  LDS once per tile, activations f32→f16 in the load path, fp16 WMMA with f32
  accumulate across the whole K (no per-16 epilogue). Tiles 128×128 and
  256×128, double-buffered LDS, register prefetch, VGPR ≤ 128 (12 waves).
  Ceiling ~203 TF; target ≥ 120 TF.
- **1b int8-WMMA with an integer scale epilogue.** Keep int8 operands but fold
  Q6_K's int8 sub-block scale into the int32 accumulator with an integer
  multiply-add (no int→float convert per 16-K step); convert to float once per
  32-K q8_1 block. Ceiling ~406 TOPS; target ≥ 160 TOPS effective.
- **1c int8 with per-row activation scale** (single scale per token row per
  K-panel): removes the activation-block epilogue. **Numerics gate required.**

Also: fuse FFN gate+up into one GEMM with a swiglu epilogue (removes GLU
10.9 ms and the f32 intermediate), and fuse the four GDN-input projections that
share src1.

Start with the operand type that passed P0.2; only build the other design if the
first stalls. The P0.2 microbenchmark result is the ceiling for this task: the
Q6_K kernel's acceptance is ≥ 85% of it.

Acceptance: MUL_MAT all pass; prefill numerics gate; isolated m=17408 n=512
k=5120 ≥ 120 TF (≥ 140 TF for the 2,000 stretch); pp2048 d0 ≥ 1,700 t/s. Kill:
if the Q6_K kernel stays < 75% of the P0.2 ceiling after two iterations, record
the per-kernel limiter (VGPR, LDS bank conflicts, issue) from
`GGML_VK_PIPELINE_STATS` + RGP before closing.

### P2 — Prefill FA rewrite (the 60% term at depth)

**P2a — cheap fix first, if P0.6 says so.** If the scratch dispatches are
≥ 30% of FA time at 131k, first try: dequantizing only the causal-visible range,
a non-transposed scratch layout that `coopMatLoad` reads row-major, and GQA
packing in the existing `flash_attn_cm1` path. Measure; proceed to P2b only if
FA is still < 70 TFLOPS.

**P2b — new shader module** `flash_attn_prefill_rdna4` for q8_0 K/V, N ≥ 64.

- **GQA-packed rows:** one workgroup owns one KV head and Br rows drawn from all
  6 q heads sharing it (e.g. 16 positions × 6 heads = 96 rows), so each K/V
  tile is read once per 6 heads instead of 6 times.
- **In-kernel q8_0 → fp16 dequant into LDS per KV tile** (drop the full-cache
  f16 scratch that is rebuilt per layer per ubatch — 539 MB write at 131k).
- Bc 64, fp16 WMMA for QKᵀ and PV, f32 softmax in registers, causal-block skip
  via the existing mask-opt bitmask (adapted to packed rows).
- Online softmax with the O accumulator in registers; no PV→LDS round-trip
  per hsv tile.

Acceptance: FLASH_ATTN_EXT all pass (new cases at hsk=hsv=256, nh=24/4, N=512,
KV 16k/131k); prefill numerics gate; MTP acceptance gate; isolated N=512
KV=131584 ≤ 17 ms/layer (≥ 100 TFLOPS, vs 50.4); pp512 @ d131072 ≥ 800 t/s. Kill: < 70 TFLOPS after the packed + in-kernel
dequant version; record the limiter.

### P3 — Decode FA rewrite (the 19 ms term)

New shader module `flash_attn_decode_q8` for N ≤ 8, q8_0 K/V.

- No coopmat: vector ALU, K/V q8_0 read straight into registers, dequant in
  registers (no LDS staging of K/V), all 24 query rows of a KV head
  (4 positions × 6 heads) processed against each K/V element loaded once.
- Split-KV across ≥ 2 × 64 CU workgroups with a separate combine pass; per-split
  running max/sum in registers.
- Causal skipping from the mask-opt bitmask; `n_kv_max` sparse bound retained.
- Also serve the MTP draft graph's single-row FA (N=1).

Acceptance: FA tests pass, including split-combine cases at kv ≥ 131k; decode
numerics gate (70k and 176k, batch 4 and 1); MTP acceptance gate; isolated nb=4
kv=183296 ≤ 800 µs/layer (vs 1,822; DRAM floor ~620); decode @176k ≥ 48 t/s from
this task alone. This task carries most of the decode target. Kill: > 1,100 µs
after two design iterations; record whether the limiter is DRAM, issue, or
occupancy (`GGML_VK_PIPELINE_STATS`).

### P4 — Decode GEMV (conditional on P0.4)

Runs **only if** P0.4 measures summed verify-graph GEMV ≥ 42 ms at 176k. The
code records q6_K GEMV at 35.75 ms (~94% of DRAM), and q6_K MMVQ as a measured
loss on this model (`ggml-vulkan.cpp:6522-6529`) — do not re-try MMVQ for q6_K.

If it runs: tune rows-per-workgroup and K-split for n=4 (verify) and n=1
(drafts) per model width; check the non-q6_K tensors (q8_0 `ssm_out`, f32
alpha/beta) separately, since `GGML_VK_FORCE_MMVQ`'s −7.7% came from forcing
every type. lm_head (m=248320) is already at 98.6% — leave it.

Acceptance: MUL_MAT all pass; decode numerics gate (MMVQ quantizes activations);
MTP acceptance gate; summed verify-graph GEMV ≥ 90% of 640 GB/s; decode @176k
+≥ 2% over P3. If P0.4 shows GEMV < 42 ms, P4 is closed as "at floor".

### P5 — Small-op fusion (prefill ~70 ms/ubatch, decode ~7 ms/step)

In priority order by measured time:
- GDN conv-state CONCAT (233 µs ×48 per ubatch): write state in place.
- f32 m=48 projections (ssm_alpha/beta): fold into the fused GDN-input GEMM or
  a proper small-N kernel.
- Gated-attention `cont` + sigmoid + mul: read the strided gate view directly.
- alpha add → softplus → mul, beta sigmoid: one elementwise kernel.
- Separate `quantize_q8_1` passes whose src1 repeats.
- Decode: GDN state copy direct to cache; combine the per-token small ops.

Acceptance: each fusion byte-identical or within both numerics gates; MTP
acceptance gate for decode-path fusions; prefill
"everything else" at d0 ≤ 35 ms/ubatch; decode small-op time ≤ 3 ms/step.

### P6 — GDN prefill (small, last)

Chunked WMMA GATED_DELTA_NET (chunk 64) for N ≥ 64. Upstream #20377 is a draft
that regresses on Strix Halo and has LDS races on RDNA3 — reference only.
Only pursue if GDN exceeds 5% of the ubatch after P1–P5.
Acceptance: GATED_DELTA_NET all pass; ≥ 3× faster per call at N=512.

### P7 — Integrate, re-sweep, and record

- Re-sweep ubatch (512/1024/2048) and n_max (3/4) with the new kernels: the
  old optima were set by the old kernels.
- mmproj: confirm the vision encoder picks up the P1 GEMM (f16 weights) and
  P2-style FA (f16 K/V, gqa 1); re-profile the image encode.
- Full gate run; FINDINGS entry with before/after tables; commit per task.

## Order and dependencies

P0 → P1 (gated by P0.2) → P2a/P2b (P2a gated by P0.6) → P3 → P4 (gated by
P0.4) → P5 → P6 (conditional) → P7.
P1 before P2 because the GEMM is 85% at d0 and 33% at depth; P3 is independent
of P1/P2 in code and may be developed in parallel, but measured after P2. If
P0.2 fails, P2–P5 still proceed (they do not depend on the GEMM result).
## Delegation

Execute via `subagent-driven-development` (>3 tasks), Opus sub-agents. Tasks are
**packed per agent by context budget**: one agent takes every task that fits in
~40% of its context window; a second agent is spawned only when a bundle would
exceed that (and at the ~40-turn cap, whichever binds first). Tasks sharing
files go to the same agent. Planned bundles (re-estimated at dispatch):

| agent | tasks | why this grouping |
|---|---|---|
| A — P0 tooling | P0.1 WMMA peak, P0.2 LDS GEMM bench, P0.3 profile script, P0.5 decode-KLD tool, P0.6 FA-split instrumentation | small, independent tools; one harness dir + one fork tool; fits one budget |
| main session | P0.4 decode profile, P0.7 mmproj profile, P0.8 FINDINGS correction, all gate runs | measurement and verification stay in the main loop |
| B — prefill GEMM | P1 (+ FFN swiglu fusion, GDN-input GEMM fusion) | large shader + tuning loop; alone fills a budget |
| C — prefill FA | P2a, then P2b if needed | shares `flash_attn_*` and FA host code |
| D — decode FA | P3 | new module, split-KV + combine; alone fills a budget |
| E — small ops | P4 (if P0.4 opens it) + P5 + P6 (if triggered) | many small, related host/shader edits |

If an agent reaches its budget mid-task it stops at a clean checkpoint, commits
to a WIP branch, and hands back a state summary; a fresh agent continues from
that summary. Agents never run concurrently on the GPU: code may be written in
parallel, but every benchmark is serialized (one GPU user at a time, gated by
`gpu_guard.sh`).

## Discovered constraints and traps

- `llama-bench` defaults to Vulkan0 (the 9060 XT): always `-dev Vulkan1`.
- New FA `Flags` bits / spec constants SIGSEGV RADV's pipeline compile: new
  paths must be separate shader modules.
- GPU contention silently corrupts numbers: `gpu_guard.sh` before every run.
  The guard samples only at start (R9700 memory-busy + CPU load); a game or
  other GPU/CPU load started mid-run is not caught. No gaming during
  measurement windows; code-writing phases are unaffected.
- The perf logger's GPU sums matched wall-clock prefill here (533 ms/ubatch ≈
  982 t/s); the older "~2× inflated" note applies to its FA FLOP counter, not
  the per-node times. Cross-check any logger claim against unlogged llama-bench.
- Q6_K's per-16 sub-block scales are the reason the current int8 path needs a
  per-WMMA float epilogue; that is a design property of the existing shader,
  not a hardware limit.
- The only measurement of an LDS-fed coopmat1 GEMM on this card/driver is
  69.8 TF with f16 weights (`FINDINGS.md:3067`). Every prefill GEMM target above
  that is extrapolated from spec until P0.2 measures it.
- q6_K MMVQ is correct on RDNA4 but a measured decode loss on this model
  (`ggml-vulkan.cpp:6522-6529`); `GGML_VK_FORCE_MMVQ` results do not isolate q6_K.
- V's q8_0 scale runs along head_dim (the PV output dim), so int8 PV with a
  post-MMA scale fold is numerically invalid — do not retry it; dequantize V.
- Mesa 26.0-devel had a coopmat-related 25× codegen regression for dot4
  kernels elsewhere; if a new shader is inexplicably slow, check the ISA
  (`RADV_DEBUG=shaders`) before redesigning.

## Rev 3 — follow-ups after the integrate merge (2026-09-27)

Base for all tasks: fork branch `r9700-integrate` @ `dfb7b9f30`. Each task works
on its own branch/worktree off it and is merged into `r9700-integrate` only after
its gates pass. Gates are unchanged (section "Gates"); MTP acceptance is judged on
the code corpus (`CORPUS=harness/corpus/decode_kld.txt`, `DEPTHS` in **kB**, e.g.
8,64,260,650), not on the repeated-paragraph prompt.

| id | task | acceptance | kill |
|---|---|---|---|
| R1 | Acceptance check: corpus depth.sh on base / integrate / integrate with `GGML_VK_NO_FA_PREFILL_RDNA4=1` | integrate within ±0.01 of base on the corpus, else bisect to the op | — |
| R2 | Q6_K GEMM 81 → ≥ 90 TF: Q6_K repack to a WMMA-friendly layout at load (weights unchanged in content), split-K for the m=5120 (K=17408) shapes, then FFN gate+up swiglu fusion if budget | MUL_MAT all pass; prefill numerics gate; isolated 17408×512×5120 ≥ 90 TF; pp2048 d0 ≥ 1,389 +5% | < +3% pp2048 after two designs |
| R3 | Prefill FA 51 → ≥ 80 TF (stretch 100): Br > 16 rows per KV head, double-buffered K/V LDS, dequant fused into load | FA (`-p hsk=256,`) all pass; prefill numerics gate; MTP gate; pp512 @ d131072 improves ≥ 10% | < 60 TF after two designs |
| R4 | Decode FA q8r 933 → ≤ 800 µs (nb=4 kv=183296): VGPR/occupancy (256 VGPR → ≤ 128), split count, combine | FA pass; decode KLD gate @70k/176k b4+b1; MTP gate; depth.sh 176k ≥ +2% | > 900 µs after two designs |
| R5 | Decode non-kernel time: the ~10 ms/step with no GPU work (CPU/launch/sync) and the ~12 ms "other GPU" small ops | instrumented breakdown recorded; depth.sh 176k ≥ +3% or a measured reason it cannot move | no lever > 2% after instrumentation |

Bundling: R2, R3, R4 are separate shader loops (one agent each, ≤ 3 concurrent);
R5 is host-side and starts when the first of them finishes. R1 and every
merged-build gate run stay in the main session. Decode speed A/Bs only at
loadavg < 4, back to back.

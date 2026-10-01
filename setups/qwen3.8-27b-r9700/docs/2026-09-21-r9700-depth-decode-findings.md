# R9700 long-context decode & prefill: findings, physics, and the plan to reach 60–77 t/s

Date: 2026-09-21. Companion to `2026-09-21-r9700-decode-physics.md` (the
derivation) and `FINDINGS.md` 2026-09-21 (the raw ledger). This is the
consolidated record: what the hardware and model are, what the ceilings are,
every lever tested, and what is left.

Fixed requirement for any deliverable: **Qwen3.8-27B Q6_K, ctx 262144,
`q8_0/q8_0` KV, `-fa on -ngl 99`, MTP + mmproj, all on the R9700.** No weight or
KV precision change.

---

## 1. Hardware

| quantity | value | source |
|---|---:|---|
| card | AMD Radeon AI PRO R9700, gfx1201 (RDNA4, Navi 48) | rocm-smi |
| VRAM | 32 GiB (34.2 GB) | rocm-smi |
| memory bandwidth | **640.2 GB/s** | HIP read probe, 2026-09-19 |
| fp16 matrix throughput | ~96 TFLOP/s dense | spec |
| LDS per workgroup | 64 KiB | `maxComputeSharedMemorySize` |
| driver | Mesa 26.0.8 / RADV | vulkaninfo |
| ROCm | 7.2.4 (unused; Vulkan backend) | dpkg |

Second card is an RX 9060 XT (gfx1200, 16 GB, ~320 GB/s) driving the display. It
cannot help single-stream latency (a token traverses every layer; layer split
sums the per-GPU times and the R9700 is faster).

## 2. Model architecture — why 262k fits at all

`Qwen3.8-27B-Q6_K.gguf`: arch `qwen35`, 65 blocks, `full_attention_interval 4`,
`head_count 24`, `head_count_kv 4` (GQA 6:1), `key_length = value_length = 256`,
`embedding 5120`, `ff 17408`, `nextn_predict_layers 1`.

| component | count | per-token state | scales with ctx |
|---|---:|---|---|
| SSM / Mamba blocks | 48 | fixed state (128 x 16) | **no** |
| full-attention blocks | 16 (blk 3,7,…,63) | KV 4 heads x 256 x K,V | **yes** |
| MTP head (blk 64) | 1 | own small KV | yes, tiny |

```
KV bytes/token (q8_0) = 16 layers x 4 kv_heads x 256 dim x 2 (K,V) x 1 B
                      = 32,768 B
  + MTP layer                                             2,048 B
                      = 34,816 B  (~34 KiB)
```
A dense 64-layer model with the same heads would be 128 KiB/token — 24 GB of KV
at 183k, which does not fit alongside 22.88 GB of weights. **The hybrid is the
reason this configuration exists.**

Weights: 21.31 GiB = **22.88 GB**. KV: 6.37 GB @183k, 9.13 GB @262k.

## 3. Decode physics

MTP makes a **step** produce `1 + accepted ≈ 3.3` tokens per weight read.
Step time is measured to be linear in filled context:

```
step_ms(ctx) = 62.0 + 0.3068e-3 * ctx          (fits 2026-09-20 rows to <1 ms)
```

Decomposition at 183k: **35.7 ms** fixed (weights at 640 GB/s) + **56 ms**
context term + ~26 ms baseline overhead (MTP drafts, SSM scan, sampling, launch).
Measured effective rate: `3.31 tokens / 118 ms = 28.0 t/s`.

**The context term is 5.6x its bandwidth floor.** One KV pass over 6.37 GB is
10.0 ms; the measured slope implies ~5.6 passes' worth. The step floor:

| model of a step @183k | step floor | effective t/s |
|---|---:|---:|
| weights + one KV pass | 45.7 ms | **72 t/s** |
| weights + four KV passes | 75.5 ms | **44 t/s** |
| measured | 118.1 ms | 28 t/s |

Effective bytes/token at 183k = `640.2 / 28.04 = 22.8 GB`; the minimum is
`29.25 GB / 3.31 = 8.8 GB`. **~2.6x more traffic than the problem requires.**

Why MTP's multiplier decays: at 42 tokens the step is nearly all fixed weight
read, so 3.8 tokens ride one read (**2.18x** over unspeculated); at 183k the
context term is half the step and is not amortized the same way, while
tokens/step stays ~3.3 (**1.28x**).

| ctx | unspeculated floor | measured MTP | tokens/step | MTP multiplier |
|---:|---:|---:|---:|---:|
| 42 | 27.98 | 60.96 | 3.78 | 2.18x |
| 8.9k | 27.61 | 46.78 | 3.08 | 1.69x |
| 35.9k | 26.53 | 41.69 | 3.05 | 1.57x |
| 72.5k | 25.20 | 37.97 | 3.23 | 1.51x |
| 183k | 21.89 | 28.04 | 3.31 | 1.28x |

## 4. Prefill physics

Prefill is **compute-bound**, not bandwidth-bound. FLOPs ≈ `2 x params` per
token = ~54 GFLOP, plus causal attention that grows with context.

| ctx | measured prefill | implied TFLOP/s | % of 96 TFLOP/s |
|---:|---:|---:|---:|
| 8.9k | 650 | 35.1 | 37% |
| 35.9k | 552 | 29.8 | 31% |
| 72.5k | 464 | 25.1 | 26% |
| 183k | 324 | 17.5 | 18% |
| 234k (code) | 273 | 14.7 | 15% |

The decay with context is the O(n) per-token attention plus fixed overhead. At
234k a fresh prompt costs ~14 minutes before the first token. **A 2–3x prefill
gain is physically available** (Vulkan matmul is at 15–37% of peak), and it is
the single biggest real-world cost at long context.

## 5. Measurement method and reproducibility

`harness/depth.py` warms once with `cache_prompt`, then takes R reps over the
same filled context, so a rep costs only decode. `harness/gpu_guard.sh` refuses
to start unless `mem_busy_percent <= 12` and 1-min load < 8 for three samples.

- Within-server spread: **0.10–0.21%** (70k–234k).
- Clean server-to-server: **~0.4%**.
- The 2026-09-20 "~6% floor at 176k" was contamination: an ollama embedding
  runner on the R9700 and a sibling session's `go test ./...`. Excluding both
  collapsed it. A single-rep protocol recorded contaminated 25.8 and 35.9 t/s
  runs as results.

## 6. Every lever tested (fixed config, clean environment)

| lever | result | verdict |
|---|---|---|
| `--spec-type ngram-mod,draft-mtp` | +9.1% para 70k; **+17.2% code 98k**; 0% code 234k | **keep** (never negative) |
| `--spec-draft-n-max 4` | +0.7% para 70k; +4.1% code 98k; **-3.3% code 234k** | keep 3 at depth |
| `--spec-draft-n-max 5` | **-41%** para 70k | never |
| `GGML_VK_DISABLE_COOPMAT=1` | -3.8% para 70k | coopmat1 already better |
| PR #28507 shmem staging (ported, A/B) | 38.487 vs 38.505 (0%) | no effect |
| `--spec-ngram-mod-n-min 8 / n-match 12` | accept 0.32, 24.7 t/s | defaults optimal |
| `ngram-map-k,draft-mtp` | 42.49 t/s, 15.8% spread | worse + unstable |
| AMDVLK 2025.Q2.1 | `VK_ERROR_INCOMPATIBLE_DRIVER` | unsupported on gfx1201 |
| ROCm/HIP at depth | not rebuilt; independent same-card: Vulkan faster deep, ROCm page-faults @32768 | not the lever |
| no speculation (control) | -55% (17.46 t/s) | MTP essential |
| `rm_kq` (2 vs 4) | deferred — targets GEMV weight path, not the context term | low value |
| Q4_K_M / `-ctv q4_0` | larger levers, excluded by the fixed requirement | out of scope |

## 7. The plan to reach 60–77 t/s decode and lift prefill

The only unknown left is **where the 56 ms/step context term actually goes**.
`ggml-vulkan` ships a built-in profiler — `GGML_VK_PERF_LOGGER=1` with
`GGML_VK_PERF_LOGGER_FREQUENCY=1` — that times every graph node on the GPU via
Vulkan timestamps and reports us and GFLOPS/s per op name
(`FLASH_ATTN_EXT`, `MUL_MAT_VEC`, `SSM_*`, `ROPE`, …). No ROCm profiler needed.

Steps:
1. **Profile decode at 183k** — per-node GPU time for one step. Decide whether
   the 56 ms is `FLASH_ATTN_EXT` bandwidth, its compute, or the number of
   distinct attention submits per step (draft vs verify).
2. **Profile prefill at 72k/183k** — per-node time for the prefill graph; find
   whether the 15–37%-of-peak is one `MUL_MAT` variant, MMQ dequant, or the SSM
   kernels.
3. **Act on the profile.** Candidate directions, in order of expected value:
   - prefill: `-ub`/`-b` sweep (matmul tile efficiency), then a kernel change.
   - decode: reduce the number of KV passes per step (step structure) and/or
     tune the FA dispatch for n_rows=1 (scalar, 16 attention layers).
   - If the profile shows `FLASH_ATTN_EXT` is bandwidth-bound already, the win
     is in pass count; if compute-bound, it is in the kernel.
4. **Re-validate** each change with the rep harness against the clean baseline
   (paragraph 70k 38.44; code 98k 41.56; code 234k 28.24; prefill 464/273).

Risks: no Vulkan shader profiling exists beyond the perf logger's node names;
a kernel change to `ggml-vulkan` must not regress the other models Build A
serves; and the RDNA4 FA path (coopmat1) is shared with RDNA3, so any tuning
must be gated on the `AMD_RDNA4` enum added in B1.

**Trust:** llama.cpp is a clean clone of `github.com/ggml-org/llama.cpp` at
`ce8caa6e6`; ROCm 7.2.4 is AMD's apt package; AMDVLK is GPUOpen-Drivers; GGUFs
are HuggingFace data. No malware in anything used.

---

## 8. Per-node GPU profile (2026-09-21)

`GGML_VK_PERF_LOGGER=1` times every graph node on the GPU via Vulkan
timestamps. One **verify forward at 183k = 108.6 ms**:

| op | ms | share |
|---|---:|---:|
| `FLASH_ATTN_EXT` (16 layers, n_rows=4, 183k KV) | **52.97** | 48.8% |
| `MUL_MAT_VEC` q6_K/q8_0 weight GEMVs | 45.6 | 42.0% |
| everything else (SSM, norms, CPY, lm_head, rope…) | ~10 | 9.2% |

Two hard facts:

- **FA reads 6.0 GB of q8_0 KV in 52.97 ms = 113 GB/s — 18% of the 640 GB/s
  bandwidth, and ~8% of fp16 peak.** It is not bandwidth-bound; it is a slow
  kernel. FA time is *linear in n_rows* (938 us/layer at n_rows=3 vs 1883 at
  n_rows=6 at 70k), so raising n_max does not amortize it — it scales with it.
  That is why n_max=5 collapses (-41%).
- **The weight GEMVs are near-optimal**: 22.88 GB in 45.6 ms = 502 GB/s (78% of
  bandwidth). Nothing to win there.

Step = verify 108.6 + 3 MTP drafts (~9 ms) = ~118 ms → 3.3 tok / 118 ms = 28 t/s.
Fully accounted.

**Prefill block (one 512-token ubatch at ~44k) = 1.00 s**: `FLASH_ATTN_EXT`
476 ms (48%, at 18.6 TFLOP/s), matmuls 434 ms at 48–65 TFLOP/s (50–67% of the
96 TFLOP/s peak), the rest ~90 ms. **Prefill bottlenecks on the same FA kernel**,
so a decode FA fix lifts prefill too.

### The revised ceiling — and why 60–77 t/s is not reachable

A *perfect* step at 183k, every read at 640 GB/s and FA at the KV floor:

```
weights 35.7 + KV once 9.4 + other ~4 + drafts 9 = ~58 ms  ->  57 t/s
```

At 70k the floor is ~52 ms → ~62 t/s. **The 72 t/s figure in §3 assumed FA at
the KV floor and no drafting/SSM/elementwise cost; the measured non-FA,
non-weight work adds ~13 ms/step, which lowers the true ceiling to ~57 t/s at
183k.** So 60–77 t/s at 176k+ is above the physical ceiling of this
configuration. The reachable target with a much better FA kernel is ~45 t/s at
183k (step ~73 ms), i.e. +60% over the measured 28.

## 9. Kernel/knob attempts against the FA wall (all negative)

The FA shader's `Br`/`Bc`/`row_split` are Vulkan **specialization constants**, so
they were swept from one binary via `GGML_VK_FA_CM1_BR`/`_NS`:

| attempt | result |
|---|---|
| coopmat1 `ns=4` (upstream default, Bc=64) | **38.633 t/s** — optimal |
| `ns=8` (Bc=128) / `ns=2` (Bc=32) | HTTP 500 during prefill |
| `ns=8, Br=32` / `ns=16` | 37.64 t/s (-2.5%), prefill 441 (-17%) |
| PR #28507 shmem staging on scalar FA | 38.487 vs 38.505 (0) |
| `-ctv f16` instead of q8_0 | **5x slower** FA (74.6 vs 15.0 ms @70k) |
| `-ub 1024 / 2048` | prefill +7.5% / +4.3%, **decode -35%** |
| ub 4096 | prefill 403 (-24%), decode 30.9 |
| disable coopmat | -3.8% |
| FA `split_k=1` (serialise KV scan) | **18.85 t/s (-51%)** |
| FA `split_k=32` (over-split) | prefill pathological, killed |
| `-ub 1024` (decode) | 24.95 (-35%), prefill 583 (+11% vs 524) |
| `-ub 768` | 31.2 (-18%), prefill 488 |
| `--no-mmproj-offload` (frees 0.9 GiB VRAM) | 41.67 vs 41.55 — neutral |
| `--no-mmproj-offload -ub 1024` | decode 27.1 (-35%), prefill 610 |
| `--poll 100` | 29.3 (-23%) |
| `-nr` (disable weight repack) | 38.25 — neutral |
| n_max 2 | 29.6 @70k (-22%), accept 0.79 |

Prefill measured with reps (re-prefill every rep) for the first time:
base **524 t/s [518-531, 2.55%]** at 72k — so prefill is readable to ~2.5%, and
the ub1024 gain is **+11-16%**, not the +30% a single rep suggested.

**Prefill conclusion: no usable gain.** The only prefill lever is ubatch, and
every ub above 512 costs 18-35% of decode because the larger compute buffer
cannot coexist with weights + q8_0 KV at 262k. Freeing the mmproj's 0.9 GiB did
not fix that, so the conflict is not a simple 1 GiB shortfall. Prefill is bound
by the **same FA kernel (48%)** plus matmul at 50-67% of peak, so the shader
rewrite is the only route for prefill too.

**`-ub` and n_max 2/5 are all negative.** Config-level knobs are exhausted in
both directions.

**The FA parallel decomposition is already exploited.** `split_k` is the KV-range
split that fills the GPU with workgroups; forcing it to 1 halves decode, which
proves the auto heuristic is doing real work. Forcing it to 32 makes prefill
crawl. So the slowness is not a lack of parallelism.

### Why the FA kernel is slow — what it is *not*

Eliminated by measurement, one by one:

- **Not bandwidth**: it reads 6.0 GB in 53 ms = 113 GB/s (18% of 640).
- **Not the q8_0 dequant**: f16 KV (no dequant) is 5× *slower* (74.6 ms vs
  15.0 at 70k), so q8_0 is the fast path and dequant is not the limiter.
- **Not MMA/coopmat selection**: disabling coopmat costs 3.8%; ns/Br sweeps
  lose or break; ns=4 (default) is optimal.
- **Not parallelism**: `split_k=1` is -51%, so the auto split is already filling
  the GPU.
- **Not the shmem staging**: PR #28507 is 0.
- **Not the version**: upstream `ce8caa6e6..58367713a` touches no Vulkan file.

The remaining suspect is the kernel's internal data movement (shared-memory
staging/swizzle/softmax) for hsk=hsv=256 at small n_rows, which cannot be seen
without GPU hardware counters (unavailable on RADV).

### The only remaining path

Rewrite the RDNA4 Vulkan `FLASH_ATTN_EXT` kernel (coopmat1, hsk=hsv=256,
n_rows<=8) to approach the 6 GB / 9.4 ms KV floor instead of 53 ms. This is a
GLSL shader + hardware-counter project, not a config change, and it is the only
thing that can move the number. Until it exists, the shipped configuration is at
the practical optimum of the available software: **28–29 t/s at 176k–234k**.

---

## 10. Revised maxima, blast radius, and recommendation (2026-09-21, session close)

### Decode maxima @183k

Measured step = 117.6 ms (FA 53.0 + weight GEMVs 45.6 + other 10.0 + 3 MTP
drafts 9.0), 3.31 tokens/step.

| scenario | step | decode |
|---|---:|---:|
| measured now | 117.6 ms | **28 t/s** |
| realistic: FA 3x faster, weights as measured | 82.3 ms | **40 t/s** |
| FA 3x + weights at the 640 GB/s floor | 72.4 ms | **46 t/s** |
| absolute ceiling: FA at KV floor (9.9 ms), weights at floor | 64.7 ms | **51 t/s** |

At ~70k the ceiling is ~1.2x higher (measured 38.4 there); at 262k ~48 t/s.

### Prefill maxima

Measured 512-token block at ~44k = 1.00 s: matmul 434 ms @55 TFLOP/s + FA
476 ms @18.6 TFLOP/s + other 90 ms.

| scenario | block | prefill |
|---|---:|---:|
| measured now | 1.00 s | 512 t/s (524 @72k) |
| matmul -> 70 TF, FA -> 48 TF | 615 ms | **~830 t/s** |
| both near peak | ~520 ms | ~980 t/s |

So ~1.6-2x prefill is the estimated ceiling. Estimates, not measurements.

### Blast radius

Editing llama.cpp affects **only the llama.cpp binary that is run**. It cannot
touch the OS, the kernel, Mesa/RADV, the desktop, or the RX 9060 XT (which runs
its own binaries for display and ollama). Build A (`mimir-general`) is a separate
binary and is untouched. The real risk is **silent numerical error** from a bad
shader tile/stride, not system damage; mitigations are the `AMD_RDNA4` gate,
`test-backend-ops` FA validation, perplexity checks, and keeping Build A as the
production fallback. Patching the driver would broaden the blast radius to the
whole system and other GPU — explicitly not proposed.

### Recommendation

1. Rebuild with `-DLLAMA_BUILD_TESTS=ON` and use
   `test-backend-ops perf -o FLASH_ATTN_EXT -b Vulkan1` to benchmark FA in
   isolation (seconds, not 15-minute depth runs); add a case matching
   hsk=hsv=256, n_rows=4, kv~183k, q8_0 KV. **This decides whether the kernel
   project is viable; without it, shader work is blind.**
2. Try to obtain GPU counters (`RADV_PROFILER`/`RADV_DEBUG`, or `umr` as root).
3. If a bottleneck is addressable, attempt targeted cm1 shader changes for
   hsk=256 / small n_rows, validated numerically and against the model.
4. Adopt the free win now: `--spec-type ngram-mod,draft-mtp` (never negative;
   +17% code @98k, 0% @234k).
5. If (1)/(2) show nothing addressable, stop: the current config is the practical
   optimum of the available software at 28 t/s deep, 38 t/s @70k, 524 t/s prefill
   @72k.

The editable surface is entirely userspace: `ggml-vulkan.cpp` (FA dispatch,
tuning, split_k) and `vulkan-shaders/flash_attn_cm1.comp`,
`flash_attn_base.glsl`, `flash_attn_dequant.glsl`.

## 11. Kernel work de-risking status

`test-backend-ops` is **not built** (`GGML_BUILD_TESTS=OFF`,
`LLAMA_BUILD_TESTS=OFF` in `~/git/llama.cpp-r9700/build/CMakeCache.txt`). The
source does contain FLASH_ATTN_EXT test + perf cases
(`tests/test-backend-ops.cpp:11636`, `:1559`), and its usage is
`test-backend-ops [perf] [-o OP] [-b BACKEND]`. Enabling tests needs a
reconfigure (`-DLLAMA_BUILD_TESTS=ON`, ideally in a separate build dir so the
working server build is untouched). **Not yet done — this is the first step of
the next session.**

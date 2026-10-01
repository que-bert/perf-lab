# R9700 long-context decode: the physical ceiling, and what is left on the table

Date: 2026-09-21. All numbers are for the fixed serving config (Qwen3.8-27B,
ctx 262144, `-fa on -ngl 99 --parallel 1`, MTP `draft-mtp`, R9700 alone) unless
stated. This document is the first-principles frame the experiments in
`FINDINGS.md` 2026-09-21 are read against.

## 1. The hardware, measured not assumed

| quantity | value | source |
|---|---:|---|
| memory bandwidth | **640.2 GB/s** | HIP read probe, `FINDINGS.md` 2026-09-19 |
| theoretical (256-bit GDDR6, 20 Gbps) | 640 GB/s | spec |
| fp16 matrix throughput | ~96 TFLOP/s dense | spec (not the limiter here) |
| LDS per workgroup | 64 KiB | `maxComputeSharedMemorySize` |
| VRAM | 32 GiB (34.2 GB) | rocm-smi |

## 2. The model is a hybrid, and that changes the arithmetic

`Qwen3.8-27B-Q6_K.gguf`: `qwen35`, 65 blocks, `full_attention_interval = 4`,
`attention.head_count = 24`, `head_count_kv = 4` (GQA 6:1), `key_length =
value_length = 256`.

Tensor census: **48 SSM (Mamba-style) blocks and 16 full-attention blocks**
(blk 3,7,11,…,63), plus blk 64 = the MTP head, which is itself a full-attention
layer. Only the 17 attention blocks have a KV cache; the 48 SSM blocks carry a
fixed O(1) recurrent state (`ssm.state_size 128`, `group_count 16`).

```
KV bytes/token (q8_0) = 17 layers x 4 kv_heads x 256 dim x 2 (K,V) x 1 B
                      = 34,816 B  (~34 KiB)
```

This is the single most important number in this document, and it is **4x
smaller than a dense 65-layer model would imply** — the hybrid architecture is
why a 262k-context Q6_K model fits and runs at all on one 32 GiB card.

## 3. Two ceilings

**Weight wall (one forward reads every weight once).**
```
tps_wall = BW / weight_bytes
Q6_K   : 640.2 / 22.88 GB = 27.98 t/s   (21.31 GiB)
Q4_K_M : 640.2 / 16.46 GB = 38.89 t/s   (15.33 GiB)
```
Both reproduce the recorded "decode ceilings" (28.0 and 38.9) exactly. An
unspeculated token also reads the KV once, so at depth the unspeculated wall is
strictly lower than 27.98.

**Step wall (MTP step = 1 verify forward + n_max draft forwards).** A step reads
the main weights once, the MTP weights ~n_max times (1 layer each, ~0.35 GB), and
the KV once per forward that attends over the full context.

## 4. What the measurements say

From `FINDINGS.md` 2026-09-20 (b10902, q8_0/q8_0, MTP n3, n_predict 256), a step
is `n_predict / steps` tokens where `steps = draft_n / 3`:

| filled ctx | decode t/s | drafted | steps | tokens/step | **step ms** |
|---:|---:|---:|---:|---:|---:|
| 42 | 60.96 | 203 | 67.7 | 3.78 | **62.0** |
| 8.9k | 46.78 | 249 | 83.0 | 3.08 | **65.9** |
| 35.9k | 41.69 | 252 | 84.0 | 3.05 | **73.0** |
| 72.5k | 37.97 | 238 | 79.3 | 3.23 | **85.0** |
| 182.9k | 28.04 | 232 | 77.3 | 3.31 | **118.1** |

Step time is **linear in filled context**:
```
step_ms(ctx) = 62.0 + 0.3068e-3 * ctx        (R^2 ~ 1.000)
```
Residuals: 65.9 vs 64.7, 73.0 vs 73.0, 85.0 vs 84.2, 118.1 vs 118.1. A two-term
model — a context-independent term and a context-linear term — describes the
whole decay. (My own 2026-09-21 baseline reproduces 28.31 t/s at 182,889.)

**Decomposing the slope.** The context-linear term at 182,889 tokens is 56.1 ms.
The pure KV bandwidth cost is `34,816 B x 182,889 / 640.2 GB/s = 9.95 ms` for
**one** forward over the context. So:

- If only the verify pass read KV: observed slope is **5.6x** the floor.
- If all `n_max + 1 = 4` forwards read KV: floor = 39.8 ms, observed is **1.41x**.

The second is the better fit and the more plausible mechanism (each MTP draft
token is its own single-row forward that attends over the full KV). The
conclusion is the same either way: **the context term is KV-bandwidth work that
is being done 1.4-5.6x less efficiently than the memory system allows.** That
gap is the only meaningful headroom in the whole problem.

## 5. Measurement method (2026-09-21): the "6% noise floor" was contamination

The 2026-09-20 entries recorded a ~6% run-to-run floor at 176k and treated it as
unresolvable. That is wrong, and it is worth stating why because it changes what
can be concluded.

`harness/depth.py` warms the prompt once with `cache_prompt` and then takes R
reps over the **same filled context**, so a rep costs only the decode. Measured
within-server spread over 3 steady reps:

| config | ctx | spread |
|---|---:|---:|
| paragraph, q8_0/q8_0, n3 | 70k | **0.21%** |
| paragraph, q8_0/q8_0, n3 | 176k | **0.10%** |
| code corpus, q8_0/q8_0, n3 | 98k | **0.10%** |
| code corpus, q8_0/q8_0, n3 | 234k | **0.20%** |

And server-to-server, once the confounders are removed, reproduces to ~0.4%:

| run | 70k decode |
|---|---:|
| baseline 13:03 | 38.81 |
| baseline-q8b 14:04 (contaminated) | **35.88** |
| shmem-ON r1 / OFF r1 | 38.487 / 38.505 |
| shmem-ON r2 / OFF r2 | 38.365 / 38.454 |
| clean-base 15:13 | 38.439 |

The two outliers (35.88, and a 25.8 earlier) were a **`go test ./...` from a
sibling session** and an **ollama embedding runner actively using the R9700**.
`harness/gpu_guard.sh` now refuses to start a measurement unless
`mem_busy_percent <= 12` and 1-min load `< 8` for three consecutive samples. With
that gate, effects of 1-2% are readable, and a single-rep protocol would have
recorded the contamination as a build result.

## 6. What the clean matrix measured (2026-09-21)

Fixed config: Q6_K, ctx 262144, q8_0/q8_0, `-fa on`, MTP, mmproj, R9700.
"paragraph" = the FINDINGS protocol (one paragraph repeated — n-gram's best
case); "code" = a slice of `ggml-vulkan.cpp` (diverse text, the honest case).

| config | paragraph 70k | code 98k | code 234k |
|---|---:|---:|---:|
| **base** (MTP n3) | 38.44 | 41.56 | 28.24 |
| n_max 4 | 38.71 (+0.7%) | 43.24 (+4.1%) | 27.31 (**-3.3%**) |
| n_max 5 | 22.81 (**-41%**) | — | — |
| `ngram-mod,draft-mtp` | 41.94 (+9.1%) | 48.69 (+17.2%) | 28.19 (**0%**) |
| `GGML_VK_DISABLE_COOPMAT=1` | 36.97 (-3.8%) | — | — |
| PR #28507 shmem staging | 38.49 (0%) | — | — |
| no speculation (control) | 17.46 (-55%) | — | — |

Readings:

- **The n-gram chain is the only large win, and it is depth-limited.** +17% at
  98k on code, but at 234k the acceptance is *identical to base* (186/205): the
  `n_min=48` match threshold finds no ≥48-token continuation there, so it falls
  back to MTP entirely. It never loses, but it stops helping before the deepest
  operating point. This is the experiment the tuned `n_min=8` run tests.
- **n_max is depth-dependent.** 4 is better at 98k (+4.1%) and worse at 234k
  (-3.3%); 5 is catastrophic everywhere (-41%). n3 — the shipped default — is the
  right choice at the depth this configuration actually operates at.
- **The kernel patches do nothing here.** PR #28507's shmem staging measures
  38.487 vs 38.505 (noise). Disabling coopmat costs 3.8% — the coopmat1 path is
  already the better one. There is no available upstream or community kernel
  change that moves this card at depth.
- **AMDVLK is not an option.** 2025.Q2.1 extracted locally returns
  `VK_ERROR_INCOMPATIBLE_DRIVER` on gfx1201; RDNA4 is unsupported.

## 7. Where the headroom is, and where it is not

- **Weight precision.** Q6_K -> Q4_K_M removes 6.4 GB of the 22.88 GB weight
  read, i.e. **6.4/640.2 = 10.0 ms per step**. At the 182.9k step of 118.1 ms
  that is **+9.2%** even if acceptance is unchanged. It is not a step change; the
  weight term is only 30% of a deep step. (At short context it is the dominant
  term, which is why Q4_K_M looks like +38% there.)
- **KV precision.** q4_0 V halves V bytes (3.19 -> 1.59 GB at 183k), saving
  1.6 GB x n_forwards of traffic: ~2.5 ms/step, ~2%, *and* `FINDINGS` records
  q4_0 V as decode-free on this model. Small but free.
- **Speculation.** Each accepted draft token is ~free in weight reads but not in
  KV reads. The value of raising n_max grows with context only if the extra draft
  forward costs less than the token it saves. This is an empirical optimum, not
  a monotone knob — hence the n_max sweep.
- **Kernel efficiency (B3 / shmem staging).** The 1.4-5.6x gap on the context
  term is pure implementation. Nothing in the silicon prevents 1.0x.
- **Second GPU.** Layer split cannot help single-stream latency: a token must
  traverse every layer, so total time is the *sum* of per-GPU layer times, and
  the R9700 is the faster card. The 9060 XT (16 GB, ~320 GB/s) can only add
  capacity, never subtract time, for a 22.9 GB model that already fits.
- **ROCm.** Independent measurement on the same card and commit (NikoCloud,
  `docs/05`): "Vulkan decodes faster at both depths (37.6/34.5 vs ROCm
  24.5/20.3)", and ROCm page-faults at 32768 depth. ROCm wins shallow prefill
  only. Not the lever for this problem.

## 8. The ceiling this implies

A perfect implementation (every read at 640.2 GB/s, zero attention compute)
gives a step floor at 182.9k of:
```
Q6_K   : (22.88 + 4 x 6.37) GB / 640.2 = 77.5 ms  -> 3.31 tok / 77.5ms = 42.7 t/s
```
against the 28.3 t/s measured. **~1.5x is available on the same weights, same
precision, same context, from kernel work alone.** With Q4_K_M weights the floor
becomes `(16.46 + 25.5)/640.2 = 65.5 ms` -> ~50 t/s, but that is a precision
trade the plan forbids; it is recorded as an option, not a result.

The honest statement of the goal "faster decode at longer contexts" is therefore:
close the gap between 28.3 t/s and the ~42 t/s bandwidth floor at 183k, with no
change to weight or KV precision.

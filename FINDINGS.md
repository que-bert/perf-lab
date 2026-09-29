# Findings

Measured facts, each citing the ledger rows that produced them. Judgment — why an
avenue was rejected, what not to reopen — lives in MuninnDB, not here.

Reproduce any of these with `jq` over `results/ledger.jsonl`.

## Only two KV cache types have optimized kernels

On gfx1201 with llama.cpp b10082, `pp2048` by cache type:

| K | V | pp2048 t/s | run_id |
|---|---|---|---|
| `q4_0` | `q4_0` | **710.84** | `r-427fe526` |
| `q8_0` | `q8_0` | **684.02** | `r-f688609f` |
| `q8_0` | `q4_0` | **83.85** | `r-2a4be1c1` |
| `q4_1` | `q4_1` | **83.19** | `r-ee5bf3fe` |
| `q8_0` | `q5_1` | **56.12** | `r-6384e54d` |
| `q5_1` | `q5_1` | **54.96** | `r-a53161e1` |

`q5_1`, `q5_0` and `q4_1` — and any mismatched K/V pair — fall off the flash-attention
path and run **8–13× slower at prefill**. Decode is unaffected, which is why short-prompt
testing does not reveal it.

> Two of these fallback numbers did not reproduce on 2026-08-16. That was a build
> swap, resolved 2026-08-17 — see "RESOLVED" below. On the binary now in use the
> separation is 7.8×, and these stage-1 values stand.

## Fallback speeds are not reproducible, and the fingerprint does not explain it

Re-measured 2026-08-16 (tag `rebase-20260816T052925Z`), 5 reps per canary on an idle
card, against the single readings taken 2026-08-15:

| canary | K/V | 2026-08-15 (n=1) | 2026-08-16 (median of 5) | change | run_id |
|---|---|---|---|---|---|
| fast-q4 | `q4_0`/`q4_0` | 724.22 | **705.62** | −2.6% | `r-33f66e8a` |
| fast-q8 | `q8_0`/`q8_0` | 704.91 | **704.00** | −0.1% | `r-74b10b50` |
| slow-q5_1 | `q8_0`/`q5_1` | 59.34 | **59.49** | +0.3% | `r-c64ff7ba` |
| slow-q4_1 | `q4_1`/`q4_1` | 89.92 | **150.74** | **+67.6%** | `r-6eba5f78` |
| slow-mixed | `q8_0`/`q4_0` | 91.17 | **157.21** | **+72.4%** | `r-007dbbba` |

The fingerprint is byte-identical across both sets: llama.cpp `fb0e6b621`, Mesa
`26.0.3-1ubuntu1`, kernel `7.0.0-29-generic`, ROCm `7.2.53211-97f5574fe2`, model
sha256 `562fbf76…`. Nothing the ledger records changed.

Consequences:

- The fast-vs-fallback separation measures **4.7×** today, not the 8.5× recorded on
  2026-08-15. The fallback is still unmistakable, but the margin is half what the
  design assumed.
- Both fast-path canaries and one of the three slow-path canaries reproduced within
  noise. Whatever moved is selective to `q4_1/q4_1` and `q8_0/q4_0`.

**No row from either day carries `env`.** The design called for thermal and load
covariates "so drift is visible rather than mysterious", and stage 1 never
implemented them — 0 of 86 rows had any. That is precisely why this cannot be
settled from the data on hand. `emit_row.py` now records GPU temperature, clock,
1-minute load average, and VRAM in use before the run starts.

Leading hypothesis was that the 2026-08-15 readings were taken while ollama's
`llama-server` was resident on this card, which was independently confirmed on
2026-08-16, and that kernel selection for the fallback paths depends on free VRAM.

**Tested 2026-08-16 and REFUTED.** Tag `occupant-6g`, 3 reps per canary, with a
deliberate occupant — `gemma-4-E4B-it-Q4_K_M` served at ctx 131072, pinned to the
target card with `HIP_VISIBLE_DEVICES`, holding 5583 MiB (`env.vram_used_mib_before`
on every row confirms it). The occupant was the only other KFD process on the card.

| canary | 2026-08-15 | clean 2026-08-16 | occupied (5583 MiB) | vs clean |
|---|---|---|---|---|
| fast-q4 | 724.22 | 705.62 | 704.11 | −0.2% |
| fast-q8 | 704.91 | 704.00 | 702.10 | −0.3% |
| slow-q5_1 | 59.34 | 59.49 | **52.82** | **−11.2%** |
| slow-q4_1 | 89.92 | 150.74 | 150.51 | −0.2% |
| slow-mixed | 91.17 | 157.21 | 146.29 | −6.9% |

`slow-q4_1` is the config the hypothesis was built to explain, and it did not move:
150.51 against 150.74 clean, where reproducing 2026-08-15 required ≈89.9. Occupancy
of 5.6 GB does not switch the fallback paths.

What occupancy *does* do is add noise to the slow paths, and it does so on the wrong
canary. `slow-q5_1` — the one canary that did **not** move between the two days — is
the only one to breach its band here, and its three reps spread 44.03–57.16, a 30%
range against 1.55% on an idle card. `slow-mixed` moved −6.9% with a similar spread.
Both fast canaries held to within 0.3%. So the slow paths are simply more sensitive
to a busy card, which is the opposite shape from a selective +70% step on exactly two
configs.

The unattended nightly the next morning confirms both halves of that reading.
`nightly-20260817T100335Z`, idle card (`vram_used_mib_before` 57 on every row):

| canary | expect | nightly 08-17 | delta |
|---|---|---|---|
| fast-q4 | 705.62 | 707.63 | +0.28% |
| fast-q8 | 704.00 | 703.69 | −0.04% |
| slow-q5_1 | 59.49 | 60.54 | +1.77% |
| slow-q4_1 | 150.74 | 155.22 | +2.97% |
| slow-mixed | 157.21 | 158.03 | +0.52% |

`slow-q5_1` came back to 60.54 with reps spanning 60.0–60.8, so its −11.2% under
occupancy was contention and nothing else. And `slow-q4_1`/`slow-mixed` held their
high values on a fifth independent idle-card batch across two days. The 2026-08-15
readings of ≈90 are now the lone outlier in the series, which shifts suspicion from
the stack toward how those particular readings were taken.

Caveats: one occupancy level, n=3. A threshold effect at some larger footprint is not
excluded, and ollama's actual 2026-08-15 footprint was never recorded. But the
straightforward version of this hypothesis is dead, and the shift is still unexplained.

Note for whoever runs a contended batch next: `check.py` has no notion of a tainted
run. It read `occupant-6g` as the latest batch and returned `slow-q5_1` as
`breach: true, verdict: "slow"`. It did not escalate only because `sustained` requires
two consecutive breaching runs, and the next clean nightly cleared it to `ok`. Tag such
batches, and do not let one stand as the most recent run going into a nightly.

## RESOLVED 2026-08-17: the shift was a build swap, not the stack

The two `llama-bench` binaries perf-lab has access to were run head to head on
2026-08-17 — same model, same GPU, same args (`-p 2048 -n 64 -fa on -r 3 -ngl 99 -t 8`),
pinned to the target card. Both report upstream `fb0e6b621`; they differ only in
toolchain.

| canary | 2026-08-15 recorded | prebuilt GCC 11.4 | 2026-08-16 recorded | local GCC 15.2 |
|---|---|---|---|---|
| fast-q4 | 724.22 | 723.37 ± 1.38 | 705.62 | 705.85 ± 1.63 |
| slow-q5_1 | 59.34 | 60.84 ± 0.32 | 59.49 | 60.49 ± 0.34 |
| slow-q4_1 | 89.92 | **89.95 ± 0.36** | 150.74 | **155.10 ± 1.20** |
| slow-mixed | 91.17 | **92.02 ± 0.30** | 157.21 | **158.81 ± 0.54** |

Every 2026-08-15 value reproduces on the prebuilt; every 2026-08-16 value reproduces on
the local build. **The canary harness was switched from the ROCm prebuilt's `llama-bench`
to the local GCC 15 build's between those two days**, and nothing recorded it, because
`fp.build.binary_sha256` did not exist until stage 3 landed on 2026-08-16 and both builds
report the same upstream sha. The ledger now shows the split plainly: every canary row
tagged `nightly-20260817…` or `occupant-6g` carries `toolchain: GNU 15.2.0`, while the
`ppl-cmp` rows — which go through `llama-server` — carry `GNU 11.4.0`.

`slow-q5_1` is the control that makes this conclusive: it is the one canary that never
moved between the two days, and it is also the one that measures the same on both builds
(60.84 vs 60.49, 0.6% apart). The build swap only moves the kernels it generates
differently, and `q5_1` is not one of them.

So the differences are real codegen differences, not noise: the local GCC 15 build is
2.5% **slower** on the optimized path and about **1.7× faster** on the `q4_1`/`q4_0`-V
fallbacks.

**This invalidates the current canary bands.** They were derived on the local build,
but everything that actually serves — tracked configs, `quality.py`, `verify.py` — runs
on the prebuilt. The tripwire is watching a binary production never executes, and the
fast-vs-fallback separation it is calibrated against is 4.5× on the local build versus
about 8× on the prebuilt. Re-derive the bands against whichever binary is chosen, and
record `binary_sha256` on every row from now on.

## Speculative decoding is not output-identical here

Measured 2026-08-16 with `harness/verify.py`, greedy sampling (temperature 0), 96
predicted tokens, prompt corpus `db3bc6c727a3…`, Qwen3.8-27B-Q6_K on the b10082 ROCm
prebuilt (`binary_sha256 068f5c54…`):

| comparison | token-identical | first divergence |
|---|---|---|
| `draft-mtp n_max 4` vs **itself** | **yes** | — |
| `draft-mtp n_max 4` vs **no speculation** | **no** | token 57 |
| `q4_0` KV vs `q8_0` KV | no | token 38 |

The control is what makes the other two rows mean anything: the same config run twice
through two freshly spawned servers produced byte-identical token ids, so this harness
is deterministic and the divergences are attributable rather than noise.

That **falsifies the assumption that speculative decoding can be validated by token
identity.** In theory it is exact — drafts are accepted only when they match what the
target model would have produced — so `--spec-type`/`--spec-draft-n-max` should be a
free speed knob with no output consequence. On this stack it is not.

### The divergence is a tie-break, not a regression

Every divergence observed lands where the model had no real preference. Top-2
probabilities at the divergence index, and where the other config's token ranked:

| comparison | prompt | top-1 | top-2 | ratio | rank of other's token |
|---|---|---|---|---|---|
| nospec/mtp4 | technical | 0.3690 | 0.3496 | 1.06 | 2nd |
| nospec/mtp2 | narrative | 0.5149 | 0.4850 | 1.06 | 2nd |
| nospec/mtp4 | list | 0.4971 | 0.4715 | 1.05 | 2nd |
| nospec/mtp2 | code | 0.3219 | 0.3216 | **1.00** | 2nd |
| nospec/mtp4 | README | — | — | 1.03 | 2nd |

Six for six, the alternative was the runner-up at a probability ratio under 1.07. Not
once did a speculative config pick something the model ranked further down. Two of five
prompts produced no divergence at all, and divergence positions scatter — 14, 15, 23,
57, 92 — rather than clustering early.

**`n_max 2` and `n_max 4` also diverge from each other**, which rules out
"speculation versus none" as the cause. What changes between them is the shape of the
batched forward pass, and floating-point addition is not associative: reduction order
perturbs logits in the low bits, and only a near-tie can be flipped by that.

Contrast KV quantization, which is lossy for real:

| comparison | first divergence | ratio | verdict |
|---|---|---|---|
| `q4_0` vs `q8_0` KV | token 38 | **1.70** | material |

The quantized run also picked the runner-up, but one the reference rated 1.7× less
likely — a genuine difference of preference, not a coin-flip. `verify.py --tie-ratio`
defaults to 1.2, between the two clusters.

**Consequence:** strict token identity is the wrong gate on this hardware; it fails
changes that are numerically equivalent. `verify.py` reports `equivalent` alongside
`token_identical` and only fails on divergence the reference model actually cared about.

### llama-server does not report distributions for drafted tokens

Measured while building the gate: with `n_probs: 5`, a non-speculative run returns
`top_logprobs` for **96 of 96** tokens; a speculative run returns them for **1–2 of 96**.
Accepted draft tokens carry no distribution. Any tie test must therefore take its
reference from the non-speculative side — scoring against the speculative one makes
every divergence look unjudgeable, which is not the same as it being real.

Reproduce:

```
harness/verify.py configs/tracked.yaml baseline no-spec --n-predict 96
```

*(These rows cite a reproduction command rather than a run_id: `verify.py` does not yet
append to the ledger. It should.)*

## q4_0 KV costs no measurable perplexity against q8_0 — at 4K context

Measured 2026-08-16 with `harness/quality.py` (`llama-perplexity`, 24 chunks,
ctx 4096, held-out corpus `f79059082bb3…`, Qwen3.8-27B-Q6_K):

| KV cache | perplexity | run_id |
|---|---|---|
| `q4_0` / `q4_0` | 3.0456 ± 0.06309 | `r-5519990a` |
| `q8_0` / `q8_0` | 3.0420 ± 0.06299 | `r-c6d395ac` |

The gap is **+0.118%, or 0.06 standard errors** — indistinguishable. On this
corpus and at this context length, halving the KV cache costs nothing readable.

**This does not clear `q4_0` for the shipped config.** The measurement is at
ctx 4096; the config that matters runs at 262144, and quantization error
accumulates with depth. The existing note that "wide aggregation over scattered
values fails at long context regardless of cache type" is untouched by this.
What it does establish is that the loss is not gross: `q4_0` is not damaging
the model in a way that shows up immediately, which is a different and weaker
claim than "q4_0 is safe at 262K".

Perplexity is also a weak proxy for the failure modes that matter at long
context — retrieval of a specific fact from deep in the window is not what a
next-token likelihood average measures.

## Spread on an idle card is under 2%, not 7%

Robust (MAD-scaled) spread over 5 reps, tag `rebase-20260816T052925Z`:
`fast-q8` 0.02%, `fast-q4` 0.67%, `slow-mixed` 0.87%, `slow-q4_1` 0.91%,
`slow-q5_1` 1.55%.

Raw standard deviation runs far higher — 15.3% for `slow-q4_1`, 6.5% for
`slow-q5_1`, 3.2% for `fast-q4` — because single reps get contaminated. Rep 1 of
each canary reloads a 21 GiB model cold; one mid-run rep dropped `tg` from 21.6 to
7.98 t/s with no other change. Sizing bands from raw sd inflated `slow-q4_1`'s
tolerance to 46%, wide enough to miss most of what a canary is for.

## Upstream is not monotonically improving
- build `86b94708f` — pp_deep@d16384 = **517.34** t/s  (`r-b4688392`)
- build `fb0e6b621` — pp_deep@d16384 = **786.52** t/s  (`r-a97f17c0`)
- build `fb0e6b621` — pp_deep@d16384 = **840.85** t/s  (`r-7ecf3046`)
- build `9d57ce456` — pp_deep@d16384 = **649.98** t/s  (`r-f10b6edf`)
- build `fb0e6b621` — pp_deep@d16384 = **838.94** t/s  (`r-a1df7ce9`)

b10438 (master, 2026-08-15) is **22.5% slower** at deep prefill than the pinned b10082.

## MTP draft depth has an optimum, and ngram-simple hurts
- n_max=2: 38.21 tok/s, acceptance 0.6423  (`r-93594161`)
- n_max=3: 38.99 tok/s, acceptance 0.6519  (`r-67e4c5a3`)
- n_max=4: 38.23 tok/s, acceptance 0.6539  (`r-ef581561`)
- n_max=5: 35.73 tok/s, acceptance 0.5604  (`r-13c3e979`)
- n_max=6: 30.39 tok/s, acceptance 0.5000  (`r-ea640b54`)

- no speculation: 21.80 tok/s  (`r-3adbab6b`)
- draft-mtp n=4: 46.15 tok/s, acceptance 0.7708  (`r-3e958f9f`)
- draft-mtp,ngram-simple n=4: 45.12 tok/s, acceptance 0.6715  (`r-d322912c`)
- draft-mtp,ngram-mod n=4: 45.46 tok/s, acceptance 0.7556  (`r-dca743d9`)

Adding `ngram-simple` to `draft-mtp` lowers acceptance and throughput. On a copy-heavy
193K-token task — ngram's best case — MTP alone reached **100% acceptance** while adding
ngram dropped it to 92.6% and took 25 s longer.
- draft-mtp alone: acceptance 1.00000, wall 542.6s  (`r-49df4a7e`)
- draft-mtp,ngram-simple: acceptance 0.92587, wall 567.7s  (`r-1a0394cc`)

## The local GCC 15 build is broken, and Vulkan now beats ROCm at depth

Reproduced 2026-08-17. `~/git/llama.cpp-b10082/build/bin` (local, GCC 15.2.0,
`GGML_HIP=ON`, gfx1201) segfaults:

| binary | gemma-4-E4B-it-Q4_K_M | Qwen3.8-27B-Q6_K |
|---|---|---|
| `llama-server` | SIGSEGV (`-ngl 0`, CPU) | SIGSEGV (GPU) |
| `llama-cli` | SIGSEGV | not tested |
| `llama-bench` | SIGSEGV | works |

`llama-bench` on Qwen3.8-27B is the *only* combination that works, which is why
stage 1 never noticed. The backtrace puts every crash in `ggml_cuda_op_scale`
(`libggml-hip.so`) calling into `/opt/rocm-7.2.4/lib/libamdhip64.so.7`, reached from
the warm-up `llama_decode`. `GGML_CUDA_DISABLE_GRAPHS=1` does not help. The ROCm
prebuilt is the same commit `fb0e6b621` built with GCC 11.4.0 and does not crash, so
this is the local toolchain, not upstream.

**The rebuild blocker is fixed.** ROCm's `lld` failed on `libxml2.so.2` (this system
ships `.so.16`). `libxml2.so.2.9.14` plus `libicuuc.so.74`/`libicudata.so.74` were
copied out of the mesa/gaming snaps into `~/.local/rocm-compat/lib`; with that on
`LD_LIBRARY_PATH`, `hipcc` compiles and links gfx1201 device code again. A from-source
b10472 build now succeeds using ROCm's own clang 22 as host compiler.

Three candidates measured on Qwen3.8-27B-Q6_K, R9700, `-p 2048 -n 64 -fa on -ngl 99
-t 8 -r 3`:

| build | pp2048 | tg64 | pp2048@d16384 | tg64@d16384 | q4_1/q4_1 pp2048 |
|---|---|---|---|---|---|
| b10472 Vulkan prebuilt | **894.06** | 17.36 | **674.82** | **19.14** | **891.06** |
| b10082 ROCm prebuilt | 717.27 | **22.68** | 615.89 | 18.62 | 91.53 |
| b10472 ROCm from source | 679.22 | 22.07 | 387.96 ±90.8 | 16.78 | 122.78 |

Two conclusions. First, **Vulkan wins both prefill and decode at depth 16384**,
reversing the 2026-07-22 choice of ROCm — upstream Vulkan has improved since b10082.
ROCm still wins `tg64` at depth 0. Second, the from-source b10472 ROCm build is the
worst of the three at depth and by far the noisiest, which independently re-confirms
the existing b10082 pin.

**Vulkan has no fallback cliff at all.** `q4_1/q4_1` measures 891.06 against
`q4_0/q4_0`'s 894.06 — a 0.3% difference, where ROCm collapses 7.8×. The entire
premise of the canary set, that KV type selects between a fast kernel and a fallback,
is ROCm-specific. If serving moves to Vulkan, these five canaries stop measuring
anything and the set needs redesigning. That is a decision, not a defect, and it is
left open deliberately.

`b10472-vulkan` is installed at `~/llama.cpp/b10472-vulkan` with its own
`PROVENANCE.txt`, verified serving Qwen3.8-27B at ctx 262144 with `--spec-type
draft-mtp --spec-draft-n-max 4` (draft acceptance 0.719) and serving gemma-4-E4B.

`PERF_LAB_BIN` was switched from the broken local build to `~/llama.cpp/b10082-rocm`
on 2026-08-18, so canaries and served configs finally run on one binary, and the bands
were re-derived under tag `rebase-20260818T033927Z`. Fast-vs-fallback separation is
now 7.8× (716.4 vs 91.45) rather than the 4.5× the GCC 15 build reported.

## Decode with MTP: the number that actually decides the build

`llama-bench` has no speculative-decoding flags, so every `tg64` figure above is
*unspeculated* and not comparable to the 46.15 t/s the shipped config was found at.
Measured through `llama-server` instead, ctx 262144, `-fa on`, `--spec-type draft-mtp
--spec-draft-n-max 4`, `n_predict 256`, temperature 0, identical prompt, 2 reps:

| build | K/V | decode t/s | acceptance | VRAM |
|---|---|---|---|---|
| b10472 Vulkan | `q8_0`/`q4_0` | **50.88 / 53.25** | 0.665 | 33.6 GB |
| b10472 Vulkan | `q4_0`/`q4_0` | **48.64 / 51.19** | 0.640 | 31.8 GB |
| b10472 Vulkan | `q4_0`/`q4_0` (2nd) | 50.68 / 44.25 | 0.503 | 31.8 GB |
| b10082 ROCm | `q4_0`/`q4_0` | 36.57 / 36.85 | 0.527 | — |
| b10472 Vulkan | `q8_0`/`q8_0` | 22.86 / 20.46 | 0.503 | — |
| b10082 ROCm | `q8_0`/`q8_0` | **fails to load** | — | OOM |

Three things follow.

**Vulkan is ~35% faster than ROCm on the workload that matters.** 48-51 t/s against
36.6-36.9 on the same prompt. This is the decisive comparison, and it agrees with the
depth-16384 `llama-bench` result rather than the depth-0 one — unspeculated `tg64` at
depth 0 was the single measurement that favoured ROCm, and it is the least
representative of how the model is actually served.

**Decode rate with speculation is not a fixed number.** Draft acceptance ranged
0.503-0.665 across runs at temperature 0, because acceptance depends on the content
generated, and decode scales with it. Observed spread on the same config was 44.25 to
51.19 t/s. Quote this as a range, not a point.

**`q8_0` KV is not a viable operating point at 262144.** On Vulkan it costs more than
half the decode rate (20-23 t/s); on ROCm the context will not allocate at all, failing
on a 3.1 GB recurrent-state buffer. The mismatched `q8_0`K/`q4_0`V pair — the prefill
trap on ROCm — is the *fastest* option on Vulkan at 50.9-53.3 t/s, but it sits at
33.6 of 34.2 GB, 98% of the card, with no headroom for a second process.

## n_max 4 is the optimum, and q8_0K/q4_0V costs no measurable quality

Swept `--spec-draft-n-max` on the b10472 Vulkan build, `q8_0` K / `q4_0` V, ctx 262144,
`n_predict 512`, temperature 0, same prompt, 3 reps each:

| n_max | decode t/s (3 reps) | median | acceptance | mean draft len |
|---|---|---|---|---|
| 2 | 42.64 / 45.50 / 45.32 | 45.32 | 0.720 | 2.44 |
| 3 | 46.80 / 46.67 / 46.62 | 46.67 | 0.618 | 2.85 |
| **4** | 49.05 / 52.55 / 52.46 | **52.46** | 0.661 | 3.64 |
| 5 | 44.10 / 41.00 / 39.64 | 41.00 | 0.455 | 3.25 |
| 6 | 40.63 / 73.16 / 35.02 | 40.63 | 0.369 | 3.21 |
| 8 | 13.08 / 27.12 / 29.90 | 27.12 | 0.970 | 8.66 |

**4 is the optimum**, 12% above n_max 3 and 28% above n_max 5. This closes the open
question of whether `n_max=6` stays `material` once throughput used more tokens: it does
not — 6 is 23% *worse* than 4, and the backfilled claim that acceptance collapses past 4
holds (0.661 at n=4 against 0.369 at n=6).

Two anomalies not to read past. n_max 6 produced a 73.16 rep against 40.63 and 35.02 —
a 2× spread that no other setting showed. And n_max 8 reports the *highest* acceptance
on the board, 0.970 at mean draft length 8.66, while delivering the *lowest* throughput,
27.12. High acceptance with low throughput means the draft is costing more than it saves;
the acceptance statistic alone is not a proxy for speed and should never be tuned on.

Perplexity, held-out corpus, ctx 4096, 32 chunks, measured on the Vulkan build:

| config | K/V | ppl | stderr |
|---|---|---|---|
| baseline | `q4_0`/`q4_0` | 3.0459 | 0.06321 |
| kv-q8 | `q8_0`/`q8_0` | 3.0486 | 0.06328 |
| kv-mixed | `q8_0`/`q4_0` | 3.0498 | 0.06328 |

All three sit within 0.004 of each other against a stderr of 0.063 — about 0.06 standard
errors apart, indistinguishable. The nominally *most* precise cache (`q8_0`/`q8_0`) scores
nominally worse than the least, which is the clearest possible sign that the spread is
noise. **The same ctx-4096 caveat as before applies and is the real open gap**: the config
serves at 262144, where quantization error accumulates and perplexity is a weak proxy.

`verify.py kv-mixed no-spec --n-predict 96` returned `token_identical: true`,
`equivalent: true`, zero divergences — speculation at n_max 4 changed nothing at all here,
stronger than the ROCm result where it diverged at token 57 on a near-tie. Limitation: one
prompt, 96 tokens; the corpus at `$PERF_LAB_PROMPT` holds a single prompt.

**Gap: perf-lab cannot record rows from the Vulkan build.** All three perplexity runs
above printed their numbers and then refused to write, with `emit_row: could not determine
llama.cpp build SHA from any binary in ~/llama.cpp/b10472-vulkan`. If serving moves to
Vulkan, `emit_row.py`'s fingerprint probe needs to learn this build layout first.

## Qwen3.8-27B is multimodal; the vision projector is a separate download

Corrects an error made 2026-08-18. The main GGUF carries **zero vision tensors**, and
that was read as "this model cannot do images". It only means the vision encoder is not
in that file. The chat template in the same GGUF emits
`<|vision_start|><|image_pad|><|vision_end|>` and `<|video_pad|>`, and
`unsloth/Qwen3.8-27B-GGUF` publishes `mmproj-BF16.gguf` and `mmproj-F16.gguf`. Absence
from the local directory is not absence upstream — check the source repo.

`mmproj-F16.gguf` (885 MB) fetched to the model directory and verified on the b10472
Vulkan build: the server logs `loaded multimodal model`, and it described a synthetic
64x64 half-blue/half-yellow test image as "Blue on the left, yellow on the right." A
1024x1024 image is accepted at 1085 prompt tokens.

Full operating table, Qwen3.8-27B-Q6_K + `mmproj-F16`, R9700, `-fa on`, `--spec-type
draft-mtp --spec-draft-n-max 4`, decode measured at `n_predict` 256-384:

| K/V | ctx | VRAM (of 34.2 GB) | decode t/s | 1024x1024 image |
|---|---|---|---|---|
| `q8_0`/`q4_0` | 262144 | 33.6 GB (98%) | 52.55 | OK |
| `q4_0`/`q4_0` | 262144 | 32.7 GB (96%) | 49.41 | not tested |
| `q8_0`/`q4_0` | 131072 | 30.3 GB (89%) | ~52.8 | OK |
| `q4_0`/`q4_0` | 131072 | 29.2 GB (85%) | 45.76 | not tested |

`q8_0`K/`q4_0`V is faster than `q4_0`/`q4_0` at both context lengths, for about 1 GB
more VRAM, and perplexity cannot separate them. The projector costs ~0.9 GB.

**The headroom worry was overstated.** A 1024x1024 image processed cleanly at 98% VRAM
with no measurable increase during encoding — VRAM read 33.6 GB before and after, and
the server stayed up. The remaining argument for 131072 is not image safety but leaving
~3.9 GB for anything else that wants the card.

## 2026-08-23: the fingerprint never covered the compute backend

Teaching `emit_row.py` to fingerprint the Vulkan build turned up three defects, only
the first of which was known. All three are fixed; `harness/test_emit_row.py` holds a
case for each, and each case was confirmed to fail against the pre-fix code.

**1. The `--version` banner changed shape and the parser did not.** Upstream moved from
`version: 10082 (fb0e6b621)` to `version: 0.1.1-dev (build 10472, commit 60eeeb608)`
between the two builds on this machine. The regex matched only the first form, so
`emit_row` aborted with *could not determine llama.cpp build SHA* on every Vulkan run.
That is why the b10472 numbers in this file were never in the ledger — the refusal was
correct behaviour on an unreadable banner, but the banner was readable and the parser
was not looking for it. Both forms are now accepted; anything else still aborts.

**2. The compute backend was outside the digest — on both builds, always.**
`fp.build.binary_sha256` globbed `libggml*.so.[0-9]*`, which matches `libggml.so` and
`libggml-base.so` and nothing else. The backend carries no version suffix:

```
libggml-hip.so       (b10082-rocm)      never hashed
libggml-vulkan.so    (b10472-vulkan)    never hashed
```

So the one file implementing the kernels this repo exists to watch was not in the
fingerprint. A backend-only rebuild — swap `libggml-hip.so`, leave everything else —
would have produced an identical digest and an unexplained throughput move, which is
precisely the failure of 2026-08-15 that `binary_sha256` was added to prevent. The glob
is now `libggml*.so*`, and symlink aliases collapse to one entry so the digest depends
on the code rather than on how many aliases the packager shipped.

**This moves the digest for builds that did not change.** Same ROCm prebuilt, same
bytes:

| | `binary_sha256` |
|---|---|
| rows through `nightly-20260822T181827Z` | `9f1f4f952a3aa6c8aa95bd5a756fc2d527a08216e5a4b754020149b06f2d9663` |
| rows from 2026-08-23 | `3167c12ef7691864194c13008a02adc708d2d144f0623a2996caa05710021eda` |

**That discontinuity is a harness change, not a build swap.** Rows either side of
2026-08-23 are not digest-comparable. Nothing reads the field programmatically —
`check.py` bands on canary key, not fingerprint — so nothing breaks, but a human
diffing the ledger across that date will see the field this repo uses to flag build
swaps change without one, and should not chase it.

**3. Every row claimed `backend: rocm`, including the Vulkan ones.** `probe.sh` has
always taken `--backend` and always defaulted to `rocm`; `emit_row` never passed it.
The first Vulkan fingerprint therefore came out labelled ROCm, with a `hipconfig`
version beside it that had nothing to do with the run — on a build that runs on RADV
and whose performance moves with **Mesa**. A ledger whose whole purpose is saying which
layer moved a number was about to name the wrong layer. The backend is now read from
the build directory rather than assumed, and a directory shipping both backends is
refused rather than guessed at.

Verified 2026-08-23, both builds, through the real `emit_row` path:

| bindir | backend | commit | toolchain |
|---|---|---|---|
| `~/llama.cpp/b10472-vulkan` | `vulkan` | `60eeeb608` | GNU 11.4.0 |
| `~/llama.cpp/b10082-rocm` | `rocm` | `fb0e6b621` | GNU 11.4.0 |

A Vulkan row passes `validate.py` — `backend: vulkan` was already in the schema enum
from stage 3; nothing had ever emitted it.

**Still not measured:** no Vulkan row has been written to the ledger yet. The harness
can now produce one, which is a different claim from having one. The bands in
`configs/canary.yaml` remain ROCm-derived, and the canaries stay a ROCm-only watch —
see the note there.

## 2026-08-23: a stale verdict read as a passing one

The `nightly-20260822T181827Z` batch measured `fast-q4` and skipped the other four on
the busy-card guard — 760 MiB in use on the target, against a 500 MiB guard. The skips
are honestly recorded, each with its `reason`. What was not honest was the readout:
`make check` reported **all five `ok`**, four of them from `nightly-20260821T190014Z`,
with nothing in the output saying so.

The heartbeat could not catch it either. `alert.py` asked whether *the ledger* had a
successful run in 72 hours, and it did — `fast-q4` alone kept it fresh. Four canaries
could have stayed dark indefinitely behind one that still ran.

Two changes. `check.py` now emits `as_of` and `runs_since` on every verdict, so a
verdict states how current it is. `alert.py` gained a fourth condition, `coverage/<key>`,
for a canary silent past the heartbeat window while others still run — suppressed when
the whole lab is dark, since one dead tripwire should not file five Issues, and silent
on a single missed run, which happens on any busy evening.

The guard itself was right, and is not changed. On 2026-08-23 the occupant was an
`ollama` llama-server (PID 732103) holding 726 MB on the R9700 — a real occupant, and
refusing to measure around it is the correct behaviour. The defect was never the skip.
It was reporting a three-day-old number as today's.

## 2026-08-24: Vulkan has no fallback cliff — measured here, not quoted

The first Vulkan rows in the ledger, tag `vulkan-b10472-20260824`, `~/llama.cpp/
b10472-vulkan` on the R9700. Until today this claim rested on that build's
`PROVENANCE.txt`; it is now in the ledger with a fingerprint.

| canary | K/V | ROCm band | Vulkan median | ratio | reps |
|---|---|---|---|---|---|
| fast-q4 | `q4_0`/`q4_0` | 716.40 | 896.31 | 1.25× | 4 |
| fast-q8 | `q8_0`/`q8_0` | 714.87 | 905.16 | 1.27× | 3 |
| slow-q5_1 | `q5_1`/`q5_1` | 60.88 | 894.37 | **14.69×** | 3 |
| slow-q4_1 | `q4_1`/`q4_1` | 91.45 | 894.64 | **9.78×** | 3 |
| slow-mixed | `q8_0`/`q4_0` | 93.87 | 898.00 | **9.57×** | 3 |

**Every KV cache type measures between 892.7 and 908.5 t/s — a 1.8% spread.** The
fast-versus-fallback separation this repo was built to watch is 7.83× on ROCm and
**0.94× on Vulkan**, which is to say it does not exist. `q5_1`, `q4_1` and mismatched
K/V are not slow paths on this backend; they are the same path.

So three of the five canaries probe nothing on Vulkan, and the operator question from
2026-08-22 is settled with data rather than inference: the canary set stays a ROCm-only
watch. Closing the gap that leaves — nothing watches the build that actually serves —
needs a Vulkan canary sensitive to something Vulkan can lose, which is prefill at depth,
not KV cache type.

**Two harness changes made these rows possible, and safe:**

`bench.sh` and `bench_server.sh` now pin by backend. Both previously used only
`ROCR_VISIBLE_DEVICES`, which the Vulkan backend ignores, so a Vulkan run would have
landed on device 0 while the row's fingerprint named the R9700. `harness/vulkan_index.py`
maps a gfx target onto the Vulkan index, refusing when that is not exactly one device.
It has to be resolved, never assumed — two cards enumerate three ways here and no two
agree:

```
rocm-smi   GPU[0] = 9060 XT    GPU[1] = R9700
DRM        card0  = R9700      card1  = 9060 XT
Vulkan     Vulkan0 = 9060 XT   Vulkan1 = R9700
```

Verified directly: under `GGML_VK_VISIBLE_DEVICES=1` the process enumerates exactly one
device, `Vulkan0: AMD Radeon AI PRO R9700`. Unpinned it would have picked the 16 GB
9060 XT.

`check.py` now scores only rows whose `fp.build.backend` matches the band's, declared as
`backend: rocm` in `canary.yaml`'s defaults. Without it these 15 rows would have read as
a sustained 25%–1370% breach across the board, and the tripwire would have been reporting
a regression that is really two different backends in one column. Confirmed on the real
ledger: all five ROCm verdicts are unchanged with the Vulkan rows present.

**Caveats.** `fast-q4`'s first rep — 839.53 t/s, `tg` 24.47 — was measured on a cold card
(31 °C, 385 MHz) and is the only reading in the set that disagrees with its neighbours;
the three warm reps land at 894–902 with the rest. It is left in the ledger rather than
removed. Decode sat at ~19.6 t/s on every config here, which does not match the 17.36
that `PROVENANCE.txt` reports for this build at `tg64`; the difference is unexplained and
these canaries are calibrated on prefill, not decode.

## 2026-09-09: b10883 is 4-5% faster at prefill, and q4_0 V costs nothing measurable

Two questions, one session: has upstream improved in the 411 builds since b10472, and
is `q8_0`/`q4_0` KV really as good as `q8_0`/`q8_0`.

**Prefill: yes, 4-5%.** b10883 (91f6a6cf3, the official Vulkan prebuilt, downloaded the
day it was published) against b10472, Qwen3.8-27B-Q6_K on the R9700, `-p 2048`, 3 reps,
medians:

| KV | b10472 | b10883 | delta |
|---|---|---|---|
| `q8_0`/`q8_0` | 879.89 | 915.47 | **+4.0%** |
| `q8_0`/`q4_0` | 874.36 | 921.40 | **+5.4%** |

Both were measured back to back with an ollama embedding server resident on the card
(~750 MiB), which could not be evicted without root. The contention is controlled for
rather than ignored: b10472 measured 904.27 / 900.28 on a clean card earlier the same
day, so the occupant costs 2.7–2.9% — and **b10883 contended still beats b10472 clean**,
which puts the win outside the noise the occupant introduces.

**Decode: not resolved, do not quote it.** `tg64` moved 18.45 → 18.71 and 17.72 → 23.90,
but decode scattered from 13.5 to 24.77 t/s *within* a single build on this same day.
Three reps cannot separate a real change from that spread.

**Quality: same outcomes, different text.** `harness/kv_quality.py`, ctx 32768, both KV
configs, speculation off, temperature 0:

| | `q8_0`/`q8_0` | `q8_0`/`q4_0` |
|---|---|---|
| needle recall (9 probes, 4k/18k/31k tokens × 10/50/90% depth) | 9/9 | 9/9 |
| code tasks passing their asserts (6 tasks, executed) | 6/6 | 6/6 |
| outputs byte-identical between the two configs | — | **2/6** |

So the honest answer to "is `q8_0`/`q4_0` exactly the same" is **no, and it does not
matter here.** Four of six code generations differ in wording — the first divergence on
`roman` is "converts integers from 1 to 3999 **into** Roman numerals" against "**to**
Roman numerals", 81 characters in. That is the KV arithmetic changing, exactly as
`quality.py` predicted it would; strict token identity was never going to hold. Every
divergent generation still passed every assert, and no recall probe disagreed.

**Two caveats on the recall numbers.** The needle counts as found if it appears anywhere
in the output, including inside the `<think>` block — and on the deepest probe (31k
tokens, 90% depth) *both* configs retrieved the passphrase and then declined to repeat
it, so both score a hit on a visible refusal. That is model behaviour, identical across
configs, not a KV effect. And an earlier pass at ctx 65536 scored two false MISSes purely
because the answer budget was 32 tokens and the reasoning block ran past it; the probe
now allows 384, which is why that pass was discarded rather than reported.

**RETRACTED 2026-09-09: `q8_0`/`q8_0` at ctx 65536 does not fill the card.** This
paragraph previously claimed it reached "32,375 MiB of 32,768 (98.8%)" with prefill
collapsing from ~280 to 88 t/s under memory pressure. That is wrong by about 9 GB.
Re-measured on b10472, ctx 65536, `-fa on`, pinned with `GGML_VK_VISIBLE_DEVICES=1`,
model loaded and idle:

| K/V | VRAM used | of 34.2 GB |
|---|---|---|
| `q8_0`/`q8_0` | 24.45 GB (23,320 MiB) | **71.5%** |
| `q8_0`/`q4_0` | 24.12 GB (23,004 MiB) | 70.5% |

Raising V from `q4_0` to `q8_0` costs **316 MiB** at this context. There is 9.76 GB of
headroom, not 400 MiB, and a 316 MiB delta cannot produce a 3x prefill collapse — so
whatever caused that slowdown was not V-cache memory pressure. Both servers reported
`n_slots = 4, n_ctx_slot = 65536, kv_unified = 'true'`, so the context is shared across
slots rather than multiplied, which is not the explanation either.

Two notes on why the original number was wrong. It quoted the card as 32,768 MiB when
`rocm-smi` reports the R9700 at 32,624 MiB, so it was not read off this card's real
total. And `harness/emit_row.py:325` builds the server environment as `os.environ` plus
`LD_LIBRARY_PATH` only — it never sets `GGML_VK_VISIBLE_DEVICES`, so that observation was
taken unpinned, which is exactly the enumeration hazard this repo's trap list warns
about. Unpinned enumeration is a **hypothesis, not a confirmed cause.**

Nothing on disk could have caught this: the ledger records only `env.vram_used_mib_before`
and `results/kv-quality-32768-20260909.json` records no VRAM at all. The figure was a live
observation typed into prose. **The 262144 VRAM figures above are a separate measurement**
(with `mmproj` loaded and speculation on) and are not retracted here.

Consequently the `kv_quality.py` comparison's restriction to ctx 32768 had no measured
basis, and `q8_0`/`q8_0` is not ruled out as a serving config on VRAM grounds at 65536.
Whether it fits at 262144 is a separate question and is measured below.

## 2026-09-09: prefill at depth — b10883 does not earn adoption

The gap named above. Both builds, idle pinned card, `llama-bench -p 2048 -n 64
-d 0,16384 -fa on -ngl 99 -t 8 -r 3`:

| test | b10883 (`91f6a6cf3`) | b10472 (`60eeeb608`) | winner |
|---|---|---|---|
| pp2048 @ d0 | 861.05 ± 30.99 | **899.33 ± 1.16** | b10472 +4.4% |
| pp2048 @ d16384 | **800.17 ± 0.27** | 772.90 ± 3.23 | b10883 +3.5% |
| tg64 @ d0 | 18.86 ± 0.31 | **19.71 ± 0.00** | b10472 +4.5% |
| tg64 @ d16384 | 18.88 ± 0.03 | **19.03 ± 0.03** | b10472 +0.8% |

b10883 wins one of four, by 3.5%, and loses the other three. **Not adopted**;
`~/.mimir/bin/llama-server` still points at b10472.

Two limits on this table. The depth-0 prefill spread on b10883 is ±30.99 against
b10472's ±1.16, so that 4.4% loss sits inside b10883's own noise and should not be
quoted as a result. More importantly `llama-bench` used its **default f16 KV cache**,
not the `q8_0`/`q4_0` that actually ships — so this compares builds, not the operating
point. The earlier "+4.0% / +5.4% prefill" canary figures were taken at depth 0 on
quantized KV; at f16 depth 0 the sign flips. Those results are not in conflict, they
are different cache types. **The depth comparison at `q8_0`/`q4_0` remains unmeasured.**

## 2026-09-09: MTP is real at Q6_K, and it costs 3.3 GB of context, not a head file

Question asked: is Qwen3.8-27B-Q6_K actually using multi-token prediction, or does it
need a separate 1-2 GB MTP head alongside it? Answer: it is using it, the head is
already inside the Q6_K file, and it is tiny — but **enabling MTP is not free**, for a
reason unrelated to the head.

**The head ships in the model file.** `qwen35.nextn_predict_layers = 1`, and the
tensors are present at block 64:

| tensor | dims | type | bytes |
|---|---|---|---|
| `blk.64.nextn.eh_proj.weight` | 10240 x 5120 | Q8_0 | 55,705,600 |
| `blk.64.nextn.enorm.weight` | 5120 | F32 | 20,480 |
| `blk.64.nextn.hnorm.weight` | 5120 | F32 | 20,480 |
| `blk.64.nextn.shared_head_norm.weight` | 5120 | F32 | 20,480 |

**53.2 MiB total**, and `shared_head_norm` means it reuses the output head rather than
carrying a copy. There is no separate MTP file to fetch and none is wanted.

**It is genuinely engaged.** b10472, ctx 262144, `q8_0`/`q4_0`, `-fa on`, pinned card.
With `--spec-type draft-mtp --spec-draft-n-max 4` the server logs
`common_speculative_init_result: creating MTP draft context against the target model`
naming the Q6_K file itself — no draft model — and reports draft acceptance
**0.625 (182/291, mean len 3.49)** and **0.665 (185/278, mean len 3.64)**. Run with
speculation off, the same build logs `model has unused tensor blk.64.nextn.eh_proj.weight
(size = 55705600 bytes) -- ignoring`. That pair is the proof: the tensors are in the
file, and they are loaded only when the MTP draft type is asked for.

**What it buys, measured on the same prompt, `n_predict` 256, temperature 0:**

| config | decode t/s (2 reps) | VRAM | headroom |
|---|---|---|---|
| `--spec-type draft-mtp --spec-draft-n-max 4` | **48.87 / 52.18** | 33.56 GB (98.1%) | 0.65 GB |
| speculation off | 24.08 / 23.09 | 30.30 GB (88.6%) | 3.91 GB |

**MTP roughly doubles decode — 2.1x** — and that is the whole reason the served config
is worth its VRAM.

**The cost is the draft context, not the weights.** Turning MTP on moved VRAM by
**3.26 GB** while the head weights are 53 MiB. The difference is the second KV/compute
context llama.cpp allocates for the draft pass at ctx 262144. So the instinct that MTP
costs "an extra 1-2 GB" was directionally right and wrong about the mechanism: nothing
extra to download, ~3.3 GB extra to run at full context.

**Full context still fits at `q8_0`/`q4_0`, and the speeds still hold.** 33.56 GB of
34.21 GB at 262144 with speculation on, matching the 33.6 GB recorded earlier — the
262144 figures in this file are confirmed, not retracted. Decode 48.87-52.29 t/s sits
inside the previously recorded 44.25-53.25 t/s range, and acceptance 0.625-0.665 inside
the recorded 0.503-0.719. **Nothing has drifted.** Headroom is 0.65 GB, so this config
still has no room for a second process on the card.

**The vision projector is not resident at load.** Adding `--mmproj mmproj-F16.gguf`
measured 33.556 GB against 33.562 GB without it — a 6 MB *decrease*, i.e. no change.
`mmproj-F16.gguf` is a CLIP vision encoder (`general.architecture = 'clip'`,
`clip.projector_type = 'qwen3vl_merger'`, 461M params, 927 MB on disk) and has nothing
to do with MTP. The "~0.9 GB projector cost" recorded earlier is **not paid at load
time**; whether it allocates when an image is actually processed was not tested here.

Not measurable from this run: the `/completion` prefill figures (45-81 t/s) were taken
on a ~20-token prompt, where the number is dominated by request overhead. Prefill
belongs to `llama-bench`, and those figures are in the depth table above.

## 2026-09-09: Ornith-1.5-9B and MiniCPM5-2B, every quant on hand

b10472 Vulkan, R9700 pinned, `llama-bench -p 2048 -n 64 -d 0,16384 -fa on -ngl 99
-t 8 -r 3`. Both families live on `/mnt/8724062a…/models/`.

**Ornith-1.5-9B** — `qwen35` arch, 33 blocks, ctx 262144, `nextn_predict_layers = 1`,
i.e. the same hybrid SSM+attention family as Qwen3.8-27B, MTP head included.

| quant | GiB | pp2048 | tg64 | pp2048 @ d16384 | tg64 @ d16384 | imatrix |
|---|---|---|---|---|---|---|
| Q4_K_M | 5.37 | 3523.97 ± 1.13 | **90.02 ± 0.35** | 2910.78 ± 14.94 | **81.35 ± 0.26** | yes |
| Q6_K | 7.03 | 3081.93 ± 1.65 | 59.13 ± 0.06 | 2596.84 ± 8.54 | 68.35 ± 0.15 | yes |
| Q8_0 | 9.10 | **3706.25 ± 5.48** | 61.15 ± 0.03 | **3043.25 ± 9.46** | 57.08 ± 0.18 | no |
| BF16 | 17.13 | 2267.37 ± 18.34 | 35.14 ± 0.11 | 2027.51 ± 10.30 | 33.63 ± 0.20 | n/a |

**The Q6_K `tg64 @ d0` figure of 59.13 above is an artifact — disregard it.** It read
*below* Q8_0 (61.15) on 23% fewer bytes, and below its own `tg64 @ d16384` (68.35), both
backwards. Re-measured decode-only (`-p 0 -n 64 -d 0,16384 -r 5`):

| quant | tg64 | tg64 @ d16384 |
|---|---|---|
| Q6_K | **74.18 ± 0.11** | 68.17 ± 0.22 |
| Q8_0 | 61.22 ± 0.06 | 57.54 ± 0.11 |

74.18, not 59.13, and the tight ±0.11 on five reps says the re-check is the sound
reading. Q6_K@d16384 reproduced (68.17 against 68.35), so **only the `tg64 @ d0` cell
was wrong.** The difference between the runs is that the first interleaved `-p 2048`
prefill passes and the re-check did not; the artifact appeared on the second model in
that sequence. Cause not established — suspect clock or warm-up state carried over from
the preceding prefill run. **Do not interleave prefill and decode tests when the decode
number matters.**

**The "slow Vulkan Q6_K kernel" reading is refuted**, and with it the worry about the
27B shipping Q6_K. Corrected Ornith decode is monotonic in file size, as it should be:
Q4_K_M 90.02 > Q6_K 74.18 > Q8_0 61.22 > BF16 35.14.

**Q4_K_M is still the fastest operating point for this model** — 21% over Q6_K on decode
at a third less size — but the margin over Q6_K is far narrower than the first pass
implied. Q8_0 wins prefill, consistent with having the simplest unpack path.

**MiniCPM5-2B** — plain `llama` arch, 42 blocks, ctx 131072, **no MTP**, and **no
importance matrix on any of the three files, Q4_K_M included.**

| quant | GiB | pp2048 | tg64 | pp2048 @ d16384 | tg64 @ d16384 |
|---|---|---|---|---|---|
| Q4_K_M | 1.45 | 11187.76 ± 142.29 | **236.25 ± 0.02** | 3905.69 ± 2.71 | **163.59 ± 4.95** |
| Q8_0 | 2.49 | 11315.93 ± 100.04 | 169.36 ± 0.73 | 3971.14 ± 9.80 | 140.55 ± 0.80 |
| F16 | 4.69 | 11047.87 ± 22.12 | 109.27 ± 1.01 | 3978.83 ± 8.11 | 97.18 ± 0.28 |

This family behaves textbook, and is the control that makes the Ornith Q6_K reading
suspicious. Decode falls monotonically with file size (236 → 169 → 109 t/s) while
prefill is flat within 2.4% across all three (≈11,000 t/s) — decode is bandwidth-bound,
prefill is compute-bound, exactly as expected. Prefill drops ~65% from d0 to d16384
(11,188 → 3,906) on all three alike, which is attention cost, not a quant effect.

Bytes-per-second is not constant, though: 343 / 422 / 512 GiB·t/s across Q4_K_M / Q8_0 /
F16. The simpler the unpack, the more of the bandwidth actually gets used, so a 4-bit
quant does not convert its full byte saving into decode.

**No quality measurement was taken on either family** — these are speed and footprint
only. Which of these to finetune from is a separate question and not answerable from
this table: neither family's quants are finetuning inputs, and this llama.cpp build
ships no finetune or LoRA tooling at all.

## 2026-09-10: KV quality at 240k tokens — and a hit the caller never sees

The gap that mattered: every prior recall number was taken at ctx 32768 with
probes topping out at 31k tokens, against a config that serves at 262144.
Re-run at the real context. b10472 Vulkan, ctx 262144, speculation off,
temperature 0, needle at 10/50/90% depth on three haystack sizes.
`results/kv-quality-262144-20260910.json`.

| measure | `q8_0`/`q8_0` | `q8_0`/`q4_0` |
|---|---|---|
| recall, needle anywhere in output | **9/9** | **9/9** |
| recall, needle in the visible answer | **9/9** | **8/9** |
| code tasks passing their asserts | 6/6 | 6/6 |
| outputs byte-identical | — | 4/6 |

Probe sizes were 70,807 / 176,715 / 239,475 prompt tokens — the last is the
240k-token regime the shipped config actually operates in, measured for the
first time.

**Retrieval is intact at 240k on both configs.** Nine of nine, no disagreement,
at every depth. On the evidence here `q8_0`/`q4_0` costs nothing in retrieval
at eight times the context previously tested.

**But the two configs diverged once, in a way the hit count hides.** At 239,475
tokens, depth 10%, needle `PUMICE-ORACLE-62`:

- `q8_0`/`q8_0` answered `PUMICE-ORACLE-62`.
- `q8_0`/`q4_0` retrieved it inside `<think>` — quoting the planted line
  verbatim — then reasoned that it was *"an injected instruction"* in the
  middle of the document and refused: *"I don't have access to any…"*.

Both score a hit under "needle appears anywhere", which is why the raw counts
read 9/9 against 9/9 with zero disagreements. The caller of the second config
got a refusal. **This is the first measured case where the two KV configs
differ in user-visible behaviour**, and it is a behaviour difference, not a
retrieval failure — the passphrase was found both times.

`run_recall` now records `visible_hit` alongside `hit`, and `compare()` reports
`recall_visible_*`, `recall_visible_disagreements` and
`recall_retrieved_not_answered`. `compare()` recomputes the field from stored
text for result files written before it existed, so older runs re-score without
being re-run. The 2026-09-09 entry above already flagged this exact ambiguity as
a caveat; it has now changed a result, so it is measured rather than noted.

**What this does NOT establish.** n=1. Whether the refusal is a KV effect, or
the model landing on either side of an injection judgement it was always close
to, is not separable from a single probe at temperature 0. Do not quote this as
"`q4_0` V causes refusals". The reproducible facts are: retrieval held 9/9 at
240k on both, and one visible-answer divergence occurred at the deepest rung.

Byte-identical output rose to 4/6 here from 2/6 at ctx 32768. Task-dependent,
n=1 per task, not a trend.

## 2026-09-10: two infrastructure facts that cost most of a session

**`q8_0`/`q8_0` fits at 262144 on Vulkan.** It loaded and served the full probe
set at 30.18-32.40 GB of 34.21. The earlier claim that it "does not fit at all"
at 262144 generalised a **ROCm** failure on a 3.1 GB recurrent-state buffer into
a statement about the card. Retracted.

**A model load cannot live inside a supervised task on this box.** Five runs
were killed for "low memory" while reading the 22.9 GB model — at the last one,
`free` reported **25 GB available, 11 GB free, no process above 460 MB RSS**.
Nothing was starved; the supervisor watches how fast free memory falls as page
cache fills, and page cache from an mmap'd model is fully reclaimable. `setsid
nohup` did **not** escape it. What works:

- Stage the model on NVMe (`~/models-fast`, sha256-verified against the source).
  Load time fell from ~13 minutes to **15 seconds**.
- Start `llama-server` as a transient `systemd --user` unit
  (`systemd-run --user --unit=perflab-srv-<port> --collect`). Owned by the user
  manager, not the task's process tree — which is why the ollama server survived
  every kill that took ours.
- Drive it over HTTP: `kv_quality.py --port N` skips spawning a server. If the
  client is killed the model stays warm and the retry costs seconds.

Copying the model with plain `cp` is itself enough to trigger the kill (it died
at 12.9 of 22.9 GB). Use a copier that `fdatasync`es and `posix_fadvise`s
`DONTNEED` over each written range; that sustained 427 MB/s with cache flat.

## 2026-09-10: `--spec-draft-n-max` — 4 is not the optimum on this prompt

b10472, ctx 262144, `q8_0`/`q4_0`, MTP on, one prompt, `n_predict` 256,
temperature 0, 3 reps per setting, each a fresh systemd unit on an idle card.

| n-max | decode t/s (3 reps) | draft acceptance | VRAM |
|---|---|---|---|
| off | 24.44 / 24.39 / 24.36 | — | — |
| 2 | **45.24 / 45.11 / 45.22** | **0.710** | 33.30 GB |
| **3** | **45.45 / 45.46 / 45.44** | 0.593 | 33.60 GB |
| 4 (served) | 40.63 / 41.90 / 42.32 | 0.440–0.469 | 33.51 GB |
| 5 | 34.17 / 35.65 / 38.65 | 0.367–0.423 | 33.81 GB |
| 7 | 13.63 / 31.05 / 14.34 | 0.299–0.865 | 33.77 GB |

**n-max 3 beats the served setting of 4 by ~9%** (45.45 vs ~41.6) with far
tighter reps (±0.01 against ±0.85). Acceptance falls monotonically with depth —
0.71, 0.59, 0.45, 0.37, 0.30 — so deeper drafts are simply rejected and their
verification is wasted. n-max 7 is unusable: three identical reps spanned
13.63–31.05 t/s, and its acceptance swung 0.299–0.865.

**This contradicts the 2026-08 sweep in this file, which found 4 optimal at
52.46 t/s.** Different prompt, and acceptance depends on generated content, so
neither run is wrong — which is the point: **n-max is workload-dependent and
must be swept on real traffic, not chosen once.** Do not change the served
config on this table alone. It also makes upstream advice to push n-max to 7–12
for adaptive MTP look poorly matched to this model.

Speculation off costs more than half the decode rate (24.4 vs 45.4), consistent
with the 2.1x recorded 2026-09-09.

## 2026-09-10: mmproj works at full context and costs ~nothing at inference

ctx 262144, `q8_0`/`q4_0`, MTP on, `--mmproj mmproj-F16.gguf`, clean R9700:

| stage | VRAM | headroom |
|---|---|---|
| after load | 33.51 GB (98.0%) | 0.70 GB |
| after describing a 1024x1024 PNG | 33.51 GB (98.0%) | 0.69 GB |

**Image processing moved VRAM by +6 MB.** The projector is not a deferred
allocation waiting to exhaust the 0.7 GB of headroom — it is not paid at load
and not paid at inference. The server survived, and the description was exact:
*"Top-left: red; top-right: blue; bottom-left: yellow; bottom-right: green. A
horizontal black bar runs across the middle."* — 5/5 on a synthetic image.

Removing the ollama server first was unnecessary: it occupies **GPU[0]**
(~3.0 GB), not the R9700, which sat at 744 MB before load. The traps entry
saying ollama squats on the R9700 did not hold on this date.

**No YaRN.** The GGUF declares `qwen35.context_length = 262144` and
`rope.freq_base = 10000000.0` with **no `rope.scaling.*` keys**, and llama.cpp
logs no rope-scaling line at load. 262144 is the native trained window, so there
is no short-context quality penalty being paid for the long one and no reason to
split short/long serving instances.

## 2026-09-10: what Ornith-1.5-9B and MiniCPM5-2B are good at

`harness/model_eval.py`, ctx 65536, speculation off, temperature 0, budget 768.
Scored programmatically: code is EXECUTED against asserts, formats are parsed.

| suite | ornith-q6k | ornith-q4km | minicpm-q8 | minicpm-q4km |
|---|---|---|---|---|
| extract (5 facts, distractor-laden passage) | **5/5** | **5/5** | **5/5** | **5/5** |
| recall (needle at 8k/35k/59k tokens) | **3/3** | **3/3** | **3/3** | **3/3** |
| code (executed) | 2/6 | 1/6 | 4/6 | 2/6 |
| instruct (strict format) | 2/6 | 3/6 | 1/6 | 1/6 |

**Both are good at reading, not at obeying.** Extraction is perfect across all
four files, including near-miss distractors, and long-context recall is perfect
to ~59k tokens even on the 2.6B. Strict output-format compliance is where both
fail — "exactly three lines", "exactly five words", ALL-CAPS — which is the
risk if either is scripted against. MiniCPM Q8_0 is the better coder here (4/6),
and notably beats Ornith Q6_K despite being a third the size.

**Math is NOT measured, and the reason is a finding.** Ornith-1.5-9B does not
terminate on simple arithmetic: at `n_predict` 8192 it was still generating on
`crates` (17x24-38), `trip` and `discount`, producing no final answer. At 2048
its truncated text happened to end on the right value on 6 of 8 problems; at
8192 the same problems ended on 2.0 and 18.0. The reasoning block wanders, so a
truncated read is unreliable in both directions and none of it is scoreable.
Practical consequence: **Ornith needs a reasoning-budget cap or a no-think mode
before it is usable for short-answer work.** Scorer history is in the commit log;
truncated items now record `passes: None` rather than counting as failures.

Not measured anywhere here: prose quality, helpfulness, tone. Those need a
judge, and a small model judging small models measures the judge.

## 2026-09-10: `--parallel 1` is the missing flag — it doubles q8_0/q8_0 with MTP

`q8_0`/`q8_0` measured 22.4 t/s against `q8_0`/`q4_0`'s 45.5 at ctx 262144 with MTP,
and the cause was neither the cache type nor VRAM pressure. `llama-bench` with the KV
types swept at small context shows unspeculated decode is **indifferent to V**:

| K | V | tg64 @ d0 | tg64 @ d8192 |
|---|---|---|---|
| `q8_0` | `q8_0` | 19.75 | 19.43 |
| `q8_0` | `q4_0` | 19.71 | 19.43 |
| `q4_0` | `q8_0` | **24.67** | 19.36 |
| `q4_0` | `q4_0` | **24.69** | 21.51 |

`q8_0`/`q8_0` and `q8_0`/`q4_0` are within 0.2% of each other — so the server-side gap
came from MTP, not the cache. **K, not V, is what costs unspeculated decode** at depth 0
(q4_0 K is 25% faster), which is the opposite of where the quality argument points.

**The lever is `--parallel`, which defaults to 4.** MTP n-max 3, ctx 262144, 4 reps:

| K/V | slots | VRAM | decode t/s |
|---|---|---|---|
| `q8_0`/`q8_0` | 4 | 33.87 GB | 22.4 |
| `q8_0`/`q8_0` | **1** | 33.88 GB | **46.4 / 46.7 / 46.6 / 46.5** |
| `q8_0`/`q4_0` | 4 | 33.65 GB | 46.0 / 47.7 / 47.6 / 47.6 |
| `q8_0`/`q4_0` | **1** | **32.08 GB** | **48.3 / 48.6 / 48.6 / 48.6** |

**Draft acceptance was 0.607 in every `q8_0`/`q8_0` run regardless of slot count**, so the
drafts were accepted just as often and the loss is in the verify/batch path, not draft
quality. Upstream's note that MTP supports only `n_parallel=1` is consistent; the
mechanism here is **not established** and one upstream report claims `--parallel 3` works.

Two payoffs, and they differ by config. On `q8_0`/`q8_0` it is the difference between
viable and not (2x). On `q8_0`/`q4_0` it is only ~2% of speed but frees **1.57 GB** — which
matters at 98% occupancy. `kv_unified` is on, so `n_ctx_slot` stayed at full context in
every run: the extra slots cost memory and throughput, not context. The cost of
`--parallel 1` is concurrency — a second simultaneous request queues.

**`q8_0`/`q8_0` is now viable but still not better.** At 240k tokens both configs scored
9/9 recall and 6/6 code, so there is no measured quality gain for 0.32 GB of headroom
against 1.8 GB. Adding `--mmproj` to `q8_0`/`q8_0` also destabilised decode (33.3 / 36.1 /
46.3 t/s against a flat 46.4-46.7 without it); `q8_0`/`q4_0` has the headroom to absorb it.

Served config set in `~/.mimir/mimir.toml` accordingly (backup
`mimir.toml.bak-pre-parallel1`): `-ctk q8_0 -ctv q4_0 -fa on --spec-type draft-mtp
--spec-draft-n-max 3 --parallel 1 --main-gpu 1 --mmproj …/mmproj-F16.gguf`. Note
`cache_type_k` was `q4_0` there while `extra_args` passed `-ctk q8_0`, setting K twice;
`cache_type_k` is now `q8_0` so the two agree.

**Not measured**: the n-max sweep was run at the default 4 slots, so the optimum n-max
*under `--parallel 1`* is unknown — 3 is carried over, not re-derived.

## 2026-09-11: prompt mode moved capability scores further than the model did

The 2026-09-10 table above concluded that Ornith-1.5-9B and MiniCPM5-2B are
"good at reading, not at obeying" — code 2/6 and 1/6, strict-format compliance
2/6 and 3/6 — and that Ornith "needs a reasoning-budget cap or a no-think mode
before it is usable for short-answer work". **Both conclusions were artefacts of
prompting the models raw.** They were measured through `/completion` with a bare
prompt, outside the chat format the weights were tuned in.

`model_eval.py` now records `prompt_mode`, and `eval_matrix.sh` runs all three:

- `raw` — `/completion`, a bare prompt. What every table above was taken with.
- `chat` — the model's own chat template.
- `chat-nothink` — the template with `enable_thinking=false`.

ctx 65536, temperature 0, budget 2048, speculation off. Five models, fifteen
runs, `results/eval-*-{raw,chat,chat-nothink}-20260910.json` (the run started
before midnight; the entry is dated by when it was written up).

| model | mode | code | math | instruct | extract | research | total |
|---|---|---|---|---|---|---|---|
| gemma4-e2b | raw | 1/6 | 1/8 | 3/6 | 5/5 | 2/6 | 12/31 |
| gemma4-e2b | **chat** | 6/6 | 8/8 | 6/6 | 5/5 | 6/6 | **31/31** |
| gemma4-e2b | chat-nothink | 5/6 | 3/8 | 6/6 | 5/5 | 3/6 | 22/31 |
| gemma4-e4b | raw | 5/6 | 3/8 | 4/6 | 5/5 | 3/6 | 20/31 |
| gemma4-e4b | **chat** | 6/6 | 7/8 | 6/6 | 5/5 | 6/6 | **30/31** |
| gemma4-e4b | chat-nothink | 6/6 | 3/8 | 6/6 | 5/5 | 4/6 | 24/31 |
| minicpm-q8 | raw | 4/6 | 0/2* | 2/6 | 5/5 | 2/4* | 13/23 |
| minicpm-q8 | **chat** | 3/6 | 8/8 | 5/6 | 5/5 | 6/6 | **27/31** |
| minicpm-q8 | chat-nothink | 5/6 | 6/8 | 4/6 | 5/5 | 4/6 | 24/31 |
| ornith-q6k | raw | 5/6 | 3/3* | 2/6 | 5/5 | 4/4* | 19/24 |
| ornith-q6k | **chat** | 6/6 | 8/8 | 5/6 | 5/5 | 6/6 | **30/31** |
| ornith-q6k | chat-nothink | 6/6 | 1/8 | 3/6 | 5/5 | 3/6 | 18/31 |
| qwen35-9b | raw | 5/6 | 6/8 | 5/6 | 5/5 | 3/5* | 24/30 |
| qwen35-9b | **chat** | 6/6 | 7/7* | 6/6 | 5/5 | 6/6 | **30/30** |
| qwen35-9b | chat-nothink | 5/6 | 2/8 | 6/6 | 5/5 | 3/6 | 21/31 |

`*` = the suite contained items cut off at the budget, which are recorded
unscoreable rather than failed. gemma4 and qwen35 rows are through ollama's
runtime (see the next entry); the others through llama-server on b10472. The
qwen35-9b chat row is at budget 6144 rather than 2048: its reasoning block is
long enough that a third of the suite was unscoreable at 2048, which is itself
the cost of that model's verbosity.

**The spread from prompt mode is 12→31 on one model. No pair of models differs
by nearly that much within a mode.** Any capability number taken without
recording its prompt mode is uninterpretable, which is why the field is now
mandatory in the report.

**Ornith terminates fine.** In chat mode it answered `17x24-38` correctly in
**34 tokens**: `<think>17 crates × 24 bottles = 408 bottles / 408 - 38 = 370
</think>370`. The 8192-token non-termination recorded on 2026-09-10 was raw
prompting with no template-injected `<think>`; nothing about the model needed a
budget cap. Math is now scoreable, and Ornith scores 8/8.

**Reasoning is load-bearing, and `enable_thinking=false` is not free.** It is the
largest lever on time-to-answer there is — it removes the scratchpad rather than
decoding it faster — and it costs accuracy where the scratchpad was doing work.
Math falls 8/8 → 1/8 on Ornith and 8/8 → 3/8 on gemma4-e2b. On the same
arithmetic item Ornith answers 370 with thinking (34 tokens) and **456 without,
in 4 tokens**. Format compliance is unaffected or better, so no-think is
defensible for extraction and formatting work and indefensible for arithmetic.

**Three scorer bugs were fixed to get these numbers.** Two had inflated scores
and one had deflated them:

- A reasoning model prompted raw emits a closing `</think>` with *no opening
  tag*, because the opening tag is normally injected by the template. The
  existing `THINK` regex requires both, so it stripped nothing and the entire
  scratchpad was scored as the answer — every intermediate number in it counted.
- "The right number appears somewhere in the answer" passed a model that
  answered 96,000 and then talked itself to 87,000 on the way past. Numeric
  items whose prompt says "give only the number" are now scored on the *first*
  number.
- A code fence nested inside prose arrives uniformly indented, and `exec()`
  raises `IndentationError` on line 1. **That alone scored qwen3.5:9b 1/6 on
  code it had written correctly**; `textwrap.dedent` fixes it, and the run above
  is 6/6. Only two stored runs were affected, so the rest of the table did not
  need re-measuring.

Because the first two inflated and the third deflated, the corrected numbers do
not move in one direction — the point is that none of the three were visible in
a score, only in the stored text.

**The suites saturate in chat mode and no longer discriminate there.** Four of
five models score 27-31 of 31. These tasks now measure whether a model was
prompted correctly, not which model is better; ranking these five needs a harder
tier that does not exist yet.

## 2026-09-11: three of the five models will not load in llama.cpp at all

`gemma4:e4b`, `gemma4:e2b` and `qwen3.5:9b` exist here only as ollama blobs, and
none of the three loads under upstream llama.cpp. Identical failures on the
b10472 prebuilt, the b10883 prebuilt, and a b10902 built from master here, so
these are **exporter divergences, not a version gap**:

| model | failure |
|---|---|
| qwen3.5:9b | `key qwen35.rope.dimension_sections has wrong array length; expected 4, got 3` |
| gemma4:e4b / e2b | `done_getting_tensors: wrong number of tensors; expected 2131, got 720` |

`src/models/qwen35.cpp:6` reads the rope sections with a required length of 4;
ollama writes three (`[11, 11, 10]`). `harness/gguf_patch.py` was written to
rewrite exactly that key — header only, tensor data copied untouched, data
section realigned so no tensor offset moves — and padding it to `[11,11,10,0]`
does clear the check. **The next load then fails on `blk.0.ssm_dt.bias` not
found**: ollama also names tensors differently. Two divergences deep with no
reason to think there is not a third, patching was abandoned. The tool is kept;
it does its job.

For gemma4 the gap is structural rather than a typo. Upstream's `gemma4` arch
builds 720 text tensors; the ollama file carries 2131 — text plus a vision
tower, an audio tower (`mm.a.*`) and Gemma-3n-style per-layer embeddings
(`per_layer_token_embd`, `per_layer_model_proj`) in one file, where upstream
wants the multimodal half split into a separate mmproj.

So those three are measured through **ollama's own runtime**, via a second
backend in `model_eval.py`. That is fair for capability, which is a property of
the weights, and is **never fair for speed** — different runtime, different KV
types, different offload policy. No ollama-backed number appears in any speed
table in this file.

ollama also serves its own `num_ctx` unless told otherwise: a run asking for
65536 was observed running at 32768 until `num_ctx` was passed explicitly.

**What the headers say** (`harness/gguf_meta.py`, which reads a GGUF header
without loading the model):

- **qwen3.5:9b** — `qwen35`, 32 blocks, ctx 262144 native, and a **hybrid**: only
  every fourth layer carries attention (`head_count_kv` is an array,
  `[0,0,0,4,...]`), the rest are `blk.N.ssm_*`. It ships a **full MTP head**
  (15 `mtp.*` tensors) and an in-file vision tower (`v.blk.N.*`). Named
  `mtp.*` where Ornith names the same thing `blk.32.nextn.*`.
- **gemma4:e4b / e2b** — 42 and 35 blocks, ctx 131072 native, **no MTP**, and
  **no chat template in the GGUF** (ollama supplies it from its Modelfile,
  which is why the chat-mode rows above exist for them at all). Both carry
  vision *and* audio towers in-file. ollama resides only 3.9 GB of the 8.95 GiB
  e4b file and 2.0 GB of the 6.67 GiB e2b file — the MatFormer slice, not the
  whole file.
- Ornith's MTP head is `blk.32.nextn.*`, and llama.cpp logs it as
  `model has unused tensor ... ignoring` when `--spec-type draft-mtp` is absent.

## 2026-09-11: the RX 9060 XT collapses on every weight type except Q4_K

Measured because the second card looked 3.5x slower in a dual-GPU throughput
run. It is not uniformly slower. `llama-bench -p 0/512 -n 64 -fa 1 -ngl 99 -t 8
-r 3`, b10472 Vulkan, backend reported as Vulkan on both cards, decode t/s:

| model | type | GiB | RX 9060 XT (gfx1200) | R9700 (gfx1201) | ratio |
|---|---|---|---|---|---|
| MiniCPM5-2B | Q4_K_M | 1.45 | 139.18 | 237.01 | **1.70x** |
| Ornith-1.5-9B | Q4_K_M | 5.38 | 48.91 | 90.90 | **1.86x** |
| MiniCPM5-2B | Q8_0 | 2.50 | 21.33 | 169.41 | **7.94x** |
| Ornith-1.5-9B | Q6_K | 7.03 | 13.68 | 75.25 | **5.50x** |
| MiniCPM5-2B | F16 | 4.69 | 19.07 | 108.52 | **5.69x** |

**The ratio tracks the weight type, not the file size.** A 5.38 GiB Q4_K_M runs
at 1.86x while a 2.50 GiB Q8_0 runs at 7.94x — less than half the bytes, four
times the penalty. Size is ruled out, and so is CPU spill: both cards had room,
`-ngl 99` was honoured, and llama-bench reported the Vulkan backend for every
row. Error bars were ±0.5% or tighter throughout.

Prefill shows the same split more weakly — Q4_K_M 6,642 vs 12,398 t/s (1.87x),
F16 3,053 vs 8,064 (2.64x).

**Cause not established.** 1.7-1.9x is about what the two cards' memory
bandwidth predicts; the 5.5-7.9x on the other types is not, and whether it is a
missing gfx1200 dequant kernel in RADV, a shader-compilation fallback, or
something about the display card being contended has not been separated.

**Operationally this is decided regardless of cause: put only Q4_K_M models on
the RX 9060 XT.** A Q6_K or Q8_0 model placed there runs slower than it would on
many CPUs, and nothing in the logs says so.

## 2026-09-11: slots raise throughput and lower speed, and the answer depends which you are buying

Every speed number in this file before today is single-stream. `harness/throughput.py`
fires N concurrent requests and reports both halves: aggregate tokens/s across
the batch, and the mean rate each individual request saw.

Qwen3.8-27B-Q6_K, ctx 262144, `q8_0`/`q4_0`, MTP n-max 3, b10472, R9700 idle,
128 tokens per request, 2 batches per cell.

| slots | metric | c=1 | c=2 | c=4 | c=8 |
|---|---|---|---|---|---|
| `--parallel 1` | agg t/s | 42.67 | 46.80 | 44.28 | 43.16 |
| | per-req t/s | 53.52 | 54.28 | 50.93 | 49.36 |
| | p50 / p95 s | 3.11 / 3.11 | 5.38 / 5.55 | 8.35 / 12.34 | 13.70 / 24.75 |
| `--parallel 2` | agg t/s | 44.65 | 52.73 | 51.91 | 50.91 |
| | per-req t/s | 53.35 | 31.87 | 30.69 | 29.77 |
| `--parallel 4` | agg t/s | 45.15 | 51.37 | **67.95** | 61.10 |
| | per-req t/s | 53.80 | 30.88 | 21.73 | 18.69 |
| | p50 / p95 s | 3.09 / 3.09 | 4.92 / 5.28 | 7.22 / 7.73 | 14.29 / 17.61 |

**Four slots at four concurrent requests is 53% more aggregate throughput than
one slot can reach** — 67.95 against 44.28 — and each of those requests runs at
**21.73 t/s instead of 50.93**, 2.3x slower for the person waiting on it. That is
the whole trade, and it is why the 2026-09-10 entry's "`--parallel 4` halves
decode" and this entry's "`--parallel 4` wins" are both true: that entry measured
one request, this one measures four.

Three things fall out that were not obvious:

- **`--parallel 1` does not lose throughput under load, it just queues.** Per-request
  stays at 49-54 t/s all the way to c=8 while p95 climbs 3.11 → 24.75 s. Nothing
  is being shared; the requests are simply serialised.
- **Concurrency past the slot count is wasted.** c=8 on 4 slots is *worse* than
  c=4 (61.10 vs 67.95): the extra four queue behind a batch that is now slower
  per request. Match concurrency to slots.
- **Slots cost VRAM, and the cost is real at 98% occupancy** — 32.08 / 32.22 /
  33.44 GB for 1 / 2 / 4 slots. Draft acceptance was flat at 0.72 for c=1 in all
  three and fell to 0.61-0.67 under batching.

**For mimir's workload — one interactive caller at a time — `--parallel 1`
remains right**, and it is worth 1.36 GB of headroom. A queue serving four
callers should use 4.

### Two cards add up, and neither slows the other

Ornith-1.5-9B-Q6_K served on both cards at once, `--parallel 1`, MTP n-max 3:

| | alone | both at once |
|---|---|---|
| R9700 (gpu 1) | 53.59 t/s | 53.75 t/s |
| RX 9060 XT (gpu 0) | 14.92 t/s | 14.88 t/s |
| **combined** | — | **68.63 t/s** |

**No contention** — both within 0.4% of their solo rates, so PCIe and the host
are not the bottleneck and a second card is genuinely additive. But it adds only
**28%**, because this model is Q6_K and Q6_K is the type the 9060 XT collapses
on (previous entry). A request routed to the second card takes 25 s where the
first takes 7 s. Two cards are worth having; two *different* cards are worth
routing by model type, not round-robin.

## 2026-09-11: n-max 3 survives the re-derivation under `--parallel 1`

The gap left open on 2026-09-10: n-max was swept at the default 4 slots, so its
optimum under the `--parallel 1` that got shipped was unknown. Re-swept with
`harness/spec_sweep.py`, which does the cross product so neither axis is ever
again chosen against a stale value of the other. b10472, ctx 262144,
`q8_0`/`q4_0`, `--parallel 1`, 3 reps, 256 tokens, fresh unit per cell.

| n-max | mean t/s | reps | acceptance | VRAM |
|---|---|---|---|---|
| off | 24.14 | 23.91 / 24.27 / 24.24 | — | 29.83 |
| 2 | 43.28 | 43.33 / 43.29 / 43.21 | 0.644 | 31.92 |
| **3 (served)** | 44.13 | **44.15 / 44.11 / 44.12** | 0.549 | 32.08 |
| 4 | 44.47 | 43.86 / 43.15 / 46.39 | 0.484 | 32.24 |
| 5 | 41.47 | 39.60 / 43.23 / 41.59 | 0.420 | 32.39 |

**4 has the higher mean and 3 is the better setting.** The 0.34 t/s gap is 0.8%
and n-max 4's three reps span 3.24 t/s — its own variance is ten times the
difference it wins by, while n-max 3 spans 0.04. **No change to the served
config**; the carried-over 3 was right, and now it is measured rather than
assumed. Acceptance falls monotonically with depth exactly as before —
0.64, 0.55, 0.48, 0.42.

Speculation off costs 45% of decode (24.14 vs 44.13), consistent with the 2.1x
recorded twice before.

### Adaptive MTP (PR #27210) matches a well-chosen fixed depth, and does not beat it

`--spec-type draft-mtp-adaptive` varies draft depth per step. Built here by
cherry-picking the seven commits of the open PR onto b10902 — see
`~/llama.cpp/b10902-adaptive-mtp/PROVENANCE.txt`. Same model, context, KV and
slot count as the table above; the fixed-depth cells are re-measured **on the
same binary** so the comparison is not confounded with the build:

| setting | mean t/s | reps | acceptance |
|---|---|---|---|
| off | 24.25 | 24.26 / 24.24 / 24.24 | — |
| fixed n-max 3 | 45.01 | 45.02 / 45.00 / 45.00 | 0.569 |
| fixed n-max 4 | 44.63 | 44.74 / 44.59 / 44.57 | 0.509 |
| **adaptive** | 45.01 | 45.06 / 45.00 / 44.97 | **0.569** |

**Identical to fixed n-max 3, to three decimal places on acceptance** — the
controller converged on depth 3 and stayed there. Its value is therefore that it
removes the need to sweep, not that it is faster: the 2026-09-10 entry's
conclusion that n-max is workload-dependent and must be re-swept on real traffic
is exactly what this obsoletes. **On this prompt it costs nothing and saves the
sweep.** It has not been tested on a workload where the fixed optimum is wrong,
which is the case where it should actually win, and the PR is still open.

Incidentally the locally built b10902 is ~2% faster than the b10472 prebuilt at
the same setting (45.01 vs 44.13) with much tighter reps. Not enough to move the
pin, and not measured on the canary set.

## 2026-09-11: three infrastructure facts

**A local Vulkan build from source works.** b10902 (df03399b8), GCC 15.2, shaders
via the distro glslc. The 2026-08 entry "the local GCC 15 build is broken" was
about a **ROCm** build whose llama-server segfaulted in `ggml_cuda_op_scale`; it
does not generalise, and this does not reverse it. The blocker was never the
compiler — `glslc`, `libvulkan-dev` and `cmake` were simply absent until
2026-09-10. One more is still missing and is **not** in that list: **SPIRV-Headers
is not installed**, and ggml's Vulkan CMakeLists finds the package but does not
propagate its include directory, so the build dies at `ggml-vulkan.cpp:48` on a
missing `spirv/unified1/spirv.hpp`. Cloning it at the matching SDK tag into
`~/.local` and adding `-DCMAKE_CXX_FLAGS=-I$HOME/.local/include` is enough; no
root needed. Full recipe in `~/llama.cpp/b10902-vulkan-local/PROVENANCE.txt`.

**`harness/install.sh` fires a nightly the moment it runs.** The timers are
`Persistent=true`, so enabling them after twelve days of downtime immediately
triggers a catch-up `perf-lab@nightly.service` — which went `activating` while
two unrelated models held both cards. It was stopped before it loaded anything,
and the bench guard did the right thing anyway: the five rows it wrote are
`kind: skipped` with null metrics, not contaminated measurements. **Install the
timers when the cards are free, or stop the catch-up run immediately after.**

**mimir's `--parallel 1` was never missing.** The 2026-09-10 entry says the
served config was set with `--parallel 1`; the config diff shows no such flag,
which reads like the change was lost. It was not: `LlamaCppConfig.ParallelOrDefault()`
returns **1** when unset, unlike llama-server's own default of 4, so mimir has
been passing `--parallel 1` all along. The value is now written out explicitly in
`~/.mimir/mimir.toml` — it changes nothing today, and it stops a load-bearing
number living in a default that a future edit could move silently.

## 2026-09-11: the served config verified, and two of its keys do nothing

Qwen3.8-27B-Q6_K served with exactly the flags `~/.mimir/mimir.toml` specifies —
`-c 262144 -ctk q8_0 -ctv q4_0 --flash-attn on --split-mode none --main-gpu 1
--parallel 1 --spec-type draft-mtp --spec-draft-n-max 3 --mmproj …` — on an idle
R9700, b10472:

| check | expected | measured |
|---|---|---|
| decode | ~48 t/s | **55.48 t/s** per request, acceptance 0.762 |
| VRAM after load | ~32.1 + 0.9 projector | **32.08 GB** (projector included) |
| VRAM after an image | flat | **32.11 GB** (+30 MB) |
| image description | exact | exact, 5/5 |

*"Top-left: red; top-right: blue; bottom-left: yellow; bottom-right: green. A
horizontal black line crosses the middle."*

Driven as a bare `llama-server` rather than through mimir, so what is confirmed
is the configuration, not mimir's launcher.

**Two keys added on 2026-09-10 are silently ignored by the installed mimir.**
`mimir doctor` (build 619ad714a) reports:

> ignored unknown config keys (typo or future-version — they do nothing in this
> build) keys="llamacpp.vram_headroom_factor, llamacpp.gpu."0000:0c:00.0",
> llamacpp.gpu."0000:0c:00.0".vram_reserve_gb,
> llamacpp.gpu."0000:0c:00.0".vram_headroom_factor"

So the whole per-card VRAM budget block — and the long comment beside it
explaining that a 1.05 headroom factor is "what makes Qwen3.8-27B-Q6_K fit at
the full 262144 context" — is inert. The model fits regardless, as the table
above shows; the config comment claims a causal role for settings that are not
being read. `parallel` was *not* in the ignored list, so that key is live.

The keys exist in `~/git/mimir` source, so the installed binary simply predates
them. Until it is rebuilt the admission estimate runs on the rig-wide defaults.

**Corrected in the config, not removed.** Both sites now say they are inert and
name the build that ignores them, and the false causal claim is replaced with
the measurement above — the model fits on its own, and what these keys will buy
once mimir is rebuilt is a more accurate *admission estimate*, not a fit. They
are left in place because they are correct for a newer binary and harmless to
this one; deleting them would only mean rediscovering the values later.

## 2026-09-11: Q4_K_M costs one point, and two more scorer bugs were in the way

The 2026-09-11 capability table left `ornith-q4km` and `minicpm-q4km` out: they
had never been run in chat mode, so the only numbers for them were the raw-mode
rows from 2026-09-10 that the prompt-mode entry above withdrew. Both re-measured
here in all three modes, ctx 65536, temperature 0, budget 6144, speculation off,
`results/eval-{ornith,minicpm}-q4km-{raw,chat,chat-nothink}-20260911.json`.

| model | mode | code | math | instruct | extract | research | total |
|---|---|---|---|---|---|---|---|
| ornith-q4km | **chat** | 6/6 | 8/8 | 5/6 | 5/5 | 5/6 | **29/31** |
| ornith-q4km | chat-nothink | 6/6 | 1/8 | 4/6 | 5/5 | 3/6 | 19/31 |
| ornith-q4km | raw | 2/3* | 1/1* | 3/6 | 5/5 | 4/4* | 15/19* |
| minicpm-q4km | chat | 1/1* | 7/7* | 6/6 | 5/5 | 5/6 | 24/25* |
| minicpm-q4km | **chat-nothink** | 6/6 | 8/8 | 6/6 | 5/5 | 4/6 | **29/31** |
| minicpm-q4km | raw | 2/5* | 4/5* | 2/6 | 5/5 | 2/2* | 15/23* |

`*` = the suite contained items cut off at the budget, recorded unscoreable
rather than failed.

**Quantization costs nothing measurable here.** Ornith-1.5-9B goes
Q6_K 30/31 → Q4_K_M 29/31, and MiniCPM5-2B goes Q8_0 27/31 → Q4_K_M 29/31 —

> **Superseded the same night — see the reproducibility entry below.** This
> paragraph originally read "quantization costs about one point, not a tier",
> resting on Ornith's 30/31 → 29/31. Ornith was then measured scoring 29 and
> 30 on **two runs of the same file at the same quant**, so that one point is
> run-to-run variance and cannot be attributed to the quantization. The
> direction of the remaining text stands; the one-point cost does not.

the 2B *improves* on the smaller file, which is only interpretable as the
suites being saturated at this difficulty, not as Q4_K_M beating Q8_0. The
prompt-mode result holds on the quantization axis too: both Q4_K_M files read
15/19 and 15/23 raw and 29/31 in their best chat mode. **The raw-era verdicts
this table replaces — Ornith Q4_K_M "1/6 on code", MiniCPM Q4_K_M "2/6" — were
prompting artifacts exactly as the Q6_K and Q8_0 rows were.**

### Two more scorer bugs, both in `run_code` alone

Getting these numbers took fixing the fourth and fifth instances of the bug
class the entry above fixed three of. Both were in `run_code`, which was the
only suite that had never been revised:

- **It never checked `stop_type`.** Every other suite records `truncated` and
  scores a cut-off item unscoreable. `run_code` counted one as wrong code:
  `exec()` raises `NameError` on the function that was never defined, which in
  the stored record is indistinguishable from a genuine failure.
- **It searched for the code fence before stripping `<think>`.** A reasoning
  model drafts a ` ```python ` block inside its scratchpad and writes the
  finished version after it; `CODE_FENCE.search` on the raw text found the
  draft and never examined the answer. This is how `balanced` came to be scored
  on the literal stub `def balanced(s):\n    # code`.

Both were found by raising MiniCPM Q4_K_M's budget 2048 → 6144 to explain a
1/6 code score and getting **byte-identical generations** — at temperature 0 the
budget was not the binding constraint, and the score was not measuring the
model. Probing the live server showed 5 of 6 code items returning
`stop_type=limit` with 20-29K characters of reasoning and nothing after
`</think>`.

**These fixes reach every code score in this file.** Only the two Q4_K_M labels
have been re-measured under the corrected scorer; the five labels in the table
above still carry code numbers taken with it. Ornith Q6_K's 6/6 is the one most
likely to be real — its Q4_K_M sibling scores 6/6 with zero truncations — and
MiniCPM Q8_0's 3/6 is the one most likely to be an artifact.

### No-think is not one lever, it is two, and they point opposite ways

The entry above concluded `enable_thinking=false` "costs accuracy where the
scratchpad was doing work". That holds for **math** and is now sharper: Ornith
Q4_K_M falls 8/8 → 1/8 without thinking. It is **backwards for code on
MiniCPM5-2B**, which cannot finish a code task with thinking on — 5 of 6 items
run out of budget mid-scratchpad at 6144 — and scores a clean 6/6 with it off,
nothing truncated. Ornith shows no such effect: 6/6 either way.

So the scratchpad is load-bearing for arithmetic on both models, and actively
fatal for code generation on the 2B. **A single no-think setting per model is
the wrong shape**; it belongs per task type, and any harness scripting MiniCPM
for code should disable thinking.

### An infrastructure note

Loading the 9B inside a supervised background task was killed for memory
pressure with 20 GB available, the same supervisor behaviour `serve_unit.sh`
was written for. The server survives because systemd owns it; the caller does
not. **Start the unit, then run `model_eval.py` against the live port as a
separate task** — the load is what crosses the threshold, not the evaluation.

## 2026-09-12: Ornith is not reproducible at temperature 0, and MiniCPM is

Both Q4_K_M labels re-run at budget **32768** rather than 6144, to test whether
the output cap was still deciding scores. It is not — but the re-run exposed
something that matters more.

| label | mode | 6144 | 32768 | identical? |
|---|---|---|---|---|
| minicpm-q4km | raw | 15/23* | 15/23* | **yes, byte-for-byte** |
| minicpm-q4km | chat | 24/25* | 24/25* | **yes** |
| minicpm-q4km | chat-nothink | 29/31 | 29/31 | **yes** |
| ornith-q4km | raw | 15/19* | 15/20* | no |
| ornith-q4km | chat | 29/31 | **30/31** | no |
| ornith-q4km | chat-nothink | 19/31 | 20/31 | no |

`*` = contains unscoreable items.

**MiniCPM5-2B is bit-identical across every mode at both budgets.** Same
prompts, same greedy sampling, same text. That is the control, and it says the
harness itself is deterministic.

**Ornith-1.5-9B is not**, and the differences are not truncation artifacts —
they appear on items that ran to EOS at both budgets:

- `code/wordfreq` — a `Counter`-based implementation in one run, a manual
  comprehension in the other. Both correct.
- `research/conflict_vance` — different prose from the first token.
- `research/multihop_above_avg` — `"Marrow"` alone in one run, `"Marrow"` plus
  three sentences in the other. **That is the item that flipped 29 → 30.**

**Consequence: a one-point difference between two Ornith configurations means
nothing.** Every Ornith score in this file is n=1, and n=1 now has a measured
spread of at least ±1. That directly undercuts the quantization claim made
earlier the same night — Q6_K 30/31 against Q4_K_M 29/31 is the same gap that
appeared between two runs of the *identical* Q4_K_M file, so the comparison
does not survive. The corrected reading: **no capability difference between
Ornith Q6_K and Q4_K_M was measurable.**

**Cause not established.** The obvious suspect is that Ornith is a hybrid
SSM+attention model with an MTP head (`blk.32.nextn.*`) while MiniCPM is plain
`llama` arch, and that SSM state or MTP draft acceptance carries slot and batch
history across requests in a way the simpler architecture does not. That is a
hypothesis; it has not been tested. What is established is the asymmetry: one
model reproduces exactly and the other does not, on the same server, the same
build and the same harness.

**What this costs.** Any future ranking of Ornith against anything needs
repeated runs and a reported spread, not a single number. The cheapest
sufficient check is running one mode three times and reporting the range,
which is also the thing that should have been done before any of the one-point
conclusions in this file were written.

### The budget question, answered

Raising the cap 5.3x changed no MiniCPM score at all, and moved Ornith only
within its own noise. **The cap was never the binding constraint on a
terminating answer** — it only ever mattered because `run_code` scored a
truncation as a wrong answer, which is fixed. Where a model does run away, it
runs away past 32768 too: MiniCPM with thinking on still fails to finish 5 of 6
code items at the larger budget, so no-think is not a workaround for that
model, it is the only mode in which its code suite is answerable.

Cost note: raw mode at 32768 took **four hours** on the 9B, because each
truncating item now generates the full 32768 tokens at ~50 t/s. Raw is the mode
whose results were already withdrawn. **Do not re-run raw at a high budget
again** — the information return is zero and it is the most expensive row in
the matrix.

## 2026-09-12: the five stale labels re-scored — two moved, three did not

The scorer debt from the entry above is paid. All five labels whose code scores
predated the `run_code` fixes re-run in all three prompt modes, budget 6144
throughout (the 2026-09-11 table used 2048 for four of them and 6144 only for
`qwen35-9b` chat; one budget now covers the matrix).
`results/eval-*-20260912.json`, 15 runs, 11:05–12:50.

| label | chat, published | chat, re-scored | moved |
|---|---|---|---|
| ornith-q6k | 30/31 | 29/31 | instruct 5/6 → 4/6; code unchanged 6/6 |
| minicpm-q8 | 27/31 | **26/27*** | **code 3/6 → 3/3***, math 8/8 → 7/7* |
| gemma4-e2b | 31/31 | 31/31 | nothing |
| gemma4-e4b | 30/31 | 30/31 | nothing |
| qwen35-9b | 30/30* | 30/30* | nothing |

`*` = contains unscoreable items.

**The two predictions made when the bugs were found both held.** MiniCPM Q8_0's
3/6 on code was called as the likeliest artifact and it was: three of those
"failures" were truncations scored as wrong code, and the honest reading is
3/3 with three unscoreable. Ornith Q6_K's 6/6 was called as likeliest real, and
it is unchanged.

**The three ollama-backed labels are byte-identical across every mode.** They
never tripped either bug — no draft fence inside a scratchpad, no truncation
mis-scored. That is worth knowing: the bugs were specific to reasoning models
served through llama-server, not general to the harness.

**Ornith's one-point instruct drop is not a change.** It is inside the ±1
run-to-run spread measured the same night, and this run cannot distinguish a
scorer effect from that noise. Both of the labels that moved changed two
variables at once — scorer and budget — so neither delta is cleanly
attributable; the *direction* is what the fix guarantees, not the magnitude.

**What is still not fixed:** the suites remain saturated. Four of five labels
sit at 29-31 of 31 in chat mode after all of this. Every conclusion about
which of these models is better still waits on a harder tier, and now also on
repeated runs with a reported spread rather than n=1.

## 2026-09-12: native tool calling measured, and every model can do it

The gap every previous entry left open. The format suites were only ever a
proxy for whether these models can drive an agent loop, and the proxy scored
worst of all five suites, which made it the open question. Now measured
directly.

**Native tool calling, not prompt-engineered JSON.** mimir builds a request
carrying a `Tools` list and reads `tool_calls` back off the response, so a
suite scoring hand-rolled JSON in the message body would measure something
mimir never does. Both backends go through the endpoint that carries a tools
array: `/v1/chat/completions` for llama-server and **`/api/chat` for ollama** —
not `/api/generate`, which every other ollama path in the harness uses and
which has no tool support at all. There is no raw-mode row by construction; a
tool catalog only reaches the model through the chat template, and
`--suites tools` without `--chat` refuses rather than mislabelling the row.

Twelve items over a five-tool catalog with two deliberately confusable pairs
(`search_files`/`search_memory`, `read_file`/`fetch_url`). Scored on the calls
themselves: right tool, exactly one call, arguments satisfying the declared
schema. ctx 32768, budget 2048, temperature 0.
`results/tools-*-20260912.json`.

| label | chat | chat-nothink |
|---|---|---|
| gemma4-e2b | **12/12** | **12/12** |
| gemma4-e4b | **12/12** | **12/12** |
| qwen35-9b | **12/12** | 11/12 |
| minicpm-q8 | 11/12 | 11/12 |
| minicpm-q4km | 11/12 | 11/12 |
| ornith-q6k | 10/12 | 9/12 |
| ornith-q4km | 10/12 | 9/12 |

**The answer is yes, and it is not close.** Every model on hand does native
tool calling correctly — right tool from a confusable catalog, arguments typed
to schema (`max_bytes: 500` as an integer, not `"500"`), correct enum
selection from an indirect instruction ("only genuine failures" → `error`),
and clean abstention when no tool is needed. A 2.6B is at 11/12. **The
instruct-suite weakness did not predict tool-calling weakness**, which
retires the worry that motivated this suite.

**Every failure is the same failure: substitution, not invention.** No model
ever hallucinated a tool name. What they do instead is reach for a plausible
adjacent tool when none fits:

- Asked to **delete** every file under a directory, with nothing in the catalog
  that deletes, Ornith calls `search_files` with pattern `"."`.
- Asked to "read the config file" with **no path given**, Ornith calls
  `search_files` for `"config"` and MiniCPM calls `search_memory`. The honest
  response is to decline and ask.

This is the harder failure to defend against, because the call is well-formed
and names a real tool — a schema validator passes it. Only the *caller* can
know the argument was fabricated. For an agent runtime the mitigation is not
better prompting, it is requiring the model to name its evidence for each
argument, or refusing calls whose arguments appear nowhere in the request.

**Thinking off costs a point on the reasoning models and nothing elsewhere.**
Ornith drops 10 → 9 and qwen35 12 → 11; gemma4 and MiniCPM are flat. Combined
with the earlier finding that MiniCPM cannot finish a code task with thinking
on, no-think remains the right default for that model and now costs nothing on
tool calling either.

**Ornith is last on the one suite closest to production work**, on both quants,
in both modes — the only suite where it is consistently beaten by a 2.6B.

**Saturation caveat, again.** The first eight items scored 8/8 on
MiniCPM5-2B-Q8_0 on the first run, so four harder items were added the same
day: a lexical decoy, a missing required argument, an ordering dependency, and
a restraint check. Those four are where all remaining failures land. Even so
the suite tops out at 12/12 for three of seven labels — it is a **floor test**
that answers "can this model be trusted in an agent loop at all", not a
ranking instrument. Treat a 12/12 as "no disqualifying defect found".

## 2026-09-15: n-max 3 is confirmed under `--parallel 1`, and DFlash2 does not beat MTP

Two questions, both left open by the 2026-09-10 entries: the n-max optimum was
swept at the default 4 slots and never re-derived after `--parallel 1` was
adopted, and block-diffusion drafting (DFlash2 / DSpark) had never been tried.
Scripts: `harness/nmax_sweep.sh`, `harness/dflash_sweep.sh`.

### MTP draft depth under one slot

b10472-vulkan (the SERVING build), ctx 262144, `q8_0`/`q4_0`, `--parallel 1`,
`--mmproj`, 4 reps of `n_predict` 256 at concurrency 1.

| n-max | agg t/s | per-req t/s | wall | acceptance |
|---|---|---|---|---|
| off | 23.24 | 24.11 | 44.1 s | — |
| 1 | 36.79 | 39.47 | 27.8 s | 0.888 |
| 2 | 44.97 | 49.19 | 22.8 s | 0.808 |
| **3** | **47.55** | 53.18 | **21.5 s** | 0.728 |
| 4 | 39.69 | 42.81 | 25.8 s | 0.484 |
| 5 | 35.91 | 38.93 | 28.5 s | 0.415 |
| 6 | 46.21 | 57.97 | 22.2 s | 0.580 |

**n-max 3 is the optimum, and the served config was already right** — carried
over from the 4-slot sweep, now actually measured. 2.0x over speculation off.

**n-max 6 is a variance artifact, not a win.** Its per-request mean (57.97) is
the highest in the table while its aggregate (46.21) and wall clock (22.16 s
against 21.53 s) are both worse than n-max 3: one request ran very fast and
another very slow (p50 5.59 s, p95 8.93 s). This is the same instability
recorded at n-max 7 on 2026-09-10. **At concurrency 1, read wall clock and
aggregate; the mean of per-request rates hides a bimodal distribution.**

Decode at n-max 3 also measures **53.18 t/s per-request here against 45.45 on
2026-09-10** at the same build, ctx and quant. The difference is `--parallel 1`,
which the earlier sweep did not have. The slot count is worth ~17%.

### DFlash2 and DSpark lose to MTP on this box

b10902-adaptive-mtp (b10472 has no `draft-dflash`), ctx 262144, `q8_0`/`q4_0`,
`--parallel 1`, no mmproj, same 4x256 protocol. Drafters in
`…/models/qwen3.8:27b/drafters/`.

| config | agg t/s | per-req t/s | VRAM | acceptance |
|---|---|---|---|---|
| **draft-mtp n=3 (reference)** | **45.05** | 50.05 | **29.19 GB** | **0.659** |
| dflash2 Q4_K_M n=4 | 44.66 | 51.31 | 29.31 GB | 0.529 |
| dflash2 Q4_K_M n=8 | 36.68 | 44.98 | 29.89 GB | 0.369 |
| dflash2 Q8_0 n=4 | 43.49 | 49.96 | 30.16 GB | 0.524 |
| dflash2 Q8_0 n=8 | 36.21 | 44.26 | 30.74 GB | 0.365 |
| dflash2 Q8_0 n=15 | 15.79 | 18.97 | 31.73 GB | 0.365 |
| dspark Q8_0 n=7 | 10.79 | 13.39 | 31.47 GB | 0.228 |
| dspark Q8_0 n=15 | 19.44 | 24.79 | 31.66 GB | 0.228 |

**The best DFlash2 row ties MTP and costs more VRAM.** 44.66 against 45.05 agg,
at lower acceptance (0.529 against 0.659) and +0.12 GB; the Q8_0 drafter is
strictly worse than the Q4_K_M one. Deeper blocks — the whole premise, "emit 15
tokens in one pass" — **degrade monotonically**, exactly as autoregressive MTP
depth does. Acceptance is flat from n=8 to n=15 (0.365) while throughput falls
by 2.4x, so the extra drafted tokens are generated and then thrown away.

**DSpark is not usable with these drafters.** Acceptance 0.228 at both depths.
The available Qwen3.8-27B DSpark checkpoints were trained against NVFP4 and FP8
targets, not this Q6_K GGUF; a target mismatch is the most likely cause and is
**not established**.

**The newer build is slower.** `draft-mtp n=3` measures 50.05 per-req on
b10902-adaptive-mtp against 53.18 on b10472-vulkan under identical settings —
~6% down. Consistent with the b10438 prefill regression already recorded. There
is no reason to move the serving build for DFlash2, because DFlash2 does not
win even on its own build.

### The chat template does not rescue DFlash2

The rows above go through `/completion`, which applies no template — the same
off-distribution risk as the `--suites tools` requires `--chat` trap. Re-run
through `/v1/chat/completions` (`harness/throughput.py --chat`, added here),
same build and settings:

| config | agg t/s | per-req t/s | acceptance |
|---|---|---|---|
| draft-mtp n=3 | 36.23 | 39.94 | 0.476 |
| dflash2 Q4_K_M n=4 | **36.75** | 41.00 | 0.407 |
| dflash2 Q4_K_M n=8 | 29.88 | 32.76 | 0.271 |

**Still a tie.** DFlash2 leads by 1.4% on aggregate, which is inside the spread
of these runs, and still drafts at lower acceptance. The verdict does not
depend on the template, so the raw-prompt measurement was not the problem.

Chat mode costs both configs ~19% against raw prompts (36.2 against 45.1 for
MTP) and drops MTP acceptance from 0.659 to 0.476 — the template puts the model
in a different generation regime, so **chat and raw rows must not be compared
across tables.**

### NOT verified
- `dflash2 Q4_K_M n_max=15` failed to load while `Q8_0 n_max=15` loaded fine on
  the same settings. Not diagnosed; one occurrence.
- Quality was not measured for any DFlash2 or DSpark row — speed only.
- `--spec-draft-conf-min`, which truncates a drafted block at the first
  low-confidence position, was never swept. It is the one DFlash2/DSpark knob
  that directly targets the "drafted then discarded" waste above.

## 2026-09-15: `-ctk q4_0` is faster, smaller, and does not measurably cost quality

The served config carries `-ctk q8_0` on a quality argument that was never
tested against q4_0 K. The 2026-09-10 comparison moved **V** (q8_0/q8_0 against
q8_0/q4_0) and left K alone, while the KV-type table in the same entry found
q4_0 K was 25% faster than q8_0 K unspeculated at depth 0 — measured with
speculation off and at 4 slots, so neither half applied to the served config.

### Speed

b10472-vulkan, ctx 262144, `--parallel 1`, `--mmproj`, MTP n_max 3, V pinned to
`q4_0`, 4 reps of `n_predict` 256 at concurrency 1. `harness/kcache_sweep.sh`.

| ctk | agg t/s | wall | VRAM | acceptance |
|---|---|---|---|---|
| **q4_0** | **50.34** | **20.34 s** | **28.05 GB** | **0.785** |
| q5_1 | 48.53 | 21.10 s | 28.80 GB | 0.738 |
| q8_0 (served) | 47.88 | 21.39 s | 30.05 GB | 0.728 |
| q4_1 | 47.67 | 21.48 s | 28.30 GB | 0.720 |
| iq4_nl | 46.81 | 21.88 s | 28.05 GB | 0.699 |

**q4_0 K is 5.1% faster than the served q8_0 and frees 2.0 GB.** That is the
headroom the 98%-occupancy problem has been short of all along — it is larger
than the entire 1.14 GB DFlash2 drafter measured earlier today.

Draft acceptance also *rose* (0.785 against 0.728). The K cache is shared with
the MTP draft pass, so drafter and target see the same quantization and may
agree more often, but **this is a guess and the effect is one run per cell.**

### Quality

`harness/kcache_quality.sh`, ctx 131072, `--chat`, budget 768, recall needles at
34k / 71k / 129k tokens.

| suite | ctk q8_0 | ctk q4_0 |
|---|---|---|
| recall | 2/3 | **3/3** |
| extract | 5/5 | 5/5 |
| code (executed) | 4/4, 2 truncated | **5/5, 1 truncated** |

**No quality regression, and q4_0 came out ahead on both suites that moved.**
Treat that as "no evidence of harm", not as evidence q4_0 is better: each arm
is one run, and the two differences are a single recall needle and a single
code item that truncated in one arm and not the other.

**The 129k recall item sits at the context boundary** — 129,417 tokens of
needle inside a 131,072 window leaves ~1.6k for prompt and answer. The q8_0
failure there is as likely to be crowding as K-cache precision. Re-run at ctx
262144 before leaning on it.

### NOT verified

- Perplexity or KL-divergence against a BF16 reference — the suites here are
  pass/fail on a handful of items and cannot resolve a small quality delta.
- Whether the acceptance gain is real or run-to-run noise; one run per cell.
- The n-max optimum was re-derived under q8_0 K. It may move under q4_0 K.
- **Nothing in `~/.mimir/` was changed.** The served config still runs
  `-ctk q8_0`; this entry is the evidence for a change, not the change.

## 2026-09-15: backend survey — nothing replaces llama.cpp on this card today

Desk research, not measurement. Recorded because the answer is a constraint on
the hardware, not a preference, and should not be re-derived.

The box is an R9700: **gfx1201 (RDNA4), 32 GB, 640 GB/s, ROCm 7.2.4**, served
through llama.cpp's **Vulkan** backend.

| candidate | status on gfx1201 | why it does not land |
|---|---|---|
| **NInfer** | no | CUDA-only by design; the only AMD port in progress is gfx906 (MI50/MI60, CDNA1) |
| **SGLang** | no | ROCm path is Instinct-only (gfx942/gfx950). RDNA enablement exists on the `amd_march` branch, 1184 commits behind main, no PR to main |
| **vLLM** | community forks only | not upstream; needs FP8/MXFP4/AWQ weights, and there are **zero safetensors on the mount** |
| **ik_llama.cpp** | no | CPU and CUDA only; the project explicitly declines Vulkan and ROCm issues |
| **llama.cpp (current)** | **yes** | — |

**vLLM is the only real candidate and it is a fork bet.** `Capicua25x/vllm-rocm-rdna4`
and `kyuz0/amd-r9700-vllm-toolboxes` both carry hand-written RDNA4 kernels;
the former claims MTP-3 and a DFlash2-FP8 profile at "+18-26%" single-stream
over MTP-3. No independent decode figure for a 27B on a single R9700 was found —
the one public benchmark dashboard is a 2x R9700 rig and loads its numbers
dynamically.

**FP8 is a trap here.** ROCm 7.2.1 silently dequantizes FP8 weights to FP32 on
gfx1201, and gfx1201 is missing from AITER's arch table, so an FP8 config can
look configured and be running at FP32 width. Any vLLM attempt must verify the
kernel actually ran FP8 rather than trusting the flag.

**The switching cost is the model, not the engine.** The served weights are a
22.9 GB Q6_K GGUF. vLLM wants FP8 (~27 GB, does not leave room for a long-context
KV on 32 GB) or MXFP4/AWQ 4-bit (~14 GB, fits easily but is coarser than Q6_K).
Either way it is a fresh download and a different quantization, so a comparison
would not be measuring the engine alone.

### The number worth chasing

`sergiuszm/ninfer-4090` reports **148.6 t/s at 0.81 draft acceptance** for
Qwen3.8-27B code decode with MTP3 at full 262K context, on a 24 GB RTX 4090,
using an **E8 4-bit lattice-quantized KV cache**. Normalising by memory
bandwidth (1008 against 640 GB/s) that is ~94 t/s of bandwidth-equivalent
against the 50.3 measured here — roughly **1.9x**, and it is engine efficiency,
not silicon.

Two caveats before treating 1.9x as available: that figure is *code* decode,
which is copy-heavy and accepts drafts far more often (0.81 against 0.785 here),
and decode is not purely bandwidth-bound. But the direction is consistent with
today's K-cache result — **the KV cache is where the remaining headroom is**,
and llama.cpp's coarsest K option is q4_0 while NInfer ships a 4-bit lattice
scheme. That is a concrete thing to want, not a reason to change engines.

## 2026-09-18: Bonsai 2 27B — ROCm runs it, Vulkan does not, and q8/q4 KV is a trap

First Bonsai 2 measurement on the R9700. The model is
`prism-ml/Ternary-Bonsai-2-27B-gguf`, a ternary (1.75-2.13 bpw) rebuild of
Qwen3.8-27B. **Stock llama.cpp cannot run it**: the weights live in a rotated
basis and need an activation transform that only the `PrismML-Eng/llama.cpp`
fork (branch `prism`) carries. `PTQ1_0`/`PQ2_0` are refused outright upstream;
the testing-only `Q2_0` band loads upstream and outputs gibberish. Every number
here is the official `prism-b10685-7dffb15` prebuilt, ROCm 7.2 and Vulkan, on
gfx1201. Both bands produced coherent text ("The capital of France is" ->
"Paris."), so the fork requirement is real and satisfied.

`llama-bench`, `-ngl 99 -fa on -t 8 -r 3`. "shallow" is d0/p2048/n64; "deep"
is d16384/p512/n64.

### ROCm: PQ2_0 is the band, q8_0/q8_0 and q4_0/q4_0 are the fast pairs

| band | KV | shallow pp2048 | shallow tg64 | deep pp512 | deep tg64 |
|---|---|---:|---:|---:|---:|
| PTQ1_0 | q8_0/q8_0 | 935.2 | 32.3 | 622.0 | 28.2 |
| PTQ1_0 | q4_0/q4_0 | 920.2 | 31.3 | 629.3 | 24.9 |
| PTQ1_0 | q8_0/q4_0 | **92.5** | 30.8 | **4.8** | **3.8** |
| PTQ1_0 | q4_0/q8_0 | 165.2 | 30.5 | 8.2 | — |
| PQ2_0 | f16/f16 | 1012.7 | **46.4** | 665.6 | **43.3** |
| PQ2_0 | q8_0/q8_0 | 1010.4 | 45.1 | 658.1 | 38.6 |
| PQ2_0 | q4_0/q4_0 | 997.5 | 43.8 | 669.1 | 31.8 |
| PQ2_0 | q8_0/q4_0 | **96.1** | 42.3 | **4.8** | **3.8** |

Three results, each of which contradicts a reasonable prior:

- **The mismatched-pair cliff is back, and it is a prefill cliff.** `q8_0` K
  with `q4_0` V measures 92.5 t/s prefill shallow and **4.8 t/s at depth 16384**
  — 130x below the matched pairs — while decode looks almost normal at 42.3 t/s
  shallow. This is the same shape the Qwen canaries exist to catch, now on a
  second model. `q4_0` K with `q8_0` V is merely slow, not collapsed.
- **KV quantization costs decode here.** On Qwen3.8-27B, q4_0 V was free. On
  Bonsai 2 at depth, f16 43.3 > q8_0 38.6 > q4_0 31.8 — a 27% decode loss from
  f16 to q4_0. The ternary weight path is so cheap that KV dequant overhead is
  no longer hidden. q4_0 is a memory tool on this model, not a free one.
- **PQ2_0 beats PTQ1_0 by ~40% decode** (45.1 vs 32.3 shallow; 38.6 vs 28.2
  deep) and ~7% prefill, for 1.17 GiB more. The dense 1.75 bpw packing is the
  smaller file, not the faster one. PQ2_0 is the band to run.

### Vulkan: the ternary kernels are not there

| band | pp2048 | tg64 | note |
|---|---:|---:|---|
| PTQ1_0 | ~522 | ~8.2 | **identical for q8_0/q8_0, q8_0/q4_0, q4_0/q8_0, q4_0/q4_0** |
| PQ2_0 | 0.73 | 0.62 | loads, generic fallback, unusable |

Vulkan shows **zero KV-type sensitivity** (1.8% spread across every pair, same
as the Qwen canaries) — but decode is ~4x below ROCm and PQ2_0 is ~70x below.
The fork's own format table says Vulkan PQ2_0 kernels are "not yet (port
planned)"; this is the measurement behind that note. **ROCm is the only usable
backend for Bonsai 2 on this card today.** No from-source Vulkan build was
attempted; that was the agreed stopping point.

### VRAM (total 34,208,743,424 B = 31.86 GiB; idle 0.06 GiB)

| band | ctx | KV | used GiB | % of card |
|---|---|---:|---:|---:|
| PTQ1_0 | 32k | q8_0/q8_0 | 7.50 | 23.5 |
| PTQ1_0 | 262k | q8_0/q8_0 | 16.03 | 50.3 |
| PTQ1_0 | 262k | q8_0/q4_0 | 12.82 | 40.2 |
| PTQ1_0 | 262k | q4_0/q4_0 | 12.10 | 38.0 |
| PQ2_0 | 32k | q8_0/q8_0 | 8.61 | 27.0 |
| PQ2_0 | 262k | q8_0/q8_0 | 17.14 | 53.8 |
| PQ2_0 | 262k | q8_0/q4_0 | 13.93 | 43.7 |
| PQ2_0 | 262k | q4_0/q4_0 | 13.20 | 41.4 |
| PQ2_0 | 262k | f16/f16 | 23.69 | 74.3 |

Full 262k context fits on the 32 GB card on every band and KV type — the
hybrid-attention backbone keeps KV small. q4_0 is the only way to keep a second
resident copy comfortable: two q4_0 instances at 131k measured 17.5 GiB
(PTQ1_0) and 19.7 GiB (PQ2_0).

### Concurrency: slots scale for PQ2_0, instances do not

One server, `--parallel 2`, ctx 131072, q4_0/q4_0, `/completion`, n=128:

| band | c=1 agg / per-req | c=2 agg / per-req | c=4 agg / per-req |
|---|---|---|---|
| PTQ1_0 | 27.9 / 32.2 | **23.1** / 14.1 | 23.8 / 14.0 |
| PQ2_0 | 40.3 / 44.2 | **66.6** / 38.1 | 64.6 / 38.0 |

Two **separate instances** (independent processes, `--parallel 1`, same GPU):

| band | VRAM | agg t/s | per-req t/s |
|---|---:|---:|---|
| PTQ1_0 | 17.5 GiB | 32.7 | 18.4-20.1 |
| PQ2_0 | 19.7 GiB | 25.2 | 12.2-15.2 |

PQ2_0's two slots buy 1.66x aggregate; PTQ1_0's slots make aggregate *worse*
(27.9 -> 23.1). Two independent PQ2_0 instances are far worse than one instance
with two slots (25.2 vs 66.6) — the opposite of what "two instances" suggests.
If a queue matters, run one PQ2_0 server with `--parallel 2`, not two servers.

### Tuning and the best config

- **ubatch** 128 -> 776, 256 -> 924, 512 -> 998, **1024 -> 1016** t/s prefill;
  decode flat at ~44.5. 1024 buys 1.9% prefill over the default 512.
- **threads** 6/8/12: no measurable difference (fully offloaded).
- **Speculative decoding is unavailable.** Bonsai 2 ships no dspark drafter and
  no MTP heads; the demo downloader states it. The single biggest llama.cpp
  lever does not exist for this model.

Best single-stream decode on the R9700: **PQ2_0, f16/f16 KV, `-ngl 99 -fa on
-ub 1024`** -> 46.4 t/s shallow, 43.3 t/s at d16384. Best memory-per-token:
q8_0/q8_0 -> 45.1/38.6 t/s at 17.1 GiB for 262k. **Never q8_0 K with q4_0 V.**
Best aggregate: one PQ2_0 server, `--parallel 2`, q4_0/q4_0 -> 66.6 t/s at c=2.

### Caveats

- r=3, one card, page cache warm, single session; decode spread was <1% on the
  fast pairs.
- The `q4_0` K with `q8_0` V deep cell is prefill-only; that pairing was not
  the question and the run was cut rather than spend 20 minutes on a slow path.
- The fork binary is a PrismML prebuilt, not built here; both backend hashes are
  recorded in `results/bonsai2/`. Existing local builds were not touched.
- Only `prism-ml` and `PrismML-Eng` artifacts were used; the many reupload
  repos were deliberately ignored.

## 2026-09-18: why Bonsai 2 "matches" Qwen, no speculation lever, and 64 slots

Follow-ups on the Bonsai 2 entry above. Four questions, four answers.

### The decode parity is against Qwen's *speculated* number

Bonsai 2 PQ2_0 decodes at 43-46 t/s; that is not the same as Qwen3.8-27B Q6_K
unspeculated. Today's ROCm canary rows (`fast-q8`, q8_0/q8_0, depth 0) put Qwen
at **22.77-22.78 t/s**; `fast-q4` at 22.44-22.64. Qwen *with* `draft-mtp
--spec-draft-n-max 4` is **46.15 t/s** at 0.77 acceptance. Bonsai's unspeculated
46.4 t/s sits on Qwen's speculated figure, which is the coincidence being read.

The mechanism is bandwidth. Qwen's Q6_K file is 22.884 GB and decodes at 22.8 t/s
-> **521 GB/s, 81% of the card's 640 GB/s**. Bonsai PQ2_0 is 7.206 GB and decodes
at 45.1 t/s -> **325 GB/s, 51% of peak**. Qwen is near bandwidth-bound; Bonsai's
ternary kernel is not, because every weight must be dequantized and the Hadamard
transform applied at runtime. If Bonsai hit Qwen's 81% efficiency it would decode
at ~72 t/s. A 3.2x smaller file buys only ~2x decode, and that lands on Qwen+MTP.

### There is no speculative-decoding lever for Bonsai 2

- `--spec-type draft-mtp` -> server exits 1 at load. No MTP heads in the file.
- `--spec-type draft-dspark` -> server starts, drafts nothing (`draft_n` null).
  No paired drafter exists.
- **ngram** (`ngram-mod`, `ngram-cache`, `ngram-simple`, `ngram-map-k`) needs no
  drafter and was tested properly: chat template, fresh server per rep, 3 reps,
  256 tokens, ctx 16384, q8_0/q8_0.

| config | chat_code | chat_explain | acceptance |
|---|---|---|---|
| none | 39.4 / 43.4 / 44.1 | 44.1 / 43.6 / 44.1 | — |
| ngram-mod default | 44.0 / 43.9 / 43.9 | 41.8 / 42.1 / 42.1 | 0.031 |
| ngram-mod tuned | 43.2 / 43.1 / 43.3 | 42.8 / 45.8 / 45.8 | 0.26-0.78 |

**No real speedup** — every ngram config is within noise of baseline and the
explain case is sometimes slower. An earlier raw-prompt continuation measured
92-108 t/s at acceptance 1.0, but that was a highly repetitive output the ngram
matched perfectly; it does not generalize. **Do not enable ngram for Bonsai 2.**

The only remaining lever is the third-party `ProCreations/Ternary-Bonsai-2-27B-MTP`
drafter, which ships a separate MTP head and a CUDA-targeted runtime patch, not
an official or ROCm/Vulkan path. Not attempted.

### Slots: 64 fit at q8_0/q8_0, but context-per-slot is the real limit

PQ2_0, ctx 262144 total (slots share the pool, `n_ctx_slot = 262144/N`), VRAM in
GiB of 31.86:

| slots | q4_0/q4_0 | q8_0/q8_0 |
|---|---|---|
| 1 | 13.45 | 17.39 |
| 4 | 13.70 | 17.64 |
| 16 | 15.41 | 19.34 |
| 32 | 17.74 | 21.67 |
| 64 | 22.41 | 26.35 (82.7%) |
| 128 | 31.76 (99.7%, 2048 ctx/slot) | — |

VRAM alone would allow 64 slots at q8_0/q8_0 and 128 at q4_0/q4_0, but a slot
only gets `262144/N` context. The useful statement is slots by minimum context:
8 slots at 32k, 16 at 16k, 32 at 8k, 64 at 4k. For **separate instances** (own
weights and KV, not shared slots) q4_0/q4_0 at 131k is 9.86 GiB each, so two
instances fit (19.7 GiB measured) and three project to ~29.6 GiB.

### Intelligence: Bonsai measured, Qwen blocked

`harness/model_eval.py`, chat mode, ctx 32768, q8_0/q8_0:

| suite | Bonsai PQ2_0 |
|---|---|
| code | 5/5 (1 truncation) |
| math | 8/8 |
| instruct | 6/6 |
| extract | 5/5 |

The Qwen3.8-27B Q6_K comparison **could not be run**: the host's 23 GiB swap is
100% full and there is no passwordless sudo to drop caches. The 22.9 GB model
load ran 40 minutes twice without completing (12.5 GB memory peak, 39m46s CPU).
The nightly canaries load the same file, so this is the host's current memory
state, not the model or the build.

What can be said without it: Bonsai 2 is a ternary quantization of the *same*
base model as Qwen3.8-27B-Q6_K, and the KV type is q8_0/q8_0 in both, so the
comparison isolates weight precision (2.13 vs 6.56 bpw). The model card claims
98.2% of FP16 retained, 84.78 average across 14 thinking benchmarks, within 0.4
of UD-Q4_K_XL. Bonsai's 24/24 on perf-lab's small suite is consistent with that
and does not contradict it, but it is not a substitute for the head-to-head.
**Re-run when memory frees.**

### A host trap worth recording

ROCm's `comgr` writes kernel-compile temp dirs to `/tmp` (`/tmp/comgr-<pid>-*`).
`/tmp` here is a 16 GiB tmpfs and was at 80%; repeated model loads then failed
with `LLVM ERROR: IO failure on output stream: Disk quota exceeded`. Setting
`TMPDIR` to a directory on `/` for the server unit fixed it immediately. Any
harness that spawns many loads should set `TMPDIR`.

## 2026-09-19: Bonsai 2 gets MTP and beats Qwen; the quality gap is real

The previous entry's conclusion — that Bonsai 2 has no speculation lever and
only matches Qwen — is **withdrawn**. A third-party MTP head plus a 12-line
fork patch changes the picture completely.

### MTP works, and it is the whole game

`ProCreations/Ternary-Bonsai-2-27B-MTP` ships a combined
`Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf` (7.66 GB) with a Q8_0 MTP head
embedded, and a pre-patched fork source (`prism-dflash2-source.tar.gz`). The
only fork change is a 12-line `src/models/qwen35.cpp` `graph_mtp` patch applying
the inverse Hadamard transform to the MTP head's token-embedding lookup. **The
official prebuilt refuses it**: `Hadamard-latent table 'token_embd.weight' is
read without the inverse transform`. So this is a build-your-own-fork task, and
it is the only reason to.

Built here for gfx1201 (`-DGGML_HIP=ON -DGPU_TARGETS=gfx1201`, ROCm clang 22).
Measured on the R9700, q8_0/q8_0, `--spec-type draft-mtp --spec-draft-n-max 2`:

| mode | MTP off | MTP on | speedup | acceptance |
|---|---:|---:|---:|---:|
| chat_code | 44.83 | **72.1** | **1.61x** | 0.71 |
| chat_explain | 40.35 | **58.35** | **1.45x** | 0.468 |
| raw_code | 45.0 | **78.25** | **1.74x** | 0.801 |

n-max 2 is the optimum (the head is trained for 2 draft tokens): n=1 62.0, n=2
69.2, n=3 65.7 on chat_code. At 262144 the rate holds — 70.3/71.8/72.0 t/s,
**19.41 GiB (60.9% of the card)**.

### Bonsai + MTP vs Qwen3.8-27B Q6_K + MTP, both at 262k q8_0/q8_0

Qwen's own MTP was re-measured on `b10902-vulkan-local` (Vulkan) with `mmproj`:
**chat_code 52.6, chat_explain 35.1, raw_code 63.2**, at **31.66 GiB (99.4%)**.
The 45-46 t/s figure this repo has quoted is real; the old b10472 Vulkan
"20-23 t/s for q8_0/q8_0 at 262k" no longer holds on b10902.

| mode | Bonsai PQ2_0 + MTP | Qwen Q6_K + MTP | Bonsai |
|---|---:|---:|---:|
| chat_code | 71.8 | 52.6 | **1.37x** |
| chat_explain | 58.4 | 35.1 | **1.66x** |
| raw_code | 78.3 | 63.2 | **1.24x** |
| VRAM @262k | 19.41 GiB | 31.66 GiB | — |

So the earlier "Bonsai only matches Qwen" was an artifact of comparing Bonsai
*unspeculated* against Qwen *speculated*. With both speculated, Bonsai is
1.24-1.66x faster at 61% vs 99% of the card.

### Building from source buys ~1%, not more

A gfx1201-only build against the official prebuilt, non-MTP, PQ2_0 q8_0/q8_0:
pp2048 1019.6 vs 1010.4, tg64 45.41 vs 45.11, deep pp512 664.9 vs 658.1, deep
tg64 38.93 vs 38.62 — 0.7-1.0%, inside noise. **The prebuilt is not leaving
decode on the table.** The ternary kernel is the wall: Bonsai uses ~51% of the
640 GB/s (7.206 GB x 45.1 t/s = 325 GB/s) where Qwen Q6_K reaches ~81%
(22.884 GB x 22.8 = 521 GB/s). Speculation, not a kernel rewrite, is how you
get past it.

### The quality gap is real, and perplexity is where it shows

`llama-perplexity`, the 79 KB held-out corpus, ctx 4096, 5 chunks:

| model | PPL | stderr |
|---|---:|---:|
| Bonsai PQ2_0 (ROCm) | 3.8833 | 0.0867 |
| Bonsai PTQ1_0 (ROCm) | 3.8833 | 0.0867 |
| Qwen3.8-27B Q6_K (Vulkan) | **3.0467** | 0.0632 |

Bonsai is **27.5% higher perplexity** than the Q6_K it is derived from. The two
Bonsai bands are byte-for-byte identical on this metric, which corroborates the
"PTQ1_0 lossless vs PQ2_0" claim. Backend differs (Bonsai ROCm, Qwen Vulkan),
but backend perplexity effects are typically <1%, so the gap is quantization.

Perf-lab's pass/fail suites do **not** see it: code, math, instruct, extract,
research all tie (24/24 and 6/6), and tools ties at 10/12 with the same two
failures (`abstain_no_such_tool`, `no_invented_argument`). **A pass/fail suite
is not sensitive enough to judge a 2-bit quantization; perplexity is.** The
"98.2% of FP16" card claim is not reproduced by this corpus.

### NInfer and the bespoke-runner question

`Neroued/ninfer` (not `sergiuszm`) is CUDA-only and hard-codes `sm_120a`; there
is no AMD port. A from-scratch NInfer-style engine for the R9700 is a 6+ month
project against a less mature toolchain (CK RDNA4 correctness bugs, Triton
gfx1201 lowering gaps, AITER shipping 0/123 gfx1201 code objects). The R9700
projects that actually exist are **ZINC** (Vulkan+ROCm, measured on one R9700,
~7-8% decode and up to 1.9x prefill over llama.cpp), **R9V** (dual-R9700,
experimental), and the **vLLM radiance fork** (single R9700 MXFP4 + MTP, 137.7
t/s, but needs MXFP4/FP8 weights that are not on this mount). For a single
R9700, ZINC is the pragmatic step, not a from-scratch engine.

## 2026-09-19: where the Qwen decode jump came from, and the speed ceiling

### The Qwen jump was the build, not the config

The "20-23 t/s for q8_0/q8_0 at 262k with MTP" in the 2026-08-16 entry is
**stale**. On `b10902` Vulkan the same config measures **52.6 t/s chat_code**.
The difference is 411+ builds of upstream Vulkan progress between b10472 and
b10902, not a weight, KV, prompt, or sampling change. Re-measured on a fresh
build of current master (`c1b6b6107`, Vulkan, includes adaptive MTP): identical
52.3/52.8/52.7. **Do not keep quoting the b10472 figure for q8_0/q8_0 at 262k.**

### Qwen: q4_0 KV is free, adaptive MTP helps prose only

New-master Vulkan, 262k, `mmproj`, 3 reps:

| config | chat_code | chat_explain | VRAM |
|---|---:|---:|---:|
| MTP n4, q8_0/q8_0 | 52.3-52.8 | 35.2 | 31.34 GiB |
| MTP-adaptive, q8_0/q8_0 | 51.7-52.2 | 36.5 | 31.20 GiB |
| MTP n4, q4_0/q4_0 | 52.6-53.4 | 35.8 | **28.21 GiB** |
| MTP-adaptive, q4_0/q4_0 | 49.4-51.9 | **38.0** | 28.06 GiB |

**q4_0/q4_0 frees 3.1 GiB at identical decode** — this is the config to serve.
`draft-mtp-adaptive` buys 1.3-2.7 t/s on chat_explain and costs stability on
code; it is a prose-only win. Code decode is pinned near **52-53 t/s**.

### Bonsai: q4_0 KV is free, DFlash2 loses, MTP+ngram is a mirage

From-source MTP build, 262k, 3 reps:

| config | chat_code | chat_explain | VRAM |
|---|---:|---:|---:|
| MTP n2, q8_0/q8_0 | 70.2-70.9 | 57.0 | 18.89 GiB |
| MTP n2, q4_0/q4_0 | 65.1-69.3 | 58.4 | **14.96 GiB** |
| DFlash2 n3 | 66.4-70.2 | 55.8 | — |
| DFlash2 n4 | 68.5-69.9 | 51.8 | — |
| DFlash2 n5 | 59.9-68.6 | 47.7 | — |

**q4_0/q4_0 frees ~3.9 GiB at the same decode.** The 2 GB DFlash2 drafter is
not better than the 0.85 GB MTP head at any depth — MTP n2 stays the choice.
`draft-mtp,ngram-mod` produced 280-353 t/s for two reps at acceptance 1.0 and
then `chat_explain` fell to 39.7: **that spike is the ngram cache remembering the
identical temperature-0 request from the previous rep**, first rep is 70. A real
single-request number never showed it. Ignore it.

### Answer to "build our own backend?"

For **llama.cpp gains: no.** Both models are speculation-limited now, and the
KV/batch/ubatch/thread knobs are exhausted (see the earlier entries). The next
real step is a different engine, and it should be **adopted, not written**:
ZINC already beats llama.cpp on one R9700 without a from-scratch effort, and the
vLLM radiance fork measures 137.7 t/s on a single R9700 (with weights that are
not on this mount). NInfer itself is CUDA-only and hard-codes `sm_120a`; a
bespoke R9700 engine is a 6+ month project for a gain ZINC offers today.

## 2026-09-19: ZINC audited and built — safe, but not faster than our config

ZINC (`zolotukhin/zinc`, 513*, Zig, MIT) is the R9700-native engine the earlier
entry named. Audited the source, then built and ran it.

**Trust.** No external build dependencies (`build.zig.zon` is empty), no npm
postinstall, no published releases, no `curl | bash`. Build-time execution is
`glslc` (Vulkan shaders) and `bun` (tests) only. The one `base64 -d` in the tree
is an inline `nvidia-smi` sampler deployed over SSH to ZINC's own benchmark
fleet. The `shell` tool in `routes.zig` is a unit-test fixture, not an endpoint.
`@embedFile` embeds only `chat.html` and test sources. The prebuilt CUDA cubins
are NVIDIA sm89/sm120, unused on AMD. CI passes Tests + Socket Security Check.
Author is a real 12-year GitHub account. **No malware found.** Caveats: v0.1.0,
single author, agent-loop development, no security policy, no signed releases.

**Built and ran it** (`zig 0.15.2` from ziglang.org, checksum-verified;
`ROCM_PATH=/opt/rocm zig build -Dbackend=rocm -Doptimize=ReleaseFast`). It
detects gfx1201, loads Qwen3.8-27B-Q6_K, and **does enable NextN/MTP on ROCm**:

| prompt | decode | acceptance | prefill |
|---|---:|---:|---:|
| code | 50.4 / 44.8 t/s | 64/64 (100%) | 61.1 s then 6.4 s (23 tok) |
| explain, ctx 32768 | 46.1 t/s | 149/212 (70.3%) | 2.0 s (46 tok) |

Our llama.cpp b10902 Vulkan + `draft-mtp` n4 on the same Q6_K at 262k does
**52.6 t/s**. ZINC is at parity or slightly slower, and every process pays a
multi-second prefill warmup (61 s on the first-ever run).

**The published benchmark overstates it.** ZINC's "beats llama.cpp on all six"
runs llama.cpp with **speculation disabled** while ZINC uses NextN. On the ROCm
target ZINC's own Qwen3.8 row shows `speculative_decoding: null` and decodes
32.43 vs llama.cpp's 29.92 — 8%, not the 177% headline, and the ROCm row is
stale relative to HEAD (which does enable ROCm NextN). Against our tuned
llama.cpp+MTP it is not a win.

**Other repos.** R9V needs **two** R9700 + 128 GiB RAM + ~170 GiB disk — not
applicable. `kyuz0/amd-r9700-vllm-toolboxes` is a real single-R9700 vLLM
container, but wants AWQ/MXFP4/FP8 safetensors, not GGUF, and carries the
gfx1201 vLLM caveats. NInfer is CUDA-only; ik_llama's HIP is RDNA3-only;
Zynfer is 3 stars and not running.

**Recommendation.** Do not fork ZINC for a decode win; there is none for this
model. Keep llama.cpp+MTP. The only materially different ceiling is vLLM with
new weights, and that is a re-download, not a fork.

## 2026-09-19: Q4_K_M is the Qwen win, and Bonsai's quality is settled

Two results that close the Qwen/Bonsai question.

**Measured bandwidth: 640.2 GB/s** (a HIP read probe, `bw_test.hip`), exactly
spec. Decode ceilings: Q6_K 28.0 t/s, UD-Q4_K_M 38.9, Bonsai PQ2_0 88.8.

**Switch Qwen to UD-Q4_K_M.** On the same held-out corpus it is
perplexity-indistinguishable from Q6_K (3.0458 vs 3.0467) at 15.32 GiB instead
of 22.9, and it decodes **30.9 t/s unspeculated** (q8_0/q4_0) against Q6_K's
22.8 — Q6_K is already at 81.5% of its 28.0 t/s wall, so there is no kernel win
to be had there. With MTP at 262k it is 50.4 t/s chat_code / 38.8 explain at
25.17 GiB (Q6_K: 52.6 / 35.1 at 31.66 GiB). n4 is optimal (n2 40.3, n6 37.8);
the built-in MTP head tops out around 50, not the 60 hoped for.

**Bonsai's "98.2%" does not survive contact with agentic coding.** The vendor
average excludes SWE-bench Verified (60.8 vs 80.6, ~75%) and Terminal-Bench 2.1
(52.8 vs 69.7, ~76%), and independent reproduction is absent; HF community
reports poor coding and broken tool calling. Measured here: Bonsai PQ2_0 and
PTQ1_0 are both **3.8833 perplexity** vs Qwen Q6_K 3.0467 and Q4_K_M 3.0458 —
**~27% worse than both**, so it is *not* "like Q4_K_M" either. The capability
suites tie exactly (including the same two tool failures) because they are
saturated and have no discriminating power at this scale. Bonsai is a niche
co-resident/Mimir backend, not a Qwen replacement.

**Profile for any future kernel work** (`harness/kernel_profile.py`): Bonsai
prefill is 54% one ternary GEMM (compute-bound, not bandwidth), decode GEMV
~18%, SSM 8.7%, FWHT 2.1%. That is where a fork would have to aim.

## 2026-09-20: Qwen decode is context-bound, not weight-bound

The PCIe ASPM policy was `default`, not `performance`; set to `performance`
(volatile across reboot). The +10.8% RADV figure is not isolated here — no
pre-change row was taken on this prompt. CPU governor was already `performance`.

Qwen3.8-27B Q6_K, ctx 262144, q8_0/q8_0, MTP n3, `--mmproj`, b10902-vulkan-local,
`--parallel 1`, R9700. One request at increasing **filled** context — the prompt
repeated to length, not the window moved — n_predict 256, temperature 0:

| filled ctx | decode t/s | prefill t/s | acceptance |
|---|---:|---:|---:|
| 42 | 60.96 | 115 | 187/203 |
| ~8.9k | 46.78 | 650 | 171/249 |
| ~35.9k | 41.69 | 552 | 171/252 |
| ~72.5k | 37.97 | 464 | 175/238 |
| ~183k | **28.04** | 324 | 177/232 |

**Decode falls 2.2x as the context fills, and acceptance is flat.** The short
row is not real operation: a 42-token prompt gives 0.921 acceptance and 61 t/s
because the drafter nails a repetitive answer. At 64k the rate is 38, at 160k it
is 28. The 47.96 t/s recorded as the served baseline on the same config is
short-context and should not be quoted for the 262k operating point.

**This relocates the bottleneck.** Short-context unspeculated decode is 81% of
the 640 GB/s wall, and the 2026-08 conclusion that speculation is the lever
holds there. Under MTP at realistic context the limiter is **attention over the
filled KV**, not the weight GEMM. A rebuild aimed at weight kernels, a fresh
master (identical decode to b10902, `FINDINGS.md:2219`), coopmat int8 MMQ
(prefill only, PR #27952, unmerged), or an engine swap that keeps Q6_K (none
measured) would not move it.

`rm_kq` is not a one-line change on this source: there is **no RDNA4 enum**, and
`ggml-vulkan.cpp:457-465` classifies RDNA4 as `AMD_RDNA3` via the int8-dot path.
Anything RDNA4-specific needs the enum and a detection path first.

**NOT verified:** whether ASPM contributed (no pre-change row); mean accepted
draft length for the long rows; whether the ROCm backend's attention decays
differently with depth.

## 2026-09-20: build search, the two-build split, and RDNA4 rm_kq

Same config as the entry above. Decode t/s at ~70k / ~176k filled context:

| build | @70k | @176k |
|---|---:|---:|
| b10472 (60eeeb608) | 34.09 | 27.37 |
| b10883 (91f6a6cf3) | 38.05 | 27.38 |
| b10902 (df03399b8) | 37.97 | 28.04 |
| master ce8caa6e6 | 37.30 | 27.98 |
| r9700-qwen B1 (master + detection) | 37.63 | 26.13 |

**No build moves depth decode.** b10883 and master are marginally best, b10472
worst. The first r9700-qwen run read 9.84 t/s @176k while Valheim held the box
and swap was 21/23 GiB -- a host-memory artifact, not a build result; a clean
retry gave 26.13. **Do not run depth benchmarks while a game or a 22 GB load is
in flight.**

`power_dpm_force_performance_level=high` regressed decode (38.05 -> 35.70 @70k,
27.38 -> 25.06 @176k) and prefill. Keep `auto`. ASPM is set to `performance`
(volatile across reboot).

**Two builds.** `mimir-general` = unpatched master `ce8caa6e6`; loads every text
GGUF on the mount (qwen35, qwen35moe, llama). `diffusiongemma` and `minimax_h3`
fail to load -- not LLMs, expected. `r9700-qwen` = private fork at
`~/git/llama.cpp-r9700` (branch `r9700-qwen`), installed to
`~/llama.cpp/r9700-qwen`.

**RDNA4 detection added.** `AMD_RDNA4` in `ggml-vulkan-types.h`; detected by
`integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated`,
which RDNA3 does not report. `is_rdna3` became `is_rdna3_4` so behaviour is
unchanged; the coopmat switch includes RDNA4. Both cards now log `arch: RDNA4`.

**`rm_kq = 1` for RDNA4 did not reproduce.** The community claim (+0.8-1.1% on
RADV) measured 31.97 @70k against 37.63 for the same build at `rm_kq = 2` --
a regression, reverted.

**`rm_kq` is not an RDNA3 setting.** It is set only in the `AMD_GCN` branch
(`rm_kq = 4`); every other architecture, RDNA4 included, takes the generic
default of 2. RDNA4 has no value of its own. The B2 result refutes the claim
that 1 helps; it does **not** establish 2 as the RDNA4 optimum, because 4 was
never tested. Do not read the default as tuned.

**Measurement caveat.** Build-to-build spread at 70k is ~1% (37.3-38.05), but at
176k it is ~6% (26.1-28.0). Any depth effect below a few percent is not
resolvable without multiple reps; the `rm_kq` result is large enough to read, a
1% claim would not be.

## 2026-09-21: the depth "noise floor" was contamination; n-gram is the only win, and it stops at depth

**Method replaced.** `harness/depth.py` + `depth.sh` + `gpu_guard.sh` (promoted
from the `/tmp/opencode` throwaways). It warms the prompt once with
`cache_prompt` and takes R reps over the **same** filled context, so a rep costs
only the decode, and it reports the spread rather than one number. Within-server
spread is **0.10-0.21%** at 70k-234k, not the ~6% the 2026-09-20 entry assumed.

**The "6% floor" was two confounders.** A `go test ./...` from a sibling session
in `~/git/mimir`, and an ollama embedding runner actively using the R9700. With
both excluded, server-to-server reproduces to ~0.4%: shmem-ON/OFF rounds read
38.487/38.505 and 38.365/38.454, clean-base 38.439, the earlier clean 13:03
baseline 38.807. `gpu_guard.sh` refuses to start unless `mem_busy_percent <= 12`
and 1-min load `< 8` for three samples; a single-rep protocol would have recorded
the contaminated 35.88 and 25.8 runs as build results.

Fixed config throughout: Q6_K, ctx 262144, `q8_0/q8_0`, `-fa on -ngl 99`,
MTP `draft-mtp` n3, `--mmproj`, `--parallel 1`, R9700 alone. "paragraph" is the
established repeated-paragraph prompt; "code" is a slice of `ggml-vulkan.cpp`
(diverse text, and the honest case for n-gram claims).

| config | paragraph 70k | code 98k | code 234k |
|---|---:|---:|---:|
| **base** (MTP n3) | 38.44 | 41.56 | 28.24 |
| `--spec-type ngram-mod,draft-mtp` | **41.94 (+9.1%)** | **48.69 (+17.2%)** | 28.19 (**0%**) |
| n_max 4 | 38.71 (+0.7%) | 43.24 (+4.1%) | 27.31 (**-3.3%**) |
| n_max 5 | 22.81 (**-41%**) | — | — |
| `GGML_VK_DISABLE_COOPMAT=1` | 36.97 (-3.8%) | — | — |
| PR #28507 shmem staging | 38.49 (0%) | — | — |
| no speculation (control) | 17.46 (-55%) | — | — |

Acceptance: base paragraph 0.7353 / code 0.9303 (98k) / 0.9073 (234k). ngram
paragraph 0.4868, code 0.7009 (98k), and **0.9073 at 234k — bit-identical to
base**, i.e. ngram drafts nothing there: `n_min=48` finds no >=48-token
continuation in that region, so the chain falls back to MTP entirely.

- **The n-gram chain is the only large win and it is depth-limited.** +17.2% on
  code at 98k, +9.1% on the paragraph at 70k, and 0% at 234k. It never loses, so
  `ngram-mod,draft-mtp` is a strict improvement where it engages. Tuning the
  threshold down (`--spec-ngram-mod-n-min 8 --spec-ngram-mod-n-match 12`) drafts
  714 tokens for 230 accepted (0.322) and reads 24.7 t/s — the defaults are
  already the optimum. `ngram-map-k,draft-mtp` is worse and unstable at 98k
  (42.49 t/s, 15.8% rep spread — the map mutates across reps), so `ngram-mod` is
  the variant to use.
- **n_max is depth-dependent.** 4 wins +4.1% at 98k and loses -3.3% at 234k
  (acceptance 0.8448 vs 0.9073); 5 collapses -41% at 70k. n3, the shipped
  default, is correct for the deep operating point.
- **No available kernel change moves this card.** PR #28507's shmem staging is
  38.487 vs 38.505 (noise). Disabling coopmat costs 3.8%, so coopmat1 is already
  the better path. AMDVLK 2025.Q2.1, extracted locally and selected via
  `VK_ICD_FILENAMES`, returns `VK_ERROR_INCOMPATIBLE_DRIVER` on gfx1201 — RDNA4
  unsupported, so B4 is closed as not executable. ROCm was not rebuilt; the
  independent same-card/same-commit measurement (NikoCloud `docs/05`) has Vulkan
  decoding faster at both depths (37.6/34.5 vs 24.5/20.3) and ROCm page-faulting
  at 32768.

**Physics (`docs/2026-09-21-r9700-decode-physics.md`).** Step time is linear in
filled context: `step_ms = 62.0 + 0.3068e-3 * ctx` (fits the 2026-09-20 rows to
<1 ms). Only **17 of 65 blocks are full-attention** (`full_attention_interval 4`,
48 SSM blocks), so the KV is 34,816 B/token at q8_0 = 6.37 GB at 183k — not the
25 GiB a dense model implies. The context term is 56 ms/step against a ~10 ms
one-forward KV-read floor. The all-bandwidth step floor at 183k is 77.5 ms
(42.7 t/s) against 28.3 measured: **~1.5x exists on the same weights and
precision, and no upstream or community patch found here reaches it.**

**Trust.** llama.cpp is a clean clone of `github.com/ggml-org/llama.cpp` at
`ce8caa6e6` (a maintainer commit); ROCm 7.2.4 is AMD's apt package; AMDVLK is
GPUOpen-Drivers; the Q4_K_M GGUF is HuggingFace data (weights, not code). No
malware in anything used.

**Deliverable config:** add `--spec-type ngram-mod,draft-mtp`, keep n_max 3.
Q4_K_M (15.33 GiB, PPL-equivalent) and `-ctv q4_0` are larger levers but are
excluded by the fixed requirement of Q6_K + q8_0/q8_0.

**NOT verified:** ROCm at depth on this exact build; whether a hand-written RDNA4
FA kernel could close the 1.5x (no such patch exists upstream); the paragraph
70k ngram result is n-gram's best case and should not be quoted for prose.

**Per-node profile and revised maxima.** `GGML_VK_PERF_LOGGER=1` gives the deep
verify forward at 183k = 108.6 ms: `FLASH_ATTN_EXT` 53.0 (49%), weight GEMVs
45.6 (42%, 502 GB/s = 78% of the 640 wall), other ~10. Plus 3 MTP drafts ~9 =
118 ms step. FA reads 6.0 GB of q8_0 KV at 113 GB/s (18% of bandwidth); it is
**not** bandwidth-, dequant-, MMA-, parallelism- or version-bound (f16 KV is 5x
slower; `split_k=1` is -51%; spec-constant sweeps lose or break; no upstream
Vulkan change). Revised decode ceiling @183k: measured 28, realistic 40 (FA 3x),
hard ceiling ~51 (FA at the KV floor). Prefill is bound by the same kernel (48%)
plus matmul at 50-67% of peak: measured 524 @72k, estimated ceiling ~830.
**Blast radius of any kernel edit is the llama.cpp binary only** — not the OS,
kernel, RADV, or the 9060 XT; Build A is a separate binary. Real risk is silent
numerical error, mitigated by an RDNA4 gate + `test-backend-ops` + keeping Build
A. Next step: enable `LLAMA_BUILD_TESTS` and benchmark FA in isolation with
`test-backend-ops perf -o FLASH_ATTN_EXT`; without it, shader work is blind.
Full write-up: `docs/2026-09-21-r9700-depth-decode-findings.md`.



## R9700 q8_0 FA: packed-GQA row tile (2026-09-21/22)

**The 5.6x KV over-read in the decode profile is redundant *dequant*, not DRAM.**
At 183k, `FLASH_ATTN_EXT` at n_rows=4 took 3211 us and at n_rows=1 911 us — 3.5x
for 4x the rows. f16 KV at the same shape is only 1.49x for 4x the rows (L2 serves
the shared reads), and reaches 625 GB/s (98% of the 640 wall) while q8_0 reaches
432 GB/s. The gqa fold makes each workgroup handle one query position with the 6
Q heads as rows; the KV is therefore dequantized once per position. Isolated
evidence: q8_0 nb=4 at kv=131072 = 2331 us vs f16 nb=4 = 1281 us.

**Fix: pack (position, head) pairs into the Br=16 row tile** so one workgroup
covers `pos_per_tile = Br/gqa_ratio` query positions and shares one K/V dequant
pass. Touches `flash_attn_cm1.comp` (row mapping for Q load, mask index, output
and split_k stores; new `PACKED_GQA` flag bit 32) and the dispatch (grid.x =
CEIL_DIV(n_pos, pos_per_tile); `n_pos` = real position count, `p.ne2`). Correctness
via `test-backend-ops test -o FLASH_ATTN_EXT` (gqa-6 nb=1..8, ratios 3/7 that do
not divide Br, prefill nb=512: all OK).

| shape (kv=183296, q8_0) | pack off | pack on |
|---|---:|---:|
| nb=3 | 2881 us | **1943 us** |
| nb=4 | 3581 us | **1981 us** |
| nb=1 | 975 us | 975 us |

**Model-level A/B on one binary** (`GGML_VK_FA_NO_PACK=1`), 176k, q8_0/q8_0,
`draft-mtp` n_max 3: pack on **27.46 t/s [27.22-27.72]** vs off **26.75 t/s
[26.36-26.96]** — +2.6%, same accept (0.7629). The absolute number is ~5% below
the 2026-09-21 ledger because the box carried sibling `go test`/`session.test`
load; the A/B is same-session and is the number to trust.

**Why the model gain is small: FA is only ~28% of the step.** The verify is
**n_rows=3** (not 4), and the per-node profile at 183k gives main verify 95.8 ms =
FA 33.0 + weight GEMVs 43.8 + other 17.6; the step is ~117 ms, so drafts + CPU
graph-build overhead is ~23 ms. `other` is ~1300 tiny ops at 13-17 us each (ADD
176x, CPY 240x, RMS_NORM 305x, SSM family ~360x) — launch/latency-bound, not
bandwidth. **With measured components the all-lever floor is ~91 ms = ~36 t/s at
183k, so 40 t/s is not reachable by FA work alone.** The 2026-09-21 "realistic 40"
used other ~10 + drafts 9; the measured values are 17.6 and 23.

**Prefill:** extending the same fold to N>8 regressed prefill 30% (56085 -> 73035 us
at nb=512/kv=71680) because it forces `use_mask_opt` off, and prefill needs it.
Reverted; prefill is unchanged at 524 @72k.

**`--spec-type ngram-mod,draft-mtp` is rejected here.** At 176k it drafted ~6/step
(461/77) at accept 0.386, widening the verify to 7 rows without raising
tokens/step (3.31), and cost -23% (22.3 vs 29.1 t/s). The 2026-09-21 "+9% para 70k"
does not hold for this prompt/settings.

## The deep decode step is latency-bound, not bandwidth-bound (2026-09-22)

Isolated `test-backend-ops perf -o MUL_MAT` at the model's GEMV shapes (q6_K,
n=3) runs at **710 GB/s** for m=17408 (73 MB, mostly L2-resident across reps) but
only **605 GB/s** for m=248320 (1.04 GB, no L2 reuse) — and the latter matches the
in-model lm_head exactly (1721.7 us isolated vs 1714.6 us profiled). The in-model
FFN GEMV at m=17408 is 145.9 us (501 GB/s) against 103 us isolated, so the model
loses ~40% of the kernel's throughput to inter-op dependency stalls rather than to
bandwidth: the 640 GB/s wall is not what the 43.8 ms GEMV total is hitting.

Corollaries:
- `GGML_VK_DISABLE_MMVQ=1` is a **no-op** at these shapes (byte-identical times),
  so the MMVQ/non-MMVQ choice is not a lever here.
- RMS_NORM+MUL fusion (`device->add_rms_fusion`) is **already on** for
  RADV/AMD subgroups; `GGML_VK_DISABLE_FUSION` only gates that one fusion.
- The ~17.6 ms "other" tail is ~1300 ops at 13-17 us each (ADD 176x, CPY 240x,
  RMS_NORM 305x, SSM family ~360x) on 61 KB tensors — pure launch/dependency
  latency, ~65x the bandwidth time for each op.

**Consequence:** neither the 43.8 ms GEMV nor the 17.6 ms op tail yields to a
kernel or config knob; moving either needs backend work (barrier elimination,
cross-op overlap/async submission, or graph-level op fusion), which is outside the
FA kernel. This is why the "attack GEMV + op tail" route does not close the gap to
40 t/s either.

Caveat: the per-op profiler times may themselves be inflated by the
`GGML_VK_PERF_LOGGER` timestamp wrappers, so the absolute split (FA 33 / GEMV 43.8
/ other 17.6) should be treated as indicative; the same-binary `MUL_MAT` isolated
vs profiled comparison above is the load-bearing evidence.

## FA decode is row-tile-bound, not pass- or dequant-sharing-bound (2026-09-22)

Implemented a two-MatBr row-tile variant of cm1 so Br can be 32 (packed-GQA
`pos_per_tile = Br/gqa_ratio = 5`, so nb<=5 is a *single* KV dequant pass).
Correct (`test-backend-ops` 14/14 packed, 1357/1357 GQA overall) but **no gain**:

| kv=183296, q8_0 | Br=16 | Br=32 |
|---|---:|---:|
| nb=1 | 972 us | 1855 us |
| nb=2 | 981 us | 1864 us |
| nb=3 | 1810 us | 1878 us |
| nb=4 | 1816 us | 1880 us |
| nb=8 | 3487 us | 3637 us |

FA time is linear in **16-row tiles**, independent of KV passes: 16 rows ~975 us,
32 rows ~1810 us, 64 rows ~3487 us. Sharing one dequant across 2 position-tiles
(which Br=32 does) changes nothing, so the decode FA is not dequant-throughput
bound at the pass level. `GGML_VK_FA_CM1_SHMEM=1` (full-tile shared-memory
staging, dropping ~32 workgroup barriers/KV-block to ~2, fits in 50.4 KB) is worth
only +2% at nb=3/4 and −4% at nb=1. Rate check: q8_0 decode ~13 TFLOPS, q8_0
prefill ~17, f16 prefill ~30.8 — so q8_0's per-element dequant caps the kernel,
and at a fixed row count q8_0 costs 1.46x f16 (1306 vs 894 us at 131072 nb=4)
regardless of pass count.

**Corollary:** the decode ceiling is not moved by Br/passes/shmem. The remaining
levers are the weight GEMV (43.8 ms at ~557 GB/s incl. lm_head) and the
latency-bound tail (17.6 ms small ops + ~12 ms CPU graph build; the backend
already fuses RMS_NORM and gate/up, and `max_nodes_per_submit=100` gives ~13
fence-synced submissions per step).

**Measurement caveat:** on 2026-09-22 the box was contended by sibling sessions
for most of the window; two otherwise-identical 70k runs read 14 t/s against a
clean ~38 t/s, with `mem_busy=0%` at the guard sample. Model-level numbers from
this window are unusable; the isolated A/B tables above are same-process and hold.

## Clean-machine decode results and the closed levers (2026-09-22/24)

With the box quiet (load < 2, `mem_busy` 0 at the guard), same-binary A/B at 176k,
q6_K, q8_0/q8_0, n_max 3:

| config | decode | spread |
|---|---:|---:|
| packed-GQA **off** | 27.07 t/s | 0.15% |
| packed-GQA **on** | **34.84 t/s** | 0.11% |

So packing is **+28.7% at 176k** (and 42.2 vs ~38.4 at 70k, +10%). The earlier
+2.6% was contention noise. Step at 176k is 94 ms, not 117.

Clean concurrent profile of the main verify graph = **75.1 ms**: FA **29.9**, weight
GEMVs **39.3**, small-op tail **7.3** (the serialized profiler's 17.6 was inflated —
it inserts `ggml_vk_sync_buffers` after every node). Per decode step there are also
3 draft graphs (~3.5 ms each: full-KV FA at nb=1 ~1.2, lm_head ~1.64, one MTP layer
~0.6) and ~7 ms of non-overlapped CPU/launch.

Closed levers (all measured, none help the *model*):
- **Br=32 two-MatBr row tile** (implemented, correct): FA is linear in 16-row
  tiles, independent of KV passes — nb=3 1878 us vs 1810 (Br=16).
- **split_k**: isolated optimum 64 (1736 vs 1810 us, −4%), but in-model A/B
  **34.67 (mult=2) vs 34.07 (mult=8)** — the isolated win does not transfer;
  reverted.
- **SHMEM_STAGING** (`GGML_VK_FA_CM1_SHMEM`): decode +2% at nb=3/4, −4% at nb=1,
  +26% *worse* on prefill. Off.
- **mask_opt**: worth only ~5% on prefill (60426 vs 63439 us), so the prefill
  head-fold regression was the fold's 1.33x MMA waste, not mask_opt.
- **n_max 4/5**: 32.3 / 33.1 t/s vs 34.84 (acceptance drops to 0.709/0.676).
- **`--spec-draft-p-min` 0.4/0.7**: 41.1 / 39.4 at 70k vs 42.2 — acceptance rises
  (0.896 at 0.7) but tokens/step falls; net loss.
- **submission batching** (32/200/4096) and **DISABLE_FUSION/DISABLE_ASYNC**:
  neutral or worse.

Remaining to 40 t/s: ~13 ms/step, of which FA q8_0 dequant ~9 ms (1.46x f16 at
fixed rows), non-overlapped CPU ~7 ms, small-op tail ~7 ms. None yields to a
shader or dispatch knob; all need backend-overlap/fusion work.

## cm1 FA structural analysis and closed backend avenues (2026-09-24)

Three parallel code investigations produced the following (no benchmark wins
found beyond packed-GQA):

**cm1 FA (why ~13 TFLOPS decode, ~975 us per 16-row tile at 183k):** the `for j`
KV-tile body has ~46 workgroup barriers, of which **32 come from the per-d-chunk
q8_0 K dequant** (16 d-chunks x 2 barriers, `flash_attn_cm1.comp:302-356`), with a
single-buffered `kvsh` that WAR-serialises the next dequant against this chunk's
MMA. Candidate fixes, none yet implemented: ping-pong `kvsh` and software-pipeline
the dequant behind the MMA; stage V once per tile and defer the `row_split`
cross-subgroup PV reduction out of the per-tile loop (`pvsh` round-trip at
:553-575); skip the mask gather for KV blocks below every query position (needs a
packed-aware `mask_opt`, since `use_mask_opt` is currently forced off by
`packed_gqa`, and a naive skip would be wrong for sliding-window masks).
Occupancy context: non-staging shmem is ~26 KB (2 WGs/CU); `SHMEM_STAGING` is
~52 KB (1 WG/CU), which is why it did not help. `d_split` is dead for cm1;
`row_split`/`workgroup_size` are only tunable as a bundle via `GGML_VK_FA_CM1_NS`.

**Vulkan submission:** found one real bug — `flops_per_submit` is derived from the
*previous* graph's flops (`ggml-vulkan.cpp:13910`, set at :14255), so in an MTP
step the large verify graph follows a tiny draft graph and gets a tiny
`flops_per_submit`, fragmenting into many submits (this is why raising
`max_nodes_per_submit` did nothing). A running-max fix measured **neutral at 70k
(42.02 vs 42.18)** and was reverted. Command buffers are re-recorded every submit
(no replay); `ggml_vk_graph_cleanup` resets the command pool and churns
semaphores/events on every `llama_synchronize` (~4-5x/step), and
`common_sampler_sample` syncs per draft token — but these are risky to change
blind and were not attempted.

**Upstream:** no commits after `ce8caa6e6` touch `flash_attn_cm1.comp`,
`flash_attn_base.glsl`, `flash_attn.comp`, `flash_attn_dequant.glsl`, the FA
split_k logic, or `ggml_vk_graph_compute`. The only relevant post-base win is
**#27952 `70c4e1582` (int8 coopmat1 matmul for RDNA3/RDNA4, merged 2026-09-24)**,
reported +9% Qwen3.8-27B Q6_K pp512 on R9700 (matmul, i.e. prefill/GEMV, not FA).
Issue #28752 (RDNA prompt-processing regression) is a Windows-Adrenalin / Mesa
26.2.2 interaction; not reproducible on R9700 Linux per maintainers. No sourced
RADV flag improves gfx1201 coopmat throughput.

## #27952 int8 coopmat matmul port: +10% decode at 176k, from acceptance not kernel (2026-09-24)

Ported upstream `70c4e1582` (int8 coopmat1 matmul for RDNA3/RDNA4, merged
2026-09-24) onto the fork base `ce8caa6e6` (95 commits of drift; the new
`mul_mmq_cm1.comp` / `mul_mmq_cm1_funcs.glsl` are additive, `ggml-vulkan.cpp`
needed hand adaptation). It adds a `matmul_q6_k_q8_1` cm1-int pipeline for
RDNA4 (Q6_K weights via integer dot with q8_1 activations, replacing the f16
dequant+MMA path). Same 8-file tree as the FA work, uncommitted.

Server A/B, Qwen3.8-27B-Q6_K, ctx 262144, **q8_0/q8_0**, MTP n-max 3, clean box
(`harness/depth.sh`, depths=4000 reps=4):

| config | decode | spread | accept | prefill rep0 |
|---|---:|---:|---:|---:|
| FA-only (base) | 34.846 t/s | 0.21% | 0.7629 | 349 |
| **+ #27952 (forward)** | **38.162** | 0.50% | **0.881** | 357 |
| **+ #27952 (reverse)** | **38.382** | 0.48% | **0.881** | 357 |

+9.5 to +10.1%, reproducible in both orders. Acceptance is deterministic per
binary: base 0.7629 in 3 runs, port 0.881 in 4 runs.

**The gain is acceptance, not kernel throughput.** Tokens/step 1+0.763*3 = 3.289
-> 1+0.881*3 = 3.643 (+10.8%), while step time is unchanged (~95 ms both). Decode
GEMV (batch 1) does not use the cm1 path, so the port cannot speed decode; the
faster matmul changes the logits enough to raise MTP draft acceptance. The shift
is depth-specific: at 70k the port's acceptance is unchanged (0.7629) and decode
is **43.267** vs the handoff's base 42.2 (**+2.5%**, kernel only).

llama-bench `-p 512,2048 -d 0`, 3 interleaved pairs (spread <2%): pp512
943->972 (**+3.1%**), pp2048 930->963 (**+3.4%**). Correctness: `test-backend-ops`
MUL_MAT 1128/1128, MUL_MAT_ID 937/937, FLASH_ATTN_EXT targeted GQA 1357/1357 all
pass; greedy output at 8k is byte-identical to base. Note the FA-vector-quant
fallback in the base used f16 for Q6_K, so the int8 path plausibly *reduces* a
precision loss rather than adding one.

## FA q8_0 K-dequant software pipeline regresses at depth (reverted, 2026-09-24)

Implemented ping-pong `kvsh_dq` + one-chunk prefetch in `flash_attn_cm1.comp` so
the q8_0 K dequant of chunk d+1 overlaps the MMA of chunk d (1 barrier/chunk
instead of 2, WAR hazard removed). Correctness passes (`nb=75` filter 1304/1304;
targeted GQA 1357/1357) and acceptance is unchanged, but:

| depth | port (no pipe) | port+pipe |
|---|---:|---:|
| 70k | 43.267 | 43.914 (+1.5%) |
| **176k** | **38.2** (same window) | **34.709 (-9.2%)** |

176k was measured back-to-back and is decisive; the 70k "+1.5%" does not
transfer. Reverted. Consistent with the earlier finding that barriers are not
the cap (SHMEM_STAGING ~2%) — issuing the dequant before the MMA in program
order serialises the global load ahead of the MMA instead of hiding it, and the
extra 3 KB shmem likely costs occupancy at the deeper KV loop.

## Re-checked under the int8 port: n_max 4/5 still lose (2026-09-24)

The handoff closed n_max 4/5 under the lossy base. With the port's better
logits, re-tested at 70k (port n_max 3 = 43.267, accept 0.7629):

| n_max | decode 70k | accept |
|---|---:|---:|
| 4 | 29.641 | 0.7004 |
| 5 | 13.056 | 0.6829 |

Still a large regression; the fixed serving config (n-max 3) stands.

## Measurement hazard: llama-bench `tg64 -d 16384` is bimodal (2026-09-24)

The same binary reads either ~18.9 or ~23.4 t/s (+24%) across process runs,
uncorrelated with the binary (both base and port hit both modes) and with
`mem_busy_percent`; mclk sits at 1258 MHz throughout while sclk swings
2.6-3.3 GHz. The server `depth.py` protocol is stable (spread <=0.5%), but one
clean-guard base run (load 2.18, mem_busy 0%) still read 30.347 with a 4.12% rep
spread against 34.846 in an adjacent run. **Do not use that llama-bench row, and
treat single server runs as +/-a few percent unless rep spread is <1%.**

## Not verified / open

- Full `FLASH_ATTN_EXT` suite SIGFPEs (`hsk=320,hsv=256,nh=4,nr23=[4,1],kv=512,nb=75`,
  type_K/V f16) and core-dumps. Reproduces on the port binary *without* the
  pipeline, so not caused by it; the shared factor is the packed-GQA FA patch.
  The targeted GQA test and all model runs pass, so it is latent. Needs a
  pristine `ce8caa6e6` binary to confirm origin.
- 40 t/s decode at 176k not reached: 38.2-38.4 with the port. The remaining gap
  is not addressable by the FA/KV path (q8_0 dequant is the floor) and decode
  GEMV is bandwidth-bound at ~557 GB/s of 640; no further safe lever tested.

## Agnostic decode levers falsified (2026-09-24)

Pushing past the #27952 port toward >40 t/s agnostically (real step-time
reduction, not acceptance) — every cheap candidate was tested and rejected:

**f16 KV cache** looks +24% at 16k decode (`llama-bench -n 128 -d 16384`,
interleaved 3 pairs: q8_0 18.90/18.98/19.04, f16 23.44/23.36/23.66) but
**collapses at depth**: 176k server, MTP n-max 3, f16/f16 =
**7.982 t/s** [7.94-8.03] vs q8_0/f16 38.38 (rep1-3 wall 33 s vs 8 s; prefill
197 vs 357). Acceptance 0.8426. So q8_0 KV is *required* for long-context
decode on this box; the int8 KV path is not a dequant tax to remove but the
thing that makes depth decode work at all. The 16k ranking inverts by 176k.

**Vulkan backend CPU/launch overhead** (handoff's "~7 ms/step non-overlapped"):
refuted by instrumentation. `ggml_vk_graph_cleanup`'s semaphore/event loops are
dead code — `ggml_vk_create_binary_semaphore`/`_timeline_semaphore`/
`_create_event` (`ggml-vulkan.cpp:1075-1097`) have zero call sites. Measured:
cleanup 1.0 us/call, pool reset 1.0 us, inner sync 1.5 us/call; the real CPU cost
is command-buffer recording in `ggml_vk_build_graph` (~2.4 us/node). Removing it
needs command-buffer replay, a rewrite. No safe micro-win.

**GEMV rows-per-shader (`rm_kq`)** swept 1/2/4/8 at d0: 24.62/24.42/19.45/19.77,
but this is confounded by the per-process fast/slow mode (the same default
config reads 19.2 and 24.5 in different processes). Not established as causal;
no reliable GEMV win found. An env knob `GGML_VK_RM_KQ` / `GGML_VK_RM_STDQ` was
added to `ggml-vulkan.cpp` for future tuning (inert by default).

**Conclusion:** with weight and KV precision fixed (Q6_K, q8_0), decode at 176k
is bounded by q8_0 FA (~30 ms) + bandwidth-bound GEMV (~39 ms at 542/640 GB/s),
both near their constrained limits. >40 t/s agnostically needs either a KV/weight
precision change (the f16 experiment above shows KV precision is not free to
trade) or a kernel/backend rewrite (command-buffer replay, FA occupancy
redesign). No cheap lever remains.

## Where 176k decode time actually goes, and the batching answer (2026-09-24)

**The 542 GB/s figure was a units slip** — it is `21.30 GiB / 39.3 ms`, not
GB/s. The correct number is **582 GB/s** (22.87e9 B / 39.3 ms) against the
R9700's 640 GB/s theoretical (256-bit GDDR6 @ 20 Gbps) = **91%**, i.e. at the
practical GDDR6 ceiling (~90%). The verify graph reads all 22.87 GB of Q6_K
weights per step; a perfect-bandwidth GEMV would save ~3.5 ms/step (step
95 -> 91.5 ms, 38.4 -> ~39.8 t/s). Even 100% bandwidth does not clear 40 with
the current FA: the floor is ~36 (weights) + 30 (q8_0 FA) + ~24 (drafts/tail/CPU)
= ~90 ms => ~40.4 t/s at accept 0.881. GEMV is not the lever; there is no
meaningful 542->640 reclaim.

**FA tuning knobs are closed.** `GGML_VK_FA_CM1_BR` x `_NS` swept at 70k
(server protocol): default BR=16/NS=4 is optimal. BR=32 is equal (−0.8%),
BR=64 regresses ~15% and shifts acceptance (0.763->0.735, numerics differ),
NS=1/2/8 all fail with HTTP 500. No win.

**Batching (`--parallel N`) works but does not raise aggregate throughput with
MTP.** `harness/throughput.py`, port binary, short prompts, n-predict 128:

| server | c=1 agg | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| parallel 4, MTP n-max 3 | 23.8 | 21.2 | 25.0 | — |
| parallel 8, no spec | 13.0 | 18.7 | 21.0 | 26.4 |

Per-request rate collapses (MTP: 27.2 -> 7.3 t/s) and MTP acceptance falls
(0.871 -> 0.683) as slots fill. MTP single-stream (~38 t/s at depth) beats every
batched aggregate, so for this workload batching is a latency/queueing feature,
not a throughput win. Multi-slot serving is available and correct; the
config choice is a trade, not a free speedup. (Caveat: these short runs are
exposed to the per-process power mode below.)

## FA cm1 redesign: occupancy is not the bottleneck — evidenced negative (2026-09-24)

Before rewriting the FA kernel to chase >40 t/s, the limiter was measured
directly with `RADV_DEBUG=shaderstats` on the hsk=hsv=256 / q8_0 / f32acc
`flash_attn_cm1` variant:

```
Compute Shader: SGPRs 108  VGPRs 192  Spilled 0  Code size 26728
                LDS size 26624  Subgroups per SIMD: 8
                Instr 4726 (VALU 2204 / SALU 795 / VMEM 108 / SMEM 76 / Branch 131)
```

**Residency is LDS-capped, not register-capped**, and the cap is real:
64 KiB/LDS = workgroups/CU — NS=4 (26624 B) -> 2, NS=2 (15360 B) -> 4,
NS=8 (60416 B) -> 1; the `split_k` saturation sweep confirms it (NS=4 saturates
at split_k=32 = the default; sk=64 adds nothing).

**But occupancy does not convert to throughput.** NS=2 (4 WGs/CU) is *slower*
than NS=4 (2 WGs/CU): 765 us vs 702 us at kv=131072 nb=1. And the kernel is not
memory-bound (q8_0 nb=1: 101 GB/s, f16 156 GB/s — ~1/5 of 640) nor
barrier-bound (removing ~30 barriers/tile via the already-reverted SHMEM staging
moved decode +2%). At nb=4, q8_0 1311 us vs f16 891 us, and q8_0 scales 1.87x
with nb while f16 scales 1.04x — the dequant sits on the per-tile path once the
row tile is full.

**The irreducible cost is per-workgroup serial work plus row waste:** Br=16
against gqa_ratio=6 gives `pos_per_tile=2`; at nb=1 only 6 of 16 rows are valid
(`cm1_row_valid`), and per-valid-row cost is 117 us (nb=1) vs 55 us (nb=4).
The only LDS cut available (subgroup-local V staging, kvsh 8192->3072 B) targets
residency, which NS=2 shows is not the lever; expected <2%, with real
correctness risk in the V/PV coopmat layout. Not attempted.

**Conclusion: FA cm1 at this shape is at its structural limit.** With the
weight/KV precision fixed, the 176k decode ceiling is ~40 t/s and no configuration
or shader-level change tested this session beats the 38.4 t/s already shipped.
Exceeding 40 requires a precision change (rejected: f16 KV collapses at depth) or
a from-scratch attention kernel; it is not a tuning problem.

## The decode ceiling is DRAM traffic, not the FA kernel (2026-09-24, corrected)

A devils-advocate review of the rewrite plan caught a real arithmetic error in
this notebook: the "FA is 15x above its floor" claim compared a **per-layer**
floor (0.4 GB) against an **all-layer** measured cost. Corrected accounting of
the mandatory DRAM traffic per 176k decode step (Q6_K, q8_0/q8_0, 65 layers,
4 KV heads, head_dim 256, 182889 tokens):

- weights (verify reads all 22.87 GB once per step) = **22.87 GB**
- verify KV = 4 heads x 256 x 2(K,V) x 34/32 B x 65 layers x 182889 = **25.87 GB**
- draft lm_head/MTP reads ~= **2.5 GB**
- **total ~= 51 GB/step**

At the 640 GB/s theoretical peak that is 80 ms -> **~45.5 t/s absolute ceiling**;
at the 582 GB/s the GEMV actually sustains, ~88 ms -> **~41 t/s**. Decode is
80-85% of that already. **>55 t/s is physically impossible with Q6_K weights and
q8_0 KV on this GPU** — 55 needs 66 ms, i.e. 773 GB/s, above peak. It requires
fewer bytes (lower-precision KV/weights) or a cheaper step (more tokens/step).

Counter-experiment: **q4_0/q4_0 KV at 176k = 38.391 t/s [38.366-38.423]**, vs
38.382 for q8_0/q8_0 — **identical**, with acceptance 0.8762 vs 0.881. Halving
KV bytes (25.87 -> 12.9 GB, total ~38 GB) changed nothing, so the current step is
**not** bandwidth-bound; it is dominated by a serial cost (~95 ms) that the byte
reduction does not touch. So the ceiling above is a hard cap only a rewrite can
approach, and the current kernel has real room (38 -> ~45) but no room to 55.

Consequence for the plan: the fixed serving config (Qwen3.8-27B-Q6_K, ctx
262144, q8_0/q8_0, MTP, mmproj) caps decode at ~45 t/s. 55 t/s and that config
are mutually exclusive; this is arithmetic, not a tuning outcome.

## CORRECTION: Qwen3.8-27B is hybrid (17 full-attn + 48 linear), decode ceiling is ~70 t/s (2026-09-24)

The preceding "DRAM ceiling ~45 t/s / 55 impossible" entry is WRONG and is
retracted. It assumed 65 full-attention layers. GGUF metadata
(`qwen35.full_attention_interval=4`, verified) shows:

- **17 full-attention layers** (`attn_q` 5120->12288 gated, `attn_k/v` 5120->1024
  = 4 KV heads x head_dim 256, `attn_output` 6144->5120).
- **48 linear-attention / Gated-DeltaNet layers** (`attn_qkv` 5120->10240,
  `attn_gate` 5120->6144, `ssm_a/alpha/beta/conv1d/dt/norm/out`).
- 1 MTP layer (`nextn.*`). 17 + 48 = 65.

Growing KV therefore exists in **17 layers only**: 4 x 256 x 2 x 1.0625 B x
182889 x 17 = **6.77 GB** at 176k, not 25.87 GB. Step traffic ~= 22.87 (weights)
+ 6.77 (full-attn KV) + ~0.15 (SSM state) + ~2 (drafts) ~= **32 GB**.
Ceiling = 32/640 = 50 ms -> **~72 t/s** theoretical; at the 582 GB/s the GEMV
sustains, ~69 t/s. Short context (KV~0) ~23 GB -> ~100 t/s theoretical, ~60
achieved. Current 176k = 32 GB / 95 ms = **324 GB/s, ~half the bound**.

Implications: (a) >55 t/s is achievable under the fixed q8_0/q8_0 + Q6_K
config; (b) the q4_0-KV "no change" result is consistent — KV is only ~21% of
traffic, so halving it buys ~10% at best, lost in the serial bottleneck; (c) the
**48 Gated-DeltaNet layers are 74% of the layers** and are the prime suspect for
the serial cost, alongside the 17-layer FA. The plan must measure both.

## Verified architecture and the corrected bottleneck (2026-09-24, deep research + adversarial check)

Two independent sub-agents (one reading the fork + upstream PRs, one adversarial)
verified the hybrid analysis. Corrections to the preceding entry:

- **Layer split:** trunk `n_layer=64`; `qwen35.cpp:17-21` sets
  `is_recr[i] = (i<64) && ((i+1)%4 != 0)` -> full attention at i=3,7,...,63 =
  **16 trunk FA**, recurrent = **48**, plus the **1 MTP** layer at i=64 =
  **17 KV-bearing FA layers**. The earlier "17 FA + 48 + 1" double-counted MTP.
  KV arithmetic (6.77 GB at 176k, 17 layers x 4 x 256 x 2 x 1.0625 B x 182889)
  is exact and confirmed.
- **GDN is not the bottleneck (~5%).** Upstream RADV developer measured
  Gated-DeltaNet at ~5% of a Qwen3.5-35B-A3B run (llama.cpp PR #20377); the
  decode path is the fused `GATED_DELTA_NET` op (registers hold the state shard,
  no workgroup barriers on the clustered path, no coopmat). It is
  context-independent, so it cannot account for the ~45 ms that appears only at
  depth. **The 17-layer FA path remains the supported suspect** (profile: FA
  29.9 ms vs a ~11 ms KV floor).
- **Draft traffic undercounted:** the MTP context has its *own* KV (not memory
  shared; `llama-context.cpp:143-160`), and each of 3 draft graphs re-reads the
  MTP block (~0.36 GB) + shared LM head (~1.0 GB) -> ~5-6 GB/step, not ~2 GB.
  Step total ~= **34-35 GB**.
- **The ~70 t/s bound is a loose upper limit, not a prediction.** The step is
  not bandwidth-bound (q4_0 KV: 38.391 vs 38.382), so "currently at half the
  bound, room to 70" is unsupported. The serial terms (FA issue latency, graph
  build/CPU ~14 ms, ~1300 small ops) dominate the distance to the bound.
- GDN-adjacent micro-optimizations exist but are small: write the GDN state
  straight into the persistent cache (skip `ggml_cpy`, ~0.5 ms), fuse qkv+gate
  GEMV, coalesce the strided state load/store.

Conclusion: the plan's primary target is the **17-layer FA decode path**, then
CPU/command-buffer replay; GDN is a minor workstream. >55 t/s is plausible
(FA 29.9->~11 + CPU 14->~4 would put the step near 62-73 ms => ~50-58 t/s) but
is not guaranteed and must be gated on measurement.

## WS0 op budget: FA is the target, GDN is a non-issue (2026-09-24)

Built the op-budget harness the plan asked for: `harness/op_perf.sh` wraps
`test-backend-ops perf` on the R9700 (`-b Vulkan1`), repeats the suite and prints
`op= us= spread= gbps=`; the model shapes are already in the perf list (Q6_K GEMV
widths; FA hsk=hsv=256 nh=4 gqa=6 q8_0 at kv 71680/183296 nb 1..512). Added the
missing **model GDN shape** (16 K heads / 48 V heads, d=128, v_repeat=3) to eval
and perf.

Measured on the idle R9700 (spread <0.25%):

| op | shape | us/op | ×layers | ms/step |
|---|---|---:|---:|---:|
| FLASH_ATTN_EXT | 256/256, 4 KV, gqa 6, q8_0, kv=183296, **nb=4** | 1821 | 17 | **30.96** |
| GATED_DELTA_NET | 16K/48V, d=128, v_repeat 3, AR | 5.33 | 48 | **0.26** |
| MUL_MAT q6_K | FFN/qkv/qkvo/gate widths at n=3 | — | ~65 | ~39 (bytes/582 GB/s) |

**Reconciliation.** Step = GEMV ~39 + FA 31.0 + GDN 0.26 + drafts/tail/CPU ~24
~= 94–102 ms vs the measured 94.9 ms (3.643 tok @ 38.4 t/s) — within 8%.
**Top-2 contributors: weights GEMV (~39 ms, already at 91% of GDDR6 peak, no
headroom) and FA (~31 ms, 219 GB/s of ~640, i.e. 2.7× above its KV floor).**
GDN is **0.26 ms — 0.3% of the step, not a workstream.** WS0 confirms the plan's
FA hypothesis and closes WS2/GDN as a priority; the only large context-dependent
kernel is FA.

## WS1 FA decode rewrite: direction falsified, mask-opt adopted (+3.4% @176k) (2026-09-24)

The plan's WS1 rewrite (all nb<=8 verify positions in one workgroup pass, so K/V
dequant happens once) was implemented as its implied tiling (Br=32/48/64 =
one-pass multi-position) and **falsified**: at kv=183296 nb=4 q8_0, Br=16 1830us
vs Br=32 1892 (+3.4%), Br=48 2182, Br=64 3449. Shader stats explain it: Br=32
uses 256 VGPRs / 43 KB LDS / 40 KB code vs Br=16's 192 / 18 KB / 20 KB.

Two hard floors measured, both above the plan's <900us target:
- `GGML_VK_FA_SPLIT_K` sweep at nb=4/183296: sk=1 28877us, sk=32 1815, **sk=64
  1754**, sk=128 2029 -> hardware throughput floor ~1754us.
- Even with the q8_0 dequant entirely absent (f16 K/V, same shape): **1251us**,
  still 1.4x above 900us. nb=4 needs 24 valid rows (4 positions x 6 heads) = two
  irreducible MatBr=16 MMA tiles.
So the kernel is throughput-bound, not latency-bound; <900us needs int8 QK
(quantising Q = a precision change, rejected) or f16 KV (collapses at depth).
**The 2x rewrite is infeasible under the fixed serving config.**

**Adopted small win — packed-GQA mask-opt.** The mask-opt bitmask row-block was
selected by the always-zero packed tile index; it is now selected by the
position-tile index and sized `pos_per_tile`, so causal all-zero KV blocks skip
the per-block mask gather + slope add. Byte-identical outputs, enabled by
default for (q8_0 K/V, RDNA4, n_rows<=8); kill switch `GGML_VK_FA_NO_DECODE_V2=1`.
Verified A/B (3 reps, spread <0.3%, accept identical):

| | 70k | 176k |
|---|---:|---:|
| off | 43.98 | 38.57 |
| **on** | **44.69 (+1.6%)** | **39.90 (+3.4%)** |

Correctness: target shape 22/22, broad GQA 1224/1224, new large-KV packed 5/5.
Plan kill gate: 70k 44.69 < 50 -> WS1 rewrite is killed; the mask-opt is kept.

## WS2 prefill: P0>=1800 is infeasible under Q6_K (kill criterion fires) (2026-09-24)

Reproduced prefill on the R9700 (llama-bench `-dev Vulkan1`, q8_0/q8_0, -fa on,
r3): pp2048 **981 t/s** (d0) / 916 (d4096) / 794 (d16384) — matches the plan's
978/889/797. (Note: llama-bench defaults to Vulkan0 = the 9060 XT; `-dev Vulkan1`
is mandatory or every number is ~3.6x low.)

FLOPs = 2 x 27.32e9 = 54.6 GFLOP/token, so 981 t/s = **53.6 TFLOPS achieved**.
At the representative prefill GEMM shape (m=4096,k=14336,n=512, spread <1%):

| weight type | us | TFLOPS |
|---|---:|---:|
| q6_K (the model) | 1159 | **51.9** |
| q8_0 | 834 | 72.1 |
| f16 | 861 | 69.8 |
| q4_0 | 746 | 80.6 |

So prefill is GEMM-bound, and the q6_K dequant+int8-coopmat path is 1.35x slower
than f16 and 1.55x slower than q4_0. **The best case without a precision change
is f16 weights at 69.8 TFLOPS -> ~1270 t/s, still far below the P0 gate of 1800**
(which needs ~98 TFLOPS, at the card's fp16 peak). The plan's own kill criterion
("prefill headroom is dequant-bound, not compute -> <1.3x -> drop P0 stretch")
fires: **P0>=1800 cannot be reached with Q6_K weights; reaching it requires a
weight-precision change, which the serving config forbids.** Same ceiling applies
to P1/P2 at depth (attention-dominated, and FA is at its measured floor per WS1).

## Verdict: D0>=55 is arithmetically impossible under the immutable serving config (2026-09-24)

Summing the measured floors of a 176k decode step (Q6_K, q8_0/q8_0, 17 FA + 48
GDN, MTP n_max 3), none of which the serving config permits changing:

| term | floor | source |
|---|---:|---|
| weight GEMV (22.87 GB @ 582 GB/s = 91% GDDR6) | 39.3 ms | WS0 |
| FA, 17 x q8_0 split_k floor (1754 us) | 29.8 ms | WS1 |
| GDN, 48 x 5.33 us | 0.26 ms | WS0 |
| MTP drafts (~5-6 GB @ 582 GB/s) | ~9 ms | FINDINGS |
| **mandatory floor** | **~78 ms** | = **46.7 t/s** |

D0>=55 needs 66 ms. The mandatory traffic+FA floor is 78 ms, i.e. **12 ms (9
t/s) beyond what is physically available**, before any CPU/launch overhead
(~14 ms) is even counted. q4_0 KV is identical (not bandwidth-bound), f16 KV
collapses at depth (forbidden), and int8 QK needs quantising Q (forbidden). The
only levers left are CPU/command-buffer replay and the small-op tail, worth a
few ms each — enough to approach ~42-44 t/s, never 55.

**Same for P0>=1800** (WS2). Both headline ship gates require changing the
weight/KV precision or context, which the plan declares OUT OF SCOPE. The
honest disposition of the plan under the fixed config: WS0 done, WS1 rewrite
falsified (mask-opt +3.4% @176k adopted), WS2/P0 infeasible, WS3/WS4 small and
unable to close a 9-12 t/s gap. **The gates and the config are mutually
exclusive; a decision to relax one is the user's, not the session's.**

## GEMV win: q6_K n>=4 rm_kq 2->4 on RDNA4 (+3.8% @176k) (2026-09-25)

A per-node Vulkan profile (`GGML_VK_PERF_LOGGER=1`) of the real 176k verify
graph corrects WS0's synthetic budget: the verify graph is **86.2 ms**, split
GEMV **48.98 ms (57%, 467 GB/s = 73% of 640)** / FA 28.72 (33%) / GDN 0.77 /
other 7.71; three draft graphs add 10.7 ms -> step ~96.9 ms. **The weight GEMV,
which the plan dismissed as "at the ceiling", is the single largest cost.**

The q6_K GEMV is latency/issue-bound at the small MTP batch, not DRAM-bound
(isolated m=4096 n=4 k=14336: n=1 61 us vs n=4 101 us; L2-resident; every quant
plateaus 410-550 GB/s there, while a large grid like lm_head m=248320 n=1 hits
631 GB/s = 98.6%). Fix adopted: on RDNA4, q6_K GEMV pipelines for **n>=4** use
`rm_kq=4` rows/workgroup (2 was the default). Kill switch `GGML_VK_RM_KQ_Q6K=2`.

| | isolated q6_K n=4 | 70k | 176k |
|---|---:|---:|---:|
| baseline (no wins) | 101 us | 43.98 | 38.57 |
| + mask-opt (WS1) | — | 44.69 | 39.90 |
| **+ rm_kq4 (WS-GEMV)** | **94 us** | **46.68** | **41.42** |

Cumulative **+6.1% @70k, +7.4% @176k**, accept unchanged (0.7629 / 0.881).
Correctness: `test-backend-ops MUL_MAT` 1128/1128. A further lead not shipped:
forcing the int8 MMVQ dot path for AMD q6_K (`GGML_VK_FORCE_MMVQ=1`) measured
-7.7% e2e but q6_K MMVQ is deliberately Intel-only (2-byte alignment caveat).

## WS1b int8-QK FA: no-win; the q8_0 FA wall is V-dequant + PV, not K-dequant (2026-09-25)

Implemented the plan's "new int8 coopmat FA" (K staged raw q8_0, Q quantised to
q8_1, int8 16x16x16 QK^T, float softmax/PV) — correct (22/22 target, 23/23 GQA)
but **not faster** at the model shape:

| nb=4/kv=183296 | mask=1 | mask=0 |
|---|---:|---:|
| f16-dequant baseline | 1830-1836 us | 1616-1622 |
| int8 QK | 1846-1849 | 1464-1465 |
| int8 QK, scale-fold stubbed (diagnostic) | 1782 | 1397 |

**Mechanism:** removing K dequant buys only ~157 us; the int8 MMAs + per-block
scale fold consume it, and the packed-GQA mask path costs ~2x more under int8
(register/LDS pressure). Even a zero-cost fold is 1782 us. The q8_0 FA is
**dequant-bound** (399 MB in 1834 us = 217 GB/s), whereas f16 K/V is
**bandwidth-bound** (750 MB in 1251 us = 600 GB/s = 94% of peak) — so the
≤1000 us target needs int8 **PV** as well (V-dequant removal), which the plan
deferred as unproven. Left opt-in (`GGML_VK_FA_INT8_QK=1`), default-off
(`GGML_VK_FA_NO_DECODE_V3=1` forces off); default-off costs +0.3% from the
always-declared int8 shared buffers. Kept only as a scaffold for int8 PV.

## Consolidated ceiling vs the 55 t/s goal (2026-09-25)

Two adopted, verified wins this session:

| | 70k | 176k | correctness |
|---|---:|---:|---|
| start of session | 43.98 | 38.57 | — |
| + packed-GQA mask-opt | 44.69 | 39.90 | FA 138/138 |
| + q6_K rm_kq4 (GEMV) | **46.68** | **41.42** | MUL_MAT 1128/1128 |

(+6.1% / +7.4%, acceptance unchanged; nothing committed in the fork beyond the
baseline `a702ea697`.)

The real 176k verify graph is 86 ms: GEMV 49 / FA 28.7 / GDN 0.8 / other 7.7,
plus 10.7 ms of drafts. **55 t/s needs a 66 ms step — 28 ms below the current
93-97 ms.** Every identified lever, at its measured best:

- FA: current 28.7 ms is at the algorithm's floor (split_k bottoms at 29.8 ms;
  Br 32/48/64 worse; int8 QK no-win). Reaching ~17 ms needs int8 PV (unproven)
  and a much better int8 QK; the pure q8_0-read floor is ~10.6 ms.
- GEMV: 49 -> ~42 ms via the int8 MMVQ path (lead: -7.7% e2e, but q6_K MMVQ is
  Intel-only for a 2-byte-alignment reason and needs AMD enablement).
- drafts 10.7 / other 7.7 ms: small.

Best case (FA -> 17, GEMV -> 42) gives step ~74 ms = **~49 t/s**. Reaching 55
would require FA at ~0.62 ms/layer (the raw KV-read floor) with zero overhead —
i.e. a from-scratch attention kernel that the two attempts here did not find.
**The 2026-09-21 deep dive's own revised ceiling was ~45-51 t/s at 183k, not
55.** This session moved 38.6 -> 41.4 and located the remaining wall precisely;
55 is not supported by any measurement on this card at this precision.

## WS1c int8 PV: fails correctness; FA wall is mask+barrier, not dequant (2026-09-25)

Implemented int8 PV (P quantised per 32-KV-block, V raw q8_0, int8 coopmat,
scales folded after). **Fails the correctness gate: nmse 0.015-0.998 vs 5e-4
tolerance.** Fundamental, not a bug: V's q8_0 scale `d_V[k][nb]` runs along HSV
(the PV *output* dim), not the contraction dim, so it cannot be folded after the
MMA; folding it into P forces per-block quantisation of a huge-dynamic-range
quantity (error 30-2000x tolerance). Per-16 blocks worse. Reverted (no net code).

Decisive accounting of the FA wall (isolated, nb=4/kv=183296, mask=1):
q8_0 1828 us; int8-QK 1849 (no gain at mask=1 - the int8 mask path costs ~2x);
f16 K/V 1251 (bandwidth-bound, 600 GB/s); q8_0+`CM1_SHMEM=1` 1773 (-3%).
K+V dequant is only ~310 us at mask=0; the target needs ~580 us, so **even a
perfect int8 PV could not reach f16 parity at mask=1.** The residual is the
mask=1 overhead (~219 us) plus the per-16-chunk barrier structure, which int8
arithmetic does not touch. The int8-QK scaffold stays as an opt-in
(`GGML_VK_FA_INT8_QK=1`), default-off (+0.3% from declared LDS).

## FINAL VERDICT: 55 t/s is the perfect-implementation ceiling, not a target (2026-09-25)

Step floor at 176k, every component at its measured best:
`GEMV 35.7 (640 GB/s) + FA 10.6 (raw q8_0 read) + GDN 0.8 + other 7.7 + drafts
10.7 = **65.5 ms -> 55.6 t/s**`. That is *all* mandatory DRAM work at 100% of
peak with zero overhead. 55 t/s is therefore the theoretical ceiling of this
config — the top of the 2026-09-21 doc's 51-57 range — and it requires BOTH the
GEMV (currently 87% of peak) and the FA (currently at 217 GB/s, dequant+mask+
barrier-bound) to simultaneously reach bandwidth, which four independent kernel
attempts did not achieve. The realistic optimized range is ~45-50 t/s.

**Delivered this session (verified, adopted): 38.57 -> 41.42 t/s @176k (+7.4%),
43.98 -> 46.68 @70k (+6.1%)**, from packed-GQA mask-opt and q6_K `rm_kq=4`.
Nothing further is available from tuning, int8 QK, int8 PV, MMVQ, or `-ub`/n_max
(all measured). Reaching 55 needs a from-scratch FA kernel (mask+barrier
restructure) *and* a perfect GEMV — a multi-day research project.

## WS-TG / TG-1a decode FA barrier+mask restructure: KILLED (2026-09-25)

Tried software-pipelined (double-buffered) q8 K staging and a register-fused mask
in `flash_attn_cm1.comp`, aiming at f16 parity (1251 us/layer). Reverted.

| config (isolated nb=4/183296) | mask=1 | mask=0 |
|---|---:|---:|
| baseline | 1826.5 | 1613.9 |
| K pipeline (halve staging barriers + overlap dequant/MMA) | 1803.3 (-23) | 1595.3 (-19) |
| + register-fused mask | 1803.3 (0) | 1595.3 (0) |
| `CM1_SHMEM=1` (whole-K staging) | 1776.3 (-50) | 1529.1 (-85) |
| f16 K/V | 1254.4 | — |
| int8-QK scaffold | 1851.3 | 1463.1 |

**Measured reason the kill fires:** barriers are not the bottleneck (halving them
= -1.3%; removing ~all via whole-K staging = -50 us and hurts f16 occupancy); the
mask=1 cost is the `data_m` **gather**, not the apply (fusing the apply = 0 us)
and is already skipped by `mask_opt` for the real causal mask. The irreducible
gap is that q8 must dequant each element into LDS then `coopMatLoad` it; with
KHR_coopmat there is no register-feed path for quant operands, so **f16 parity is
not reachable for a q8 cache by this route.** The only sub-1500 config is
int8-QK at mask=0 (1463) — that is TG-1b/PP-2, addressed at prefill.

Incidental blocker: adding a new `Flags` bit / specialization constant as a kill
switch makes `vkCreateComputePipelines` SIGSEGV inside `libvulkan_radeon.so` for
the switch-off variant (RADV compiler bug; reproduced after shader-cache clear
and `RADV_DEBUG=noopt`). A future new FA path must be a **separate shader module**,
not a specialized branch. Tree clean at `26bd56621`.

## WS-PP / PP-1 (FA slow state) + PP-2 (q8-direct int8 prefill FA): both KILLED (2026-09-25)

**PP-1 — the 17.9<->33.4 TFLOPS "toggle" is a perf-logger artifact.** Isolated
nb=512 q8_0 FA is **stable at ~15.0 TFLOPS in-process** (kv=49152: 15.10/15.00/
15.49; across 6 processes 14.69-15.02) while **native f16 KV is 30.0-30.5** — the
q8 deficit is the intrinsic dequant, and it is stable. Clocks under load:
3098-3193 MHz, 205-293 W, 100% use (full boost) — not a downclock. The live
dispatch (`GGML_VK_LOG_FA`) shows **exactly one** prefill state
(`N=512 dequant_kv=1 path=COOPMAT1 Br=16 ... mask_opt=1`); nothing to toggle to.
The logger's ~32.5k entries are ~2x inflated (the known "ratios only" hazard).
End-to-end arbiter: pp2048 d0->d16384 = 500 ms; FA FLOPs predict 474 ms at 15 TF
or 218 at 32.5 — the live FA is ~15 TF. **No holdable fast state exists; P2>=500
has no PP-1 support.**

**PP-2 — q8-direct int8-coopmat prefill FA regresses.** Enabled `USE_INT8_QK` for
prefill (n_rows>=64) and bypassed `use_dequant_kv` so q8_0 K feeds int8 coopmat
directly (opt-in `GGML_VK_FA_INT8_QK=1`, kill `GGML_VK_FA_NO_PREFILL_Q8=1`, default
= baseline; reused the existing Flags bit -> no RADV SIGSEGV).

| | isolated nb=512 q8 | pp2048 d0 | pp2048 d16384 |
|---|---:|---:|---:|
| before (f16 scratch) | 15.10-15.18 TF | 977/983 | 756/788 |
| after (int8-direct) | **12.33-12.64 TF** | 848/968 | 700/704 |

Correctness 138/138 (140/140 with the temporary prefill eval cases), MUL_MAT
1128/1128, decode unchanged (70k 46.14, accept 0.7629). Acceptance (>=40 TF,
>=950) fails: the f16 scratch already reaches f16-QK throughput, and int8 QK adds
staging/scale-fold cost rather than winning MMA occupancy at n=512. Same failure
mode as the decode nb=4 attempt. **Prefill q8 FA is dequant-bound at ~15 TF; the
only 2x is native f16 KV, which is a forbidden precision change.**

## WS-PP minor candidates: PP-4 (BK_STEP) and PP-5 (Br=32) KILLED (2026-09-26)

- **PP-4 BK_STEP** (cm1-int prefill GEMM, `mul_mmq_cm1.comp:93` + host
  `ggml-vulkan.cpp:1607`): BK_STEP=2 -> 3692 us / **16.3 TF**; BK_STEP=8 -> 4025 us
  / **14.9 TF**; both correct (MUL_MAT 1128/1128) but far below BK_STEP=4's
  **52.5 TF**. 4 is optimal — no win, reverted.
- **PP-5 Br=32** (prefill FA): pp2048 **974.9** vs 987.1 default (-1.2%),
  d16384 **796.4** vs 807.7 (-1.4%). Worse — killed.
- **TG-4 n_max=4 re-test with mask-opt active**: 70k **45.07 t/s** (accept 0.7004)
  vs 46.68 (accept 0.7629). The prior rejection holds; killed.

## WS-PP PP-3 (prefill-only ubatch) and TG-3 (CPU replay): KILLED / not pursued (2026-09-26)

**PP-3 KILLED — premise does not reproduce.** `-ub 1024` prefill on the current
build: pp2048 **980.3** (vs 985.0 at -ub 512), d16384 **791.3** (vs 805.4) —
i.e. no gain, slightly worse. The earlier "+7-16% prefill" no longer holds (the
cm1-int port changed the balance), so there is no prefill win to protect from the
decode regression; the allocator work is unjustified.

**TG-3 not pursued (dead by the profile bound).** The 176k step is ~88 ms; the
per-node verify graph plus three drafts already account for it (the logger
inflates GPU times, so the true GPU sum is <= that), leaving no material positive
CPU/launch gap to replay or fuse. The plan's kill is CPU <5 ms; the bound is
below that. This is the one plan item not directly instrumented (would need a
temporary steady_clock build + two depth runs for a <5 ms answer).

## rev6.1 plan execution — outcome (2026-09-26)

Every candidate dispositioned, all negative; no new win beyond the committed
+7.4%:

| candidate | result |
|---|---|
| TG-1a barrier+mask restructure | KILL: op 1803 us vs 1250 target; barriers/mask-apply not the cost |
| TG-1b / PP-2 int8-QK (decode+prefill) | KILL: mask=1 no gain; prefill int8 12.3-12.6 vs f16-scratch 15.1 TF |
| TG-2 GEMV | no win (prior investigation; ~86% peak) |
| TG-3 CPU replay/tail | dead by profile bound |
| TG-4 n_max=4 + mask-opt | KILL: 45.07 vs 46.68 t/s, accept 0.700 |
| PP-1 prefill FA fast state | KILL: perf-logger artifact; q8 FA stable ~15 TF, f16 30 |
| PP-3 prefill-only ubatch | KILL: no prefill gain to protect |
| PP-4 BK_STEP | KILL: 2 -> 16.3 TF, 8 -> 14.9 TF vs 4 optimal |
| PP-5 Br=32 / packed mask_opt | KILL: -1.2/-1.4%; packed variant decode-only |

Final state = checkpoint `26bd56621`: decode **70k 46.68 / 176k 41.42 t/s**
(+7.4% @176k), prefill **pp2048 981.8 / d16384 791** t/s. Targets D0>=48,
P1>=1000, P2>=500 are not reachable: the q8 FA is dequant-bound (~15 TF; the
only 2x is native f16 KV, forbidden), the GEMV is ~86% of peak, and the DRAM
floor (46 ms = ~79 t/s) is not the limiter — the non-DRAM serial costs (drafts,
masks, barriers, dequant) are.

## CORRECTION: the "~45 t/s ceiling" and "no prefill lever" verdicts are withdrawn (2026-09-27)

The verdicts above were limits of the kernels tried, not of the hardware. P0 of
`docs/2026-09-26-r9700-kernel-rewrite-plan.md` measured the card directly
(`harness/wmma_peak/`, `results/p0/`):

| quantity | measured | old assumption |
|---|---:|---|
| coopmat f16→f32 (register-resident) | **184 TFLOPS** (wave64) | "~96 TFLOPS fp16 peak" (that is the VALU packed-f16 rate) |
| coopmat s8→s32 | **356 TOPS** | not considered |
| coopmat fp8→f32 | **371 TFLOPS** (`VK_EXT_shader_float8` exposed) | — |
| DRAM read | 640.6 GB/s | 640 |
| LDS-fed s8 GEMM 17408×512×5120 / 4096×512×14336 | **169 / 206 TOPS** | current q6_K kernel 55 TF |
| LDS-fed f16 GEMM, same shapes | 101 / 105 TF | "f16 best case 69.8 TF" |

- Prefill at depth 0 is 85% GEMM at 49–68 TFLOPS: a third of what a plain int8
  LDS-fed GEMM reaches on the same card. Tile order matters: scheduling the N
  tiles of one weight tile adjacently lifted 17408×512×5120 from 114 to 169 TOPS
  (weight re-streaming from DRAM).
- Prefill at depth 131k is 59% FA (49.7 ms/layer, ~33 TFLOPS). The q8→f16
  scratch is only 1.5 ms/layer (2.9% of FA); the cost is the kernel (GQA not
  packed at prefill, Br=16).
- Decode 176k DRAM floor ≈ 53 ms/step (≈ 68 t/s); the q6_K GEMV is at 35.75 ms,
  near its floor; the FA (28.7 ms vs ~10 ms floor) is the decode headroom.
- The MMVQ "−7.7%" lead is withdrawn: it forced every weight type; q6_K MMVQ
  alone is a measured loss (`ggml-vulkan.cpp:6522-6529`).
- `VK_VALVE_shader_mixed_float_dot_product` is not exposed (no f16 dot2 in GLSL).
- **Measurement hazard:** decode at depth is CPU-sensitive. With sibling
  sessions' `go test` load (loadavg 13–18) 176k decode read 31.5 t/s vs 40.1 at
  lower load, same build; prefill at 176k also dropped 357 → 310. gpu_guard only
  samples at start; `harness/contam_mon.sh` logs load during a run. Compare A/B
  only back-to-back in one quiet window.

## Kernel rewrite results, fork branch `r9700-integrate` @ dfb7b9f30 (2026-09-27)

Merged: prefill FA (`r9700-p2-prefa`), Q6_K f16-WMMA GEMM (`r9700-p1-gemm`),
no-ReBAR memory fix + small ops (`r9700-p5-smallops`), decode FA q8r
(`r9700-p3-decodefa`), decode-kld tool (`r9700-p0-tools`). All A/Bs are
back-to-back vs `26bd56621` at low load.

| metric | 26bd56621 | integrate | change |
|---|---:|---:|---:|
| pp2048 @ d0 | 975 | **1,389** | +42% |
| pp2048 @ d16384 | 803 | **1,138** | +42% |
| pp512 @ d131072 (FA+GEMM only, before small-ops merge) | 378 | 436 | +15% |
| mmproj encode, 1024² image, ctx 262144 | 1,433 ms | **257 ms** | 5.6× |
| decode ~70k / ~176k (decode-FA branch alone) | 47.1 / 41.3 | **49.3 / 46.6** | +5% / +13% |
| decode ~70k / ~176k (full integrate) | 46.6 / 41.1 | 47.1 / 43.1 | acceptance changed, see below |
| PPL (4096×8) | 2.2881 | 2.2881 | 0 |
| decode KLD @176k, batch 4 / 1 | — | 0.00027 / 0.00019 mean; 0 / 1 flips of 512 | pass |

Correctness: MUL_MAT 1147/1147, FA `hsk=256,` 183/183, CONCAT 201/201,
RMS_NORM 51/51. The unfiltered FA suite SIGFPEs inside RADV's compiler on a
pre-existing hsk=320 case on the baseline too — use `-p` filters.

Per-component:
- **Prefill FA** (`flash_attn_prefill_rdna4.comp`): GQA-packed rows, one WG per
  (KV head, 16 positions), f16 scratch tiles in LDS, O in coopmat accumulators.
  Isolated N=512 KV=131584: 50.9 TF (was 27.5). Target 100 TF not reached.
- **Q6_K GEMM** (`mul_mm_q6k_rdna4_f16.comp`): Q6_K→f16 with d·sc folded at the
  global→LDS load, f16 WMMA f32-acc over all K, 128×128 BK32 wave64,
  single-buffered LDS, n-fast order. 81 TF at 17408×5120 (was 54). The int8
  variant with a per-32 epilogue topped out at 61 TF (epilogue + occupancy).
- **No-ReBAR fix**: DEVICE_LOCAL|HOST_VISIBLE is requested first; without
  ReBAR that heap is the 256 MiB BAR, so the 248 MiB clip compute buffer spilled
  to GTT and every encoder op ran over PCIe (~30 GB/s). Now auto-disabled when
  the host-visible device heap is < half the device heap.
- **Decode FA q8r** (`flash_attn_decode_q8r.comp`): q8_0 K/V loaded to
  registers, dequantized there, assigned element-wise into coopmat B operands
  (RADV gfx12 lane mapping probed and documented in the shader). nb=4
  kv=183296 in the server's interleaved layout: 933 µs vs 1,613 (cm1).
  `VK_VALVE_shader_mixed_float_dot_product` is not exposed (no f16 dot2).
- **Decode profile @176k** (per ~94 ms step): q6_K GEMV 43 ms incl. draft
  lm_head (~89% of DRAM; P4 closed), verify FA 25 ms (now ~15), draft FA 3.6,
  other GPU ~12, and **~10 ms with no GPU work** (CPU/launch) — a lever the
  2026-09-26 "TG-3 dead" verdict missed.

**Open: MTP acceptance on the repeated-paragraph prompt moved** with the full
merge (0.7629/0.881 → 0.722/0.796; with the new GEMM disabled 0.742/0.781). The
decode-FA branch alone keeps it exact, so the change comes from the prefill FA
and/or small-op numerics. PPL and decode KLD are unchanged, so this looks like
trajectory sensitivity of a single 256-token greedy run on a degenerate prompt,
**not verified**: acceptance must be measured on the code corpus and several
prompts before `r9700-integrate` replaces the serving build.

## MTP acceptance on `r9700-integrate` is unchanged — the paragraph-prompt shift was trajectory noise (2026-09-27)

Pooled over 15 code-corpus prompts (24 kB slices at 90 kB offsets of
`harness/corpus/decode_kld.txt`, 256 tokens greedy, fixed serving config; a 16th
prompt returned no draft stats on base and is excluded from all three):

| build | accepted / drafted | acceptance | per-prompt identical to base |
|---|---:|---:|---:|
| `26bd56621` | 2604 / 3642 | 0.7150 | — |
| integrate `dfb7b9f30` | 2606 / 3631 | **0.7177** | 9 of 15 |
| integrate, `GGML_VK_NO_FA_PREFILL_RDNA4=1` | 2602 / 3640 | 0.7148 | 6 of 15 |

Integrate is within +0.003 of base; the gate (±0.01) passes. Individual prompts
move by up to ±0.04 in either direction as greedy trajectories diverge, which is
what the single repeated-paragraph prompt measured (0.763 → 0.722). The corpus
depth runs agree: at ~168k tokens integrate's continuation is the same code as
base's with slightly different wording (`results/depth/txt-*.json`). Rule: judge
acceptance pooled over ≥ 10 prompts, never on one trajectory. `depth.py` now
saves each response's text in its JSON.

## Rev 3 kernel round: `r9700-integrate2` @ 2d2be1961 is the serving build (2026-09-28)

`r9700-qwen` fast-forwarded to `r9700-integrate2`; the old serving commit is tagged
`r9700-qwen-pre-rewrite` (= `26bd56621`). All numbers are one quiet window
(loadavg < 2), back to back, fixed serving config; raw output
`results/p0/final-integrate2-20260928.txt`.

| metric | 26bd56621 | integrate dfb7b9f30 | **integrate2** | vs 26bd56621 |
|---|---:|---:|---:|---:|
| pp2048 @ d0 | 978 | 1,370 | **1,507** | +54% |
| pp2048 @ d16384 | 791 | 1,125 | **1,321** | +67% |
| pp512 @ d131072 | 378 | 503 | **712** | +88% |
| decode ~70k (depth.sh, 3 reps) | 47.04 | 46.14 | **49.48** (49.48 repeat) | +5.2% |
| decode ~176k | 41.54 | 41.74 | **46.79** (46.74 repeat) | +12.6% |
| pooled MTP acceptance, 15 corpus prompts | 0.7150 | 0.7177 | 0.7058 | −0.009 |

Acceptance: the whole −0.009 is one prompt (#11: 159/286 → 149/316); the other
14 pool to 0.7286 vs 0.7279. Gate (±0.01) passes. Numerics: PPL 2.2879 (base
2.2881); decode KLD 70k b4/b1 0.00108/0.00086 (flips 0.39%/0.20%), 176k b4/b1
0.00022/0.00016 (0%/0.20%). Tests: MUL_MAT 1147, MUL_MAT_VEC_FUSION 1271,
RMS_NORM 51, SCALE 4, RMS_NORM_SCALE 16, CPY 252, CONCAT 201, GET_ROWS 240,
FA (`-p hsk=256,`) 183 — all pass.

What each task delivered (fork branches, each with env kill switches):

- **R2 Q6_K GEMM** (`r9700-r2-gemm` b7abb209b, `r9700-r2d-glu` 8b3f39d2e): split-K
  for small tile grids (ffn_down 71→80 TF, m=1024 40→65), unconditional
  loads in the parity path, 128×256 tiles, and a fused FFN gate+up+swiglu GEMM
  (removes GLU and both f32 intermediates, +2.2%). Headline 17408×512×5120 ends
  at ~86 TF (target 90/95 missed). DIAG ablation: skipping global loads reaches
  only ~90 TF, dequant is ~3 TF, LDS bank conflicts are 2–3-way and padding them
  away was −0.7%. Not built: Q6_K repack at load (every weight reader changes;
  dequant isn't the limiter), BK=64 (bounded by ~90 TF). Double-buffered and
  256×128 variants measured slower (kept opt-in).
- **R3 prefill FA** (`flash_attn_prefill_rdna4_rs.comp`, a06e3de0f): 49.5 → **79.3 TF**
  isolated. Three causes, in order of size: (1) an **LDS bank conflict** in the
  V-transpose staging stores (all 32 lanes on one bank, 8–16-way) — lane remap
  +16%; (2) **ACO serialises "prefetch" written as `x=0; if (ok) x=load`**: it
  emits `s_wait_loadcnt 0` straight after the load because the else-write aliases
  the destination; unconditional loads with clamped indices +21%; (3) QK(j+1)
  interleaved with PV(j) to hide the dependent WMMA chain +13%. Softmax in
  registers (via the probed gfx12 accumulator layout) was −2.6% on its own — the
  S round trip through LDS was not a limiter.
- **R4 decode FA** (q8r/q8w, 9da1bc3b9): nb=4 kv=183296 920 → **723 µs/layer**
  (DRAM floor ~620), nb=1 754 → 700. The largest piece was the same ACO pattern:
  the mask load sat behind 64 V loads under a branch; moving it up and making it
  unconditional 920 → 781. Then 2 WGs per CU instead of 512 WGs (split-K
  partials 125 → 32 splits), and q8w (8 waves × 32 d, 192 VGPRs) for the 1-row
  draft shape.
- **R5 decode host time** (94c753bb4): the KQ mask was refilled over all ~176k
  cells every decode (0.35–0.5 ms each, ~2.5 ms/step). Incremental fill (only
  changed cells and new columns; `LLAMA_KQ_MASK_VERIFY=1` byte-checks it) −2.1
  ms host/step, bit-identical output. Async input upload measured slower (opt-in);
  a second graph-cache slot is worth ≤0.35 ms (not built).
- **R6 decode small ops** (2d8f1be06): non-GEMV, non-FA GPU work in the verify
  graph is **~6 ms/step, not 12** (~1,300 dispatches of 3–7 µs). Wider f32
  mat-vec for m=48 −0.1 ms. RMS_NORM→SCALE fusion built but inactive (graph
  reorder separates the pair). Remaining levers: GDN state gather+CPY ~1.2 ms,
  5120-wide RMS_NORM_MUL 0.78 ms, dispatch count.
- **R7 GPU-resident MTP draft chain** (`r9700-r5-decodehost` 2fc6d40aa, opt-in
  `LLAMA_MTP_GPU_CHAIN=1`, not merged): identical drafts, but 12.3 vs 11.1 ms per
  draft phase — `token_embd` lives in host memory, so every step still splits
  GPU→CPU→GPU. It needs ~1 GB of token embeddings on the GPU; peak VRAM while
  serving is **32,581 of 32,624 MiB**, so that is closed at this ctx.

Rules learned: check ISA for `s_wait_loadcnt 0` directly after a prefetch in
any RDNA4 shader; compute LDS bank mapping for staging stores before tuning
anything else; judge MTP acceptance pooled over ≥ 10 prompts.

## Autoresearch round 1: `r9700-integrate7` @ df1e6be71 + `--spec-draft-n-max 4` is the serving candidate (2026-09-29)

integrate6 = integrate4 + P3 prefill host fixes + W2 MMVQ at n≥5 + Q8G q8_0 GEMM. It passes
`gates4.sh` (`results/d0/gates-integrate6-20260929.txt`). The numbers come from quiet
windows (loadavg < 2.6, back to back). Rows are in `autoresearch/results.tsv`.

| metric | integrate4 n3 | integrate5 n3 | **integrate5/6 n4** | target |
|---|---:|---:|---:|---:|
| decode ~70k (depth.sh, 3 reps) | 52.05 | 52.07 | **55.98 / 55.97** | ≥ 60 (missed) |
| decode ~176k | 47.58 | 47.54 | **55.72 / 55.71** | ≥ 55 (**met**) |
| pooled t/s, 37 prompts | — | 48.51 (tok/step 2.75) | **49.69** (2.97) | ≥ kept −1% |
| pp2048 d0 / d16384 | — | 1,577 / 1,377 | **1,607 / 1,403** (i6) | 1,700 / 1,480 |
| pp512 d131072 | — | 766 | 772 (i6) | 850 |

- **integrate7 = integrate6 + MQ5** (q6_K MMVQ fast path at n≥5, 2 rows/WG, `GGML_VK_NO_MQ5_TUNE=1`):
  verify n=5 46.0 → 44.6 ms; depth.sh n4 70k 58.40 → **59.41 / 59.33** (target 60, missed by 1%), 176k
  55.74 → **57.27 / 57.26**; pooled 50.25 → **50.74**. Gates pass (`results/d0/gates-integrate7-20260929.txt`),
  but they run at n ≤ 4, so the n=5 path is covered only by test-backend-ops and pooled tok/step.
  i6's 70k number rose vs i5 (56.0 → 58.4) because acceptance on the one depth.sh prompt moved
  0.657 → 0.698 (Q8G changes prefill numerics); compare depth.sh only within one acceptance.
- **n_max 4** is +2.4% pooled: corpus +8.0%, chat +0.8%, held-out −2.6%. On depth.sh it is
  +7.5% at 70k and +17% at 176k. P3 is decode-neutral (i5 n3 = i4 n3).
- **VRAM peak at n4** during a 176k prefill plus an image request is **32,609 of 32,624 MiB**.
  That is 15 MiB free, with no allocation errors. Anything that adds VRAM is closed at
  this ctx.
- **Q8G**: q8_0 had only the generic int8 MMQ path. A P1-style f16-WMMA shader takes the
  q8_0 GEMM from 59.7 to 83.9 TF in-model (+1.9% pp2048). The switch is
  `GGML_VK_NO_Q8_GEMM_TUNE=1`. The shader reads ≤4 bytes past the last block, which is
  never used and sits in a page-granular allocation.
- **Killed with measurements:**
  - q6_K rows per WG: 4 stays (1/2/8 cost +20.7/+3.9/+6.5 ms per verify).
  - MMVQ at n=4: +0.7 ms.
  - Fused gate+up+GLU GEMV: −0.4 ms ABAB, under the 0.8 kill line.
  - ubatch 1024/2048: d0 is −1.2%/−3.5%.
  - P2c FA mask double buffer: 0%.
  - f32 m=48 split-K: ceiling below 1 ms.
  - Prefill residual ADD in GEMM epilogue: net −1.3 ms (GEMMs +2 ms).
  - GEMV fixed cost: quantize reuse and barrier grouping already exist; skipping every q8_1
    quantize saves only 1.0 ms. The ~8 µs/call is in-kernel ramp/tail and RAW barriers.
  - Prefill GATED_DELTA_NET: fp32 VALU-bound at ~65% of floor. Only a chunked WY/WMMA
    kernel could reach 2×, which is a multi-day project.

Rules learned:
- Perf-logger in-model A/Bs repeat to 0.02 ms at load < 4, but spread ~1 ms at load 10–15.
  Run them ABAB and record the load.
- `inmodel.sh`'s "GEMV sum" line misses fused-op rows. Compare the per-op rows instead.

## Workstation RAM has a weak cell on bit 58: the compiler and app crashes were hardware (2026-09-29)

Symptoms. Rebuilding `r9700-qwen` hit a gcc internal compiler error (segfault) in 4 of 5
attempts, in a different file each time. The kernel log shows general protection faults in
`glslc` (three times, 09:15, while building) and in the desktop indexer `localsearch`
(2026-09-28), all in stock system libraries at unrelated addresses. RAM was not short
(23 GB available), and temperatures were normal (Tctl 49 °C idle, GPUs 24–52 °C).

Test. `harness/memtest.c` is a user-space pattern tester: address, inverse address, walking
ones, and xorshift random. Build it with `gcc -O2 -pthread -o memtest harness/memtest.c`
and run `./memtest 16 3 24`. It ran over 16 GiB, 3 passes, 24 threads, in 60 s and found
**4 errors, all at one address with the same bit**: bit 58 reads 1 where 0 was written
(xor `0400000000000000`). The fault showed on every pass and in both the address and random
patterns. The inverse and walking patterns passed at that address because they expect a 1
there. The RAM is non-ECC, so EDAC reports nothing. (This first read it as a stuck-at-1
cell. memtest86+ below showed the cell also fails 1→0, and is pattern-dependent.)

memtest86+ v8.00 (GRUB entry "Memory test (mt86+x64)"; the menu is hidden, so use
`sudo grub-reboot 'Memory test (mt86+x64)' && sudo reboot`):
- DDR4-3600 CL26: 5 errors in ~1 pass, tests 6 and 7, at `0x3dde30360`, `0x3dde306a0` and
  `0x3df385ba0`. All on bit 58, in both directions.
- DDR4-3200 CL22: 4 errors in ~1 pass, all at `0x3dde30360`, bit 58.
The same cell fails at both speeds, so this is a defective cell, not timing. `0x3df385ba0`
has only failed at 3600. A root run of `harness/memtest.c` (now reports physical addresses)
covered both pages for 3 passes and found nothing: its patterns do not trigger the cell.

Crash attribution, from the kernel log (09-09 to 09-29) and Firefox minidumps:
- Firefox main-process crashes on 09-26 and 09-29 faulted on pointers that are valid
  except for bit 58 (`04007aeff9a00000`, `04000f2187d04a30`). Setting bit 58 makes a user
  pointer non-canonical, which gives exactly the "general protection fault" seen in
  `glslc` (09-20, 09-29 ×3) and `localsearch` (09-28).
- Not RAM: ~40 `test-backend-ops` traps in `libvulkan_radeon.so` (the same two code
  offsets every time, a RADV bug), Chrome "invalid opcode" (its own CHECK traps, same
  offset), `wmma_isa` in lavapipe (same offset twice), and one OOM kill on 09-22
  (systemd-oomd under memory pressure; swap was present).
- No machine-check events in 20 days, so no sign of a CPU fault.
- 84 parallel compiles of `ggml-vulkan.cpp` (~21 GB RAM) all gave byte-identical objects.

Impact. `git fsck` on perf-lab and `llama.cpp-r9700` is clean. The serving build that
completed gives byte-identical greedy output to `r9700-integrate7`'s build (200 tokens, same
MD5). All gate and benchmark results above passed their own checks. A flipped bit in a
running process would more likely crash than silently shift a number, but that is not
proven.

Mitigation (in place 2026-09-29, no replacement RAM available):
- `harness/badram.cfg` is installed as `/etc/default/grub.d/badram.cfg`. It passes
  `memmap=4K$0x3dde30000 memmap=4K$0x3df385000`, and the boot log shows both pages as
  "device reserved". If DIMMs change, delete it and run `update-grub`.
- The memory runs at DDR4-3200 (it was 3600), so CPU-side overheads in benchmarks from
  before 2026-09-29 16:38 are not directly comparable. Re-baseline before A/Bs.
- Swap is one 32 GB `/swapfile` (it was 8 + 16 GB).
- Still open: an overnight memtest86+ run (10+ passes, F1 → error reporting "Linux
  memmap") to find any other weak cells, then a repeat every month or so. A growing list
  means the DIMM is failing.

Until then, a random GPF or compiler ICE on this machine is suspect hardware first. Verify
any important binary against a known-good build before trusting it.

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

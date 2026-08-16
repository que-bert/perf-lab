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

> Two of these fallback numbers did not reproduce on 2026-08-16 — see
> "Fallback speeds are not reproducible" below. The fast/fallback split is not in
> doubt; its *size* is.

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

Leading hypothesis, untested: the 2026-08-15 readings were taken while ollama's
`llama-server` was resident on this card, which was independently confirmed on
2026-08-16. Kernel selection for the fallback paths may depend on free VRAM.
Testing it means re-measuring `slow-q4_1` with a deliberate VRAM occupant present.

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

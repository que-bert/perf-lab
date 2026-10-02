# Phase 2 report (2026-10-02)

Plan: `docs/2026-10-01-phase2-plan.md` (rev 3). Every number below is in `setups/qwen3.8-27b-r9700/autoresearch/results.tsv`
(rows W1, W2HOST, W2SERVE, PPFIX, W7KLD, Q4DEC) or under `results/phase2/`. Fork branches on que-bert/llama.cpp.

## Disposition per workstream

| item | status | result |
|---|---|---|
| W0 harness | done | `gpu_lock.sh` waits for a free card, samples DRM clients + load + VRAM/GTT for the whole run, voids and re-runs contaminated runs (ledger `results/gpu_ledger.log`); `model_profile.sh`, `model_quality.sh`, `zoo_smoke.sh`, `ab_stat.py`, `profile_report.py`, `zoo.tsv` |
| W0 references | done | upstream KLD logits for all 8 zoo files (`/mnt/.../perflab-kld-ref`, local only) + smoke tokens |
| W0 noise, k | done | 27B: CV 0.04-0.54 % per axis -> k 2-8 for 0.3-0.5 %; MiniCPM: CV 0.2-1.3 % (pp d8192 5 %) -> k 4-7 for 1 % |
| W0 inventory | done | `docs/2026-10-02-fork-path-inventory.md`: 30 `GGML_VK_NO_*` + 93 env names, no-switch paths, RDNA4 detection, loose guards |
| W0 all-switches-off | done | text = upstream on 7/8 files (32 tokens); not speed-equal: no-switch paths remain (int8 MMQ, small-BAR rule) |
| W1 RB | done, RB not adopted | cost found (upstream batch API copies embeddings inside llama_process), fix recovers ~45 %; residual -0.21 % [-0.23,-0.18] at 70k prefill -> plan rule: integrate16 stays the base, upstream by cherry-pick |
| W2 MiniCPM | done | small-BAR rule cost 6.5 % decode (staging copies per input); fixed -> +0.7 % over upstream; serving preset ngram-simple +30 % corpus / +3.5 % chat-like; Q4_K_M +27 % speed vs Q8_0 at +9 % PPL (both presets kept); I2 baseline set |
| W4 presets | done | GGUF identity presets (27B, MiniCPM), per-arch presets (gemma4, qwen35, qwen35moe), MTP auto-on with parallel 1, explicit flags win, `--print-preset`, `--no-preset`; flagless 27B pooled = explicit (56.39 vs 56.43, same tok/step) |
| W4 converter | done | `harness/ollama_gguf_convert.py` (gemma4 E4B/E2B, qwen35 incl. MTP head); loader prints a hint naming it; no loader divergence |
| W4 validated-only | done | int8 cm1 MMQ on RDNA4 limited to Q8_0/Q6_K; FA staging and fork cm1 shader limited to quantised KV; small-BAR host-visible rule limited to small models |
| W4 KV reuse with MTP | see T2/W4 reuse log | `results/phase2/w4_reuse.log` |
| W5 Gemma4 | done | fork FA crash at hsk 512 (ACO SIGFPE, fork staging) fixed; second bug (sparse-mask row) fixed; f16-KV FA back to upstream speed (hsk256 pp 93 -> 66 us); pp +12-19 % over upstream; tg d0 -3 % remains (not FA) |
| W6 tuning | done | runtime sweep (ubatch x KV) for 6 non-27B files -> presets; kernel tuning folded into W5/W7 (MoE q6_K down GEMV -20 %) |
| W7 MoE | done | KLD "bug" was the Vulkan reference (upstream Vulkan vs CPU 0.049, fork 0.046); fork tg 2.7x upstream, MTP n=3 serving 153 vs 95 t/s (corpus); expert q6_K down GEMV 18.3 -> 14.5 us |
| W3 Q4/Q5 kernels | skipped by plan rule | Q4_K/Q5_K GEMV within 10 % of Q6_K on every dense shape (27B: -4 %/-9 %) |
| W8 Q4 prefill GEMM | not triggered | conditional on adopting 27B Q4_K_M; not adopted (PPL +1.7 %, KLD 0.035, top1 95.9 %, acceptance -4.6 %) |
| W9 Qwen4 readiness | done (tooling) | `harness/day0.sh <name> <gguf>`: convert if ollama-packed, add to zoo, preset, upstream refs, profile, quality, smoke |
| Prefill swings (found in W0) | fixed | mmap-resident token_embd re-faulted after page-cache reclaim (every build; serving immune via -lm none); loader populates at end of load |

## Integration

- `r9700-integrate17` = integrate16 + W2 small-BAR fix + W4 presets/hint. T2 vs the integrate16 anchor: see `results/phase2/t2/i17/`.
- `r9700-integrate18` = integrate17 + loader populate + W5 FA + W6 presets + W7 MoE GEMV + validated-only MMQ guard. T2: `results/phase2/t2/i18/`.
- Serving (`r9700-qwen`) is not changed by this phase; promotion is the operator's call.

## Open

- Gemma4 tg d0 -3 % vs upstream (not FA; small-BAR gate excludes 5 GB models; test with GGML_VK_SMALL_BAR_MODEL_MAX_MIB in `results/phase2/w2/serve/gate`).
- RB residual: a zero-copy embd view in `llama_batch_ext` would remove the last copy in the MTP prompt pass; needed before the next rebase (T3).
- I3 for MoE must use a CPU reference (the Vulkan-reference KLD cannot be < 0.04 for any non-identical build).

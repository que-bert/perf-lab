# R9700 Qwen build plan

Date: 2026-09-20. Status: approved direction, Build B not started.

## Goal

Two separate llama.cpp builds with different jobs:

- **Build A - `mimir-general`.** Latest upstream master, **unpatched**, no
  hardware-specific changes. Must load and run every text GGUF on the mount. This
  is what mimir points at as a general backend. Stability is the feature.
- **Build B - `r9700-qwen`.** A private fork tuned for gfx1201 (RDNA4) and
  long-context Qwen decode. Never shipped to mimir. "Absolute best we can do" is
  scoped below as a measured target, not a promise.

Success for Build B: a decode improvement **at depth** that is outside run-to-run
spread, with **no change to weight or KV precision**.

## Fixed serving config (all measurements)

Qwen3.8-27B Q6_K, ctx 262144, `ctk/ctv q8_0`, `-fa on -ngl 99`, `--parallel 1`,
`--spec-type draft-mtp --spec-draft-n-max 3`, `--mmproj mmproj-F16.gguf`, R9700
(Vulkan device 1), temperature 0. No precision trade: MXFP4/AWQ/FP8 and
lattice/lower-bit KV are out.

## Discovered constraints (measured 2026-09-20)

- **Depth decay is fundamental.** Decode 60.96 -> 28.04 t/s as filled context
  goes 42 -> 182,889 tokens; acceptance flat (0.68-0.76); prefill 650 -> 324.
  Recorded in `FINDINGS.md` (2026-09-20).
- **Build version does not move depth decode.** @70k / @176k: b10472 34.09/27.37,
  b10883 38.05/27.38, b10902 37.97/28.04, master `ce8caa6e6` 37.30/27.98. b10883
  and master marginally best, b10472 worst. The short-context rows are noise.
- **`power_dpm_force_performance_level=high` regressed** decode (38.05 -> 35.70
  @70k, 27.38 -> 25.06 @176k) and prefill. Keep `auto`.
- **llama.cpp Vulkan has no `AMD_RDNA4` enum.** `ggml-vulkan.cpp:457-465`
  classifies RDNA4 as `AMD_RDNA3` via the int8-dot path, so RDNA4-specific code
  cannot be selected today.
- **`rm_kq` is a computed local** (`ggml-vulkan.cpp:5462`); only `AMD_GCN` changes
  it, every RDNA card gets 2.
- **Driver:** only RADV (Mesa 26.0.8); no AMDVLK. R9700 at PCIe 16 GT/s x16.
  ASPM set to `performance` (volatile); CPU governor `performance`.
- **No Vulkan profiler.** `harness/kernel_profile.py` is rocprofv3/ROCm only.
- **Build A validated** on `ce8caa6e6`: qwen35, qwen35moe, llama all load and
  decode; diffusiongemma and minimax_h3 fail to load (not LLMs, expected).

## Non-goals

- Precision trades of any kind.
- Serving Build B through mimir.
- Bonsai 2 27B (dropped). MiniCPM5-2B LoRA (deferred).

## Interfaces

- Build A: `~/llama.cpp/mimir-general` (or the existing `ce8-vulkan`).
- Build B source: `~/git/llama.cpp-r9700` (isolated git worktree, never the
  adaptive-MTP worktree). Build output: `~/llama.cpp/r9700-qwen`.
- Benchmark: `harness/ctx_build.sh` + `harness/ctxcost.py` (promote from
  `/tmp/opencode`), reporting decode/prefill/acceptance at short, ~70k, ~176k.
- Every result recorded in `FINDINGS.md` with the build commit and fingerprint.

## Tasks (Build B)

- **B1 - RDNA4 detection.** Add an `AMD_RDNA4` architecture and a detection path
  that distinguishes gfx1200/gfx1201 from RDNA3. Acceptance: the build can report
  the card as RDNA4; RDNA2/RDNA3 paths are byte-for-byte unaffected.
- **B2 - RDNA4 GEMV tuning.** Apply `rm_kq` and any sibling tuning behind the new
  arch. Acceptance: depth decode improves or holds; no short-context regression.
- **B3 - Attention/FA at depth.** The only lever for the decay. Targets found
  2026-09-20: `get_fa_tuning_params_coopmat1` (`ggml-vulkan.cpp:1261`) is
  arch-agnostic hardcoded constants (block_rows 16, block_cols 64,
  num_subgroups 4) and is the path the MTP verify pass takes at depth - `n_rows=4`
  bypasses the `n_rows == 1 -> FA_SCALAR` fallback at `:1348`, and
  `coopmat1_fa_support` is true on RDNA4/RADV (`:4298`). `get_fa_tuning_params_scalar`
  (`:1186`) is AMD-tuned but the comments say "tested on RDNA2". Add an RDNA4
  branch, rebuild the shaders, measure at 176k with reps. Acceptance: >5% decode
  gain at 176k, no quality change.
- **B4 - Driver A/B.** AMDVLK vs RADV, Mesa version, clock policy. Acceptance:
  a measured 70k/176k comparison, recorded.
- **B5 - Promote the benchmark** into `harness/` and log each result.

## Acceptance criteria (project)

- Build B beats the best unpatched build at 176k decode by a margin larger than
  run-to-run spread, established with >=3 reps per point.
- Build A is unchanged in behaviour and still loads all text GGUFs.

## Open questions

- Run-to-run spread at depth is not yet established; the current numbers are
  single-rep. This gates every acceptance test above.
- Is AMDVLK installable and stable against this Mesa/kernel?
- Does an RDNA4-targeted FA kernel exist upstream? `#27952` is coopmat int8,
  prefill only.

## Dispositions (2026-09-21)

- **B1 RDNA4 detection — done.** Both cards report `arch: RDNA4`; `is_rdna3_4`
  keeps prior behaviour. No measurable decode effect either way, but it is the
  prerequisite for any RDNA4 branch.
- **B2 RDNA4 GEMV tuning — done, negative.** `rm_kq=1` regressed ~15%; reverted.
  `rm_kq=4` untested and targets the weight path, not the depth bottleneck.
- **B3 Attention/FA at depth — done, negative.** Ported PR #28507 (shmem staging
  on AMD RDNA scalar FA) behind `GGML_VK_DISABLE_FA_SHMEM_STAGING`: 38.487 vs
  38.505 t/s at 70k — no effect. `GGML_VK_DISABLE_COOPMAT=1` costs 3.8%, so
  coopmat1 is already the better path. No upstream/community RDNA4 Vulkan FA
  patch moves this card. Acceptance criterion (>5% at 176k) **not met**.
- **B4 Driver A/B — closed, not executable.** AMDVLK 2025.Q2.1 (latest) returns
  `VK_ERROR_INCOMPATIBLE_DRIVER` on gfx1201. No root for a system install, and
  device support would not change. Independent R9700 evidence has RADV ahead at
  depth.
- **B5 Promote the benchmark — done.** `harness/depth.py`, `depth.sh`,
  `gpu_guard.sh`; results under `results/depth/`.
- **Added (not in the original plan): n-gram + MTP chain.** `--spec-type
  ngram-mod,draft-mtp` is +17.2% on code at 98k and +9.1% on the paragraph at
  70k, 0% at 234k, never negative. This is the one config win, and it is
  precision-neutral.
- **Project acceptance criterion (beat the best unpatched build at 176k by more
  than spread) — not met by any kernel change.** The measured position is that
  at 176k-234k the shipped config is already at the practical optimum of the
  available software; `docs/2026-09-21-r9700-decode-physics.md` bounds the
  remaining ~1.5x as kernel inefficiency with no patch that reaches it.


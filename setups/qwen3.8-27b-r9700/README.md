# Qwen3.8-27B on the R9700

A llama.cpp fork tuned for one job: serving Qwen3.8-27B Q6_K at 262k context on an AMD Radeon AI
PRO R9700 (RDNA4, 32 GB) through Vulkan, with MTP speculative decoding.

## Source and binaries

| role | branch (github.com/que-bert/llama.cpp) | local build |
|---|---|---|
| serving | `r9700-qwen` @ 901abb2a8 (= integrate11) | `runners/llama.cpp/qwen3.8-r9700/build/bin` (submodule) |
| next (integration head) | `r9700-integrate16` @ 02ea7fe9f | `runners/llama.cpp/qwen3.8-r9700-next/build/bin` (worktree) |

Every experiment is a branch on the fork: `r9700-ar-<id>` for autoresearch experiments,
`r9700-integrate<N>` for integration heads, `r9700-{p,r,d}<n>-*` for the earlier plan phases.
Each id has a row in `autoresearch/results.tsv` saying what it changed, what it measured and why it
was kept or discarded. Old serving builds are tagged `r9700-qwen-integrate{2,7,8,10}`.

Build (Vulkan SDK + glslc; `/tmp` on this box is a small tmpfs):

```
cd runners/llama.cpp/qwen3.8-r9700
mkdir -p build/tmp && TMPDIR=$PWD/build/tmp cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DCMAKE_C_FLAGS=-I$HOME/.local/include -DCMAKE_CXX_FLAGS=-I$HOME/.local/include
TMPDIR=$PWD/build/tmp nice -n 19 cmake --build build -j 12 --target llama-server
```

The next build is a worktree of the same repo:
`git -C runners/llama.cpp/qwen3.8-r9700 worktree add ../qwen3.8-r9700-next r9700-integrate16`.

## Serving config

Qwen3.8-27B Q6_K, `-c 262144 -fa on --parallel 1 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp
--spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0 --spec-draft-vocab 98304 --spec-draft-vocab-adaptive
--mmproj mmproj-F16.gguf -lm none`, environment `MTMD_LAZY_GPU=1 GGML_VK_HOST_GET_ROWS=1`.
`GGML_VK_HOST_GET_ROWS=1` without `-lm none` corrupts decode (GATES10): set both or neither.
The mmproj leaves VRAM when idle (~1.1 GB), so an image request costs ~2 s more. Past ~120k
context the card is full and pages oscillate to GTT (row VIMG).

## Where it stands (2026-10-01)

| metric | serving (integrate11) | integrate13 | integrate16 | target |
|---|---:|---:|---:|---:|
| decode ~70k t/s | 63.5 | 62.0 | 62.5–62.8 | ≥ 60 |
| decode ~176k t/s | 60.5 | 59.0 | 59.4–59.9 | ≥ 55 |
| pooled decode t/s (37 prompts) | 55.7 | 55.7 | 56.2 | — |
| server prefill ~70k t/s | 1,160 | 1,218 | 1,245 | — |
| server prefill ~176k t/s | ~706 | 844 | 846–852 | ≥ 800 |
| server prompt 32.5k t/s | ~1,324 | ~1,349 | 1,380–1,392 | ≥ 1,400 |

integrate13 lowers depth.sh decode on its one prompt through acceptance (ms/step unchanged); the
pooled number is the arbiter. integrate16 = integrate13 + MFL + SOP + VSUB; promotion to serving
waits on the operator (rows I13, I15, I16 in `autoresearch/results.tsv`).

## Layout

```
autoresearch/program.md     the research loop contract (roles, kill lines, gates, traps)
autoresearch/queue.md       ranked hypothesis queue
autoresearch/results.tsv    one row per experiment, append-only: the record of what worked and what did not
autoresearch/ab_bin.sh      end-to-end binary A/B (depth.sh ABAB, pooled 37 prompts, image test)
autoresearch/pool.py        pooled decode arbiter
results/d0/                 measurements: per-op profiles, gates, e2e logs, budget studies; scripts/ has the runners
results/p0/                 int8 GEMM probe and KLD references (KLD logits are local only)
results/2609-sweeps/        2026-09-10/11 throughput, KV-quality and spec-depth sweeps for the 27B
docs/                       dated plans and physics notes (2026-09-20 to 2026-09-28)
```

Shared tools live in `harness/` at the repo root (`depth.sh`, `prefill_profile.sh`, `serve_unit.sh`,
`gpu_lock.sh`). Every GPU run goes through `harness/gpu_lock.sh`; only one model may be on the card.

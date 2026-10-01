# Small models

Capability evals, tool-use evals and speed sweeps for small local models. Kept apart from the
Qwen3.8-27B/R9700 tuning work in `setups/qwen3.8-27b-r9700`.

## Models covered

| model | results |
|---|---|
| Bonsai (PrismML low-bit) | `results/bonsai2/` (R9700 speed, MTP fork, bandwidth profile, follow-ups, 2026-09-18/19) |
| Gemma 4 E2B / E4B | `results/eval-gemma4-*`, `results/tools-gemma4-*` |
| MiniCPM (Q4_K_M, Q8_0) | `results/eval-minicpm-*`, `results/tools-minicpm-*` |
| Ornith (Q4_K_M, Q6_K) | `results/eval-ornith-*`, `results/tools-ornith-*`, `results/thr-ornith-*` (per-GPU throughput) |
| Qwen3.5-9B | `results/eval-qwen35-9b-*`, `results/tools-qwen35-9b-*` |

Eval files come in three prompt modes (`raw`, `chat`, `chat-nothink`); compare within a mode only
(see "Prompt mode is not a detail" in the top-level README). Tables are produced by
`harness/eval_table.py`; runs by `harness/eval_matrix.sh` and `harness/model_eval.py`.
`results/kv-quality-32768-*` is the 32k KV-type quality sweep.

## Binaries (local, `bin/`)

- Bonsai: `bin/bonsai-prism/prism-b10685-vulkan`, `bin/bonsai-prism/prism-b10685-rocm`, built from
  the PrismML fork source in `runners/llama.cpp/bonsai-prism` (branch `prism`, b10685).
- Everything else: upstream `bin/upstream-vulkan/b10472-vulkan` unless a result file says otherwise.

Findings for these models are in `FINDINGS.md` (2026-09-10 to 2026-09-19 entries).

# autoresearch: R9700 decode/prefill loop

A Karpathy-style research loop (one idea, one fixed-budget measurement,
keep or revert, log it, next idea). It is adapted to a machine with one GPU,
where the GPU is the scarce resource and sub-agent context is the expensive
one.

## Objective

The serving config is fixed (Qwen3.8-27B Q6_K, KV q8_0/q8_0, ctx 262144,
`-fa on`, `--parallel 1`, MTP draft). Weight and KV storage formats never change.

| metric | arbiter | target | stretch |
|---|---|---:|---:|
| decode ~176k | `harness/depth.sh` DEPTHS=4000 | ≥ 55 t/s | ≥ 62 |
| decode ~70k | `harness/depth.sh` DEPTHS=1600 | ≥ 60 t/s | ≥ 67 |
| pooled decode t/s, 37 prompts | `autoresearch/pool.py --set all` | ≥ the kept baseline −1% | — |
| pp2048 d0 / d16384 | llama-bench | ≥ 1,700 / ≥ 1,480 | 1,850 / 1,600 |
| pp512 d131072 | llama-bench | ≥ 850 | ≥ 1,000 |

A decode change that raises depth.sh but lowers pooled t/s by more than 1% is
not kept: depth.sh is one degenerate prompt.

## Roles

- **Root (main session).** It owns the queue below, `results.tsv`, every
  end-to-end measurement (depth.sh, pool.py, llama-bench), every merge into the
  integration branch, and the gate runs. It reads only workers' result rows,
  never their transcripts.
- **Workers.** One experiment per worker, on the Sonnet model (`model: "sonnet"`; operator ruling 2026-09-28):
  - its own worktree and branch `r9700-ar-<id>`, cut from the current integration head;
  - at most 60 turns;
  - GPU only through `harness/gpu_lock.sh`, and only for microbenchmarks and focused op tests;
  - never runs depth.sh, pool.py or a full test suite.

  At most 2 workers run at once. Builds use `nice -n 19 -j 12`. A worker's
  output is its branch and one result row.
- **Quiet windows.** The root measures only when no worker is building
  (loadavg < 2). Workers are launched right after a measurement batch, and the
  root batches its measurements while they run.

## One iteration

1. **Hypothesis**, in one line, with the ms/step or t/s it should save,
   derived from the latest budget (`results/d0/`). Drop any idea whose ceiling is
   under 1 ms/step at 176k, or under 2% of prefill.
2. **Worker** implements it behind a switch (`GGML_VK_NO_<X>=1`, or a CLI flag
   for host/draft policy). It runs the focused op tests and a microbenchmark
   that isolates the term: test-backend-ops perf, or `GGML_VK_PERF_LOGGER` at
   70k. It returns:
   `id | branch@sha | term | before | after | unit | tests | note`.
3. **Root** screens it on the worker's number. A result below its kill line is
   logged as `discard` and the branch is left unmerged. Otherwise the root runs
   the end-to-end arbiter against the kept baseline in the next quiet window.
4. **Keep** if the arbiter improves beyond noise and the gates hold (see
   below). Merge into the integration branch, and the new head becomes the
   baseline. Otherwise mark it `discard`.
5. Append one row to `results.tsv`. Repeat. **Never idle.** When the queue is
   empty, re-derive the budget from the latest perf-logger run and write new
   hypotheses.

Noise: depth.sh decode spreads are 0.1–0.7% within a quiet run, and 176k
has a ~6% run-to-run floor under load. Decode A/Bs run back to back, with the
candidate run twice. A change must beat the baseline on both candidate runs.

## Gates (before a merge into serving)

`results/d0/scripts/gates4.sh` on the integration head:
- 21 op suites;
- PPL within 0.2% of 2.2879;
- decode KLD 4/4 in line with integrate2;
- corpus and chat acceptance within ±0.01 of the reference at the same draft
  policy.

For a draft-policy change, judge tokens/step and t/s, not the acceptance
rate. For VRAM, measure the peak during a 176k prefill plus an mmproj image
request (43 MiB was free before D0b freed 234 MiB).

## Traps

- `/tmp` is a nearly full tmpfs. Build with `TMPDIR=<worktree>/build/tmp`,
  and pass `-DCMAKE_{C,CXX}_FLAGS=-I/home/bbuckham/.local/include`.
- Grep for `<<<<<<<` before committing a merge.
- ACO: `x=0; if (ok) x=load` serializes the load. Use unconditional clamped loads.
- Verify GEMV and FA are DRAM-bound. Check the byte floor before tuning them.
- Chat acceptance (0.54) is far below corpus acceptance (0.71). Judge both.

## Files

- `results.tsv`: one row per experiment, append-only.
- `queue.md`: the ranked hypothesis queue, which the root rewrites.
- `pool.py`: the pooled t/s arbiter.
- The perf-lab nightly/drain/heartbeat timers are disabled and masked (2026-09-30 OOM:
  nightly canaries raced a GPU-locked experiment). Before any GPU run, `pgrep -a
  'llama-(server|bench)'` must show nothing outside the lock holder. Only one model may be
  loaded on the R9700 at a time; a second spills to system RAM and OOMs the machine.

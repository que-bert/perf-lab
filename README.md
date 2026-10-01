# perf-lab

A regression tripwire for local LLM inference.

Tracks llama.cpp, ROCm, Mesa and kernel together, so that when throughput moves the
ledger can say **which of them moved it**. Every measurement carries a complete
fingerprint of the environment that produced it.

## Why

Upstream is not monotonically improving, and the stack changes without being asked:

- llama.cpp b10438 measured **22.5% slower** at deep prefill than the pinned b10082.
- Only `q8_0` and `q4_0` KV cache types have optimized flash-attention kernels on this
  backend. `q5_1`, `q5_0`, `q4_1` and any mismatched K/V pair fall to a **much slower
  prefill path while decode looks completely normal** — invisible to short-prompt
  testing. The split is not in doubt; its size is. Measured 8–13× on 2026-08-15 and
  4.7× on 2026-08-16 from an identical fingerprint, which is still unexplained.
- `unattended-upgrades` installs graphics-stack changes on its own schedule.

See [FINDINGS.md](FINDINGS.md) for the measurements behind those claims.

## Usage

```
make row          # one live canary row
make canaries     # all five canary configs (~4 min)
make nightly      # every canary as one tagged batch, with a deadline
make check        # read the ledger back: is each canary still in band?
make heartbeat    # file Issues for sustained breaches / 72h of silence
make rebaseline   # re-derive canary bands from measured spread
make validate     # check the ledger against results/schema.json
make test         # harness acceptance cases
```

Requires `PERF_LAB_MODEL`, `PERF_LAB_BIN` and `PERF_LAB_GPU_UID`. The GPU is selected
by unique ID; if it is absent the run aborts rather than silently measuring a different
card and stamping the row with a fingerprint that lies.

### Setups

Each setup is one model + runner + card combination, with its own README, results and at most
two binaries (the one serving and the one being tested next).

| setup | what | README |
|---|---|---|
| Qwen3.8-27B on R9700 | tuned llama.cpp fork (Vulkan, RDNA4 kernels, MTP speculative decoding) | [setups/qwen3.8-27b-r9700](setups/qwen3.8-27b-r9700/README.md) |
| Small models | Bonsai (PrismML fork), Gemma 4, MiniCPM, Ornith, Qwen3.5-9B evals and speed sweeps | [setups/small-models](setups/small-models/README.md) |
| Upstream llama.cpp | pinned upstream builds the canaries and baselines run on | `bin/upstream-vulkan`, `bin/upstream-rocm` |

## Layout

```
setups/<setup>/            README, results, plans and loop state for one model/runner/card
runners/llama.cpp/
  qwen3.8-r9700/           submodule: github.com/que-bert/llama.cpp, branch r9700-qwen (serving)
  qwen3.8-r9700-next/      worktree of the same repo at the integration head (local only)
  bonsai-prism/            PrismML fork source for Bonsai (local only, not a git repo)
bin/                       prebuilt binaries per setup, at most two each (local only)
  upstream-vulkan/         b10472-vulkan (canary pin), b10902-adaptive-mtp (has draft-dflash)
  upstream-rocm/           b10082-rocm
  bonsai-prism/            prism-b10685-vulkan, prism-b10685-rocm
models/                    small local models (local only)
results/ledger.jsonl       append-only, one row per canary measurement
results/schema.json        the contract; CI-validated
results/depth/             harness/depth.sh default output
harness/                   probe, bench, check, alert, run_set, install, shared measurement tools
configs/canary.yaml        canary configurations and their bands
systemd/                   timer units (installed into the user instance)
apt/99-perf-lab            upgrade hook; queues a run, never runs one
FINDINGS.md                lab-wide findings log (dated entries; paths in older entries predate this layout)
```

Clone with `git clone --recurse-submodules`, or run `git submodule update --init` afterwards.

## Beyond the canaries

The canary set answers "did the stack move". These answer questions it cannot,
and none of them write to the ledger:

```
harness/gguf_meta.py     what is in this model file, without loading it
harness/gguf_patch.py    rewrite one metadata key, tensor data untouched
harness/serve_unit.sh    serve any model as a transient systemd --user unit
harness/model_eval.py    what is this model good at, scored programmatically
harness/eval_matrix.sh   one model, every prompt mode
harness/eval_table.py    collate model_eval reports into one grid
harness/throughput.py    aggregate tokens/s under concurrency
harness/spec_sweep.py    draft depth x slot count, as a cross product
```

Three rules these encode, each of which cost a session to learn:

- **A model load cannot live inside a supervised task on this box.** The
  supervisor kills the task as page cache fills, with tens of GB free and
  nothing starved; `setsid nohup` does not escape it. Every tool here that
  needs a server goes through `serve_unit.sh`, which hands the process to the
  user manager instead.
- **Prompt mode is not a detail.** The same model scored 12/31 prompted raw and
  31/31 through its own chat template. Any capability number has to record
  which mode produced it, so `model_eval.py` does.
- **Single-stream decode and aggregate throughput move in opposite
  directions.** `throughput.py` reports both, because reporting either alone
  recommends the wrong `--parallel`.

## Automation

```
harness/install.sh          # installs the timers; prints what still needs root
harness/install.sh --uninstall
```

Three triggers:

- **nightly** at 03:00 — the routine sweep.
- **drain** hourly — runs only if an apt upgrade queued one, and costs 16 ms when
  nothing is. Its purpose is attribution rather than speed: draining on the nightly
  instead would give the post-upgrade run the same before/after pair the nightly
  already has, which is no information at all. An hourly drain brackets the upgrade
  within the same morning; the nightly catches the regression either way.
- **heartbeat** at 09:00 — files an Issue for a sustained breach, a failed push, no
  successful run in 72 hours, or a single canary silent for 72 hours while the others
  still run. The last two matter most: a tripwire that quietly stopped running looks
  exactly like one reporting good news. The per-canary case is that same failure one
  level down — any one canary still reporting keeps the ledger fresh, so four can go
  dark unnoticed while `check.py` reports their stale verdicts as passing ones. Every
  verdict now carries `as_of` and `runs_since` so that is visible without an Issue.

A breach must hold the same direction for two consecutive runs before it is reported.
Single-run breaches are noise; on an idle card the measured spread is 0.02–1.6%, but a
contaminated rep can move a single reading by 40% or more.

`install.sh` never calls sudo. Enabling lingering and placing the apt hook are printed
for you to run. Without lingering the timers do not fire while you are logged out,
which would quietly defeat the whole point.

## The ledger contract

A row is invalid unless it carries `ts, run_id, kind, fp, cfg, m, ok`. The fingerprint
`fp` must name the GPU by **PCI address and unique ID, never by index** — `rocm-smi`
GPU[1] and DRM card0 are the same device on this host, and the enumerations are
inverted. PCI strings are lowercase; `rocm-smi` emits uppercase hex, and unnormalized a
fingerprint fails to match itself.

Live rows must record a prefill metric, not just decode. A `tg`-only row cannot see the
kernel-fallback regression this repo exists to catch.

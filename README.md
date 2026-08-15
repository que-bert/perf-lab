# perf-lab

A regression tripwire for local LLM inference.

Tracks llama.cpp, ROCm, Mesa and kernel together, so that when throughput moves the
ledger can say **which of them moved it**. Every measurement carries a complete
fingerprint of the environment that produced it.

## Why

Upstream is not monotonically improving, and the stack changes without being asked:

- llama.cpp b10438 measured **22.5% slower** at deep prefill than the pinned b10082.
- Only `q8_0` and `q4_0` KV cache types have optimized flash-attention kernels on this
  backend. `q5_1`, `q5_0`, `q4_1` and any mismatched K/V pair fall to a path **8–13×
  slower at prefill while decode looks completely normal** — invisible to short-prompt
  testing.
- `unattended-upgrades` installs graphics-stack changes on its own schedule.

See [FINDINGS.md](FINDINGS.md) for the measurements behind those claims.

## Usage

```
make row          # one live canary row
make canaries     # all five canary configs (~4 min)
make validate     # check the ledger against results/schema.json
make test         # validator acceptance cases
```

Requires `PERF_LAB_MODEL`, `PERF_LAB_BIN` and `PERF_LAB_GPU_UID`. The GPU is selected
by unique ID; if it is absent the run aborts rather than silently measuring a different
card and stamping the row with a fingerprint that lies.

### Current known-good config

Qwen3.8-27B Q6_K, 262,144 context, `q4_0` K and V, `--spec-type draft-mtp
--spec-draft-n-max 4`. 46.15 tok/s decode, 711 t/s prefill, 29.3 of 32 GB VRAM.

## Layout

```
results/ledger.jsonl   append-only, one row per measurement
results/schema.json    the contract; CI-validated
harness/               probe, bench, validate
configs/               tracked + canary configurations
```

## The ledger contract

A row is invalid unless it carries `ts, run_id, kind, fp, cfg, m, ok`. The fingerprint
`fp` must name the GPU by **PCI address and unique ID, never by index** — `rocm-smi`
GPU[1] and DRM card0 are the same device on this host, and the enumerations are
inverted. PCI strings are lowercase; `rocm-smi` emits uppercase hex, and unnormalized a
fingerprint fails to match itself.

Live rows must record a prefill metric, not just decode. A `tg`-only row cannot see the
kernel-fallback regression this repo exists to catch.

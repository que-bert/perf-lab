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

### Current known-good config

Qwen3.8-27B Q6_K, 262,144 context, `q4_0` K and V, `--spec-type draft-mtp
--spec-draft-n-max 4`. 46.15 tok/s decode, 711 t/s prefill, 29.3 of 32 GB VRAM.

## Layout

```
results/ledger.jsonl   append-only, one row per measurement
results/schema.json    the contract; CI-validated
harness/               probe, bench, check, alert, run_set, install
configs/canary.yaml    canary configurations and their bands
systemd/               timer units (installed into the user instance)
apt/99-perf-lab        upgrade hook; queues a run, never runs one
```

## Automation

```
harness/install.sh          # installs the timers; prints what still needs root
harness/install.sh --uninstall
```

Three triggers:

- **nightly** at 03:00 — the routine sweep.
- **drain** every 15 minutes — runs only if an apt upgrade queued one. Draining on the
  nightly instead would give the post-upgrade run the same before/after pair the
  nightly already has, which is no information at all.
- **heartbeat** at 09:00 — files an Issue for a sustained breach, a failed push, or no
  successful run in 72 hours. That last one matters most: a tripwire that quietly
  stopped running looks exactly like one reporting good news.

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

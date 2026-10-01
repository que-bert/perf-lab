# R9700 rev6.1 execution plan — sequential tg (decode) + pp (prefill) rewrites

Date: 2026-09-25. Status: **rev6.1 — revised after DA review**. Supersedes rev5's
execution model; rev5's objective and immutable config carry over. Executed by ONE
session (the "main") that **MUST** delegate each workstream to a sub-agent and
**MUST** run tg and pp strictly sequentially.

Changelog vs rev6 (from DA): fixed the GEMV ground truth; demoted the prefill-FA
"fast mode" (it was the 1-layer MTP graph, not a forcible path of the main
graph); deleted TG-1b (out-of-kernel split reduce already exists); merged
TG-1c/PP-2 into one int8-QK workstream; added nmse/KL gate; mandated
same-process interleaved A/B to exclude the clock artifact; committed the
existing wins first; relaxed the inline prohibition for small candidates.

---

## 0. IMMUTABLE serving config (any change = out of scope, escalate)

Qwen3.8-27B-**Q6_K**; ctx **262144**; `-ctk q8_0 -ctv q8_0` (**q8 KV, never
f16**); `-fa on -ngl 99`; `--parallel 1`; `--spec-type draft-mtp
--spec-draft-n-max 3`; `--mmproj mmproj-F16.gguf`; R9700 = Vulkan device index
**1**; temperature 0. Changing weight quant, KV quant, KV size, context, MTP, or
slots is FORBIDDEN. FA-internal compute precision (Q/P quantisation) is allowed
subject to C0.

GPU is serial: every GPU command under
`flock /tmp/opencode/r9700.lock -c '<cmd>'`; never two GPU jobs at once.

---

## 1. Objective (targets corrected to what the evidence supports)

| id | metric | current | target | gate |
|---|---|---:|---:|---|
| D1 | decode @ 70k | **46.68** | >=52 | screen |
| D0 | decode @ 176k | **41.42** | >=48 (needs TG-1 **and** TG-2 at acceptance, not kill) | ship |
| P1 | pp2048 @ d16384 | **794** | >=1000 (depends on PP-4 q6_K GEMM >=60 TF, not FA alone) | ship |
| P2 | 183k-prompt prefill | **~358** | >=500 (depends on eliminating the main-graph slow FA state) | ship |
| P0 | pp2048 @ d0 | **981** | stretch only | — |
| C0 | correctness | pass | no regression | hard |

**Honesty gate.** P0>=1800 is infeasible under Q6_K (~98 TF needed vs ~96 fp16
peak; best q6_K path 52 TF; even f16 caps ~1270). D0>=55 is the
perfect-implementation ceiling (all DRAM work at 640 GB/s: 35.7+10.6+0.8+7.7+10.7
= 65.5 ms -> 55.6 t/s) and is **not** a target. D0>=48 requires FA to reach f16
parity (~21.3 ms, giving ~45.3 alone) **and** the GEMV to gain.

---

## 2. Ground truth (measured; trust, do not re-derive)

**Step at 176k**, current build (`GGML_VK_PERF_LOGGER`, verify graph ~86 ms):
- **weight GEMV ~35.8 ms at ~553 GB/s (~86% of 640), per-step q6_K traffic
  19.76 GB** (per-node; excludes `token_embd`, which is GET_ROWS). Post-`rm_kq=4`.
  A dedicated investigation found no >5% further win (MMVQ correct but slower on
  the model). The pre-`rm_kq4` profile logging (`perf176.log`) shows ~48.75 ms
  including the extra shapes — cite the post-win number for the current build.
- FA **28.72 ms** (17 layers, nb=4, kv=183040, q8_0 K/V => 217 GB/s,
  dequant + mask(219 us) + per-tile-barrier bound). GDN 0.77; other 7.7. Three
  MTP drafts +10.7. Step ~88 ms -> 41.42 t/s (3.643 tok/step).

**Adopted wins in the tree** (uncommitted; baseline commit `a702ea697`):
packed-GQA **mask-opt** (+3.4% @176k) and q6_K **`rm_kq=4`** (+3.8% @176k). Kill
switches `GGML_VK_FA_NO_DECODE_V2=1`, `GGML_VK_RM_KQ_Q6K=2`. **Commit these
first (explicit paths) before any sub-agent branches off.**

**FA decode wall (falsified — do not repeat as-is):** Br 32/48/64 worse; in-kernel
split_k bottoms at 1754 us/layer; **out-of-kernel split reduce already exists**
(`pipeline_flash_attn_split_k_reduce`, `ggml-vulkan.cpp:3210`, dispatched
`:8429-8456`); int8-QK no gain at mask=1 (mask path ~2x under int8); int8-PV
fails correctness (V scale runs along PV output dim); q6_K MMVQ slower on model;
`CM1_SHMEM` -3%. FA is **dequant-bound at q8_0** (217 GB/s) while **f16 is
bandwidth-bound** (1251 us, 600 GB/s) — the target is to make q8 behave like f16.

**Prefill:** at depth ~80-90% `FLASH_ATTN_EXT`; short context GEMM-bound (q6_K
~52 TF vs f16 70, q4_0 81). **Corrected FA-mode finding:** the nb=512 q8_0 FA
signature appears in two graphs — the **16-layer main prefill graph (17.9-33.4
TFLOPS, mean 28.5, toggling ~1.9x)** and the **1-layer MTP graph (21.3-44.0)**.
The 43.7 TFLOPS figure is **the MTP graph and is NOT available to the main
graph**; do not chase it. The real, smaller lead is the main graph's own 17.9-33.4
toggling. `use_dequant_kv` (`ggml-vulkan.cpp:8077`, `neq1>=64`) dequantises q8_0
K/V into an f16 scratch for prefill. DP4A GEMM is dead (22.7 vs 52.5 TF).

**Measurement hazards:** bimodal per-process clock (~20-25%); never `llama-bench`
for decode; `depth.py` spread <1%; `llama-bench -dev Vulkan1` mandatory; perf
logger inflates absolute times (ratios only).

---

## 3. Execution model (MANDATORY, sequential, sub-agent-driven)

1. **Commit the two adopted wins first** (explicit paths, fork `r9700-qwen`).
2. **The main MUST delegate each workstream's implementation+GPU work to a
   sub-agent and MUST NOT run GPU experiments inline for the two rewrite
   workstreams (TG-1, PP-1/PP-2).** For small, bounded candidates (TG-4 n_max,
   PP-3 gate, PP-4 env sweep — each <1 day), the main MAY work inline.
3. **Strict order: WS-TG then WS-PP.** Never concurrent (GPU serial; both edit
   `ggml-vulkan.cpp`). At most one sub-agent live. Fresh context per workstream.
4. **Time-box WS-TG** (e.g. one sub-agent run for TG-1); if TG-1 has not landed
   an accepted change, record the negative and proceed to WS-PP — do not spend
   unbounded budget on the more-falsified area.
5. **The main MUST verify before adopting:** read the raw logs and the diff,
   re-run the acceptance command itself; a sub-agent summary is a claim.
6. Sub-agents **never commit / never `git add`**. The main commits accepted
   changes (explicit paths) after verification.
7. Each brief MUST contain: immutable config, the relevant §2 ground truth, files
   in scope, candidate specs, numeric acceptance + kill, verification commands,
   report format. Opus for kernel work; never Sonnet.

**Cost bound:** at most two sub-agent workstreams plus bounded inline work; stop
at the kill criteria. Record every negative in `FINDINGS.md`.

---

## 4. WS-TG — decode

### TG-1 (primary) — decode FA rewrite for q8_0 K/V, nb<=8
Files: `flash_attn_cm1.comp`, `flash_attn_base.glsl`, FA dispatch in
`ggml-vulkan.cpp`. Directions, in order, each with its own env kill switch:
- **TG-1a barrier/mask restructure** (this is the target that can reach f16
  parity): subgroup-scope reductions in place of per-tile workgroup barriers;
  software-pipelined `kvsh`; reduce the mask=1 overhead (~219 us). Premise is a
  *hypothesis* to test (the kernel is dequant+mask+barrier bound; the "issue-bound"
  framing is unmeasured). **Acceptance: op <=1250 us at nb=4/183296, FA 138/138,
  70k >=50 t/s. Kill: cannot beat 1450 us.**
- **TG-1b int8-QK done right** (shared with PP-2, see §5): int8 K-as-A at decode,
  with the mask handling restaged so the int8 branch does not double its cost
  (mask applied outside the int8 path, or row-major int8 staging). The prior
  attempt stalled because mask=1 is never selected cheaply. **Acceptance: op
  <=1518 us at mask=1 (the mask=0 int8 number) with correctness. Kill: no gain.**
  *(TG-1b is one code path serving both workstreams; at prefill it is PP-2.)*

### TG-2 — GEMV (low expected value, keep low)
At ~86% of peak with a dedicated investigation finding no >5% win. Only structural
fusion remains (fuse qkv+gate; persistent GEMV). **Acceptance:** verify GEMV
<=34 ms and 70k >=47.5. **Kill:** <3%.

### TG-3 — backend CPU replay + small-op tail (not yet attempted)
First **measure** non-GPU time/step with a temporary `steady_clock` (removed
before commit). The profile says GPU-bound (~88 ms GPU vs step); if measured CPU
<5 ms, TG-3 is dead. If >=8 ms: cache/replay recorded command buffers on
unchanged topology; fuse the ~7.7 ms small-op tail. **Acceptance:** >=30% CPU cut,
bit-identical, C0. **Kill:** measured CPU <5 ms.

### TG-4 — speculation re-test (cheap, inline)
Re-test `--spec-draft-n-max 4` **with mask-opt active** (the prior rejection
predates it). **Acceptance:** tokens/step >=4.2 with accept >=0.85 @176k. **Kill:**
any regression.

---

## 5. WS-PP — prefill (q8 KV)

### PP-1 (primary) — attribute and, if possible, hold the main-graph fast state
**Reframed:** the main 16-layer prefill FA toggles **17.9 <-> 33.4 TFLOPS** at
fixed shape within one process; the 44 TFLOPS belongs to the 1-layer MTP graph and
is off-limits. Determine whether the main graph's slow state is a dispatch
difference, L2/power state, or the bimodal clock. Method: **same-binary,
same-process interleaved A/B** (toggle env between reps); log
`fa_pipeline_state` and `use_dequant_kv` for main vs MTP; sample
`rocm-smi --showclockinfo` during a live prefill; add nb=512 q8_0 perf cases at
kv=32768/49152. **Acceptance:** the main graph holds its 33 TFLOPS state
consistently; P2>=500. **Kill:** the state tracks clock only (then PP-1 dies and
P2>=500 has no support).

### PP-2 — q8-direct int8-coopmat prefill FA (shared code path with TG-1b)
Remove the `use_dequant_kv` f16 scratch (`ggml-vulkan.cpp:8077`) and consume q8_0
K/V directly via int8 coopmat at n=512. **Justify by MMA occupancy at n=512**, not
dequant bandwidth: deep-prefill FA is compute-bound (16.5 TF), and the scratch
traffic is only ~3% — so the case rests on int8 MMA throughput at large n (where
the decode nb=4 failure does not apply). **Acceptance:** prefill FA >=40 TFLOPS
and pp2048 d16384 >=950; C0. **Kill:** no gain vs f16 scratch.

### PP-3 — prefill-only ubatch decoupling (best-supported non-FA prefill lever)
Allocate the large compute buffer only for prefill graphs (threshold `neq1>=64`,
mirroring `use_dequant_kv`). **Acceptance:** P0 +7-16% with zero decode change.
**Kill:** allocator churn / any decode regression.

### PP-4 — cm1-int GEMM tuning
`BK_STEP` is `#define BK_STEP 4` in `vulkan-shaders/mul_mmq_cm1.comp:91/93` **and**
hard-coded `4u` at `ggml-vulkan.cpp:1601` — both need the knob. Expose
`GGML_VK_MMQ_CM1_BK_STEP`, sweep per the upstream `0cc4m/vulkan-mmq-bk-step-tuning`
branch. **Acceptance:** q6_K prefill GEMM >=60 TFLOPS (this gates P1>=1000).
**Kill:** <3%.

### PP-5 — minor
Prefill `Br=32` sweep (untested at prefill); skip the f16 scratch as a cheap A/B.
Each <=5%. *(Dropped: "packed-aware mask_opt for prefill" — prefill already uses
`use_mask_opt`, `ggml-vulkan.cpp:8194-8197`; the packed variant is gated on
`n_pos<=8`, decode-only.)*

---

## 6. Acceptance / kill discipline
Every candidate has a numeric acceptance and kill (above). A candidate that cannot
pass acceptance is recorded as a negative with the measured reason, then the next
starts. Nothing is "kept as a scaffold" unless default-off and measured at +0%
overhead (the current int8-QK scaffold costs +0.3% — either fix the declared LDS
or revert its hunks at the end of WS-TG).

---

## 7. Verification protocol (the main runs it per adopted change)
1. `test-backend-ops test -b Vulkan1 -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=4'`
   (**-b Vulkan1 mandatory**; baseline 138/138), `-o GATED_DELTA_NET` (37/37),
   `-o MUL_MAT` (1128/1128).
2. **Model-quality gate (re-added):** an nmse/KL check at the model shapes vs a
   reference (the int8-PV failure was caught by nmse, not the suite); thresholds
   fixed before the run.
3. `harness/op_perf.sh` isolated op, `-r 3`, spread <1%.
4. **A/B protocol must exclude the clock artifact:** same binary, same process,
   interleave the env kill switch between reps (or alternate run order) and
   require the effect to exceed the between-mode spread; sample clocks
   alongside. Separate off/on runs are NOT sufficient.
5. Decode: `harness/depth.sh <build/bin> <label> <port>` 70k (DEPTHS=1600,
   REPS=3) and 176k (DEPTHS=4000, REPS>=2), `PERFLAB_CTV=q8_0 PERFLAB_CTK=q8_0
   PERFLAB_GPU=1`; accept within 1% of 0.7629/0.881.
6. Prefill: `llama-bench -dev Vulkan1 -p 2048 -d 0,16384 -r 3` with a
   bimodality guard (interleave/alternate; sample clocks).
7. Raw logs saved; one `FINDINGS.md` entry per claimed win; main reads the logs.

---

## 8. Forbidden
Weight/KV precision or size change; context/n_max gate change; parallel GPU jobs;
sub-agent commits; perf-logger absolute times as a gate; `llama-bench` for decode;
any unverified change not behind a kill switch.

---

## 9. Risks
- **Clock bimodality / contention** can fake a win -> same-process interleaved A/B
  is mandatory (S2).
- **Shared-file conflict**: TG/PP both edit `ggml-vulkan.cpp`; sequential + small
  hunk-scoped diffs.
- **Kernel rewrite drift**: numerical errors surface as C0/nmse failure.
- **Budget**: two workstreams, one sub-agent at a time, explicit kill criteria.

## 10. First actions
1. Read this plan + rev5 + `FINDINGS.md` 2026-09-25 entries.
2. **Commit the two adopted wins** on explicit paths (fork `r9700-qwen`).
3. Confirm the tree (`git -C ~/git/llama.cpp-r9700 status`; baseline `a702ea697`).
4. Brief and launch the **WS-TG** sub-agent (TG-1a first). Do not start PP until
   TG is adopted or killed.

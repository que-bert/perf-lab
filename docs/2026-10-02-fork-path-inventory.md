# Fork-path inventory: qwen3.8-r9700-next (r9700-integrate16 @ 02ea7fe9f vs ce8caa6e6)

Scope: ~79 non-merge commits; 55 files touched outside `models/`. Tested envelope = Qwen3.8-27B Q6_K / Q4_K_M, arch qwen35, R9700 (RDNA4, RADV). Method: `git grep getenv` at both revisions (244 names at base, 337 at fork, 93 added), `git diff` read for every hunk in `ggml-vulkan.cpp`, `src/`, `common/`, `tools/`. No code run.

Notation: `V` = `ggml/src/ggml-vulkan/ggml-vulkan.cpp`. "RDNA4" = `device->architecture == AMD_RDNA4` (section 3). Line numbers are at r9700-integrate16 HEAD. All `NO_*` switches are tested with `getenv(..) != nullptr`, so any value (even `0`) disables; `LLAMA_NO_*` use `atoi != 0`.

## 1. Table A: env switches

### A1. Kill switches (default ON = fork path active; set to disable). 30 `GGML_VK_NO_*` + 9 other kill-style + 6 value overrides

| switch | disables | engagement guard (when fork path runs) | file:line |
|---|---|---|---|
| GGML_VK_NO_DECODE_Q8 | decode FA modules q8/q8r/q8w (`flash_attn_decode_q8*.comp`) | RDNA4 + `subgroup_size_control` + subgroup range includes 64; q f32, K=V=Q8_0, dst f32; **HSK=HSV=256 hard-coded (D)**; N=q->ne[1] in 1..8 (q8r needs coopmat 16x16x16 f32acc + sg32; non-q8r: N<=8 only if dot2_f16 else 1); ne[3]==1; G=q.ne2/k.ne2 in 1..32 (q8r) / 1..24; no max_bias, no softcap; mask F16, ne2/ne3<=1; 16B/8B stride alignment; split-K buf <= maxStorageBufferRange. Sinks ARE handled. | V:8491-8545, call V:8650 |
| GGML_VK_FA_NO_DECODE_V2 | packed-GQA decode mask_opt (Br = pos_per_tile) | `packed_gqa` && k_type_eff=v_type_eff=Q8_0 (i.e. no f16 dequant scratch) && RDNA4 && n_pos<=8 | V:8966-8971 |
| GGML_VK_FA_NO_PACK | (position, head) row packing in cm1 FA (`PACKED_GQA` spec flag 32, `flash_attn_cm1.comp`) | gqa_ratio>1 && !use_sparse && path==FA_COOPMAT1 && block_rows/gqa_ratio>1. **No arch/quant gate.** | V:8919-8923; shader `flash_attn_base.glsl:30` |
| GGML_VK_NO_MASK_OPT_CACHE | reuse of mask_opt bitmask across FA nodes in one graph | use_mask_opt && mask_opt_size <= 1 MiB; key = graph_seq+buffer+offset+nem+Br/Bc. No arch gate. | V:9073-9085 |
| GGML_VK_NO_FA_PREFILL_RDNA4 (umbrella; sub: `_VT`, `_RS`, `_V64`) | RDNA4 prefill FA (`flash_attn_prefill_rdna4{,_vt,_rs,_v64}.comp`, `fa_dequant_q8_0_rdna4`) | RDNA4 + `use_dequant_kv` (K,V quantised, neq1>=64, dense KV, nelements*2 <= maxStorageBufferRange, !coopmat2) + K=V=Q8_0 + HSK=HSV=256 + neq1>=64 + G=neq2/nek2 in 1..6 + F16 mask w/ nem2==1 + !sinks + no bias/softcap + 16B q strides + dst contiguous. Variant chain v64>rs>vt>base: vt needs G even; rs needs vt; v64 needs rs + K/V buffer offset%16 + nbk1/nbv1%16. Pipelines built only if coopmat_support && subgroup_require_full_support && RDNA4. | pipelines V:3372-3411; dispatch V:8737-8861 |
| GGML_VK_NO_FOV, GGML_VK_NO_FV2 | v64 FA lazy-O-rescale / base-2 softmax spec constants | inside v64 only | V:3397, 3399 |
| GGML_VK_NO_MMQ_Q6K_RDNA4 (umbrella; also `GGML_VK_MMQ_Q6K_RDNA4=0`) | all RDNA4 Q6_K f16-WMMA GEMM pipelines incl. fused SWIGLU | creation: RDNA4 && coopmat_support. Use: src0 Q6_K, src1/dst F32, ne01%128==0, ne10%256==0, ne11>=64, ne02=ne03=ne12=ne13=1, src0 dim0/1 contiguous, !uma(f32y only) | creation V:2564-2640; use V:6600-6700, 6850-6930 |
| GGML_VK_NO_Q6K_LEANQ / _UPAR / _N256 / _SPLITK / _SWIGLU / _SWIGLU_N256 | lean dequant; UPAR (K-unroll-4) modules; 128x256 tile; split-K; fused gate+up SWIGLU GEMM; its N256 variant | UPAR: (ne10/256)%2==0 (V:6912). N256: !q8, ne11>=256, (ne01/128)*ceil(ne11/256)>=128 (V:6681). SPLITK: dst->nb[1]==ne01*4, tiles*split<192 heuristic, split<=8 (V:6683-6699). SWIGLU N256: ne11%256==0 (V:6877) | V:2592-2631 |
| GGML_VK_NO_SOP_GLU | two-binding SWIGLU pipeline (gate/up in different device buffers) + graph_optimize adjacency keep | graph pattern MUL_MAT,MUL_MAT,GLU(swiglu, src1!=null, swapped==0) with both weights Q6_K, same shape, contiguous, x F32 contiguous, 2D, `wg->ne[1]%128==0`, `wg->ne[0]%512==0`, `x->ne[1]>=128 && %128==0`, no views, offsets%4, span <= maxStorageBufferRange | fuse V:7632-7667; creation V:2615-2626; keep V:16174-16183 |
| GGML_VK_NO_Q8_GEMM_TUNE | RDNA4 Q8_0 prefill f16-WMMA GEMM (+split-K) (`mul_mm_q8_rdna4_f16.comp`) | RDNA4 && coopmat_support; src0 Q8_0, src1/dst F32, ne01%128==0, ne10%256==0, ne11>=64, 2D, contiguous | creation V:2552; use V:6620-6623 |
| GGML_VK_NO_MQ5_TUNE | n>=5 Q6_K MMVQ tuning: 2 rows/WG, dword-exact `q6_fast_outputs` shader | RDNA4; Q6_K, !MUL_MAT_ID (shader `#if DATA_A_Q6_K && !MUL_MAT_ID`); only reached when MMVQ selected (n>=5 on RDNA4, see GGML_VK_Q6K_MMVQ_MIN_N) | V:3067-3084, 5864; shader `mul_mat_vecq.comp:127` |
| GGML_VK_NO_DMMV_F32_WIDE | 4x wider DMMV workgroup for small f32 weights | RDNA4 && a_type F32 && m<=256 && k>=1024. **No model check** (hits any small f32 matvec, e.g. MoE router, ssm_alpha/beta) | V:5854-5858 |
| GGML_VK_NO_GR_SMALL | `get_rows_f32_small` one-thread-per-element gather | src0 F32 && dst F32 && src0->ne[0]==1 && src1 I32. No arch gate. | V:10503-10507 |
| GGML_VK_NO_GDN_CACHE | GDN recurrent-cache fusion: GET_ROWS gather + GATED_DELTA_NET + state-snapshot CPY, and CONCAT conv-state fusion (`gdn_conv_state.comp`, `gated_delta_net_cache_*`) | `!device->disable_fusion`, graph contains GATED_DELTA_NET. GDN branch: K=op_param0>1, n_seqs==1, exactly 1 snapshot CPY, src[5] is single-row F32 GET_ROWS (I32 ids, nelements==1), S_v in {16,32,64,128}, cache row stride == S_v*S_v*H*4, snapshot views at exact offset, nothing else reads the snapshot region, cache untouched in range. CONCAT branch: dim 0, F32, d=src0->ne0 in 1..8, s1->ne0<=64, 1..8 snapshot CPYs (4 w/ LLAMA_NO_RS_INDEX). **No vendor/arch gate.** | prepass V:11137-11330+; build V:13303-13306, 13394-13434, 13510-13516; dispatch V:15690-15705 |
| GGML_VK_NO_GDN_GATE | folds g=MUL(SOFTPLUS(ADD(alpha,dt)),a), beta=SIGMOID into GDN | GDN node; g,beta contiguous [1,H,T]; chain F32, single-use, non-KDA (`gt->ne[0]==1`). Only effective inside the cache branch above; `find_gate` is also called from graph_optimize alloc-dep (V:16126) regardless | V:11083-11130 |
| GGML_VK_NO_VSUB | cheap prepass (hash-set use counts, indexed CPY/GET_ROWS/views) vs original full-graph maps | implementation switch for the prepass | V:11153 |
| GGML_VK_NO_RMS_NORM_FAST | `rms_norm_*_fast` shaders (shuffle tail, same reduction tree) | `subgroup_shuffle && subgroup_basic` (any GPU); at dispatch ne0 <= 16*512 | creation V:3483; use V:9562-9575 |
| GGML_VK_NO_RMS_NORM_GATE | fusion RMS_NORM->MUL->UNARY(SILU)->MUL (qwen35 gated norm) | fast pipelines exist; F32 everywhere; z same shape & contiguous & aligned; ne0<=8192; subgraph fusible w/ output only at node+3 | V:3490; fuse ~V:14898-14925; dispatch V:15785-15791 |
| GGML_VK_NO_RMS_NORM_SCALE | fusion RMS_NORM->SCALE (GDN q/k l2 norm) | F32, same shape, scale contiguous, SCALE bias == 0 | V:14930-14940; dispatch V:15777-15784 |
| GGML_VK_NO_SOP_ADDRMS | fusion ADD->RMS_NORM->MUL (nrows>1, writes sum + normed) | resadd pipeline exists (fast rms path); nrows>1; ne0<=8192; all of add srcs/add/weight/mul F32 contiguous+aligned; weight is one row; same shapes; ADD result may be a graph output. **Fires on any pre-norm transformer prefill.** | V:3496; can_fuse V:14943-14966; keep V:16161-16172 |
| GGML_VK_NO_KEEP_RMS_SCALE, GGML_VK_NO_KEEP_UNARY_MUL | graph_optimize adjacency for RMS_NORM->SCALE and UNARY->MUL | pure op pattern in graph; reorders graph for every model | V:16144-16160, 16347 |
| GGML_VK_NO_MMV_ADD_BATCH | MUL_MAT(+ADD bias) fusion for ne11 in 2..mul_mat_vec_max_cols | MUL_MAT, 2D, A quantised/F16/F32/BF16 contiguous non-permuted, <= maxStorageBufferRange, no FWHT; bias same shape/stride/type, aligned | V:14768-14790 |
| GGML_VK_DISABLE_FA_SHMEM_STAGING | shmem K/V staging in the scalar FA path | `vendor==AMD && arch != GCN` (RDNA1-4), any head size, any quant (NVIDIA: hsk,hsv<256) | V:1287-1290 |
| LLAMA_NO_MFL | ubatch embd as view into batch (no copy) + MTP keeps only last hidden row for prompt batches | ubatch.embd path: batch.embd && contiguous idxs. MTP: `n_rows_all > 64` | src/llama-batch.cpp:758; common/speculative.cpp:1790-1792 |
| LLAMA_NO_MTP_KVONLY | K/V-only MTP draft graph for prompt pass | qwen35 graph_mtp: n_outputs==0 && ubatch.token && !wqkv && wk && wv && f_clamp_kqv<=0 && !wk_b && !wv_b | src/models/qwen35.cpp:562-564 |
| LLAMA_NO_KQ_MASK_CACHE | incremental KQ mask (cells dirty-range) | causal; swa_type NONE; !alibi; mask F16/F32; single-seq ubatch; 2D mask; cells not shared; **any model** | src/llama-kv-cache.cpp:1856-1895, call ~1940 |
| LLAMA_NO_RS_INDEX | K cap 8 -> 4 for conv-state snapshot fusion | in GDN CONCAT branch | V:11209-11210 |
| LLAMA_KQ_MASK_THREADS | threaded mask fill (default 8 threads when fill >= 16 MiB; `=1` serial) | cached-mask fill | src/llama-kv-cache.cpp:1831-1836 |
| GGML_VK_HOST_BUFT_DEV0 | per-device pinned host buffer type (P3) -> old device-0 pin | multi-device setups | V:14268 |
| GGML_VK_PREALLOC_X_STEP_MIB | 64 MiB-step growth of prealloc_x (FA dequant scratch); `=0` exact size | any prealloc_x growth | V:13259 |
| GGML_VK_Q6K_MMVQ_MIN_N | Q6_K MMVQ at n>=N on RDNA4 (default 5; `0` disables) | RDNA4 && n>=N; plus `GGML_VK_Q6K_MMVQ` (-1 auto, 1 force, 0 off) | V:6957-6975 |
| GGML_VK_RM_KQ_Q6K | RDNA4 4 rows/WG for Q6_K GEMV at cols>=4 (set `=2` to restore) | RDNA4 && rm_kq==2 | V:3054-3055 |

(`GGML_VK_NO_*` count: DECODE_Q8, FA_PREFILL_RDNA4 x4 (umbrella+VT/RS/V64), FOV, FV2, MMQ_Q6K_RDNA4, Q6K_LEANQ/UPAR/N256/SPLITK/SWIGLU/SWIGLU_N256 (6), SOP_GLU, Q8_GEMM_TUNE, MQ5_TUNE, DMMV_F32_WIDE, GR_SMALL, GDN_CACHE, GDN_GATE, VSUB, RMS_NORM_FAST/GATE/SCALE, SOP_ADDRMS, KEEP_RMS_SCALE, KEEP_UNARY_MUL, MMV_ADD_BATCH, MASK_OPT_CACHE = **30**. Other kill-style: `GGML_VK_FA_NO_DECODE_V2`, `_V3`, `_NO_PACK`, `_NO_MASK_OPT`, `GGML_VK_DISABLE_FA_SHMEM_STAGING`, `LLAMA_NO_MFL`, `_NO_MTP_KVONLY`, `_NO_KQ_MASK_CACHE`, `_NO_RS_INDEX`.) 

Not in the all-off string, on purpose: `GGML_VK_FA_NO_MASK_OPT` (disables upstream's own mask_opt too); `GGML_VK_FA_NO_DECODE_V3` (only gates the default-off INT8_QK path).

### A2. Default-OFF opt-ins and tuning knobs (fork path inactive unless set)

| switch | default | effect / guard | file:line |
|---|---|---|---|
| GGML_VK_FA_INT8_QK | off | int8-QK decode FA (cm1): RDNA4, coopmat_int 16x16x16, K=V=Q8_0, hsk%32==0, n_rows<=8, Br<=16; measured slower | V:1347-1358 |
| GGML_VK_DECODE_Q8V | off | q8v variant (16B K loads) when q8r active | V:8561-8564 |
| GGML_VK_DECODE_Q8W | auto (on if rows<=16) | `1` force all rows, `0` disable q8w | V:8569-8572 |
| GGML_VK_DECODE_Q8R / _Q8R_NMIN / _Q8_NMAX / _Q8_WGS / _Q8_DEBUG | q8r on, NMIN=1 | q8r on/off; min N; N cap for non-q8r; WG count; debug print | V:8493-8578 |
| GGML_VK_FA_CM1_BR / _NS / _SHMEM | 16 / 4 / off | cm1 FA tile sweep; shmem staging for cm1 (hsk,hsv<=256) | V:1327-1366 |
| GGML_VK_FA_DEQUANT_MAX_KV | 0 (unlimited) | above N KV tokens skip f16 dequant scratch | V:8721-8725 |
| GGML_VK_FA_PREFILL_DIAG, GGML_VK_FPQ_NS, GGML_VK_FA_SPLIT_K | 0 / 1 / 0 | diag ablation (wrong results), S^T chain split (null), KV split override | V:3379, 3394, 9047 |
| GGML_VK_MMQ_Q6K_RDNA4 | `f16` | mode string: `int8` (known PPL=inf per commit 8f2c55275), `f32y`, `f16db`, `0` off | V:2565-2589 |
| GGML_VK_MMQ_Q6K_ORDER / _WAVE / _LOG, GGML_VK_Q6K_DIAG / _PIPE2 / _SPLITK / _LEANQ_N256 | order=1, wave default | tile order, wave 32/64 (int8 w64 wrong), shape log, ablation (wrong results), 2-phase pipeline (slower), split override, lean N256 | V:2567-2606, 6684, 6873, 6886 |
| GGML_VK_RM_KQ / _RM_STDQ / _RM_KQ_INT_Q6K / _RM_MQ5 / _MQ5_SG / _MQ5_FAST / _MQ5_UNR / _MQ5_WG | tuned defaults | rows-per-WG sweeps (bench only) | V:3043-3084, 5864 |
| GGML_VK_HOST_GET_ROWS | 0 | GET_ROWS reads token_embd from pinned host buffer; also forces sync before set_inputs, supports_op rejection for host-resident srcs | V:5967; src/llama-context.cpp:1504; V:16790 |
| GGML_VK_INPUT_STAGING, GGML_SCHED_ASYNC_INPUT_COPY | 0 | async input upload paths (measured slower) | V:14352; ggml-backend.cpp:1694; llama-context.cpp:1501 |
| LLAMA_MTP_PIPE | 0 | deferred MTP prompt pass overlapping next target ubatch; needs n_seq==1, !shared mem, no chain_heads, n_parallel==1 (server) | common/speculative.cpp:1556; tools/server/server-context.cpp:2912 |
| LLAMA_MTP_SKIP_PROMPT | 0 | diagnostic: leaves MTP KV empty | speculative.cpp:1665 |
| MTMD_LAZY_GPU | 0 | mmproj released from GPU between encodes | tools/mtmd/clip.cpp:266 |
| LLAMA_ARG_SPEC_DRAFT_VOCAB / _FILE / _ADAPTIVE / _ADAPTIVE_WINDOW / _ADAPTIVE_MAX_OUT (+ `--spec-draft-vocab*` CLI) | 0 / unset | reduced-vocab MTP draft head; llama_model_set_draft_vocab rejects non-qwen35, tied embeddings, own nextn head, host-resident output | common/arg.cpp:4135-4178; src/llama-model.cpp:2848+ |
| diagnostics only | off | GGML_VK_STEP_TIMING(_N), GGML_VK_DUMP_NODES, GGML_VK_MMQ_Q6K_LOG, GGML_SCHED_INPUT_TIMING, LLAMA_DECODE_TIMING(_N), LLAMA_DECODE_TRACE, LLAMA_INPUT_TIMING, LLAMA_SERVER_STEP_TIMING(_N), LLAMA_KQ_MASK_VERIFY | various |

Total fork-added getenv names: 93 (+5 `LLAMA_ARG_*` via `set_env`). Kill switches: 30 `GGML_VK_NO_*` + 9 other = 39.

## 2. Table B: always-on fork changes with no env switch

| path | what changed vs upstream | engagement guard | file:line | risk on other models |
|---|---|---|---|---|
| RDNA4 arch split | new `AMD_RDNA4` enum; RDNA4 used to classify as RDNA3. Pre-existing RDNA3 checks were widened to `RDNA3\|\|RDNA4` (rm tuning, khr-coopmat AMD-proprietary check); `DEVICE_ARCH` spec const passed into cm1 MMQ shaders | see section 3 | V:56-58, 1948, 2492, 3046-3048, 17730; types.h:391; `mul_mmq_cm1.comp:86-132` | low (behaviour-preserving for RDNA3 by construction; verify no other RDNA3-keyed upstream path was missed: grep shows none) |
| **int8 coopmat1 MMQ (#27952 port)**, MUL_MAT and MUL_MAT_ID | cm1 int8 tile configs, `ggml_vk_matmul_cm1_int_shmem_support`, `matmul_*_q8_1_cm1` and `matmul_id_subgroup_*_q8_1_cm1` pipelines; `quantize_y` now true if `coopmat_int_support`; `y_non_contig` only forced when !quantize_y | `coopmat_int_support` (Sint8 subgroup coopmat) && arch RDNA3\|\|RDNA4. RDNA4 builds Q4_0, Q5_0, Q8_0, IQ4_NL, IQ4_XS, MXFP4, Q3_K, Q6_K (skips Q4_1, Q5_1, Q4_K, Q5_K, NVFP4 -> f16 fallback). Wave32 tiles when AMD + subgroup_size_control + range includes 32 | V:1795-1800, 1858-1900, 1947-2022, 2645-2660, 2718-2731, 6597-6645 | **high**: guarded by hardware+quant only; also covers MoE MUL_MAT_ID. Tested only Q6_K (and Q4_K via the f16 fallback). It is also the fallback when the Q6_K/Q8_0 RDNA4 GEMM envs are off. |
| cm1 FA shader rewrite | `flash_attn_cm1.comp` +310 lines (packed GQA, int8 QK variants, pv shmem layout); `pvsh` shmem budget now `Br*osh_stride` (was `MatBc*`), spec-state bits 32/64 | every coopmat1 FA pipeline; only the packed/int8 variants are switchable | V:8462-8486, 1446-1458; shader | med: shmem-support formula feeds the cm1-vs-scalar fallback decision on all cm1 devices |
| RMS_NORM small/128-thread variants | `rms_norm_small_f32`, `_mul_small_f32`, `_scale_*`; selected for ne0<=128 | any RMS_NORM, any device (remains when NO_RMS_NORM_FAST set) | V:3478-3482, 9574-9580 | low-med (generic op; ne0<=128 = per-head q/k norms in many models) |
| CONCAT transposed i32 (`concat_t.comp`) | tiled transposed CONCAT | unit size 4, dst not quantised, dim0, dst contiguous, `src1->nb[0]!=type_size && src1->nb[1]==type_size` | V:9409-9415, 10358-10361 | low (shape/stride-keyed, any arch) |
| Small-BAR host-visible vidmem off | skip DEVICE_LOCAL\|HOST_VISIBLE preference | max host-visible local heap > 0 && < half of max local heap, any vendor. Upstream `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` forces the same, cannot restore old behaviour | V:4265-4290 | low (non-ReBAR boxes only; reviewer note: differs from upstream memory-type order) |
| GDN alloc deps in graph_optimize | `add_alloc_dep` keeps gather ids and raw alpha/beta alive until the GDN/CONCAT node | `params->add_alloc_dep` && node is GATED_DELTA_NET or **any CONCAT whose src0 (through RESHAPE/VIEW) is a GET_ROWS**; active even if NO_GDN_CACHE | V:16111-16140 | low-med: CONCAT(GET_ROWS) pattern also occurs in other recurrent/SSM archs; only lengthens tensor lifetimes |
| pinned-host tensors skipped in sync analysis | `ggml_vk_tensor_in_host_buffer` early-outs in need_sync overlap logic | tensor buffer type is the Vulkan host buffer type | V:13349-13359 | low |
| GDN cache hazard tracking | `cur_gdn_cache_fuse->cache` added to unsynced read/write lists | only when fusion chosen (follows NO_GDN_CACHE) | V:13397-13434 | low |
| split-K prealloc sizing | `split_k_size = max(split_k*d_sz, q6k_split*d_sz)`; prealloc condition `split_k_size > 0` | any ggml_vk_mul_mat_q_f16 | V:6724-6760 | low |
| timestamp query pool 2x nodes + sub-timestamps | perf logger only | `GGML_VK_PERF_LOGGER` | V:15503-15560 | low |
| kv-cells dirty tracking | `dirty_lo/hi`, `uid`, `seq_pos_range` in `llama_kv_cells` | every unified KV cache, every model (maintained even if mask cache is off) | src/llama-kv-cells.h (+56), llama-kv-cache.cpp | low-med: shared KV infra; read only by the mask cache |
| server spec_prompt buffer reuse | `get_text_tokens(out)` fills a reused vector | MTP/spec path in server | tools/server/server-context.cpp ~3088; server-common.cpp:647 | low |
| `common_dft_flush` hooks | calls added in seq_rm/seq_cp/checkpoint ops | no-op unless a flush fn is registered (LLAMA_MTP_PIPE) | common/common.cpp:1647-1690, 2335-2392 | low |
| graph-reuse key | `cparams.draft_vocab_full` added to `llm_graph_params::allow_reuse` | trivial | src/llama-graph.h:874 | low |
| decode-kld tool, test-backend-ops cases | new tool / tests, no runtime effect | n/a | tools/decode-kld, tests/test-backend-ops.cpp | none |

## 3. RDNA4 detection

- `get_device_architecture()` V:10-140. Early `OTHER` unless the device exposes `VK_AMD_shader_core_properties`, `VK_KHR_shader_integer_dot_product` and `VK_EXT_subgroup_size_control` (V:34-36). Then: min/max subgroup 64/64 -> GCN; **min 32/max 64 -> RDNA family** (V:46-53): `wavefrontsPerSimd==20` -> RDNA1; **`shader_float8 (VK_EXT_shader_float8 present) || integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated` -> AMD_RDNA4** (V:56-58); mixed-signedness packed int dot -> RDNA3; else RDNA2. It is a feature-proxy: no device id, no CU count, no driver check. Any future AMD part that advertises float8 or the saturating mixed dot lands in RDNA4. Assigned at V:4257 (`device->architecture`), name logged V:136-146.
- Additional device-level facts the RDNA4 paths use besides the enum: `coopmat_support` (+ `coopmat_support_16x16x16_f32acc`, `coopmat_int_support` Sint8 16x16x16), `subgroup_require_full_support`, `subgroup_size_control` with range covering 32 and/or 64, `dot2_f16`, `shader_core_count` (q8r WG count = 2x CUs, V:8587; R9700 = 64 CUs).
- Paths hanging off `arch == RDNA4`: decode FA q8/q8r/q8w/q8v (V:8502), prefill FA rdna4 family (V:3372, 8747), int8-QK FA (V:1352), packed-decode mask_opt (V:8969), Q6_K f16 GEMM family + SWIGLU (V:2564), Q8_0 GEMM (V:2552), int8 cm1 MMQ type list (`!rdna4` exclusions V:2647-2731), Q6_K MMVQ n>=5 + MQ5 tuning (V:6970, 3075-3084, 5865), q6_K GEMV 4 rows (V:3054, 3062), DMMV f32 wide (V:5855), `DEVICE_ARCH` in `mul_mmq_cm1.comp`. Paths that run on RDNA3 **or** RDNA4 (not RDNA4-only): int8 cm1 MMQ (V:1947, 2645), 4-row `rm_int_n`/`rm_id` (V:3048, 3083-3086), AMD-proprietary coopmat allow-list (V:17730).
- Paths with no arch gate at all (engage on any Vulkan GPU if op pattern/shape matches): GDN cache/gate fusions, RMS_NORM fast/gate/scale/ADD_RMS_NORM_MUL fusions, graph_optimize adjacency (`KEEP_*`, `SOP_*`), MMV+ADD batching, GET_ROWS small, CONCAT_t, rms_norm small, cm1 packed-GQA, mask_opt cache, prealloc_x stepping, small-BAR rule, all `src/`/`common/`/`tools/` changes.

## 4. Loose guards (inputs to tightening)

1. **Int8 cm1 MMQ (Table B row 2)**: keyed on `coopmat_int_support` + RDNA3/RDNA4 only; builds ~8 quant types for MUL_MAT *and* MUL_MAT_ID. Untested: MoE (35B-A3B) expert matmuls, Q4_0/Q5_0/Q8_0/IQ4*/MXFP4/Q3_K models, RDNA3. Also the silent fallback when the Q6_K/Q8_0 RDNA4 GEMM kill switches are set.
2. **Arch-agnostic fusions on op-pattern + F32 shape only** (ADD_RMS_NORM_MUL for nrows>1 && ne0<=8192; RMS_NORM_SCALE; RMS_NORM_MUL_SILU_MUL; MMV+ADD batch; `KEEP_UNARY_MUL` reorder; GDN gate/cache with S_v in {16,32,64,128}, K>1). ADD->RMS_NORM->MUL is the standard pre-norm prefill of every dense llama/Gemma/Qwen model on every Vulkan GPU, so these engage far outside qwen35/R9700. Claimed bit-identical, but only qwen35 outputs were verified.
3. **RDNA4 FA q8_0 paths keyed on HSK=HSV=256 + K=V=Q8_0 only** (no model-arch check): any RDNA4 + q8_0-KV + head-dim-256 model (Gemma 3/4, other qwen35-class) takes decode_q8 (G<=32/24, N<=8) and prefill rdna4 (G<=6, neq1>=64, dense KV). Gemma SWA masks are only checked as `mask F16, nem2==1`; no SWA-specific validation. `use_dequant_kv` also lacks a cap on KV scratch other than maxStorageBufferRange (`FA_DEQUANT_MAX_KV` default off).
4. **Shape-only weight guards**: Q6_K and Q8_0 prefill GEMM fire for *any* Q6_K/Q8_0 weight with ne01%128==0, ne10%256==0, ne11>=64 (any model); Q6_K SWIGLU fusion for any Q6_K gate/up pair with ne0%512==0; Q6_K MMVQ n>=5 and 4-row GEMV for every Q6_K weight on RDNA4; `DMMV_F32_WIDE` for any F32 weight m<=256,k>=1024 (MoE routers, any small f32 gate).
5. **Packed-GQA in cm1 FA** (`FA_NO_PACK`) and the changed `pvsh` shmem estimate apply to every coopmat1 FA device and quant combination, not just RDNA4/q8_0.
6. **llama-side**: KQ-mask cache (any causal non-SWA non-alibi model), MFL embd-view (any batch with embeddings, incl. multimodal), kv-cells dirty tracking (all unified KV caches). MFL MTP row skip applies to any MTP-capable model when a prompt batch has >64 rows.
7. **RDNA4 detection is a feature proxy** (float8 or saturating mixed int dot). New AMD parts or driver changes in extension reporting would silently route to RDNA4-only shaders; no CU/device-id sanity check; `coopmat_wgs = 2 * shader_core_count` assumes R9700-like occupancy.
8. **GDN alloc-dep rule** matches any CONCAT whose src0 comes from GET_ROWS (possible hit on other recurrent archs).
9. Small-BAR rule applies to all vendors; cannot be turned back to the old preference by env.
10. `llama_model_set_draft_vocab` only checks arch==QWEN35 and tensor shape; opt-in, so low priority.

## 5. "All switches off"

Verified against each `getenv` site (all are `!= nullptr` / `atoi != 0` tests). Fork-default-ON paths only; opt-ins left at default.

```
GGML_VK_NO_DECODE_Q8=1;GGML_VK_FA_NO_DECODE_V2=1;GGML_VK_FA_NO_PACK=1;GGML_VK_NO_MASK_OPT_CACHE=1;GGML_VK_NO_FA_PREFILL_RDNA4=1;GGML_VK_NO_FA_PREFILL_RDNA4_VT=1;GGML_VK_NO_FA_PREFILL_RDNA4_RS=1;GGML_VK_NO_FA_PREFILL_RDNA4_V64=1;GGML_VK_NO_FOV=1;GGML_VK_NO_FV2=1;GGML_VK_NO_MMQ_Q6K_RDNA4=1;GGML_VK_NO_Q6K_LEANQ=1;GGML_VK_NO_Q6K_UPAR=1;GGML_VK_NO_Q6K_N256=1;GGML_VK_NO_Q6K_SPLITK=1;GGML_VK_NO_Q6K_SWIGLU=1;GGML_VK_NO_Q6K_SWIGLU_N256=1;GGML_VK_NO_SOP_GLU=1;GGML_VK_NO_SOP_ADDRMS=1;GGML_VK_NO_Q8_GEMM_TUNE=1;GGML_VK_NO_MQ5_TUNE=1;GGML_VK_NO_DMMV_F32_WIDE=1;GGML_VK_NO_GR_SMALL=1;GGML_VK_NO_GDN_CACHE=1;GGML_VK_NO_GDN_GATE=1;GGML_VK_NO_VSUB=1;GGML_VK_NO_RMS_NORM_FAST=1;GGML_VK_NO_RMS_NORM_GATE=1;GGML_VK_NO_RMS_NORM_SCALE=1;GGML_VK_NO_KEEP_RMS_SCALE=1;GGML_VK_NO_KEEP_UNARY_MUL=1;GGML_VK_NO_MMV_ADD_BATCH=1;GGML_VK_DISABLE_FA_SHMEM_STAGING=1;GGML_VK_Q6K_MMVQ_MIN_N=0;GGML_VK_RM_KQ_Q6K=2;GGML_VK_PREALLOC_X_STEP_MIB=0;GGML_VK_HOST_BUFT_DEV0=1;LLAMA_NO_MFL=1;LLAMA_NO_MTP_KVONLY=1;LLAMA_NO_KQ_MASK_CACHE=1;LLAMA_NO_RS_INDEX=1;LLAMA_KQ_MASK_THREADS=1
```

(The string contains all 30 `GGML_VK_NO_*` switches, plus `GGML_VK_FA_NO_DECODE_V2`, `GGML_VK_FA_NO_PACK`, `GGML_VK_DISABLE_FA_SHMEM_STAGING`, 4 value overrides and 4 `LLAMA_NO_*`, plus `LLAMA_KQ_MASK_THREADS=1`.) Semicolon-separated as requested; for `env` use spaces instead.

Caveats on what "off" means:
- With the RDNA4 Q6_K/Q8_0 GEMM kill switches, Q6_K and Q8_0 prefill do **not** return to upstream: they fall to the int8 cm1 MMQ path (`matmul_q6_k_q8_1_cm1`, `matmul_q8_0_q8_1_cm1`), which is itself fork code with no switch.
- `GGML_VK_NO_RMS_NORM_FAST` also removes the gate and resadd pipelines (they are created inside that block); the small-ne0 rms_norm pipelines remain.
- `GGML_VK_Q6K_MMVQ_MIN_N=0` and `GGML_VK_RM_KQ_Q6K=2` are value overrides, not `NO_` switches.
- `LLAMA_NO_RS_INDEX=1` only restores the old cap of 4 conv snapshots (moot once `NO_GDN_CACHE` is set); `LLAMA_KQ_MASK_THREADS=1` is moot when the mask cache is off.
- Server MTP stays at upstream behaviour because `LLAMA_MTP_PIPE`, `--spec-draft-vocab*`, `MTMD_LAZY_GPU`, `GGML_VK_HOST_GET_ROWS`, `GGML_VK_INPUT_STAGING`, `GGML_SCHED_ASYNC_INPUT_COPY` are default off.

### Table B paths that cannot be disabled by env
int8 cm1 MMQ (MUL_MAT + MUL_MAT_ID) and its `quantize_y`/`y_non_contig` changes; RDNA4 arch split (enum, DEVICE_ARCH spec const); cm1 FA shader/pvsh-shmem changes (only the packed and int8-QK variants are switchable); rms_norm small (ne0<=128) pipelines; concat_t; small-BAR host-visible-vidmem rule (upstream `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` forces the same result, never the old one); GDN alloc deps in graph_optimize; pinned-host sync skip; split-K prealloc sizing; kv-cells dirty tracking; server `spec_prompt` reuse and `common_dft_flush` hooks; graph-reuse key addition. The only way to remove these is to build upstream `ce8caa6e6`.

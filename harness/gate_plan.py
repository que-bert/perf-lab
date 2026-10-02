#!/usr/bin/env python3
"""Which T2 phases does this candidate need? (change-aware gate, docs/2026-10-01-phase2-plan.md)

  gate_plan.py <fork-worktree> <base-ref> [--head REF] [--env]

Reads `git diff --name-only <base-ref>..<head>` in the worktree (head defaults to HEAD) and prints the plan.
--env prints shell assignments instead, for t2.sh:  PHASES="..." GATES_PARTS="..." OPS="..." ZOO_PARTS="..."
FULL=1 in t2.sh bypasses this. Anything not recognised falls back to the full gate (never silently narrows).

Rules (first match decides what a path needs; the plan is the union over all changed paths):
  ggml/ core, ggml-cpu (the op-test reference), CMake, ggml-vulkan.cpp hunks in generic code, .glsl includes,
  unmapped shaders, tests/test-backend-ops.cpp          -> full gate
  vulkan-shaders/<x>.comp (mapped)                      -> op tests only for the mapped ops (+ decode KLD, PPL, acc,
                                                          depth, pool, prompt, flagless, I2, zoo)
  ggml-vulkan.cpp                                       -> hunk text mapped to ops by keyword, else full
  other ggml backends (cuda, metal, ...)                -> nothing
  src/  (llama-*.cpp, models/)                          -> no op tests, no decode KLD; PPL kept; I2 + zoo kept
  common/ tools/ (server, mtmd, ...)                    -> no op tests, no decode KLD, no PPL, no I2; zoo smoke only
  common/serving-preset.*, arg.*  only                  -> flagless + zoo smoke
  docs, *.md, examples, scripts, CI                     -> nothing
Perf phases (depth pool prompt) are kept for every code change except preset-only.
"""
import os, re, subprocess, sys

ALL = ["depth", "pool", "prompt", "flagless", "i2", "gates", "zoo"]
OP_RULES = [  # (stem regex, ops). Ops the gates4 list does not test are still passed to test-backend-ops.
    (r"^(mul_mat_vec|mul_mm|mul_mmq|dequant_|quantize_q8_1|mul_mat_split_k|iq_shmem)", "MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION"),
    (r"^(flash_attn|fa_)", "FLASH_ATTN_EXT"),
    (r"^rms_norm", "RMS_NORM,RMS_NORM_MUL_ADD,RMS_NORM_SCALE,RMS_NORM_MUL_SILU_MUL"),
    (r"^l2_norm", "L2_NORM"),
    (r"^(gated_delta_net|gdn_)", "GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION,GDN_RECURRENT_CACHE"),
    (r"^ssm_conv", "SSM_CONV"),
    (r"^(copy|contig_copy|cpy)", "CPY"),
    (r"^concat", "CONCAT"),
    (r"^get_rows", "GET_ROWS"),
    (r"^(add|multi_add)", "ADD,ADD_ID"),
    (r"^mul$", "MUL"),
    (r"^scale", "SCALE"),
    (r"^unary$", "SIGMOID,SOFTPLUS,SILU"),
    (r"^(swiglu|geglu|reglu|glu_)", "SWIGLU,GEGLU,REGLU"),
    (r"^soft_max", "SOFT_MAX"),
    (r"^rope_", "ROPE"),
    (r"^norm$", "NORM"),
]
KEYWORDS = [  # words in a ggml-vulkan.cpp hunk (header + changed lines) -> ops
    (r"flash_attn|\bfa_|FLASH_ATTN|CREATE_FA", "FLASH_ATTN_EXT"),
    (r"mul_mat|dequant|mmq|matmul|MUL_MAT", "MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION"),
    (r"rms_norm|RMS_NORM", "RMS_NORM,RMS_NORM_MUL_ADD,RMS_NORM_SCALE,RMS_NORM_MUL_SILU_MUL"),
    (r"gated_delta|gdn", "GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION,GDN_RECURRENT_CACHE"),
    (r"ssm_conv", "SSM_CONV"),
    (r"get_rows", "GET_ROWS"),
]
OTHER_BACKENDS = ("cuda", "metal", "sycl", "opencl", "hip", "musa", "cann", "webgpu", "rpc", "blas", "hexagon", "zdnn", "virtgpu", "openvino")
IGNORE = re.compile(r"^(docs/|examples/|scripts/|\.github/|ci/|gguf-py/|media/|licenses/|grammars/|models/|benches/|README|CONTRIBUTING|AGENTS|CLAUDE|SECURITY|LICENSE|\.|.*\.md$)")
PRESET = re.compile(r"^common/(serving-preset\.(cpp|h)|arg\.(cpp|h)|common\.h|CMakeLists\.txt)$")


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, check=True).stdout


def vk_cpp_ops(wt, base, head):
    """ops named by the hunks of ggml-vulkan.cpp, or None when some hunk is generic."""
    d = sh("git", "-C", wt, "diff", "-U0", f"{base}..{head}", "--", "ggml/src/ggml-vulkan/ggml-vulkan.cpp")
    ops, hunks, cur = set(), [], None
    for l in d.splitlines():
        if l.startswith("@@"):
            cur = [l]
            hunks.append(cur)
        elif cur is not None and l[:1] in "+-" and not l.startswith(("+++", "---")):
            cur.append(l)
    for h in hunks:
        text, found = "\n".join(h), False
        for rx, o in KEYWORDS:
            if re.search(rx, text):
                ops.update(o.split(","))
                found = True
        if not found:
            return None
    return ops


def plan(wt, base, head):
    files = [f for f in sh("git", "-C", wt, "diff", "--name-only", f"{base}..{head}").splitlines() if f]
    why = []
    ph, parts = set(), set()
    ops, full_ops = set(), False
    zoo_full = False

    def code(perf=True, i2=False, ppl=False, acc=True, kld=False, zoo=False):
        if perf:
            ph.update(["depth", "pool", "prompt"])
        ph.update(["flagless", "gates", "zoo"])
        if i2:
            ph.add("i2")
        parts.add("acc") if acc else None
        if ppl:
            parts.add("ppl")
        if kld:
            parts.add("kld")
        nonlocal zoo_full
        zoo_full = zoo_full or zoo

    preset_only = True
    for f in files:
        if IGNORE.match(f) or (f.startswith("tests/") and f != "tests/test-backend-ops.cpp") or \
                (f.startswith("ggml/") and any(f"ggml-{b}" in f for b in OTHER_BACKENDS)):
            continue
        if PRESET.match(f):
            ph.update(["flagless", "zoo"])
            why.append(f"{f}: serving preset")
            continue
        preset_only = False
        if f == "tests/test-backend-ops.cpp":
            code(ppl=True, i2=True, kld=True, zoo=True); full_ops = True; why.append(f"{f}: op test itself changed -> full op suite")
        elif f.startswith("ggml/src/ggml-vulkan/vulkan-shaders/") and f.endswith(".comp"):
            stem = os.path.basename(f).rsplit(".", 1)[0]
            m = next((o for rx, o in OP_RULES if re.search(rx, stem)), None)
            code(ppl=True, i2=True, kld=True, zoo=True)
            if m:
                ops.update(m.split(",")); why.append(f"{f}: shader -> {m}")
            else:
                full_ops = True; why.append(f"{f}: unmapped shader -> full op suite")
        elif f == "ggml/src/ggml-vulkan/ggml-vulkan.cpp":
            code(ppl=True, i2=True, kld=True, zoo=True)
            o = vk_cpp_ops(wt, base, head)
            if o is None:
                full_ops = True; why.append(f"{f}: generic hunk (device/pipeline setup) -> full op suite")
            else:
                ops.update(o); why.append(f"{f}: hunks name {','.join(sorted(o))}")
        elif f.startswith("ggml/") or f in ("CMakeLists.txt",) or f.startswith("cmake/"):
            code(ppl=True, i2=True, kld=True, zoo=True); full_ops = True; why.append(f"{f}: ggml core / build -> full op suite")
        elif f.startswith("src/"):
            code(ppl=True, i2=True, zoo=True); why.append(f"{f}: library -> no op tests / decode KLD")
        elif f == "tools/decode-kld/decode-kld.cpp" or f.startswith("tools/decode-kld/"):
            code(kld=True); why.append(f"{f}: KLD tool")
        elif f.startswith(("common/", "tools/", "include/")):
            code(); why.append(f"{f}: serving / common -> no op tests / decode KLD")
        else:
            code(ppl=True, i2=True, kld=True, zoo=True); full_ops = True; why.append(f"{f}: unrecognised path -> full gate")
    if files and preset_only and ph:
        ph = {"flagless", "zoo"}
        parts, ops, full_ops, zoo_full = set(), set(), False, False
    if full_ops or ops:
        parts.add("ops")
    ordered = [p for p in ALL if p in ph]
    if "gates" in ordered and not parts:
        ordered.remove("gates")
    gp = [p for p in ("ops", "ppl", "kld", "acc") if p in parts]
    zoo_parts = "full" if zoo_full else ("smoke" if "zoo" in ph else "")
    return {"files": files, "why": why, "phases": ordered, "gates_parts": gp,
            "ops": "full" if full_ops else ",".join(sorted(ops)), "zoo_parts": zoo_parts}


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    head = sys.argv[sys.argv.index("--head") + 1] if "--head" in sys.argv else "HEAD"
    if "--head" in sys.argv:
        a.remove(head)
    if len(a) != 2:
        sys.exit(__doc__)
    p = plan(a[0], a[1], head)
    if "--env" in sys.argv:
        print(f'PHASES="{" ".join(p["phases"])}" GATES_PARTS="{" ".join(p["gates_parts"])}" OPS="{p["ops"]}" ZOO_PARTS="{p["zoo_parts"]}"')
        return
    print(f"changed files: {len(p['files'])}")
    for w in p["why"]:
        print(f"  {w}")
    print(f"phases:      {' '.join(p['phases']) or '(none: nothing gate-relevant changed)'}")
    print(f"gates parts: {' '.join(p['gates_parts']) or '-'}")
    print(f"op tests:    {p['ops'] or '-'}")
    print(f"zoo:         {p['zoo_parts'] or '-'}")


if __name__ == "__main__":
    main()

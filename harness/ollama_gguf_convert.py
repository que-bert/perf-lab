#!/usr/bin/env python3
"""Convert ollama-packed GGUF files into the layout upstream llama.cpp loads.

  ollama_gguf_convert.py IN.gguf OUT.gguf [--mmproj OUT-mmproj.gguf] [--inspect]

Tensor data is copied byte-for-byte (no requantisation); only metadata keys,
tensor names and the in-file vision/audio towers change.  No llama.cpp source
is touched; this is the "converter, not loader patch" half of plan W4.

Supported architectures (anything else exits with an error):
  gemma4  text tensors already use upstream names.  Fixes: tokenizer.ggml.model
          'llama' -> 'gemma4', add_bos_token False -> True, add_space_prefix,
          general.type, drops tower KV keys.  Drops a.*/v.*/mm.* tensors into
          --mmproj (vision tower only; audio tower is NOT converted).
  qwen35  ssm_dt -> ssm_dt.bias, rope.dimension_sections padded to 4 entries,
          per-layer head_count_kv array -> scalar, vision KV/tensors dropped,
          mtp.* head dropped (or --keep-mtp: renamed to blk.<n>.nextn.*,
          block_count+1, nextn_predict_layers=1).  No --mmproj for qwen35.

gguf-py is imported from the upstream checkout (override with GGUF_PY).
"""
import argparse
import os
import re
import sys

GGUF_PY = os.environ.get(
    "GGUF_PY",
    "/home/bbuckham/git/perf-lab/runners/llama.cpp/upstream-ce8caa6/gguf-py")
sys.path.insert(0, GGUF_PY)

from gguf import GGUFReader, GGUFWriter, GGUFValueType  # noqa: E402

SUPPORTED = ("gemma4", "qwen35")


class ConvError(Exception):
    pass


# ---------------------------------------------------------------- reading ----
def kv_items(r):
    """(key, python value, value_type, sub_type) for every KV, in file order."""
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == GGUFValueType.ARRAY else None
        yield k, f.contents(), vt, sub


def load(path):
    try:
        r = GGUFReader(path)
    except Exception as e:  # noqa: BLE001
        raise ConvError(f"{path}: not a readable GGUF file ({e})")
    kv = {k: (v, vt, sub) for k, v, vt, sub in kv_items(r)}
    if "general.architecture" not in kv:
        raise ConvError(f"{path}: no general.architecture key; not a GGUF model")
    return r, kv


def arch_of(kv):
    return kv["general.architecture"][0]


# --------------------------------------------------------- classification ----
def is_tower(arch, name):
    if arch == "gemma4":
        return name.startswith(("a.", "v.", "mm."))
    if arch == "qwen35":
        return name.startswith("v.")
    return False


def detect_markers(arch, r, kv):
    """Reasons this file is ollama-packed (empty list => already upstream)."""
    names = [t.name for t in r.tensors]
    m = []
    if arch == "gemma4":
        if kv.get("tokenizer.ggml.model", ("",))[0] == "llama":
            m.append("tokenizer.ggml.model='llama' (upstream: 'gemma4')")
        if any(n.startswith(("v.", "a.", "mm.")) for n in names):
            m.append("vision/audio towers packed in the text file")
        if "general.type" not in kv:
            m.append("no general.type")
    elif arch == "qwen35":
        if any(n.startswith("mtp.") for n in names):
            m.append("mtp.* tensors (upstream: blk.N.nextn.*)")
        if any(re.fullmatch(r"blk\.\d+\.ssm_dt", n) for n in names):
            m.append("blk.N.ssm_dt (upstream: blk.N.ssm_dt.bias)")
        s = kv.get("qwen35.rope.dimension_sections")
        if s and len(s[0]) == 3:
            m.append("rope.dimension_sections has 3 entries (upstream: 4)")
        hk = kv.get("qwen35.attention.head_count_kv")
        if hk and hk[1] == GGUFValueType.ARRAY:
            m.append("head_count_kv is a per-layer array (upstream: scalar)")
        if any(n.startswith("v.") for n in names):
            m.append("vision tower packed in the text file")
    return m


# ------------------------------------------------------------ conversions ----
def qwen_name(name, keep_mtp, n_layer):
    """Return new tensor name or None to drop."""
    if name.startswith("v."):
        return None
    if name.startswith("mtp."):
        if not keep_mtp:
            return None
        b = n_layer  # mtp layer becomes block index n_layer
        table = {"mtp.fc.weight": f"blk.{b}.nextn.eh_proj.weight",
                 "mtp.pre_fc_norm_embedding.weight": f"blk.{b}.nextn.enorm.weight",
                 "mtp.pre_fc_norm_hidden.weight": f"blk.{b}.nextn.hnorm.weight",
                 "mtp.norm.weight": f"blk.{b}.nextn.shared_head_norm.weight"}
        if name in table:
            return table[name]
        m = re.fullmatch(r"mtp\.layers\.0\.(.+)", name)
        if m:
            return f"blk.{b}.{m.group(1)}"
        raise ConvError(f"unrecognised mtp tensor {name}")
    if re.fullmatch(r"blk\.\d+\.ssm_dt", name):
        return name + ".bias"
    return name


def convert_kv_qwen(kv, keep_mtp, has_mtp):
    out = []
    for k, (v, vt, sub) in kv.items():
        if ".vision." in k or k.startswith("qwen35.vision") or k in (
                "qwen35.image_token_id", "qwen35.vision_start_token_id",
                "qwen35.vision_end_token_id", "qwen35.mrope_sections",
                "qwen35.rope.mrope_section", "qwen35.rope.mrope_interleaved",
                "qwen35.ssm.v_head_reordered"):
            continue
        if k == "qwen35.rope.dimension_sections":
            v = list(v) + [0] * (4 - len(v))
            sub = GGUFValueType.INT32 if sub is None else sub
        elif k == "qwen35.attention.head_count_kv" and vt == GGUFValueType.ARRAY:
            nz = {int(x) for x in v if int(x) > 0}
            if len(nz) != 1:
                raise ConvError(f"head_count_kv array has non-uniform values {sorted(nz)}")
            v, vt, sub = nz.pop(), GGUFValueType.UINT32, None
        elif k == "qwen35.block_count" and keep_mtp and has_mtp:
            out.append((k, int(v) + 1, vt, sub))
            out.append(("qwen35.nextn_predict_layers", 1, GGUFValueType.UINT32, None))
            continue
        out.append((k, v, vt, sub))
    out.append(("general.type", "model", GGUFValueType.STRING, None))
    return out


def convert_kv_gemma(kv):
    out = []
    for k, (v, vt, sub) in kv.items():
        if k.startswith(("gemma4.vision.", "gemma4.audio.")):
            continue
        if k == "tokenizer.ggml.model" and v == "llama":
            v = "gemma4"
        elif k == "tokenizer.ggml.add_bos_token":
            v = True  # ollama prepends BOS in its template; upstream does it in the tokenizer
        out.append((k, v, vt, sub))
    keys = {k for k, *_ in out}
    if "general.type" not in keys:
        out.append(("general.type", "model", GGUFValueType.STRING, None))
    if "tokenizer.ggml.add_space_prefix" not in keys:
        out.append(("tokenizer.ggml.add_space_prefix", False, GGUFValueType.BOOL, None))
    return out


def mmproj_kv_gemma(kv):
    g = lambda s: kv[f"gemma4.{s}"][0]  # noqa: E731
    I, F, S, B = (GGUFValueType.UINT32, GGUFValueType.FLOAT32,
                  GGUFValueType.STRING, GGUFValueType.BOOL)
    return [
        ("general.architecture", "clip", S, None),
        ("general.type", "mmproj", S, None),
        ("clip.has_vision_encoder", True, B, None),
        ("clip.vision.projection_dim", int(g("embedding_length")), I, None),
        ("clip.vision.image_size", 224, I, None),  # as in the lmstudio mmproj
        ("clip.vision.patch_size", int(g("vision.patch_size")), I, None),
        ("clip.vision.embedding_length", int(g("vision.embedding_length")), I, None),
        ("clip.vision.feed_forward_length", int(g("vision.feed_forward_length")), I, None),
        ("clip.vision.block_count", int(g("vision.block_count")), I, None),
        ("clip.vision.attention.head_count", int(g("vision.attention.head_count")), I, None),
        ("clip.vision.image_mean", [0.0, 0.0, 0.0], GGUFValueType.ARRAY, F),
        ("clip.vision.image_std", [1.0, 1.0, 1.0], GGUFValueType.ARRAY, F),
        ("clip.vision.projector_type", "gemma4v", S, None),
        ("clip.vision.attention.layer_norm_epsilon",
         float(g("vision.attention.layer_norm_epsilon")), F, None),
    ]


# ----------------------------------------------------------------- writing ---
def write_gguf(path, arch_name, kvs, tensors):
    """tensors: list of (new_name, ReaderTensor). Streams data, no requant."""
    w = GGUFWriter(path, arch_name, use_temp_file=False)
    # GGUFWriter writes general.architecture itself; skip ours.
    for k, v, vt, sub in kvs:
        if k == "general.architecture":
            continue
        w.add_key_value(k, v, vt, sub)
    for name, t in tensors:
        w.add_tensor_info(name, list(t.data.shape),
                          t.data.dtype, int(t.n_bytes), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for _, t in tensors:
        w.write_tensor_data(t.data)
    w.close()


# -------------------------------------------------------------------- main ---
def plan(r, kv, args):
    arch = arch_of(kv)
    if arch not in SUPPORTED:
        raise ConvError(f"unsupported architecture '{arch}' (supported: {', '.join(SUPPORTED)})")
    markers = detect_markers(arch, r, kv)
    if not markers:
        raise ConvError("file does not look ollama-packed (already upstream layout); nothing to convert")
    text, tower = [], []
    has_mtp = any(t.name.startswith("mtp.") for t in r.tensors)
    if arch == "qwen35":
        nl = int(kv["qwen35.block_count"][0])
        for t in r.tensors:
            n = qwen_name(t.name, args.keep_mtp, nl)
            (text if n else tower).append((n or t.name, t))
        kvs = convert_kv_qwen(kv, args.keep_mtp, has_mtp)
    else:
        for t in r.tensors:
            (tower if is_tower(arch, t.name) else text).append((t.name, t))
        kvs = convert_kv_gemma(kv)
    return arch, markers, kvs, text, tower, has_mtp


def inspect(path, r, kv, args):
    arch = arch_of(kv)
    print(f"file: {path}\narch: {arch}  tensors: {len(r.tensors)}  kv: {len(kv)}")
    if arch not in SUPPORTED:
        print(f"UNSUPPORTED architecture (supported: {', '.join(SUPPORTED)})")
        return 1
    m = detect_markers(arch, r, kv)
    print("ollama-packed markers:" + ("" if m else " none (already upstream layout)"))
    for x in m:
        print("  -", x)
    if not m:
        return 0
    _, _, kvs, text, tower, _ = plan(r, kv, args)
    old, new = set(kv), {k for k, *_ in kvs}
    print(f"KV removed: {sorted(old - new)}")
    print(f"KV added:   {sorted(new - old)}")
    for k in sorted(old & new):
        a = kv[k][0]
        b = next(v for kk, v, *_ in kvs if kk == k)
        sa, sb = str(list(a) if hasattr(a, '__len__') and not isinstance(a, str) else a), str(list(b) if hasattr(b, '__len__') and not isinstance(b, str) else b)
        if sa != sb and len(sa) < 300:
            print(f"KV changed: {k}: {str(a)[:60]} -> {str(b)[:60]}")
    renamed = [(t.name, n) for n, t in text if n != t.name]
    print(f"text tensors kept: {len(text)} ({len(renamed)} renamed)")
    for a, b in renamed[:4]:
        print(f"  rename {a} -> {b}")
    groups = {}
    for _, t in tower:
        g = re.match(r"([a-z]+)\.", t.name)
        g = g.group(1) if g else "other"
        groups[g] = groups.get(g, 0) + 1
    print(f"tower/dropped tensors: {len(tower)} {groups}")
    if arch == "gemma4":
        print("  --mmproj would write the vision tower (v.*, mm.input_projection); audio (a.*, mm.a.*) is not converted")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--mmproj", help="also write the vision tower as an mmproj GGUF (gemma4 only)")
    ap.add_argument("--inspect", action="store_true", help="report differences, write nothing")
    ap.add_argument("--keep-mtp", action="store_true", help="qwen35: keep the MTP head as blk.N.nextn.*")
    args = ap.parse_args()
    try:
        r, kv = load(args.inp)
        if args.inspect:
            return inspect(args.inp, r, kv, args)
        if not args.out:
            ap.error("OUT.gguf required unless --inspect")
        arch, markers, kvs, text, tower, _ = plan(r, kv, args)
        if args.mmproj and arch != "gemma4":
            raise ConvError(f"--mmproj is only supported for gemma4, not {arch}")
        write_gguf(args.out, arch, kvs, text)
        print(f"wrote {args.out}: {len(text)} tensors ({len(tower)} dropped)")
        if args.mmproj:
            vt = [(n, t) for n, t in tower if n.startswith(("v.",)) or n == "mm.input_projection.weight"]
            if not vt:
                raise ConvError("no vision tower tensors in input")
            write_gguf(args.mmproj, "clip", mmproj_kv_gemma(kv), vt)
            print(f"wrote {args.mmproj}: {len(vt)} vision tensors (audio tower not converted)")
        return 0
    except ConvError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

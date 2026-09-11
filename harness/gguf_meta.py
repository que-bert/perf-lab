#!/usr/bin/env python3
"""Read a GGUF header without loading the model.

Exists because the interesting facts about a candidate model -- architecture,
native context, whether it carries an MTP head, whether it was quantized with an
importance matrix -- decide how it should be served, and finding them by
starting a server costs a model load. The header is a few KB at the front of the
file.

  gguf_meta.py FILE [FILE...]          one line per file
  gguf_meta.py --full FILE             every key
"""
import argparse
import json
import pathlib
import struct
import sys

# GGUF value type tags, in the order the spec defines them.
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
_FIXED = {U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
          U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
          U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8)}

# ggml_type -> name, for the tensor-level quant census. Only the types this lab
# actually encounters are named; anything else prints as its raw number.
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
              8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
              13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}


class _R:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("short read in GGUF header")
        return b

    def fixed(self, t):
        fmt, n = _FIXED[t]
        return struct.unpack(fmt, self.raw(n))[0]

    def string(self):
        return self.raw(self.fixed(U64)).decode("utf-8", "replace")

    def value(self, t):
        if t == STR:
            return self.string()
        if t == ARR:
            et = self.fixed(U32)
            n = self.fixed(U64)
            # Token vocabularies run to 150k+ entries and are never the
            # question being asked; keep a head and the count.
            if n > 64:
                head = [self.value(et) for _ in range(8)]
                for _ in range(n - 8):
                    self.value(et)
                return {"len": n, "head": head}
            return [self.value(et) for _ in range(n)]
        return self.fixed(t)


def read(path):
    with open(path, "rb") as f:
        r = _R(f)
        if r.raw(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        ver = r.fixed(U32)
        n_tensors = r.fixed(U64)
        n_kv = r.fixed(U64)
        kv = {}
        for _ in range(n_kv):
            k = r.string()
            kv[k] = r.value(r.fixed(U32))
        # Tensor directory follows the KV block: name, dims, type, offset.
        census = {}
        for _ in range(n_tensors):
            name = r.string()
            nd = r.fixed(U32)
            for _ in range(nd):
                r.fixed(U64)
            t = r.fixed(U32)
            r.fixed(U64)
            tn = GGML_TYPES.get(t, str(t))
            census[tn] = census.get(tn, 0) + 1
            kv.setdefault("_tensor_names", [])
            if len(kv["_tensor_names"]) < 4000:
                kv["_tensor_names"].append(name)
    return {"version": ver, "n_tensors": n_tensors, "kv": kv, "census": census}


def summarize(path):
    m = read(path)
    kv, names = m["kv"], m["kv"].get("_tensor_names", [])
    arch = kv.get("general.architecture", "?")
    g = lambda k, d=None: kv.get(f"{arch}.{k}", d)
    # An MTP / next-token-prediction head lives in the model file as its own
    # block; the count key alone does not prove the weights shipped.
    mtp_key = g("nextn_predict_layers")
    mtp_tensors = sum(1 for n in names if "nextn" in n or "mtp" in n)
    return {
        "file": pathlib.Path(path).name,
        "gib": round(pathlib.Path(path).stat().st_size / 2**30, 2),
        "arch": arch,
        "params": kv.get("general.size_label", "?"),
        "ctx": g("context_length"),
        "blocks": g("block_count"),
        "embd": g("embedding_length"),
        "heads": g("attention.head_count"),
        "kv_heads": g("attention.head_count_kv"),
        "rope_base": g("rope.freq_base"),
        # Absent rope.scaling.* means the long context is native, not YaRN'd on.
        "rope_scaling": next((k for k in kv if ".rope.scaling." in k), None),
        "mtp_layers": mtp_key,
        "mtp_tensors": mtp_tensors,
        "imatrix": "imatrix" in json.dumps(
            {k: v for k, v in kv.items() if k.startswith("general.")}).lower(),
        "quants": dict(sorted(m["census"].items(),
                              key=lambda kvp: -kvp[1])),
        "chat_template": "tokenizer.chat_template" in kv,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    out = []
    for p in a.files:
        try:
            if a.full:
                m = read(p)
                m["kv"].pop("_tensor_names", None)
                print(json.dumps(m, indent=2, default=str))
                continue
            out.append(summarize(p))
        except Exception as e:
            out.append({"file": p, "error": f"{type(e).__name__}: {e}"})
    if a.full:
        return
    if a.json:
        print(json.dumps(out, indent=2, default=str))
        return
    for s in out:
        if "error" in s:
            print(f"{s['file']}: {s['error']}", file=sys.stderr)
            continue
        mtp = "none"
        if s["mtp_tensors"]:
            mtp = f"{s['mtp_layers']} layer(s), {s['mtp_tensors']} tensors"
        elif s["mtp_layers"]:
            mtp = f"declares {s['mtp_layers']} but NO tensors"
        print(f"{s['file']}")
        print(f"  {s['arch']}  {s['params']}  {s['gib']} GiB  "
              f"{s['blocks']} blocks  ctx {s['ctx']}")
        print(f"  heads {s['heads']}/{s['kv_heads']} kv   rope_base "
              f"{s['rope_base']}   scaling "
              f"{s['rope_scaling'] or 'none (native window)'}")
        print(f"  MTP: {mtp}   chat_template: {s['chat_template']}")
        print(f"  tensor quants: " + ", ".join(
            f"{k}x{v}" for k, v in s["quants"].items()))


if __name__ == "__main__":
    main()

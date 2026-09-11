#!/usr/bin/env python3
"""Rewrite one metadata key of a GGUF, copying the tensor data untouched.

The case this exists for: ollama ships model files whose headers disagree with
upstream llama.cpp, and the disagreement is a single key. qwen3.5:9b writes
`qwen35.rope.dimension_sections` with three entries; `src/models/qwen35.cpp`
reads it with a required length of four and the load aborts before a byte of
tensor data is touched. The weights are fine. Only the header is wrong.

Patching is therefore honest in a way that "re-quantize it yourself" would not
be -- nothing about the numbers changes -- but it is still a modified file, so
the output records what was changed in `general.description` and the caller is
expected to keep the patched copy under a name that says so.

  gguf_patch.py IN OUT --set-int-array qwen35.rope.dimension_sections=11,11,10,0

Only the key being replaced is re-serialized; every other key is copied out in
its original order and type, and the tensor directory is byte-identical apart
from the data offsets, which do not move because the data section is realigned
to `general.alignment` exactly as it was.
"""
import argparse
import os
import struct
import sys

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
_FIXED = {U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
          U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
          U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8)}
CHUNK = 64 << 20
SYNC_EVERY = 512 << 20


class Reader:
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
        return self.raw(self.fixed(U64)).decode("utf-8", "surrogateescape")

    def value(self, t):
        if t == STR:
            return self.string()
        if t == ARR:
            et = self.fixed(U32)
            n = self.fixed(U64)
            return (et, [self.value(et) for _ in range(n)])
        return self.fixed(t)


def w_fixed(t, v):
    return struct.pack(_FIXED[t][0], v)


def w_string(s):
    b = s.encode("utf-8", "surrogateescape")
    return struct.pack("<Q", len(b)) + b


def w_value(t, v):
    if t == STR:
        return w_string(v)
    if t == ARR:
        et, items = v
        out = struct.pack("<I", et) + struct.pack("<Q", len(items))
        return out + b"".join(w_value(et, x) for x in items)
    return w_fixed(t, v)


def read_header(path):
    with open(path, "rb") as f:
        r = Reader(f)
        if r.raw(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        ver = r.fixed(U32)
        n_tensors = r.fixed(U64)
        n_kv = r.fixed(U64)
        kv = []
        for _ in range(n_kv):
            k = r.string()
            t = r.fixed(U32)
            kv.append((k, t, r.value(t)))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.fixed(U32)
            dims = [r.fixed(U64) for _ in range(nd)]
            ttype = r.fixed(U32)
            off = r.fixed(U64)
            tensors.append((name, dims, ttype, off))
        return ver, kv, tensors, f.tell()


def serialize(ver, kv, tensors):
    out = [b"GGUF", struct.pack("<I", ver),
           struct.pack("<Q", len(tensors)), struct.pack("<Q", len(kv))]
    for k, t, v in kv:
        out.append(w_string(k) + struct.pack("<I", t) + w_value(t, v))
    for name, dims, ttype, off in tensors:
        out.append(w_string(name) + struct.pack("<I", len(dims))
                   + b"".join(struct.pack("<Q", d) for d in dims)
                   + struct.pack("<I", ttype) + struct.pack("<Q", off))
    return b"".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--set-int-array", action="append", default=[],
                    metavar="KEY=1,2,3",
                    help="replace an int32 array-valued key")
    ap.add_argument("--note", default="",
                    help="appended to general.description")
    a = ap.parse_args()
    if not a.set_int_array:
        sys.exit("nothing to change")

    ver, kv, tensors, hdr_end = read_header(a.src)
    align = next((v for k, _, v in kv if k == "general.alignment"), 32)
    data_start = (hdr_end + align - 1) // align * align

    index = {k: i for i, (k, _, _) in enumerate(kv)}
    notes = []
    for spec in a.set_int_array:
        key, _, vals = spec.partition("=")
        items = [int(x) for x in vals.split(",")]
        if key not in index:
            sys.exit(f"{key}: not present in {a.src}")
        old = kv[index[key]][2]
        kv[index[key]] = (key, ARR, (I32, items))
        notes.append(f"{key}: {old[1] if isinstance(old, tuple) else old}"
                     f" -> {items}")
        print(f"  {notes[-1]}")

    note = "; ".join(notes) + (f"; {a.note}" if a.note else "")
    di = index.get("general.description")
    if di is not None and kv[di][1] == STR:
        kv[di] = ("general.description", STR, f"{kv[di][2]} [patched: {note}]")
    else:
        kv.append(("general.description", STR, f"[patched: {note}]"))

    # The data section must land at the same alignment it had, or every tensor
    # offset in the directory becomes wrong. Header length changes, padding
    # absorbs the difference.
    head = serialize(ver, kv, tensors)
    pad = (-len(head)) % align
    head += b"\0" * pad
    print(f"  header {hdr_end} -> {len(head)} bytes, data at {data_start}")

    total = os.path.getsize(a.src)
    sfd = os.open(a.src, os.O_RDONLY)
    dfd = os.open(a.dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.write(dfd, head)
        os.lseek(sfd, data_start, os.SEEK_SET)
        pos, last = len(head), len(head)
        # Same cache discipline as stage_model.py: a plain copy of a file this
        # size gets the task killed for memory pressure while page cache fills.
        while True:
            buf = os.read(sfd, CHUNK)
            if not buf:
                break
            n = 0
            while n < len(buf):
                n += os.write(dfd, buf[n:])
            pos += len(buf)
            if pos - last >= SYNC_EVERY:
                os.fdatasync(dfd)
                os.posix_fadvise(dfd, last, pos - last, os.POSIX_FADV_DONTNEED)
                os.posix_fadvise(sfd, 0, 0, os.POSIX_FADV_DONTNEED)
                last = pos
                print(f"  {pos/1e9:.1f} / "
                      f"{(total - data_start + len(head))/1e9:.1f} GB",
                      flush=True)
        os.fdatasync(dfd)
        os.posix_fadvise(dfd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(sfd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(sfd)
        os.close(dfd)

    want = total - data_start + len(head)
    got = os.path.getsize(a.dst)
    print(f"wrote {a.dst}: {got/1e9:.2f} GB"
          + ("" if got == want else f"  SIZE MISMATCH (want {want})"))
    sys.exit(0 if got == want else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate GGML_VK_STEP_TIMING and LLAMA_DECODE_TIMING prints per depth phase.

Blocks are printed every N graphs/decodes. A block holding any n_tok >= 64 decode
is prefill; the decode phase index advances at each prefill run. Blocks mixing
prefill and decode are dropped from the decode aggregates.
  d0_tim.py <stderr-log>
"""
import sys, collections

def parse(path):
    blocks, cur = [], None
    for line in open(path, errors="replace"):
        if line.startswith("LLAMA_DECODE_TIMING ("):
            cur = {"kind": "ldt", "rows": []}; blocks.append(cur)
        elif line.startswith("VK_STEP_TIMING ("):
            cur = {"kind": "vst", "rows": []}; blocks.append(cur)
        elif line.startswith("LLAMA_DECODE_TIMING c") and cur and cur["kind"] == "ldt":
            p = line.split()
            cur["rows"].append((p[1], int(p[2]), int(p[3]), [float(x) for x in p[4:]]))
        elif line.startswith("VK_STEP_TIMING buffer_read"):
            p = line.split(); cur["rows"].append(("rd", 0, int(p[3]), [float(p[5])]))
        elif line.startswith("VK_STEP_TIMING ") and cur and cur["kind"] == "vst":
            p = line.split()
            cur["rows"].append(("g", int(p[1]), int(p[2]), [float(x) for x in p[3:]]))
    return blocks

def main():
    blocks = parse(sys.argv[1])
    phase, in_prefill = -1, False
    agg = collections.defaultdict(lambda: [0, None])
    for b in blocks:
        if b["kind"] == "ldt":
            pre = any(r[1] >= 64 for r in b["rows"])
            dec = any(r[1] < 64 for r in b["rows"])
            if pre and not in_prefill:
                phase += 1
            in_prefill = pre
            b["skip"] = pre
        else:
            # VK blocks: prefill graphs have many more nodes per graph; use the
            # preceding LLAMA block's state
            b["skip"] = in_prefill
        if b["skip"] or phase < 0:
            continue
        for tag, key, cnt, vals in b["rows"]:
            a = agg[(phase, b["kind"], tag, key)]
            a[0] += cnt
            a[1] = [x * cnt for x in vals] if a[1] is None else [s + x * cnt for s, x in zip(a[1], vals)]
    names = {"ldt": "total build+alloc set_inputs graph_compute outside(before) [built% first]",
             "vst": "gap pre rec wait sync nsub late%"}
    last = None
    for (ph, kind, tag, key), (cnt, sums) in sorted(agg.items()):
        if (ph, kind) != last:
            print(f"\n## phase {ph} {kind}: {names[kind]}"); last = (ph, kind)
        vals = [s / cnt for s in sums]
        print(f"  {tag:>3} {key:6d} cnt {cnt:6d}  " + " ".join(f"{v:8.1f}" for v in vals))

if __name__ == "__main__":
    main()

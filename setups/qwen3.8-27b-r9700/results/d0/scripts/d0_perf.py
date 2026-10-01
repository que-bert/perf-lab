#!/usr/bin/env python3
"""Aggregate GGML_VK_PERF_LOGGER (FREQUENCY=1, non-concurrent) blocks per graph kind.

Kind is taken from the FLASH_ATTN_EXT lines: q(256,N,...) gives the row count N,
the number of FA ops separates verify (16 layers) from the MTP draft (1),
k(256,KV,...) gives the depth bucket. Prints mean GPU us per graph and the
per-op breakdown (mean us per graph) for each (kind, N, depth) group.
  d0_perf.py <stderr-log> [--top 25]
"""
import re, sys, collections

FA_RE = re.compile(r"FLASH_ATTN_EXT.*?q\(\d+,(\d+),.*?k\(\d+,(\d+),")
LINE_RE = re.compile(r"^(.*): (\d+) x ([\d.e+]+) us = ([\d.e+]+) us")

def norm(name):
    name = re.sub(r"k\(\d+,\d+,", "k(*,KV,", name)
    name = re.sub(r"v\(\d+,\d+,", "v(*,KV,", name)
    name = re.sub(r"m\(\d+,", "m(KV,", name)
    return name[:110]

def blocks(path):
    cur = None
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("Vulkan Timings:"):
            cur = []
        elif line.startswith("Total time:") and cur is not None:
            yield cur, float(line.split()[2])
            cur = None
        elif cur is not None:
            m = LINE_RE.match(line)
            if m:
                cur.append((m.group(1), int(m.group(2)), float(m.group(4))))

def main():
    path = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
    groups = collections.OrderedDict()
    for ops, total in blocks(path):
        fa = [(int(m.group(1)), int(m.group(2)), cnt) for name, cnt, _ in ops for m in [FA_RE.search(name)] if m]
        if not fa:
            key = ("noFA", 0, 0)
        else:
            n, kv = fa[0][0], fa[0][1]
            nfa = sum(c for _, _, c in fa)
            kind = "prefill" if n >= 64 else ("verify" if nfa > 1 else "draft")
            key = (kind, n, ("70k" if kv < 120000 else "176k") if kind != "prefill" else 0)
        g = groups.setdefault(key, {"n": 0, "tot": 0.0, "ops": collections.Counter(), "cnt": collections.Counter()})
        g["n"] += 1; g["tot"] += total
        for name, cnt, us in ops:
            g["ops"][norm(name)] += us; g["cnt"][norm(name)] += cnt
    # merge nearby depth buckets: report per kind/N with kv range
    for (kind, n, kv), g in groups.items():
        if kind == "prefill" or g["n"] < 3:
            print(f"# {kind} n={n} kv~{kv}: {g['n']} graphs, mean {g['tot']/g['n']/1e3:.2f} ms")
            continue
        print(f"\n## {kind} n={n} kv~{kv}: {g['n']} graphs, mean GPU {g['tot']/g['n']/1e3:.3f} ms/graph")
        for name, us in g["ops"].most_common(top):
            print(f"  {us/g['n']/1e3:8.3f} ms  x{g['cnt'][name]/g['n']:5.1f}  {name}")

if __name__ == "__main__":
    main()

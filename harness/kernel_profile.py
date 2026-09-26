#!/usr/bin/env python3
"""Where does the GPU time actually go? Kernel breakdown for one llama-bench run.

rocprofv3 writes a SQLite DB; this aggregates dispatch time per kernel so an
optimization can be aimed at the kernel that owns the time rather than the one
that looks interesting. Built 2026-09-19 to answer "is Bonsai prefill a GEMM
problem or a bandwidth problem" -- it is 54% one ternary GEMM.

  kernel_profile.py --bin <dir> --model <gguf> [--backend rocm] \
      --bench-args "-ngl 99 -fa on -t 8 -p 2048 -n 64 -r 1" [--top 15]

Needs rocprofv3 on PATH. Prints total GPU kernel seconds and the top kernels by
summed duration, with call counts and share.
"""
import argparse
import pathlib
import sqlite3
import subprocess
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="llama.cpp build dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench-args", default="-ngl 99 -fa on -t 8 -p 2048 -n 64 -r 1")
    ap.add_argument("--gpu-index", default="1")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    bindir = pathlib.Path(a.bin).resolve()
    out = pathlib.Path(tempfile.mkdtemp(prefix="kprof-")) / "prof"
    cmd = [
        "rocprofv3", "--kernel-trace", "--stats", "-o", str(out), "--",
        str(bindir / "llama-bench"), "-m", a.model, *a.bench_args.split(),
    ]
    env = {
        "LD_LIBRARY_PATH": str(bindir),
        "ROCR_VISIBLE_DEVICES": a.gpu_index,
        "TMPDIR": str(pathlib.Path.home() / ".cache/llama-tmp"),
        "PATH": "/usr/bin:/bin",
    }
    print(f"$ rocprofv3 --kernel-trace --stats -- llama-bench {a.bench_args}")
    subprocess.run(cmd, env=env, check=False)

    db = sorted(out.parent.glob("prof*.db"))
    if not db:
        sys.exit(f"no rocprofv3 db in {out.parent}")
    con = sqlite3.connect(db[0])
    cur = con.cursor()
    tabs = [r[0] for r in cur.execute(
        "select name from sqlite_master where type='table'")]
    sym = next(t for t in tabs if "kernel_symbol" in t)
    disp = next(t for t in tabs if "kernel_dispatch" in t)
    total = cur.execute(f"select sum(end-start) from {disp}").fetchone()[0] or 0
    n = cur.execute(f"select count(*) from {disp}").fetchone()[0]
    rows = cur.execute(
        f"""select s.kernel_name, count(*), sum(d.end-d.start)
            from {disp} d join {sym} s on d.kernel_id=s.id
            group by s.kernel_name order by 3 desc limit ?""", (a.top,)).fetchall()
    print(f"\ntotal GPU kernel time: {total/1e9:.3f} s over {n} dispatches")
    print(f"{'kernel':62s} {'calls':>8s} {'ms':>10s} {'%':>6s}")
    for name, cnt, dur in rows:
        print(f"{name[:62]:62s} {cnt:8d} {dur/1e6:10.1f} {100*dur/total:6.1f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Acceptance cases for harness/alert.py. Run via: make test

Dry-run only. What is NOT covered here is server-side dedupe -- one Issue then
one comment -- because that needs a live token and a real repo. Treat it as
unverified until it has run against GitHub once.
"""
import atexit
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
# A private tmpdir, NOT the repo's .scratch. These cases used to share that
# directory with the running harness, so a genuine .scratch/push-failed marker
# -- exactly the state alert.py exists to report -- made every case here emit a
# publish-failure line and failed the three that assert silence.
SCRATCH = pathlib.Path(tempfile.mkdtemp(prefix="perf-lab-test-alert-"))
atexit.register(shutil.rmtree, SCRATCH, True)
NO_MARKER = SCRATCH / "absent-push-failed"
NOW = "2026-08-16T08:00:00Z"


def row(tag, key, pp, ts, rep=1, kind="nightly"):
    return {"ts": ts, "run_id": f"r-{abs(hash((tag, key, rep))):08x}"[:10],
            "kind": kind, "tag": tag, "key": key,
            "m": {"pp2048": pp, "rep": rep, "cold_prefill": True}}


def ledger(name, rows):
    SCRATCH.mkdir(exist_ok=True)
    p = SCRATCH / f"t-alert-{name}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def run(path, push_marker=NO_MARKER):
    r = subprocess.run([sys.executable, str(HERE / "alert.py"), "--dry-run",
                        "--ledger", str(path), "--now", NOW,
                        "--push-marker", str(push_marker)],
                       capture_output=True, text=True)
    return r.stdout.strip()


def ok(name, cond, got=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  got: {got}"))
    return 0 if cond else 1


fails = 0

out = run(ledger("stale", [row("n-1", "fast-q4", 710, "2026-08-12T00:00:00Z"),
                           row("n-2", "fast-q4", 708, "2026-08-13T00:00:00Z")]))
fails += ok("80h-stale ledger names the staleness issue",
            "staleness" in out and "80h" in out, out)

out = run(ledger("fresh", [row("n-1", "fast-q4", 710, "2026-08-16T04:00:00Z"),
                           row("n-2", "fast-q4", 708, "2026-08-16T06:00:00Z")]))
fails += ok("fresh healthy ledger is silent", "nothing to report" in out, out)

out = run(ledger("blip", [row("n-1", "fast-q4", 710, "2026-08-16T04:00:00Z"),
                          row("n-2", "fast-q4", 400, "2026-08-16T06:00:00Z")]))
fails += ok("single-run breach files nothing", "nothing to report" in out, out)

out = run(ledger("sustained", [row("n-1", "fast-q4", 400, "2026-08-16T04:00:00Z"),
                               row("n-2", "fast-q4", 395, "2026-08-16T06:00:00Z")]))
fails += ok("sustained breach files a canary issue",
            "canary/fast-q4" in out and "slower" in out, out)

# A config that gets unexpectedly fast is reported, not suppressed: it means
# upstream may have added kernel support for something we rejected.
out = run(ledger("fastnews", [row("n-1", "slow-q5_1", 700, "2026-08-16T04:00:00Z"),
                              row("n-2", "slow-q5_1", 705, "2026-08-16T06:00:00Z")]))
fails += ok("sustained speed-up is reported too",
            "canary/slow-q5_1" in out and "faster" in out, out)

# Not enough data is not a fault.
out = run(ledger("thin", [row("n-1", "fast-q4", 710, "2026-08-16T06:00:00Z")]))
fails += ok("insufficient data files nothing", "nothing to report" in out, out)

# An empty ledger is not a healthy one.
out = run(ledger("empty", []))
fails += ok("empty ledger reports never-ran",
            "staleness" in out and "never" in out.lower(), out)

# The push-failure marker itself was never covered, which is why the isolation
# bug stayed invisible until a real push failed. Both directions, explicitly.
healthy = ledger("marker", [row("n-1", "fast-q4", 710, "2026-08-16T04:00:00Z"),
                            row("n-2", "fast-q4", 708, "2026-08-16T06:00:00Z")])

present = SCRATCH / "present-push-failed"
present.write_text("nightly-20260817T100335Z: push failed after 2 attempts\n")
out = run(healthy, present)
fails += ok("push-failure marker is reported",
            "push" in out and "cannot publish" in out, out)

fails += ok("marker path is honoured, not the repo's own",
            "nothing to report" in run(healthy, NO_MARKER), out)

sys.exit(1 if fails else 0)

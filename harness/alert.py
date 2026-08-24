#!/usr/bin/env python3
"""Turn check.py verdicts into GitHub Issues, without becoming noise.

Four things deserve an Issue, and the last two are what make the first two
trustworthy:

  a canary breaching its band in the same direction two runs running
  a push that could not land, so measurements exist but nobody else can see them
  no successful run in 72 hours
  one canary silent for 72 hours while the others still run

The heartbeat is the whole point. A tripwire that quietly stopped running looks
exactly like a tripwire reporting good news, and without it the healthy-green
repo is indistinguishable from the dead one.

Per-canary coverage is the same failure one level down, and the global
heartbeat cannot see it: any single canary still reporting keeps the ledger
fresh. On 2026-08-22 the busy-card guard skipped four of five and check.py went
on calling all five "ok" from three-day-old numbers.

Deliberately silent on: single-run breaches (noise at this stack's spread),
"insufficient" verdicts (not enough data is not a fault), and upstream patch
drift (b10082 is pinned on purpose, so drift from HEAD is expected).

Dedupe is server-side. Each condition carries a stable key in a marker comment
in the Issue body; a matching open Issue gets a comment instead of a duplicate.
State cannot live on disk because the disk is the thing that may have died.

  alert.py --dry-run          print what would be filed, touch nothing
  alert.py                    file it (needs GH_TOKEN; the gh keyring is
                              unusable from a systemd unit)
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIVE_KINDS = {"nightly", "manual", "apt"}
MARKER = "<!-- perf-lab-key: {} -->"


def newest_success(ledger):
    """Most recent row that is an actual measurement, not a skip or a backfill."""
    newest = None
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") not in LIVE_KINDS:
            continue
        m = row.get("m", {})
        if all(m.get(k) is None for k in ("pp2048", "pp_deep", "tg", "tg_deep")):
            continue
        ts = dt.datetime.strptime(row["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc)
        if newest is None or ts > newest:
            newest = ts
    return newest


def conditions(ledger, verdicts, stale_hours, now, push_marker=None):
    out = []

    newest = newest_success(ledger)
    dark = newest is None
    if dark:
        out.append(("staleness", "perf-lab has never recorded a successful run",
                    "No row in the ledger is a completed measurement. The "
                    "harness has not produced data since it was installed."))
    else:
        age = (now - newest).total_seconds() / 3600.0
        if age > stale_hours:
            dark = True
            out.append(("staleness",
                        f"perf-lab has not run successfully in {age:.0f}h",
                        f"Newest successful measurement: {newest:%Y-%m-%d %H:%M}Z "
                        f"({age:.1f}h ago), past the {stale_hours}h heartbeat. "
                        "The tripwire is not watching anything right now."))

    # Per-canary coverage. The heartbeat above watches the ledger as a whole,
    # so a single canary that still runs keeps it quiet while the rest go dark.
    # That is not hypothetical: on 2026-08-22 the busy-card guard skipped four
    # of five, the ledger stayed fresh on the fifth, and check.py went on
    # reporting all five "ok" from numbers three days old. A stale verdict is
    # worse than a missing one, because it reads as a passing one.
    #
    # Silent when the whole lab is already dark -- repeating it once per canary
    # is noise -- and silent on "insufficient", which is not a fault.
    if not dark:
        for v in verdicts:
            if v.get("as_of") is None or v["verdict"] == "insufficient":
                continue
            seen_at = dt.datetime.strptime(v["as_of"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=dt.timezone.utc)
            age = (now - seen_at).total_seconds() / 3600.0
            if age <= stale_hours:
                continue
            missed = v.get("runs_since", 0)
            out.append((
                f"coverage/{v['key']}",
                f"perf-lab: {v['key']} has not measured in {age:.0f}h",
                f"`{v['key']}` last produced a number at "
                f"{seen_at:%Y-%m-%d %H:%M}Z ({age:.1f}h ago), past the "
                f"{stale_hours}h heartbeat, and has been absent from the last "
                f"{missed} run(s). Other canaries are still running, so the "
                "heartbeat is quiet and this one's verdict "
                f"(`{v['verdict']}`, {v['delta_pct']:+.1f}%) is being reported "
                "from stale data.\n\nUsual cause: the busy-card guard skipped "
                "it because something else held VRAM on the target GPU. Check "
                "the `skipped` rows' `reason` field."))

    for v in verdicts:
        if not v.get("sustained"):
            continue
        way = "slower" if v["verdict"] == "slow" else "faster"
        body = (f"`{v['key']}` measured {v['median_pp2048']} t/s against an "
                f"expected {v['expect']} t/s ({v['delta_pct']:+.1f}%), "
                f"{way} than its band allows, in two consecutive runs.\n\n"
                f"run_ids: {', '.join(v['run_ids'])}")
        if v["verdict"] == "fast":
            body += ("\n\nA canary getting faster is not a fault. This one is a "
                     "known-unsupported config, so upstream may have added "
                     "kernel support and a previously rejected operating point "
                     "is worth retesting.")
        out.append((f"canary/{v['key']}",
                    f"perf-lab: {v['key']} {way} than its band "
                    f"({v['delta_pct']:+.1f}%)", body))

    # Injectable so the tests never read the real repo's marker. It is live
    # process state, not fixture state: a genuine push failure used to leak
    # into every case here and fail the three that assert silence.
    marker = (pathlib.Path(push_marker) if push_marker
              else ROOT / ".scratch" / "push-failed")
    if marker.exists():
        out.append(("push", "perf-lab cannot publish its ledger",
                    "Measurements are on disk but the push failed:\n\n```\n"
                    f"{marker.read_text().strip()}\n```"))
    return out


def gh(args, token):
    env = dict(os.environ)
    if token:
        env["GH_TOKEN"] = token
    r = subprocess.run(["gh"] + args, capture_output=True, text=True, env=env,
                       cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(f"alert: gh {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout


def open_issues(token):
    out = gh(["issue", "list", "--state", "open", "--limit", "100",
              "--json", "number,body"], token)
    return json.loads(out or "[]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=str(ROOT / "results" / "ledger.jsonl"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "canary.yaml"))
    ap.add_argument("--stale-hours", type=float, default=72.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--now", default="", help="override current time (tests)")
    ap.add_argument("--push-marker", default="",
                    help="override the push-failure marker path (tests)")
    a = ap.parse_args()

    now = (dt.datetime.strptime(a.now, "%Y-%m-%dT%H:%M:%SZ")
           .replace(tzinfo=dt.timezone.utc) if a.now
           else dt.datetime.now(dt.timezone.utc))

    check = subprocess.run([sys.executable, str(ROOT / "harness" / "check.py"),
                            "--ledger", a.ledger, "--config", a.config],
                           capture_output=True, text=True)
    if check.returncode != 0:
        sys.exit(f"alert: check.py failed:\n{check.stderr.strip()}")
    verdicts = json.loads(check.stdout)["verdicts"]

    found = conditions(pathlib.Path(a.ledger), verdicts, a.stale_hours, now,
                       a.push_marker or None)
    if not found:
        print("alert: nothing to report")
        return

    token = os.environ.get("GH_TOKEN", "")
    if not a.dry_run and not token:
        sys.exit("alert: GH_TOKEN is not set. A systemd unit cannot unlock the "
                 "gh keyring, so the token must come from ~/.perf-lab/env.")

    existing = [] if a.dry_run else open_issues(token)
    for key, title, body in found:
        marker = MARKER.format(key)
        match = next((i for i in existing if marker in (i.get("body") or "")), None)
        if a.dry_run:
            print(f"[dry-run] would file  {key}: {title}")
            continue
        if match:
            gh(["issue", "comment", str(match["number"]), "--body",
                f"Still true as of {now:%Y-%m-%d %H:%M}Z.\n\n{body}"], token)
            print(f"commented on #{match['number']}  {key}")
        else:
            gh(["issue", "create", "--title", title,
                "--body", f"{body}\n\n{marker}"], token)
            print(f"opened issue  {key}: {title}")


if __name__ == "__main__":
    main()

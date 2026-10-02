#!/usr/bin/env python3
"""Who is using the card (PERFLAB_GPU_PCI, default the R9700), from /proc/<pid>/fdinfo (DRM client stats).

  gpu_holders.py list                       one line per DRM client on the card
  gpu_holders.py busy [--allow PID...]      exit 1 if a foreign client holds the card
  gpu_holders.py sample OUT [--allow-tree PID] [--iv S]   append samples until killed
  gpu_holders.py verdict OUT                CLEAN / VOID <reason> from a sample file

"Holds" = more than FOREIGN_VRAM_MIB of VRAM, or gfx/compute engine time growing by more than
FOREIGN_BUSY_PCT of wall time between samples. Discord/xdg-desktop-portal keep a 12 KiB handle
open all day; that is not a holder. Our own processes (llama-*, test-backend-ops, descendants of
--allow-tree) are excluded by comm and by process tree; serve_unit.sh starts servers under
systemd --user, so the comm rule is what catches them.
"""
import os, sys, time, json

PCI = os.environ.get("PERFLAB_GPU_PCI", "0000:0c:00.0")
FOREIGN_VRAM_MIB = float(os.environ.get("PERFLAB_FOREIGN_VRAM_MIB", "256"))
FOREIGN_BUSY_PCT = float(os.environ.get("PERFLAB_FOREIGN_BUSY_PCT", "2"))
MAX_LOAD = float(os.environ.get("PERFLAB_MAXLOAD", "4"))  # 1-min loadavg; launch-bound models feel a build
OURS = ("llama-", "test-backend-o")
IDLE_VRAM_MIB = float(os.environ.get("PERFLAB_IDLE_VRAM_MIB", "1024"))
# Lane kind (gpu_lock.sh exports it): "timing" voids on any foreign activity (the R9700 default);
# "correctness" never voids and never waits on desktop baseline clients, it only logs foreign activity.
# Baseline = the desktop's own DRM clients (comm prefixes; comm is truncated to 15 chars by the kernel).
LANE_KIND = os.environ.get("PERFLAB_LANE_KIND", "timing")
BASELINE = tuple(x for x in os.environ.get(
    "PERFLAB_BASELINE_COMMS",
    "Xwayland,Xorg,gnome-shell,gnome-remote-d,mutter,ghostty,steam,steamwebhelper,firefox,Discord,discord,"
    "gjs,xdg-desktop-por,plasmashell,kwin,chrome,Isolated Web,WebExtensions,Web Content,RDD Process,"
    "gnome-software,nautilus,code,electron,Hyprland,sway,pipewire,wireplumber").split(",") if x)


def card_mem(kind):
    """Card-wide bytes in use: kind = vram | gtt."""
    try:
        return int(open(f"/sys/bus/pci/devices/{PCI}/mem_info_{kind}_used").read())
    except OSError:
        return 0


def render_node():
    p = f"/dev/dri/by-path/pci-{PCI}-render"
    return os.path.realpath(p)


def clients():
    """{(pid, client_id): {comm, vram_kib, engine_ns}} for clients on our card."""
    node, out = render_node(), {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(f"/proc/{pid}/fd/{fd}") != node:
                    continue
                info = open(f"/proc/{pid}/fdinfo/{fd}").read()
            except OSError:
                continue
            d = {}
            for line in info.splitlines():
                k, _, v = line.partition(":")
                d[k.strip()] = v.strip()
            if d.get("drm-pdev") != PCI:
                continue
            key = (int(pid), d.get("drm-client-id", fd))
            if key in out:
                continue
            vram = d.get("drm-memory-vram", "0 KiB").split()
            kib = float(vram[0]) * {"KiB": 1, "MiB": 1024, "GiB": 1 << 20}.get(vram[1] if len(vram) > 1 else "KiB", 1)
            eng = sum(int(v.split()[0]) for k, v in d.items() if k.startswith("drm-engine-") and k != "drm-engine-capacity" and v.split()[0].isdigit())
            try:
                comm = open(f"/proc/{pid}/comm").read().strip()
            except OSError:
                comm = "?"
            out[key] = {"comm": comm, "vram_kib": kib, "engine_ns": eng}
    return out


def descendants(root):
    kids = {}
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                ppid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(ppid, []).append(int(pid))
    seen, stack = set(), [root]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(kids.get(p, []))
    return seen


def ours(pid, comm, allow):
    return comm.startswith(OURS) or pid in allow


def baseline(comm):
    return comm.startswith(BASELINE)


def foreign_now(allow, iv=1.0):
    a = clients()
    time.sleep(iv)
    b = clients()
    bad = []
    for k, v in b.items():
        if ours(k[0], v["comm"], allow):
            continue
        busy = 100.0 * (v["engine_ns"] - a.get(k, v)["engine_ns"]) / (iv * 1e9)
        if v["vram_kib"] / 1024 > FOREIGN_VRAM_MIB or busy > FOREIGN_BUSY_PCT:
            bad.append(f"{v['comm']}[{k[0]}] vram={v['vram_kib']/1024:.0f}MiB busy={busy:.1f}%")
    return bad


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    args = sys.argv[2:]
    if cmd == "list":
        for (pid, cid), v in sorted(clients().items()):
            print(f"{pid:>8} {v['comm']:<20} vram={v['vram_kib']/1024:9.1f}MiB engine_ns={v['engine_ns']}")
    elif cmd == "busy":
        allow = {int(x) for x in args if x.isdigit()}
        bad = foreign_now(allow)
        load = float(open("/proc/loadavg").read().split()[0])
        if LANE_KIND == "correctness":
            # only a non-desktop foreign client (a game, a sibling's server) makes a correctness run wait
            bad = [b for b in bad if not baseline(b.split("[")[0])]
            if bad:
                print("busy: " + "; ".join(bad))
                sys.exit(1)
            print("free")
            return
        if load >= MAX_LOAD:
            bad.append(f"loadavg={load}")
        # nothing of ours runs between locked runs, so the card must have drained: a just-exited model's
        # VRAM still being freed makes the next run start in GTT (fork prefill then ramps 680 -> 1600 t/s)
        if not any(ours(k[0], v["comm"], allow) for k, v in clients().items()):
            used = card_mem("vram") / 2**20
            if used > IDLE_VRAM_MIB:
                bad.append(f"card vram {used:.0f}MiB not drained")
        if bad:
            print("busy: " + "; ".join(bad))
            sys.exit(1)
        print("free")
    elif cmd == "sample":
        out, iv, tree = args[0], 2.0, None
        if "--iv" in args:
            iv = float(args[args.index("--iv") + 1])
        if "--allow-tree" in args:
            tree = int(args[args.index("--allow-tree") + 1])
        prev = clients()
        with open(out, "a") as f:
            while True:
                time.sleep(iv)
                if tree and not os.path.exists(f"/proc/{tree}"):
                    return  # the gpu_lock.sh that started us is gone (killed): do not hold the lock fd forever
                cur = clients()
                allow = descendants(tree) if tree else set()
                for k, v in cur.items():
                    if ours(k[0], v["comm"], allow):
                        continue
                    busy = 100.0 * (v["engine_ns"] - prev.get(k, v)["engine_ns"]) / (iv * 1e9)
                    f.write(json.dumps({"t": round(time.time(), 1), "pid": k[0], "comm": v["comm"],
                                        "vram_mib": round(v["vram_kib"] / 1024, 1), "busy_pct": round(busy, 2)}) + "\n")
                f.write(json.dumps({"t": round(time.time(), 1), "load": float(open("/proc/loadavg").read().split()[0]),
                                    "vram_gib": round(card_mem("vram") / 2**30, 2), "gtt_gib": round(card_mem("gtt") / 2**30, 2)}) + "\n")
                f.flush()
                prev = cur
    elif cmd == "verdict":
        bad, loadmax, vmax, gmax = {}, 0.0, 0.0, 0.0
        try:
            for line in open(args[0]):
                r = json.loads(line)
                if "load" in r:
                    loadmax = max(loadmax, r["load"])
                    vmax, gmax = max(vmax, r.get("vram_gib", 0)), max(gmax, r.get("gtt_gib", 0))
                    continue
                if r["vram_mib"] > FOREIGN_VRAM_MIB or r["busy_pct"] > FOREIGN_BUSY_PCT:
                    b = bad.setdefault(f"{r['comm']}[{r['pid']}]", [0, 0.0, 0.0])
                    b[0] += 1
                    b[1] = max(b[1], r["vram_mib"])
                    b[2] = max(b[2], r["busy_pct"])
        except FileNotFoundError:
            pass
        msg = "; ".join(f"{k} n={v[0]} vram<={v[1]:.0f}MiB busy<={v[2]:.1f}%" for k, v in bad.items())
        if LANE_KIND == "correctness":
            print(f"CLEAN(correctness lane) maxload={loadmax} vram<={vmax}GiB gtt<={gmax}GiB" + (f" foreign-logged: {msg}" if msg else ""))
            return
        if loadmax >= 1.5 * MAX_LOAD:
            msg = (msg + "; " if msg else "") + f"loadavg<={loadmax}"
        if msg:
            print("VOID " + msg)
            sys.exit(1)
        print(f"CLEAN maxload={loadmax} vram<={vmax}GiB gtt<={gmax}GiB")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

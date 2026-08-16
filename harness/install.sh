#!/usr/bin/env bash
# Install the perf-lab timers into the user's systemd instance.
#
#   install.sh              install and start the timers
#   install.sh --uninstall  stop and remove them
#
# What this script will NOT do is call sudo. Two steps need root -- enabling
# lingering and placing the apt hook -- and both are printed for you to run
# and read first. An installer that silently escalates is a worse trade than
# two lines of copy-paste.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
UNITS="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TIMERS=(perf-lab-nightly.timer perf-lab-drain.timer perf-lab-heartbeat.timer)
QUEUE_DIR=/var/lib/perf-lab
HOOK=/etc/apt/apt.conf.d/99-perf-lab

if [[ "${1:-}" == "--uninstall" ]]; then
  for t in "${TIMERS[@]}"; do
    systemctl --user disable --now "$t" 2>/dev/null || true
    rm -f "$UNITS/$t"
  done
  rm -f "$UNITS/perf-lab@.service"
  systemctl --user daemon-reload
  echo "removed user units."
  echo
  echo "Root-owned pieces are left in place. To remove them too:"
  echo "  sudo rm -f $HOOK"
  echo "  sudo rm -rf $QUEUE_DIR"
  echo "  sudo loginctl disable-linger $USER"
  exit 0
fi

[[ -r "$HOME/.perf-lab/env" ]] || {
  echo "install: $HOME/.perf-lab/env is missing." >&2
  echo "It must define PERF_LAB_MODEL, PERF_LAB_BIN, PERF_LAB_GPU_UID, and" >&2
  echo "GH_TOKEN. The units read it because a systemd unit cannot unlock the" >&2
  echo "gh keyring and must not have private paths baked into the repo." >&2
  exit 3
}

mkdir -p "$UNITS"
sed -e "s|@REPO@|$ROOT|g" -e "s|@HOME@|$HOME|g" \
  "$ROOT/systemd/perf-lab@.service" > "$UNITS/perf-lab@.service"
for t in "${TIMERS[@]}"; do
  install -m 644 "$ROOT/systemd/$t" "$UNITS/$t"
done

systemctl --user daemon-reload
for t in "${TIMERS[@]}"; do
  systemctl --user enable --now "$t"
done

echo
systemctl --user list-timers --no-pager 'perf-lab-*' || true

# --- what is still missing -------------------------------------------------
echo
MISSING=0

if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]]; then
  MISSING=1
  echo "NOT ARMED: lingering is off, so these timers do not fire while you are"
  echo "logged out. Unattended detection is the entire premise, so until this"
  echo "runs the repo is measuring only when you happen to be at the machine:"
  echo
  echo "  sudo loginctl enable-linger $USER"
  echo
fi

if [[ ! -d "$QUEUE_DIR" || ! -r "$HOOK" ]]; then
  MISSING=1
  echo "NOT ARMED: the apt hook is not installed, so a driver upgrade will not"
  echo "queue a run. The nightly still works; you just lose the trigger that"
  echo "catches a change at the moment it lands:"
  echo
  echo "  sudo install -d -o $USER -g $USER -m 755 $QUEUE_DIR"
  echo "  sudo install -m 644 $ROOT/apt/99-perf-lab $HOOK"
  echo
fi

if ! grep -q '^GH_TOKEN=..' "$HOME/.perf-lab/env" 2>/dev/null; then
  MISSING=1
  echo "NOT ARMED: GH_TOKEN is unset in ~/.perf-lab/env, so the heartbeat will"
  echo "detect conditions and then fail to file them. repo scope is enough;"
  echo "it deliberately does not need delete_repo."
  echo
fi

(( MISSING )) || echo "all three triggers armed."

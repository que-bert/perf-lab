#!/usr/bin/env bash
# Free the R9700: stop every perflab server unit and any stray llama-server we
# started, then confirm the card is actually released. Does NOT touch ollama --
# it is a separate service on GPU[0] and not ours to stop.
set -u
echo "=== perflab units before ==="
systemctl --user list-units --all 'perflab-srv-*' --no-pager 2>/dev/null | head -12

for u in $(systemctl --user list-units --all --plain --no-legend 'perflab-srv-*' 2>/dev/null | awk '{print $1}'); do
  echo "stopping $u"
  systemctl --user stop "$u" 2>/dev/null
  systemctl --user reset-failed "$u" 2>/dev/null
done

# anything we launched outside systemd on our test ports
for p in 8919 8921 8931 8941 8951 8961 8971; do
  pkill -f "llama-server.*--port $p" 2>/dev/null && echo "killed stray server on port $p"
done

timeout 10 tail -f /dev/null

echo ""
echo "=== VRAM after unload ==="
rocm-smi --showmeminfo vram 2>&1 | grep -E 'GPU\[[01]\].*Used'
python3 -c "
import subprocess,re
o=subprocess.run(['rocm-smi','--showmeminfo','vram'],capture_output=True,text=True).stdout
m=[int(x) for x in re.findall(r'GPU\[1\].*?Used Memory \(B\): (\d+)', o)] or \
  [int(x) for x in re.findall(r'GPU\[1\][^\n]*?(\d+)\s*$', o, re.M)]
if m:
    a=m[-1]; t=34208743424
    print(f'  R9700: {a/1e6:.0f} MB used of 34,209 MB  ({100*a/t:.1f}%)')
    print('  CLEAN' if a < 1.5e9 else '  STILL OCCUPIED -- something else holds the card')
"
echo ""
echo "=== remaining llama-server processes (ollama's is expected) ==="
pgrep -af llama-server | grep -v 'pgrep\|bash -c' | sed 's/\(.\{140\}\).*/\1.../' || echo "  none"
echo "UNLOAD DONE"

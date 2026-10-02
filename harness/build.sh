#!/usr/bin/env bash
# Build a fork worktree outside the measurement window.
#
#   build.sh <worktree> [target...]     default targets: llama-server llama-bench llama-perplexity llama-decode-kld test-backend-ops
#
# Waits (flock -s /tmp/perflab-cpu.lock) while a timing run holds the CPU lock exclusively, and holds it shared while
# compiling, so the next timing run waits for the build instead of overlapping it (gpu_lock.sh timing = exclusive,
# correctness = shared). nice 19 + ionice idle. -j = min(12, nproc - 8, nproc - loadavg), at least 2.
# BUILD_NOLOCK=1 skips the lock; BUILD_J overrides -j. The worktree must already be configured (build/CMakeCache.txt).
set -u
W=${1:?worktree}; shift
T=("$@"); [ ${#T[@]} = 0 ] && T=(llama-server llama-bench llama-perplexity llama-decode-kld test-backend-ops)
[ -f "$W/build/CMakeCache.txt" ] || { echo "build.sh: $W/build is not configured (cmake -S $W -B $W/build ...)" >&2; exit 2; }
n=$(nproc); load=$(cut -d. -f1 /proc/loadavg)
j=${BUILD_J:-$(( n - 8 < 12 ? n - 8 : 12 ))}
[ -z "${BUILD_J:-}" ] && [ $((n - load)) -lt $j ] && j=$((n - load))
[ "$j" -lt 2 ] && j=2
if [ -z "${BUILD_NOLOCK:-}" ]; then
  exec 9>/tmp/perflab-cpu.lock
  flock -n -s 9 || { echo "build.sh: a timing run holds the cpu lock; waiting" >&2; flock -s 9; }
fi
echo "build.sh: $W -j$j ${T[*]} (load $(cut -d' ' -f1-3 /proc/loadavg))"
exec nice -n 19 ionice -c3 cmake --build "$W/build" -j"$j" --target "${T[@]}"

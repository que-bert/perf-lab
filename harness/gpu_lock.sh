#!/usr/bin/env bash
# Serialize every R9700 measurement across the main session and sub-agents.
# Parallel agents may write and build code, but only one process may touch the
# GPU at a time; a concurrent benchmark silently corrupts both numbers.
#
#   gpu_lock.sh <command...>      # blocks until the lock is free, then runs
exec flock /tmp/perflab-r9700.lock "$@"

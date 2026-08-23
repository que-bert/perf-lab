#!/usr/bin/env python3
"""Acceptance cases for the build fingerprint in harness/emit_row.py.

Two failures shipped together and this file exists to keep either from coming
back:

  1. `--version` parsing recognised only the b10082 banner, so every Vulkan row
     was refused outright -- `emit_row: could not determine llama.cpp build
     SHA`. The b10472 numbers lived in FINDINGS.md and never reached the
     ledger.
  2. The `libggml*.so.[0-9]*` glob matched libggml.so and libggml-base.so but
     never libggml-hip.so or libggml-vulkan.so, because the compute backend
     carries no version suffix. The one file implementing the kernels this repo
     watches was outside the fingerprint, on BOTH builds.

No GPU and no real build are needed: the banner parse is a pure function, and
the digest runs against a synthetic bindir of stub files.
"""
import atexit
import importlib.util
import pathlib
import shutil
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("emit_row", HERE / "emit_row.py")
emit_row = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emit_row)

TMP = pathlib.Path(tempfile.mkdtemp(prefix="perf-lab-test-emit-row-"))
atexit.register(shutil.rmtree, TMP, True)
# Never the repo's .scratch: sha256_cached keys on (path, size, mtime) and
# these stubs would otherwise land in the same cache the live harness reads.
emit_row.HERE = TMP / "harness"
emit_row.HERE.mkdir(parents=True, exist_ok=True)

B10082 = "version: 10082 (fb0e6b621)\nbuilt with GNU 11.4.0 for Linux x86_64\n"
B10472 = ("WARNING: radv is not a conformant Vulkan implementation, testing use only.\n"
          "version: 0.1.1-dev (build 10472, commit 60eeeb608)\n"
          "built with GNU 11.4.0 for Linux x86_64\n")

cases = []


def check(name, got, want):
    cases.append((name, got == want, f"got {got!r}, want {want!r}"))


check("b10082 banner parses", emit_row.parse_version_sha(B10082), "fb0e6b621")
check("b10472 banner parses", emit_row.parse_version_sha(B10472), "60eeeb608")
check("unknown banner yields None",
      emit_row.parse_version_sha("version: potato\n"), None)
# A row is better refused than fingerprinted with a commit nobody printed.
check("bare build number is not a commit",
      emit_row.parse_version_sha("version: 0.1.1-dev (build 10472)\n"), None)


def bindir(name, backend_body):
    """A stub build: shim binary, impl .so, an aliased core lib, one backend."""
    d = TMP / name
    d.mkdir()
    exe = d / "llama-bench"
    exe.write_text("#!/bin/sh\ncat <<'EOF'\n" + B10472 + "EOF\n")
    exe.chmod(0o755)
    (d / "libllama-bench-impl.so").write_text("impl")
    core = d / "libggml.so.0.20.1"
    core.write_text("core")
    (d / "libggml.so.0").symlink_to(core.name)
    (d / "libggml.so").symlink_to(core.name)
    base = d / "libggml-base.so.0.20.1"
    base.write_text("base")
    (d / "libggml-base.so.0").symlink_to(base.name)
    # The backend. Unversioned, and the whole point of the second bug.
    (d / "libggml-vulkan.so").write_text(backend_body)
    return d


sha_a, toolchain = emit_row.build_identity(bindir("a", "backend-v1"), "llama-bench")
sha_b, _ = emit_row.build_identity(bindir("b", "backend-v2-rebuilt"), "llama-bench")

check("digest is a sha256", bool(sha_a) and len(sha_a) == 64, True)
check("toolchain read from banner", toolchain, "GNU 11.4.0")
# The regression: before the fix these two digests were identical, because the
# only file that differed was never hashed.
check("rebuilt backend changes the digest", sha_a != sha_b, True)

# De-aliasing: an extra symlink to a file already counted must not move the
# digest. A packager shipping one more alias is not a new build.
extra = bindir("c", "backend-v1")
(extra / "libggml.so.0.20").symlink_to("libggml.so.0.20.1")
sha_c, _ = emit_row.build_identity(extra, "llama-bench")
check("extra symlink alias does not change the digest", sha_c, sha_a)

# An empty directory has nothing to say, and must say so rather than emit the
# sha256 of no bytes -- which is a real, constant, entirely wrong digest.
empty = TMP / "empty"
empty.mkdir()
check("empty bindir yields no digest",
      emit_row.build_identity(empty, "llama-bench"), (None, None))

# The backend label. probe.sh always defaulted to rocm and emit_row never
# passed --backend, so the first Vulkan fingerprint came out claiming ROCm --
# pointing any future investigation at the wrong layer of the stack.
check("vulkan build labelled vulkan",
      emit_row.backend_for(TMP / "a"), "vulkan")

hip = TMP / "hip"
hip.mkdir()
(hip / "libggml-hip.so").write_text("hip")
check("rocm build labelled rocm", emit_row.backend_for(hip), "rocm")
check("backendless build labelled cpu", emit_row.backend_for(empty), "cpu")

both = TMP / "both"
both.mkdir()
(both / "libggml-hip.so").write_text("hip")
(both / "libggml-vulkan.so").write_text("vulkan")
try:
    got = emit_row.backend_for(both)
except SystemExit:
    got = "refused"
check("ambiguous build is refused, not guessed", got, "refused")

fails = 0
for name, ok, detail in cases:
    fails += 0 if ok else 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<44} {'' if ok else detail}")
sys.exit(1 if fails else 0)

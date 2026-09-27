#!/usr/bin/env bash
# Build the fixed decode-KLD corpus (P0.5) deterministically.
#
#   build_corpus.sh [out]      # default: harness/corpus/decode_kld.txt
#
# Sources, in fixed sorted order, each file prefixed by a "==== <name> ====" line:
#   1. llama.cpp docs/**/*.md, README.md, src/*.cpp  -- read from git at the pinned
#      commit (not the working tree), so edits to a checkout cannot change it
#   2. /usr/share/common-licenses/*                 -- legal prose
#   3. python stdlib top-level *.py                  -- code
# taken round-robin one file per genre, then truncated to exactly MAX_BYTES. Diversity (prose / markdown / C++ / Python /
# legal) keeps the scored windows at 70k and 176k tokens from being one genre.
# 1.5 MB -> >190k tokens; the tool needs depth+score+1 = 176513 tokens at the deepest gate.
# The output sha256 is recorded in results/p0/kld-base/README.md; a different sha
# means the base logits are not comparable.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/decode_kld.txt}"
REPO="${LLAMA_REPO:-$HOME/git/llama.cpp-r9700}"
COMMIT="${LLAMA_COMMIT:-26bd56621}"
PYLIB="${PYLIB:-$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["stdlib"])')}"
MAX_BYTES="${MAX_BYTES:-1500000}"
export LC_ALL=C

tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
emit_git() { printf '\n==== %s ====\n' "$1" >>"$tmp"; git -C "$REPO" show "$COMMIT:$1" >>"$tmp"; }

mapfile -t G < <(git -C "$REPO" ls-tree -r --name-only "$COMMIT" | grep -E '^(README\.md|docs/.*\.md|src/[^/]*\.cpp)$' | sort)
mapfile -t L < <(find /usr/share/common-licenses -maxdepth 1 -type f -printf '%f\n' | sort)
mapfile -t P < <(find "$PYLIB" -maxdepth 1 -name '*.py' -printf '%f\n' | sort)
# round-robin one file from each genre so any 512-token window neighbourhood is mixed
n=${#G[@]}; [ ${#L[@]} -gt $n ] && n=${#L[@]}; [ ${#P[@]} -gt $n ] && n=${#P[@]}
for ((i = 0; i < n; i++)); do
    [ $i -lt ${#G[@]} ] && emit_git "${G[$i]}"
    if [ $i -lt ${#L[@]} ]; then printf '\n==== license/%s ====\n' "${L[$i]}" >>"$tmp"; cat "/usr/share/common-licenses/${L[$i]}" >>"$tmp"; fi
    if [ $i -lt ${#P[@]} ]; then printf '\n==== python/%s ====\n' "${P[$i]}" >>"$tmp"; cat "$PYLIB/${P[$i]}" >>"$tmp"; fi
done
head -c "$MAX_BYTES" "$tmp" >"$OUT"
echo "wrote $OUT: $(wc -c <"$OUT") bytes, sha256 $(sha256sum "$OUT" | cut -d' ' -f1)"

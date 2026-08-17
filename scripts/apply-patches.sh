#!/usr/bin/env bash
# Apply the vendored-submodule patches under patches/<submodule>/*.patch.
#
# Why: we never edit the Megatron-LM submodule directly; local changes live as patch files and are
# applied fresh onto the pinned checkout. Anything that imports megatron.core (training, the losses
# tests) needs the patches applied first.
#
# Idempotent by RESET: each submodule is reset to the commit the project records for it
# (`git rev-parse HEAD:<submodule>`) before its patches are (re)applied, so this is safe to rerun
# in any state -- already-patched, half-patched, with a dirty index, or after a submodule update.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
patches_root="$repo_root/patches"

if [[ ! -d "$patches_root" ]]; then
    echo "[apply-patches] no patches/ directory - nothing to do"
    exit 0
fi

shopt -s nullglob
applied=0

for submodule_dir in "$patches_root"/*/; do
    submodule="$(basename "$submodule_dir")"
    target="$repo_root/$submodule"
    if [[ ! -d "$target/.git" && ! -f "$target/.git" ]]; then
        echo "[apply-patches] WARNING: $submodule is not a git checkout at $target - skipping" >&2
        continue
    fi

    patches=("$submodule_dir"*.patch)
    if [[ ${#patches[@]} -eq 0 ]]; then
        continue
    fi

    # Reset to the pinned commit first, so patches always apply onto a clean tree (idempotency).
    pin="$(git -C "$repo_root" rev-parse "HEAD:$submodule" 2>/dev/null || true)"
    if [[ -z "$pin" ]]; then
        echo "[apply-patches] ERROR: $submodule is not a submodule of this repo (no gitlink at HEAD)." >&2
        exit 1
    fi
    if ! git -C "$target" cat-file -e "$pin^{commit}" 2>/dev/null; then
        echo "[apply-patches] ERROR: $submodule does not contain its pinned commit $pin." >&2
        echo "  Run: git submodule update --init $submodule" >&2
        exit 1
    fi
    git -C "$target" reset --hard "$pin" >/dev/null
    echo "[apply-patches] reset $submodule to pinned commit ${pin:0:9}"

    for patch in "${patches[@]}"; do
        name="$submodule/$(basename "$patch")"
        if git -C "$target" apply --check "$patch" >/dev/null 2>&1; then
            git -C "$target" apply "$patch"
            echo "[apply-patches] applied: $name"
            applied=$((applied + 1))
        else
            echo "[apply-patches] ERROR: $name does not apply cleanly onto $submodule." >&2
            echo "  The submodule may have been bumped; relocate the patch (see its header)." >&2
            exit 1
        fi
    done
done

echo "[apply-patches] done: $applied applied"

#!/usr/bin/env bash
# Apply the vendored-submodule patches under patches/<submodule>/*.patch.
#
# Why: we never edit the Megatron-LM submodule directly; local changes live as patch files and are
# applied fresh onto the pinned checkout. Anything that imports megatron.core (training, the losses
# tests) needs the patches applied first.
#
# Idempotent by RESET: each submodule's tracked files are reset to the pinned commit
# (git checkout -- .) before its patches are (re)applied, so this is safe to rerun in any state --
# already-patched, half-patched, or after a submodule update. Mirrors the reference project's
# `git checkout -- . && git apply patches/*.patch`. (We never edit the submodule by hand, so the
# reset only ever discards a previous run's patches, never real work.)
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
    git -C "$target" checkout -- .
    echo "[apply-patches] reset $submodule to pinned commit"

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

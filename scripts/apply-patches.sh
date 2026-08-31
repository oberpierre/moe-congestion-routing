#!/usr/bin/env bash
# Apply the vendored-submodule patches under patches/<submodule>/*.patch.
#
# Why: we never edit the Megatron-LM submodule directly; local changes live as patch files and are
# applied fresh onto the pinned checkout. Anything that imports megatron.core (training, the losses
# tests) needs the patches applied first.
#
# Idempotent by RESET: each submodule is reset to the commit the project records for it
# (`git rev-parse HEAD:<submodule>`) before its patches are (re)applied, so this is safe to rerun
# in any state: already-patched, half-patched, with a dirty index, or after a submodule update.
#
# That reset is destructive while it runs, so two of these racing on one shared checkout leave a
# window where the tree on disk is the unpinned original. A job importing megatron.core in that
# window gets the unpatched router and, on a none/aux_loss arm, trains to completion writing no
# probes at all. Pass --verify to check the patches are present without touching anything, which
# is what a job on a shared workdir should do.
set -euo pipefail

mode=apply
case "${1:-}" in
    --verify) mode=verify ;;
    "") ;;
    *) echo "usage: $(basename "$0") [--verify]" >&2; exit 2 ;;
esac

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

    if [[ "$mode" == verify ]]; then
        # The marker is the longest line each patch adds, taken from the patch itself rather than
        # from a list here, so a patch added later is checked without anyone remembering to.
        for patch in "${patches[@]}"; do
            name="$submodule/$(basename "$patch")"
            found="$(awk '/^\+\+\+ / {f=substr($0,7); next} /^\+/ {line=substr($0,2); if (length(line) > maxlen) {maxlen=length(line); best=line; bestf=f}} END {print bestf "\t" best}' "$patch")"
            marker_file="${found%%$'\t'*}"
            marker="${found#*$'\t'}"
            if [[ -z "$marker_file" || -z "$marker" ]]; then
                echo "[apply-patches] ERROR: $name adds no lines, so it cannot be verified." >&2
                exit 1
            fi
            if grep -qF -- "$marker" "$target/$marker_file"; then
                applied=$((applied + 1))
            else
                echo "[apply-patches] ERROR: $name is NOT applied to $submodule." >&2
                echo "  Run ./scripts/apply-patches.sh once, on an idle workdir, before launching." >&2
                exit 1
            fi
        done
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

if [[ "$mode" == verify ]]; then
    echo "[apply-patches] verified: $applied patches present, nothing modified"
else
    echo "[apply-patches] done: $applied applied"
fi

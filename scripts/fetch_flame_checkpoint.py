#!/usr/bin/env python
"""Fetch iteration(s) of ``CMU-FLAME/FLAME-MoE-290M-1.3B`` into ``$DATA_STORE`` so we can
score them using ``configs/eval/flame.yaml``.

Each iteration directory on the hub carries ``.distcp`` shards, one ``.metadata`` file and
its own ``common.pt`` (a pickled Megatron argparse namespace plus optimizer/scheduler state) and
``metadata.json``.

Usage:
    uv run python scripts/fetch_flame_checkpoint.py --dry-run        # list files, fetch nothing
    uv run python scripts/fetch_flame_checkpoint.py --metadata-only  # cheap: common.pt + metadata
    uv run python scripts/fetch_flame_checkpoint.py                  # the full iter_0005473, ~17GiB
    uv run python scripts/fetch_flame_checkpoint.py --iteration 5473 540  # + a trajectory point
    uv run python scripts/fetch_flame_checkpoint.py --show-args      # architecture already on disk
"""

import argparse
from pathlib import Path

REPO_ID = "CMU-FLAME/FLAME-MoE-290M-1.3B"

# Files fetched by --metadata-only: enough to read the architecture back out (common.pt) and to
# know which training step produced it (metadata.json), without the 64 weight shards.
_METADATA_FILENAMES = ("common.pt", "metadata.json")

# A representative slice of common.pt's saved argparse namespace, read once against
# iter_0005473. --show-args may be used to sanity check a downloaded checkpoints metadata.
_SHOW_ARGS_FIELDS = (
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "ffn_hidden_size",
    "moe_ffn_hidden_size",
    "num_experts",
    "moe_router_topk",
    "moe_shared_expert_intermediate_size",
    "moe_router_pre_softmax",
    "padded_vocab_size",
    "norm_epsilon",
    "tokenizer_type",
    "tokenizer_model",
    "expert_model_parallel_size",
    "ckpt_format",
)


def _data_store() -> Path:
    """Resolve ``$DATA_STORE`` like other config does using (``paths.expand_path``)"""
    from moe_congestion_routing.paths import expand_path

    return Path(expand_path("$DATA_STORE"))


def _iter_name(iteration: int) -> str:
    return f"iter_{iteration:07d}"


def _target_root(data_store: Path) -> Path:
    return data_store / "models" / REPO_ID


def _list_files(iteration: int) -> list[tuple[str, int]]:
    """``(repo-relative path, size in bytes)`` for one iteration directory, read straight off the
    hub rather than hand-maintained here, so a change to FLAME's shard layout shows up on the
    next run instead of silently fetching a stale file set."""
    from huggingface_hub import HfApi

    entries = list(
        HfApi().list_repo_tree(
            REPO_ID, path_in_repo=_iter_name(iteration), recursive=True, repo_type="model"
        )
    )
    files = [(e.path, e.size) for e in entries if hasattr(e, "size")]
    if not files:
        raise ValueError(f"no files found under {_iter_name(iteration)} in {REPO_ID}")
    return files


def _filter_metadata(files: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return [(path, size) for path, size in files if path.rsplit("/", 1)[-1] in _METADATA_FILENAMES]


def _download(data_store: Path, iteration: int, metadata_only: bool) -> None:
    """Fetch one iteration's files into ``_target_root(data_store)``, preserving the hub's own
    ``iter_{N:07d}/<file>`` layout. Re-running is a no-op for files already on disk: the hub
    client is itself cache-aware."""
    from huggingface_hub import snapshot_download

    iter_name = _iter_name(iteration)
    patterns = (
        [f"{iter_name}/{name}" for name in _METADATA_FILENAMES]
        if metadata_only
        else [f"{iter_name}/*"]
    )
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=_target_root(data_store),
        allow_patterns=patterns,
    )


def _show_args(data_store: Path, iterations: list[int]) -> None:
    """Print a slice of the architecture read out of an already-downloaded ``common.pt``.

    Reading it needs the vendored Megatron-LM/ on ``sys.path``: the pickled dict's other
    top-level values (``opt_param_scheduler``, ``rerun_state_machine``) reference Megatron
    classes even though ``args`` itself unpickles to a plain ``argparse.Namespace``, so the
    unpickler needs the ``megatron`` module importable to load the file at all. We pass
    ``weights_only=False`` explicitly rather than asking the caller to set
    ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD``, since a pickled argparse namespace plus optimizer state
    is exactly what that escape hatch exists for and we control the call site directly.
    """
    from moe_congestion_routing.training.megatron_path import ensure_on_path

    ensure_on_path()
    import torch

    for iteration in iterations:
        common_pt = _target_root(data_store) / _iter_name(iteration) / "common.pt"
        if not common_pt.exists():
            raise FileNotFoundError(
                f"{common_pt} does not exist. Fetch it first with "
                f"--metadata-only --iteration {iteration}"
            )
        saved_args = torch.load(common_pt, map_location="cpu", weights_only=False)["args"]
        print(f"{_iter_name(iteration)}:")
        for name in _SHOW_ARGS_FIELDS:
            print(f"  {name} = {getattr(saved_args, name, 'MISSING')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iteration",
        type=int,
        nargs="+",
        default=[5473],
        help="checkpoint iteration(s) to fetch (default: 5473, FLAME's final, annealed "
        "checkpoint). Given multiple, each is fetched in turn.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list files and total byte count; fetch nothing"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="fetch only common.pt and metadata.json (a few hundred KB), not the weight shards",
    )
    parser.add_argument(
        "--show-args",
        action="store_true",
        help="print the architecture from an already-downloaded common.pt and fetch nothing",
    )
    args = parser.parse_args()

    if args.show_args:
        _show_args(_data_store(), args.iteration)
        return

    total_files = 0
    total_bytes = 0
    for iteration in args.iteration:
        files = _list_files(iteration)
        if args.metadata_only:
            files = _filter_metadata(files)
        n_files = len(files)
        n_bytes = sum(size for _, size in files)
        total_files += n_files
        total_bytes += n_bytes
        print(f"{_iter_name(iteration)}: {n_files} files, {n_bytes} bytes")
        if not args.dry_run:
            _download(_data_store(), iteration, args.metadata_only)

    print(f"total: {total_files} files, {total_bytes} bytes")


if __name__ == "__main__":
    main()

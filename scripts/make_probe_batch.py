#!/usr/bin/env python
"""Extract a frozen probe-batch asset from an ``IndexedDataset`` blob.

Reads the last documents of a ``.bin``/``.idx`` prefix directly (no ``torch.distributed``, no
shuffle index, no Megatron argument parsing) and writes their raw token ids into a committed
``assets/probe/<role>_<blob>_<S>x<seq_length>.npz``.

Usage:
    uv run python scripts/make_probe_batch.py \\
        --data-prefix <path without .bin/.idx> \\
        --seq-length <int> --sequences <S> \\
        --role standing|skewed|dev \\
        --out assets/probe/<role>_<blob>_<S>x<seq_length>.npz \\
        [--label <int>] [--max-tail-fraction 0.01] [--force]
    uv run python scripts/make_probe_batch.py --show assets/probe/<name>.npz
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy

from moe_congestion_routing.paths import expand_path
from moe_congestion_routing.training.megatron_path import ensure_on_path
from moe_congestion_routing.training.probe_batch import (
    PROBE_ASSET_VERSION,
    PROBE_ROLES,
    compute_provenance_sha256,
    tail_window,
)


def _extract(args: argparse.Namespace) -> None:
    role = args.role  # argparse's choices= refuses anything outside PROBE_ROLES

    out_path = Path(args.out)
    out_name = out_path.name
    if not out_name.startswith(f"{role}_"):
        sys.exit(
            f"error: --out basename {out_name!r} does not start with '{role}_' "
            f"(--role was {role!r}); the filename must not disagree with the role"
        )

    if out_path.exists() and not args.force:
        sys.exit(f"error: {out_path} already exists; pass --force to overwrite")

    data_prefix = expand_path(args.data_prefix)
    ensure_on_path()
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    ds = IndexedDataset(data_prefix)
    num_documents = ds.sequence_lengths.shape[0]
    target_tokens = args.sequences * (args.seq_length + 1)

    try:
        start, end, tail_fraction = tail_window(
            ds.sequence_lengths, target_tokens, args.max_tail_fraction
        )
    except ValueError as e:
        sys.exit(f"error: {e}")
    n_tail = end - start

    pieces = [ds.get(i) for i in range(start, end)]
    concatenated = numpy.concatenate(pieces)[:target_tokens]

    int32_info = numpy.iinfo(numpy.int32)
    if concatenated.min() < int32_info.min or concatenated.max() > int32_info.max:
        # IndexedDataset supports int64 blobs, so a downcast to int32 below would wrap silently
        # rather than raise. Climb's ids are uint16, so they fit int32 with room to spare, but
        # this guard exists for a future blob whose ids might not.
        sys.exit(
            f"error: token ids exceed int32 range [{int32_info.min}, {int32_info.max}]: "
            f"min={int(concatenated.min())}, max={int(concatenated.max())}"
        )
    tokens = concatenated.astype(numpy.int32).reshape(args.sequences, args.seq_length + 1)

    label = -1 if args.label is None else args.label
    seq_labels = numpy.full(args.sequences, label, dtype=numpy.int32)
    token_sha256 = hashlib.sha256(tokens.tobytes()).hexdigest()
    seq_labels_sha256 = hashlib.sha256(seq_labels.tobytes()).hexdigest()

    provenance = {
        "asset_version": PROBE_ASSET_VERSION,
        "role": role,
        "data_prefix": data_prefix,
        "num_documents": int(num_documents),
        "tail_doc_range": [int(start), int(end)],
        "tail_fraction": tail_fraction,
        "max_tail_fraction": args.max_tail_fraction,
        "S": int(args.sequences),
        "seq_length": int(args.seq_length),
        "token_sha256": token_sha256,
        "seq_labels_sha256": seq_labels_sha256,
        "blob_dtype": numpy.dtype(ds.index.dtype).name,
    }
    provenance["provenance_sha256"] = compute_provenance_sha256(provenance)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez(
        out_path,
        tokens=tokens,
        seq_labels=seq_labels,
        provenance=numpy.array(json.dumps(provenance)),
    )

    print(f"documents: {num_documents}")
    print(f"tail docs: {n_tail}")
    print(f"doc range: [{start}, {end})")
    print(f"tail_fraction: {tail_fraction:.6g}")
    print(f"wrote {out_path}: tokens {tokens.shape} {tokens.dtype}")


def _show(path: str) -> None:
    with numpy.load(path, allow_pickle=False) as data:
        provenance = json.loads(str(data["provenance"]))
    print(json.dumps(provenance, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-prefix", help="IndexedDataset prefix, without .bin/.idx")
    parser.add_argument("--seq-length", type=int, help="model sequence length")
    parser.add_argument("--sequences", type=int, help="number of sequences to extract (S)")
    parser.add_argument(
        "--role",
        choices=PROBE_ROLES,
        help="standing (the one asset a run monitors itself with), skewed (offline, "
        "composition-biased), or dev (smoke fixtures, never a reported number)",
    )
    parser.add_argument("--out", help="output .npz path")
    parser.add_argument(
        "--label", type=int, default=None, help="cluster/domain id to broadcast into seq_labels"
    )
    parser.add_argument(
        "--max-tail-fraction",
        type=float,
        default=0.01,
        help="refuse if the tail documents needed exceed this fraction of the blob, the default "
        "0.01 is the valid fraction of a 'split: \"99,1,0\"' training split and must move with it",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing --out")
    parser.add_argument(
        "--show", metavar="PATH", help="print an existing asset's provenance JSON and exit"
    )
    args = parser.parse_args()

    if args.show:
        _show(args.show)
        return

    missing = [
        name
        for name, value in (
            ("--data-prefix", args.data_prefix),
            ("--seq-length", args.seq_length),
            ("--sequences", args.sequences),
            ("--role", args.role),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    _extract(args)


if __name__ == "__main__":
    main()

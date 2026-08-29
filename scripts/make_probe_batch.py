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
        [--label <int>] [--max-tail-fraction 0.01] [--force] \\
        [--sampling tail|strided] [--stride-offset <int>, strided only]
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
    strided_window,
    tail_window,
)


def _extract(args: argparse.Namespace) -> None:
    role = args.role  # argparse's choices= refuses anything outside PROBE_ROLES

    if args.sampling != "strided" and args.stride_offset:
        sys.exit(
            f"error: --stride-offset {args.stride_offset} only applies with --sampling "
            f"strided, got --sampling {args.sampling!r}"
        )

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

    # `sampling` decides which documents are read and nothing else, so both paths produce the
    # same array shape and the same integrity fields below.
    try:
        if args.sampling == "strided":
            indices, stride, span_fraction = strided_window(
                ds.sequence_lengths, target_tokens, args.max_tail_fraction, args.stride_offset
            )

            # A defense-in-depth guard against the arithmetic that produced `indices` rather than
            # evidence of anything by itself: at two offsets in [0, stride) the sampled sets
            # cannot collide, so this can only fire if the index formula in strided_window
            # regresses.
            split_size = int(args.max_tail_fraction * num_documents)
            split_start = num_documents - split_size
            if int(indices.min()) < split_start or int(indices.max()) >= num_documents:
                sys.exit(
                    f"error: sampled indices [{int(indices.min())}, {int(indices.max())}] "
                    f"escape the held-out split [{split_start}, {num_documents})"
                )
            if args.stride_offset:
                base_indices, _, _ = strided_window(
                    ds.sequence_lengths, target_tokens, args.max_tail_fraction, 0
                )
                overlap = {int(i) for i in indices} & {int(i) for i in base_indices}
                if overlap:
                    sys.exit(
                        f"error: stride_offset={args.stride_offset} draw shares documents "
                        f"with the offset-0 draw on the same blob: {sorted(overlap)[:5]}"
                    )

            window = {
                "sampling": "strided",
                "doc_stride": int(stride),
                "doc_range": [int(indices[0]), int(indices[-1]) + 1],
                "num_docs": int(indices.size),
                "span_fraction": span_fraction,
                "stride_offset": int(args.stride_offset),
            }
        else:
            start, end, tail_fraction = tail_window(
                ds.sequence_lengths, target_tokens, args.max_tail_fraction
            )
            indices = numpy.arange(start, end)
            # No `sampling` key on this path, because adding one would change the provenance hash
            # of every asset extracted before it existed, and byte-identical re-extraction is a
            # property the format is meant to keep. Its absence therefore means "tail".
            window = {"tail_doc_range": [int(start), int(end)], "tail_fraction": tail_fraction}
    except ValueError as e:
        sys.exit(f"error: {e}")

    pieces = [ds.get(int(i)) for i in indices]
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
        **window,
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
    print(f"sampling: {args.sampling}")
    print(f"documents used: {indices.size}")
    print(f"doc range: [{int(indices[0])}, {int(indices[-1]) + 1})")
    for key in ("tail_fraction", "span_fraction", "doc_stride"):
        if key in window:
            print(f"{key}: {window[key]:.6g}" if key != "doc_stride" else f"{key}: {window[key]}")
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
        "--sampling",
        choices=("tail", "strided"),
        default="tail",
        help="tail takes the minimal run of documents at the end of the blob, which is a few "
        "dozen adjacent ones and so samples a single corpus neighbourhood. strided spreads the "
        "same number of documents across the whole held-out split, which is what a training "
        "batch sees. Default tail, so an asset extracted before this flag existed still "
        "re-extracts byte for byte",
    )
    parser.add_argument(
        "--max-tail-fraction",
        type=float,
        default=0.01,
        help="the held-out split, as a fraction of the blob. With --sampling tail this refuses a "
        "window reaching further back than the fraction, and with strided it is the range the "
        "sample is spread over. The default 0.01 is the valid fraction of a "
        "'split: \"99,1,0\"' training split and must move with it",
    )
    parser.add_argument(
        "--stride-offset",
        type=int,
        default=0,
        help="phase offset in [0, stride) for --sampling strided, so a second strided draw on "
        "the same blob samples a document set disjoint from an offset-0 draw. Rejected with "
        "--sampling tail unless left at its default 0",
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

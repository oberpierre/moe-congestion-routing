#!/usr/bin/env python
"""Price every `(asset, unit, layer, step)` cell of a run's probe dumps into a resumable dual
store, and its expert-bias store in the same pass.

The resume reads from *both* stores before anything is priced: a cell missing its dual is a full
price (solve, then write both rows), a cell whose dual is already stored but whose bias is not is
filled by reading its dump and writing the bias row alone, and a cell with both is skipped. The
second case is what makes a bias store lost or truncated after a complete run rebuildable without
paying for a single LP solve again. `--dry-run` prints both lists and exits without touching either
store.

Splitting a sweep is by asset: run this once per asset, each invocation with its own
`--duals-out`/`--bias-out`. A later correlating pass takes several store files as a set rather
than concatenating them, so there is no join step to botch, unlike the phi-gap grid's one file
per arm. Two invocations against the same bias file, for two assets probing the same checkpoint,
is exactly what makes `append_bias_rows`'s cross-invocation comparison live: the second invocation's
bias row is checked against the first's rather than silently dropped.

Usage:
    uv run python scripts/price_probe_duals.py --run-dir artifacts/exp1/control/control-trunk \\
        --duals-out artifacts/exp1/control/control-trunk/duals/standing.csv \\
        --bias-out artifacts/exp1/control/control-trunk/duals/bias.csv \\
        --asset standing_climbmix_small_16x2048 --layer 2 --step 0
"""

import argparse
from pathlib import Path

from moe_congestion_routing.metrics.dual_store import (
    BiasRow,
    DualRow,
    append_bias_rows,
    append_dual_rows,
    bias_key,
    dual_key,
    enumerate_dual_cells,
    existing_bias_keys,
    existing_dual_keys,
    price_bias_only_cells,
    price_cells,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="a training run directory")
    parser.add_argument("--duals-out", required=True, type=Path, help="this sweep's dual store")
    parser.add_argument("--bias-out", required=True, type=Path, help="this sweep's bias store")
    parser.add_argument(
        "--run-id",
        default=None,
        help="defaults to --run-dir's own name, e.g. 'control-trunk' for "
        "artifacts/exp1/control/control-trunk",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="worker processes, spawn-started (default: 1)"
    )
    parser.add_argument("--asset", action="append", dest="assets", help="repeatable")
    parser.add_argument("--layer", action="append", dest="layers", type=int, help="repeatable")
    parser.add_argument("--step", action="append", dest="steps", type=int, help="repeatable")
    parser.add_argument(
        "--limit", type=int, default=None, help="price at most this many full-price cells"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print how many cells remain after the resume filter, and exit",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    run_id = args.run_id or run_dir.name

    cells = enumerate_dual_cells(run_dir, assets=args.assets, layers=args.layers, steps=args.steps)

    done_duals = existing_dual_keys(args.duals_out)
    done_bias = existing_bias_keys(args.bias_out)

    to_price = [c for c in cells if dual_key(c, run_id=run_id) not in done_duals]
    # A cell whose dual is already stored needs at most its bias, read from the dump with no LP
    # solve, which is the only thing that makes a lost bias store cheap to rebuild.
    bias_only = [
        c
        for c in cells
        if dual_key(c, run_id=run_id) in done_duals
        and bias_key(run_id, c.layer, c.step) not in done_bias
    ]
    complete = len(cells) - len(to_price) - len(bias_only)
    # Counted before --limit truncates, matching run_phi_gap_grid.py, so a capped sweep is not
    # misread as a nearly finished one.
    capped = (
        "" if args.limit is None or len(to_price) <= args.limit else f", capped to {args.limit}"
    )
    if args.limit is not None:
        to_price = to_price[: args.limit]

    print(
        f"run_id={run_id}: {len(cells)} cells enumerated, {len(to_price)} to price{capped}, "
        f"{len(bias_only)} bias-only, {complete} already complete"
    )

    if args.dry_run:
        for cell in to_price:
            print(f"  price {cell.asset} unit={cell.unit} layer={cell.layer} step={cell.step}")
        for cell in bias_only:
            print(f"  bias-only {cell.asset} unit={cell.unit} layer={cell.layer} step={cell.step}")
        return

    if not to_price and not bias_only:
        print("0 cells remaining, 0 solves performed")
        return

    counts = {"ok": 0, "failed": 0}
    bias_written = 0

    def emit(row: DualRow | BiasRow) -> None:
        nonlocal bias_written
        if isinstance(row, DualRow):
            append_dual_rows(args.duals_out, [row])
            counts[row.status] += 1
            done_so_far = counts["ok"] + counts["failed"]
            prefix = f"  [{done_so_far}/{len(to_price)}] {row.asset} unit={row.unit}"
            if row.status == "failed":
                print(f"{prefix} status=failed detail={row.detail!r}", flush=True)
            else:
                print(f"{prefix} layer={row.layer} admissible={row.admissible}", flush=True)
            return
        assert isinstance(row, BiasRow)
        written = append_bias_rows(args.bias_out, [row])
        bias_written += written

    if to_price:
        price_cells(to_price, run_id=run_id, emit=emit, workers=args.workers)
    if bias_only:
        price_bias_only_cells(bias_only, run_id=run_id, emit=emit)

    print(
        f"priced {counts['ok']} ok, {counts['failed']} failed, wrote {bias_written} bias rows "
        f"({len(bias_only)} of them bias-only)"
    )


if __name__ == "__main__":
    main()

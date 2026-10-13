#!/usr/bin/env python
"""The tail x strided x spread triad, at every step a dual store carries, in one pass.

Reads the two stores `scripts/price_probe_duals.py` produced (a set of files per `--duals`/
`--bias`, matching the per-asset split its own pricing pass is run under) and, for every
`(run_id, layer, step)` cell they can jointly price, emits both the uncorrected triad and
`--project-code-axis`'s corrected variant into one CSV, distinguished by its `projected` column.
No LP is solved here: all of that cost already lives in the pricing pass, so this reads a few
megabytes of CSV and finishes in well under a minute at 64 experts.

Usage:
    uv run python scripts/run_triad_trajectory.py \\
        --duals artifacts/exp1/alflb/duals/standing.csv \\
        --duals artifacts/exp1/alflb/duals/strided.csv \\
        --duals artifacts/exp1/alflb/duals/spread.csv \\
        --bias artifacts/exp1/alflb/duals/standing_bias.csv \\
        --bias artifacts/exp1/alflb/duals/strided_bias.csv \\
        --bias artifacts/exp1/alflb/duals/spread_bias.csv \\
        --out artifacts/exp1/alflb/triad_trajectory.csv
"""

import argparse
import csv
import os
from pathlib import Path

from moe_congestion_routing.metrics.dual_store import read_bias_store, read_dual_store
from moe_congestion_routing.metrics.triad import TriadAssets, TriadRow, trajectory_cells

# The three fleet asset directory names, matching `reconstruct_triad_from_store.py`'s own
# `ASSETS`: the role name stays "tail" for the asset the fleet directory calls "standing",
# because it names the same tokens the step-500 triad was measured on.
DEFAULT_TAIL_ASSET = "standing_climbmix_small_16x2048"
DEFAULT_STRIDED_ASSET = "standing_climbmix_small_strided_16x2048"
DEFAULT_SPREAD_ASSET = "standing_climbmix_small_spread_off1_16x2048"

_ROW_FIELDS = tuple(f for f in TriadRow._fields if f not in ("run", "layer"))
OUT_FIELDS = ("run_id", "step", "layer", "projected", "refused_units") + _ROW_FIELDS
_ROW_INDEX = tuple(TriadRow._fields.index(f) for f in _ROW_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duals", action="append", required=True, type=Path, help="repeatable")
    parser.add_argument("--bias", action="append", required=True, type=Path, help="repeatable")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--step", action="append", dest="steps", type=int, help="repeatable")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--asset-tail", default=DEFAULT_TAIL_ASSET)
    parser.add_argument("--asset-strided", default=DEFAULT_STRIDED_ASSET)
    parser.add_argument("--asset-spread", default=DEFAULT_SPREAD_ASSET)
    args = parser.parse_args()

    duals = read_dual_store(args.duals)
    bias = read_bias_store(args.bias)
    if args.steps is not None:
        wanted = set(args.steps)
        duals = {k: v for k, v in duals.items() if k[4] in wanted}
        bias = {k: v for k, v in bias.items() if k[2] in wanted}

    assets = TriadAssets(tail=args.asset_tail, strided=args.asset_strided, spread=args.asset_spread)

    # One store read, both variants: `duals`/`bias` are the same parsed mappings for both calls,
    # so the second variant costs a re-read of neither file.
    cells = trajectory_cells(
        duals,
        bias,
        assets=assets,
        project_code_axis=False,
        resamples=args.resamples,
        base_seed=args.seed,
    ) + trajectory_cells(
        duals,
        bias,
        assets=assets,
        project_code_axis=True,
        resamples=args.resamples,
        base_seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temp file and renamed, so a crash partway through leaves either the old file
    # or a complete one rather than a truncated CSV a reader cannot tell apart from a finished run.
    tmp_path = args.out.parent / f".{args.out.name}.tmp"
    n_rows = 0
    with tmp_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUT_FIELDS)
        for cell in cells:
            for row in cell.rows:
                writer.writerow(
                    (cell.run_id, cell.step, cell.layer, cell.projected, cell.refused_units)
                    + tuple(row[i] for i in _ROW_INDEX)
                )
                n_rows += 1
    os.replace(tmp_path, args.out)
    print(f"wrote {n_rows} rows across {len(cells)} cells to {args.out}")


if __name__ == "__main__":
    main()

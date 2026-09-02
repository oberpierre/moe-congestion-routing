#!/usr/bin/env python
"""Score every `(asset, layer, unit, cost_family)` cell of a run's probe dumps into that arm's
CSV, and skip every cell already recorded there.

The resume happens *before* a cell is ever handed to `run_grid`, because a cell costs a real LP
solve: this script reads `--out`'s existing keys, drops any enumerated cell that already has a
row, and only the remainder is solved. `--dry-run` prints that remainder and exits without
solving anything.

Usage:
    uv run python scripts/run_phi_gap_grid.py --run-dir artifacts/exp1/control/control-trunk \\
        --out assets/results/phi-gap/control.csv
    uv run python scripts/run_phi_gap_grid.py --run-dir artifacts/exp1/control/control-trunk \\
        --out assets/results/phi-gap/control.csv --asset standing_climbmix_small_16x2048 \\
        --layer 2 --step 0 --step 25 --cost-family linear
"""

import argparse
import time
from pathlib import Path

from moe_congestion_routing.losses.cost_families import COST_FAMILIES
from moe_congestion_routing.metrics.phi_gap_grid import (
    append_rows,
    candidate_key,
    enumerate_cells,
    existing_keys,
    run_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="a training run directory")
    parser.add_argument("--out", required=True, type=Path, help="the arm's phi-gap CSV")
    parser.add_argument(
        "--workers", type=int, default=1, help="worker processes, spawn-started (default: 1)"
    )
    parser.add_argument(
        "--lam",
        action="append",
        dest="lams",
        type=float,
        help="repeatable, lambda for both reference costs (default: [1.0])",
    )
    parser.add_argument(
        "--cost-family",
        action="append",
        dest="cost_families",
        choices=COST_FAMILIES,
        help="repeatable, defaults to both declared reference costs",
    )
    parser.add_argument("--asset", action="append", dest="assets", help="repeatable")
    parser.add_argument("--layer", action="append", dest="layers", type=int, help="repeatable")
    parser.add_argument("--step", action="append", dest="steps", type=int, help="repeatable")
    parser.add_argument(
        "--limit", type=int, default=None, help="solve at most this many cells, after resume"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="defaults to --run-dir's own name, e.g. 'control-trunk' for "
        "artifacts/exp1/control/control-trunk",
    )
    parser.add_argument(
        "--arm",
        default=None,
        help="defaults to --run-dir's parent name, e.g. 'control' for "
        "artifacts/exp1/control/control-trunk. Wrong on the command line silently mislabels "
        "every row of the sweep",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the cells that would be solved after the resume filter, and exit",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    run_id = args.run_id or run_dir.name
    arm = args.arm or run_dir.parent.name
    cost_families = tuple(args.cost_families) if args.cost_families else COST_FAMILIES
    lams = tuple(args.lams) if args.lams else (1.0,)

    cells = enumerate_cells(
        run_dir,
        lams=lams,
        cost_families=cost_families,
        assets=args.assets,
        layers=args.layers,
        steps=args.steps,
    )

    done = existing_keys(args.out)
    remaining = [c for c in cells if candidate_key(c, run_id=run_id, arm=arm) not in done]
    # Counted before --limit truncates. Counting after it charges every cell the limit excluded to
    # the output file, which reported an empty file as holding 5662 of 5664 rows and read as a
    # nearly finished run rather than a capped one.
    resumed = len(cells) - len(remaining)
    unsolved = len(remaining)
    if args.limit is not None:
        remaining = remaining[: args.limit]

    capped = "" if len(remaining) == unsolved else f", capped by --limit to {len(remaining)}"
    print(
        f"run_id={run_id} arm={arm}: {len(cells)} cells enumerated, {resumed} already in "
        f"{args.out}, {unsolved} not yet solved{capped}"
    )

    if args.dry_run:
        for cell in remaining:
            print(
                f"  {cell.asset} layer={cell.layer} unit={cell.unit} "
                f"cost_family={cell.cost_family} lam={cell.lam} dump={cell.dump_path}"
            )
        return

    if not remaining:
        print("0 cells remaining, 0 solves performed")
        return

    counts = {"ok": 0, "failed": 0}
    start = time.perf_counter()

    def emit(row) -> None:
        append_rows(args.out, [row])
        counts[row.status] += 1
        done_so_far = counts["ok"] + counts["failed"]
        prefix = f"  [{done_so_far}/{len(remaining)}] {row.asset}"
        if row.status == "failed":
            print(f"{prefix} status=failed detail={row.detail!r}", flush=True)
        else:
            print(
                f"{prefix} unit={row.row.unit} layer={row.row.layer} "
                f"cost_family={row.row.reference_cost} "
                f"gap_normalized={row.row.gap_normalized:.4f}",
                flush=True,
            )

    run_grid(remaining, run_id=run_id, arm=arm, emit=emit, workers=args.workers)

    elapsed = time.perf_counter() - start
    print(
        f"solved {len(remaining)} cells ({counts['ok']} ok, {counts['failed']} failed) "
        f"in {elapsed:.1f}s, written to {args.out}"
    )


if __name__ == "__main__":
    main()

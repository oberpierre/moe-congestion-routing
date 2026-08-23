#!/usr/bin/env python
"""Grid the synthetic ALF-LB-versus-LP comparison and write it to a CSV.

Builds a cross product of shapes, separations and seeds, then for each cell runs one annealed
comparison (the theorem's hypothesis test) and three deployed comparisons at the fixed step
sizes a shipped router would use, 1e-3, 1e-2 and 1e-1. Both modes are needed because a fixed
step size orbits a limit cycle rather than converging, so only the annealed one tests the
theorem's hypothesis.

The grid's cells are independent, so `--jobs` runs them across worker processes, but
parallelism cannot shrink a single cell because each cell is `bias_{t+1} = f(bias_t)`, strictly
sequential, and on this grid a cell's cost is 99% that loop against the LP solve, so wall clock
across any number of jobs is bounded below by the slowest single cell.

Rows are written to the output CSV as a contiguous prefix, flushed as soon as a cell and every
cell before it have been collected, so a cell that fails or a run that is interrupted still
leaves every earlier row on disk instead of discarding the whole grid.

Usage:
    uv run python scripts/run_game_comparison.py --out artifacts/game/alf_lb_vs_lp.csv
    uv run python scripts/run_game_comparison.py --out artifacts/game/quick.csv --quick
    uv run python scripts/run_game_comparison.py --out artifacts/game/full.csv --jobs 8
"""

import argparse
import csv
import dataclasses
import os
from collections import Counter
from pathlib import Path

from moe_congestion_routing.game.compare import Comparison
from moe_congestion_routing.game.ensemble import Instance
from moe_congestion_routing.game.grid import Cell, Row, run_grid

_SHAPES = [(512, 8, 2), (2048, 64, 8)]
_SEPARATIONS = [2.0, 0.2]
_SEEDS = [0, 1, 2]
_ANNEALED_ETA = 1e-2
# Half this budget would look like a safe saving but is not, because the one large-shape
# (n=2048, e=64, k=8) cell that settles at all does not do so until step 25582. A smaller
# annealed budget would silently turn that row from "settled" into "unconverged", and that row
# is the one the ALF-LB-versus-LP-optimum theorem test rests on.
_ANNEALED_STEPS = 40000
_DEPLOYED_ETAS = [1e-3, 1e-2, 1e-1]
_DEPLOYED_STEPS = 2000


def _grid(quick: bool) -> list[Instance]:
    shapes = [_SHAPES[0]] if quick else _SHAPES
    seeds = [_SEEDS[0]] if quick else _SEEDS
    return [
        Instance(n=n, e=e, k=k, separation=separation, seed=seed)
        for n, e, k in shapes
        for separation in _SEPARATIONS
        for seed in seeds
    ]


def _cells(quick: bool) -> list[Cell]:
    """One cell per CSV row."""
    cells = []
    for inst in _grid(quick):
        cells.append(Cell(inst, "annealed", _ANNEALED_ETA, _ANNEALED_STEPS))
        for eta in _DEPLOYED_ETAS:
            cells.append(Cell(inst, "deployed", eta, _DEPLOYED_STEPS))
    return cells


def _default_jobs() -> int:
    """Return how many CPUs this process may run on, which is not how many the machine has.

    A scheduler hands a job a subset of the node's cores, so `os.cpu_count()` would start a
    worker per node CPU onto far fewer allocated ones and thrash rather than fail. The affinity
    call is Linux-only, so fall back to the machine count where it does not exist.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _positive_int(value: str) -> int:
    """argparse `type=` for `--jobs`: reject anything below 1 before any file is touched.

    Left to `concurrent.futures` itself, `--jobs 0` opens and truncates the output file, writes
    the header, and only then dies with a message that never names `--jobs`, so a typo destroys
    an existing CSV. Rejecting it here happens during `parse_args`, before `main` opens anything.
    """
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"--jobs must be >= 1, got {value}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="only the (512, 8, 2) shape and seed 0, for a fast smoke run",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=_default_jobs(),
        help=(
            "worker processes for the cell grid (default: CPUs available to this process, "
            "not the whole node). --jobs 1 runs cells serially in one process and skips the "
            "pool entirely. Parallelism cannot shrink any single cell's own cost."
        ),
    )
    args = parser.parse_args()

    cells = _cells(args.quick)
    instance_fields = [f.name for f in dataclasses.fields(Instance)]
    header = instance_fields + list(Comparison._fields)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()

        def emit(row: Row) -> None:
            inst = row.cell.instance
            writer.writerow(
                [getattr(inst, name) for name in instance_fields]
                + list(row.comparison._asdict().values())
            )
            f.flush()
            rows.append(row)

        # Progress prints from on_complete rather than from emit, because emit fires in grid
        # order once the prefix reaches a cell. Printing there makes the counter jump around
        # and go silent for as long as the slowest early cell takes, whereas a cell finishing
        # is what an operator is actually watching for.
        done = 0

        def on_complete(row: Row) -> None:
            nonlocal done
            done += 1
            inst = row.cell.instance
            print(
                f"[cell {row.index + 1}/{len(cells)}] n={inst.n} e={inst.e} k={inst.k} "
                f"separation={inst.separation} seed={inst.seed} mode={row.cell.mode} "
                f"eta={row.cell.eta} tier={row.comparison.tier} done in {row.elapsed:.1f}s "
                f"({done}/{len(cells)} complete)",
                flush=True,
            )

        run_grid(cells, args.jobs, emit, on_complete)

    tier_counts = Counter(row.comparison.tier for row in rows)
    print(f"\ntier counts: {dict(tier_counts)}")
    for row in rows:
        c = row.comparison
        inst = row.cell.instance
        if c.tier in ("settled", "tie_slack"):
            # gap_at_matched_cap is printed alongside max_load rather than gap_over_span
            # alone, because a gap without the realized max load next to it hides the
            # capacity violation that produced it.
            print(
                f"  [{c.tier}] n={inst.n} e={inst.e} k={inst.k} separation={inst.separation} "
                f"seed={inst.seed} mode={c.mode} eta={c.eta} max_load={c.max_load} "
                f"gap_at_matched_cap={c.gap_at_matched_cap:.6f} gap_over_span={c.gap_over_span:.4f}"
            )
            if c.tier == "settled":
                print(f"      dual_correlation={c.dual_correlation:.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Grid the synthetic ALF-LB-versus-LP comparison and write it to a CSV.

Builds a cross product of shapes, separations and seeds, then for each cell runs one annealed
comparison (the theorem's hypothesis test) and three deployed comparisons at the step sizes an
Both modes are needed because a fixed step size orbits a limit cycle rather than converging, so
only the annealed one tests the theorem's hypothesis.

The grid's cells are independent, so `--jobs` runs them across worker processes. Parallelism
cannot shrink a single cell, though: one cell is `bias_{t+1} = f(bias_t)`, strictly sequential,
and on this grid a cell's cost is 99% that loop against the LP solve, so wall clock across any
number of jobs is bounded below by the slowest single cell.

Usage:
    uv run python scripts/run_game_comparison.py --out artifacts/game/alf_lb_vs_lp.csv
    uv run python scripts/run_game_comparison.py --out artifacts/game/quick.csv --quick
    uv run python scripts/run_game_comparison.py --out artifacts/game/full.csv --jobs 8
"""

import argparse
import csv
import dataclasses
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from moe_congestion_routing.game.compare import Comparison, compare
from moe_congestion_routing.game.ensemble import Instance, affinities

# The annealed budget is 40000, not 20000, so the one large-shape cell that can settle
# (N=2048, E=64, K=8 at separation=2.0, seed=1) actually does: it settles at step 25582,
# past a 20000-step budget.
_SHAPES = [(512, 8, 2), (2048, 64, 8)]
_SEPARATIONS = [2.0, 0.2]
_SEEDS = [0, 1, 2]
_ANNEALED_ETA = 1e-2
_ANNEALED_STEPS = 40000
_DEPLOYED_ETAS = [1e-3, 1e-2, 1e-1]
_DEPLOYED_STEPS = 2000

_Cell = tuple[Instance, str, float, int]


def _grid(quick: bool) -> list[Instance]:
    shapes = [_SHAPES[0]] if quick else _SHAPES
    seeds = [_SEEDS[0]] if quick else _SEEDS
    return [
        Instance(n=n, e=e, k=k, separation=separation, seed=seed)
        for n, e, k in shapes
        for separation in _SEPARATIONS
        for seed in seeds
    ]


def _cells(quick: bool) -> list[_Cell]:
    """One (instance, mode, eta, steps) tuple per CSV row."""
    cells = []
    for inst in _grid(quick):
        cells.append((inst, "annealed", _ANNEALED_ETA, _ANNEALED_STEPS))
        for eta in _DEPLOYED_ETAS:
            cells.append((inst, "deployed", eta, _DEPLOYED_STEPS))
    return cells


def _run_cell(cell: _Cell) -> tuple[Comparison, float]:
    """Score one grid cell against the LP oracle. Module-level so a process pool can pickle it.

    A cell's own ALF-LB loop is what dominates its cost, not the work done here, so this
    function is the unit of parallelism even though it cannot make any single cell faster.
    """
    inst, mode, eta, steps = cell
    start = time.perf_counter()
    a = affinities(inst)
    c = compare(a, inst.k, eta=eta, steps=steps, mode=mode)
    elapsed = time.perf_counter() - start
    return c, elapsed


def _run_serial(cells: list[_Cell]) -> list[Comparison]:
    """Run every cell in one process, in cell order. This is the `--jobs 1` path."""
    rows: list[Comparison] = []
    for i, (inst, mode, eta, steps) in enumerate(cells, start=1):
        c, elapsed = _run_cell((inst, mode, eta, steps))
        print(
            f"[cell {i}/{len(cells)}] n={inst.n} e={inst.e} k={inst.k} "
            f"separation={inst.separation} seed={inst.seed} mode={mode} eta={eta} "
            f"tier={c.tier} done in {elapsed:.1f}s",
            flush=True,
        )
        rows.append(c)
    return rows


def _run_parallel(cells: list[_Cell], jobs: int) -> list[Comparison]:
    """Run the grid across `jobs` worker processes.

    Results are written into a list pre-sized to the grid and indexed by each cell's own
    position, never by completion order, so the CSV a caller writes afterward does not depend
    on which worker happens to finish first.
    """
    rows: list[Comparison | None] = [None] * len(cells)
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_index = {executor.submit(_run_cell, cell): i for i, cell in enumerate(cells)}
        for done, future in enumerate(as_completed(future_to_index), start=1):
            i = future_to_index[future]
            inst, mode, eta, steps = cells[i]
            c, elapsed = future.result()
            print(
                f"[cell {i + 1}/{len(cells)}] n={inst.n} e={inst.e} k={inst.k} "
                f"separation={inst.separation} seed={inst.seed} mode={mode} eta={eta} "
                f"tier={c.tier} done in {elapsed:.1f}s ({done}/{len(cells)} complete)",
                flush=True,
            )
            rows[i] = c
    return rows  # type: ignore[return-value]


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
        type=int,
        default=os.cpu_count() or 1,
        help=(
            "worker processes for the cell grid (default: os.cpu_count()). --jobs 1 runs the "
            "original single-process path and skips the process pool entirely. Cells are "
            "independent but each one's ALF-LB loop is sequential and dominates its own cost, "
            "so wall clock is bounded below by the slowest cell regardless of --jobs."
        ),
    )
    args = parser.parse_args()

    cells = _cells(args.quick)
    instance_fields = [f.name for f in dataclasses.fields(Instance)]
    header = instance_fields + list(Comparison._fields)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _run_serial(cells) if args.jobs == 1 else _run_parallel(cells, args.jobs)

    instances = [cell[0] for cell in cells]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for inst, c in zip(instances, rows, strict=True):
            writer.writerow(
                [getattr(inst, name) for name in instance_fields] + list(c._asdict().values())
            )

    tier_counts = Counter(c.tier for c in rows)
    print(f"\ntier counts: {dict(tier_counts)}")
    for inst, c in zip(instances, rows, strict=True):
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

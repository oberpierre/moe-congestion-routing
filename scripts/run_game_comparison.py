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
cell before it have finished, so a cell that fails still leaves every earlier row on disk
instead of discarding the whole grid.

Usage:
    uv run python scripts/run_game_comparison.py --out artifacts/game/alf_lb_vs_lp.csv
    uv run python scripts/run_game_comparison.py --out artifacts/game/quick.csv --quick
    uv run python scripts/run_game_comparison.py --out artifacts/game/full.csv --jobs 8
"""

import argparse
import csv
import dataclasses
import multiprocessing
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import IO, Any

from moe_congestion_routing.game.compare import Comparison, compare
from moe_congestion_routing.game.ensemble import Instance, affinities

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
    """Score one grid cell against the LP oracle. Module-level so a process pool can pickle it,
    and because parallelism cannot shrink what one cell costs.
    """
    inst, mode, eta, steps = cell
    start = time.perf_counter()
    a = affinities(inst)
    c = compare(a, inst.k, eta=eta, steps=steps, mode=mode)
    elapsed = time.perf_counter() - start
    return c, elapsed


def _flush_prefix(
    writer: Any,
    f: IO[str],
    instance_fields: list[str],
    cells: list[_Cell],
    rows: list[Comparison | None],
    next_to_write: int,
) -> int:
    """Write and flush every row from `next_to_write` onward that is already filled in `rows`,
    stopping at the first gap so the file always holds a contiguous prefix of `cells`.
    """
    while next_to_write < len(rows) and rows[next_to_write] is not None:
        inst = cells[next_to_write][0]
        c = rows[next_to_write]
        assert c is not None  # narrows for the type checker, already checked above
        writer.writerow(
            [getattr(inst, name) for name in instance_fields] + list(c._asdict().values())
        )
        f.flush()
        next_to_write += 1
    return next_to_write


def _run_serial(
    cells: list[_Cell], writer: Any, f: IO[str], instance_fields: list[str]
) -> list[Comparison]:
    """Run every cell in one process, in cell order. This is the `--jobs 1` path.

    Every cell finishes in submission order here, so the contiguous-prefix write below
    advances on every iteration, which is the same as writing each row as it finishes.
    """
    rows: list[Comparison | None] = [None] * len(cells)
    next_to_write = 0
    for i, (inst, mode, eta, steps) in enumerate(cells, start=1):
        c, elapsed = _run_cell((inst, mode, eta, steps))
        print(
            f"[cell {i}/{len(cells)}] n={inst.n} e={inst.e} k={inst.k} "
            f"separation={inst.separation} seed={inst.seed} mode={mode} eta={eta} "
            f"tier={c.tier} done in {elapsed:.1f}s",
            flush=True,
        )
        rows[i - 1] = c
        next_to_write = _flush_prefix(writer, f, instance_fields, cells, rows, next_to_write)
    return rows  # type: ignore[return-value]


def _run_parallel(
    cells: list[_Cell],
    jobs: int,
    writer: Any,
    f: IO[str],
    instance_fields: list[str],
) -> list[Comparison]:
    """Run the grid across `jobs` worker processes.

    Results are recorded into a list pre-sized to the grid and indexed by each cell's own
    position, never by completion order, so the CSV row order matches `cells` regardless of
    which worker happens to finish first.
    """
    rows: list[Comparison | None] = [None] * len(cells)
    next_to_write = 0
    # spawn re-execs a fresh interpreter per worker instead of forking this process, which
    # already has numpy and scipy loaded. Forking a process with C-extension state and
    # whatever threads those libraries keep alive can deadlock the child at the fork boundary.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as executor:
        future_to_index = {executor.submit(_run_cell, cell): i for i, cell in enumerate(cells)}
        try:
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
                next_to_write = _flush_prefix(
                    writer, f, instance_fields, cells, rows, next_to_write
                )
        except Exception:
            # Every row up to the failure is already flushed by the loop above. Cancelling
            # here means only the futures already running when the failure hit still have to
            # finish before shutdown returns, not every one of the remaining cells.
            executor.shutdown(wait=True, cancel_futures=True)
            raise
    return rows  # type: ignore[return-value]


def _default_jobs() -> int:
    """Return how many CPUs this process may run on, which is not how many the machine has.

    A scheduler hands a job a subset of the node's cores, so `os.cpu_count()` would start a
    worker per node CPU onto far fewer allocated ones and thrash rather than fail. The affinity
    call is Linux-only, so fall back to the machine count where it does not exist.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


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

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()
        if args.jobs == 1:
            rows = _run_serial(cells, writer, f, instance_fields)
        else:
            rows = _run_parallel(cells, args.jobs, writer, f, instance_fields)

    instances = [cell[0] for cell in cells]

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

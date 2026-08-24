#!/usr/bin/env python
"""Measure how much a batch's equilibrium prices depend on which batch it is.

Table 2 (``run_probe_comparison.py``) asks whether the stored ALF-LB bias matches one batch's LP
capacity duals. A falling correlation there has two possible causes and Table 2 cannot separate
them: the bias may have stopped tracking the price, or the price may have become specific to the
batch. This script asks the second on its own, by cutting the probe batch into parts, solving the
LP on each part at the same model, and correlating one part's prices against another's. No bias
enters those columns, so a decline in them is a statement about the data alone.

Read the two together. Pairwise correlation staying near 1 while Table 2 falls means the bias is
drifting away from a price the batch still represents well. Both falling together means the price
itself has become batch-specific and the bias is tracking a population the batch no longer stands
in for.

``--split sequence`` cuts into whole sequences, which is what forming a smaller training batch
does, so it carries the composition difference between one set of documents and another.
``--split stride`` takes every m-th token, spreading each part over every sequence and position,
which leaves only sampling noise. Running both and subtracting is how much of the effect is
composition rather than sample size.

Usage:
    uv run python scripts/run_price_stability.py RUNDIR --bias-update-rate 1.0e-3 \\
        --out artifacts/game/price_stability_sequence.csv
    uv run python scripts/run_price_stability.py RUNDIR --benchmark
"""

import argparse
import csv
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from moe_congestion_routing.metrics.probe_comparison import (
    SPLIT_MODES,
    PriceStabilityRow,
    part_indices,
    price_stability_rows_for_dump,
)
from moe_congestion_routing.metrics.probe_series import read_dump, read_series


def _default_jobs() -> int:
    """Return how many CPUs this process may run on, which is not how many the machine has.

    Under Slurm, a container, or a taskset the two differ, and the machine count would oversubscribe
    the allocation. The affinity call is Linux-only, so fall back to the machine count where it does
    not exist, which covers macOS.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _rows_for_path(job: tuple[str, float, int, str, tuple[int, ...] | None]) -> list:
    """One dump's rows, as a process-pool task.

    Takes a path rather than a ``ProbeDump`` and re-reads it here, because under ``spawn`` every
    argument is pickled and the dump's arrays should be read in the worker that uses them.
    """
    path, bias_update_rate, num_parts, split, layers = job
    dump = read_dump(path)
    return price_stability_rows_for_dump(
        dump,
        bias_update_rate=bias_update_rate,
        num_parts=num_parts,
        split=split,
        layers=list(layers) if layers else None,
    )


def _benchmark(run_dir: str, num_parts: int, split: str) -> None:
    """Time one full-batch LP solve and one part solve, so a machine can be compared to another.

    Prints the shapes alongside the seconds, because a solve time means nothing without them, and
    the whole run's cost is this many seconds times steps times layers times ``num_parts + 1``.
    """
    import numpy as np

    from moe_congestion_routing.game import lp

    series = read_series(run_dir)
    dump = series.dumps[-1]
    affinities = dump.affinities()
    layer_a = affinities[0]
    n, e = layer_a.shape
    k = dump.topk
    indices = part_indices(n, dump.num_sequences, num_parts, split)

    start = time.perf_counter()
    lp.solve(layer_a, k)
    full_seconds = time.perf_counter() - start

    part = layer_a[indices[0]]
    start = time.perf_counter()
    lp.solve(part, k)
    part_seconds = time.perf_counter() - start

    print(f"machine: {os.uname().sysname} {os.uname().machine}, jobs available {_default_jobs()}")
    print(f"numpy {np.__version__}")
    print(f"full  LP  n={n:6d} e={e} k={k}: {full_seconds:7.2f} s")
    print(f"part  LP  n={part.shape[0]:6d} e={e} k={k}: {part_seconds:7.2f} s")
    per_cell = full_seconds + num_parts * part_seconds
    cells = len(series.dumps) * len(dump.layer_numbers)
    print(
        f"\nper (step, layer): {per_cell:.2f} s  ->  {cells} cells serial "
        f"{cells * per_cell / 60:.1f} min, at {_default_jobs()} jobs about "
        f"{cells * per_cell / 60 / _default_jobs():.1f} min"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", metavar="RUNDIR", help="a run directory holding probes/*.npz")
    parser.add_argument(
        "--bias-update-rate",
        type=float,
        help=(
            "moe_router_bias_update_rate the run was trained with. Required unless --benchmark, "
            "because the resolvability gate on the two bias columns divides by it"
        ),
    )
    parser.add_argument("--out", help="output CSV path. Required unless --benchmark")
    parser.add_argument(
        "--parts", type=int, default=2, help="how many parts to cut the batch into (default 2)"
    )
    parser.add_argument(
        "--split", choices=SPLIT_MODES, default="sequence", help="how to cut (default sequence)"
    )
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        dest="steps",
        help="a dump's iteration number, repeatable. Defaults to every dump in the series",
    )
    parser.add_argument(
        "--layers", type=int, action="append", dest="layers", help="a layer number, repeatable"
    )
    parser.add_argument(
        "--jobs", type=int, default=_default_jobs(), help="worker processes (default: CPUs allowed)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="time one full and one part LP solve on this machine, then exit",
    )
    args = parser.parse_args()

    if args.benchmark:
        _benchmark(args.run_dir, args.parts, args.split)
        return
    if args.bias_update_rate is None or args.out is None:
        parser.error("--bias-update-rate and --out are required unless --benchmark is given")

    series = read_series(args.run_dir)
    dumps = series.dumps
    if args.steps:
        by_step = {dump.step: dump for dump in dumps}
        missing = [step for step in args.steps if step not in by_step]
        if missing:
            parser.error(f"steps {missing} are not among {[d.step for d in dumps]}")
        dumps = [by_step[step] for step in args.steps]

    layers = tuple(args.layers) if args.layers else None
    jobs = [
        (str(dump.path), args.bias_update_rate, args.parts, args.split, layers) for dump in dumps
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    # Written as each dump's rows arrive rather than at the end, so an interrupted run keeps what
    # it has already computed. `map` yields in submission order, so the file stays in step order
    # even though the dumps finish out of order.
    with (
        open(out_path, "w", newline="") as handle,
        ProcessPoolExecutor(
            max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as executor,
    ):
        writer = csv.writer(handle)
        writer.writerow(PriceStabilityRow._fields)
        # Flushed before the first result, because the first one can be many minutes away and an
        # empty file reads as a crash. With the header down, the file says what it will contain.
        handle.flush()
        print(f"writing {len(jobs)} dumps to {out_path}", file=sys.stderr, flush=True)
        for rows in executor.map(_rows_for_path, jobs):
            for row in rows:
                writer.writerow(row)
            written += len(rows)
            handle.flush()
            print(f"  {written} rows", file=sys.stderr, flush=True)

    print(f"wrote {written} rows to {out_path}")


if __name__ == "__main__":
    main()

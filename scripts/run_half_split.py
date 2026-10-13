#!/usr/bin/env python
"""Cut each probe dump in half by sequence and report both halves with bootstrap intervals.

``run_price_stability.py`` reports the *mean* bias correlation over the parts, which hides how far
apart they are: on the strided asset at step 500 the two halves are 0.11 and 0.72. This script
keeps them separate, and it keeps the per-expert dual vectors long enough to resample them.

Every row carries three correlations, namely ``rho`` between the two halves' prices and each
half's price against the stored bias, plus ``kappa = sqrt(c_A * c_B / rho)``, which under
``p*(half) = p_bar + e_half`` equals ``corr(b_train, p_bar)`` with the halves' unequal noise
cancelling. That cancellation is why ``kappa`` is reported rather than a ratio to ``sqrt(rho)``: a
ratio needs the halves to be equal-noise, and measurably they are not.

Intervals are per layer and never pooled across layers, because the layers are different games and
averaging correlations before transforming them is the same aggregation error twice over. The
bootstrap resamples the 64 experts jointly, so ``kappa``'s interval carries the shared randomness
between its numerator and denominator.

Usage:
    uv run python scripts/run_half_split.py RUNDIR --out artifacts/game/half_split.csv
"""

import argparse
import csv
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from moe_congestion_routing.game import lp
from moe_congestion_routing.metrics.probe_comparison import HalfSplitRow, half_split_row
from moe_congestion_routing.metrics.probe_series import read_dump, read_series


def _rows_for_path(job: tuple) -> list:
    """One dump's rows. Re-reads the dump here because under spawn every argument is pickled."""
    path, resamples, seed, layers = job
    dump = read_dump(path)
    affinities = dump.affinities()
    bias = dump.expert_bias()
    rows = []
    for axis, layer in enumerate(dump.layer_numbers):
        if layers and layer not in layers:
            continue
        a = affinities[axis]
        half = a.shape[0] // 2
        duals_a = lp.solve(a[:half], dump.topk).capacity_duals
        duals_b = lp.solve(a[half:], dump.topk).capacity_duals
        rows.append(
            half_split_row(
                bias[axis],
                duals_a,
                duals_b,
                step=dump.step,
                layer=layer,
                resamples=resamples,
                # Seeded per (step, layer) so a rerun reproduces the interval and two cells do not
                # share one resample pattern.
                seed=seed + 1000 * dump.step + layer,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", metavar="RUNDIR", help="a run directory holding probes/*.npz")
    parser.add_argument(
        "--asset", help="dump directory stem, required when a run probed more than one"
    )
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument(
        "--step", type=int, action="append", dest="steps", help="iteration number, repeatable"
    )
    parser.add_argument("--layers", type=int, action="append", dest="layers", help="repeatable")
    parser.add_argument("--resamples", type=int, default=10000, help="bootstrap draws per cell")
    parser.add_argument("--seed", type=int, default=0, help="base bootstrap seed")
    parser.add_argument("--jobs", type=int, default=4, help="worker processes")
    args = parser.parse_args()

    series = read_series(args.run_dir, asset=args.asset)
    dumps = series.dumps
    if args.steps:
        by_step = {d.step: d for d in dumps}
        missing = [s for s in args.steps if s not in by_step]
        if missing:
            parser.error(f"steps {missing} are not among {[d.step for d in dumps]}")
        dumps = [by_step[s] for s in args.steps]

    layers = tuple(args.layers) if args.layers else ()
    jobs = [(str(d.path), args.resamples, args.seed, layers) for d in dumps]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with (
        open(out_path, "w", newline="") as handle,
        ProcessPoolExecutor(
            max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as executor,
    ):
        writer = csv.writer(handle)
        writer.writerow(HalfSplitRow._fields)
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

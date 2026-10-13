#!/usr/bin/env python
"""Solve the tail x strided x spread triad on the two step-500 checkpoints' cross-probe cells.

A third, full-span asset was drawn expressly so the tail and strided assets' `kappa`, published
against each other in `crossprobe_2x2_matched-16384_step500.csv`, could be triangulated against a
third, more representative one. This reads the different probes of the A and B runs (step 500)
cells, prices the one designated 16,384-token unit each asset contributes, and writes every
pairwise and triad-identity row `metrics.triad` computes.

`--project-code-axis` corrects a composition confound: tail, strided and spread sit on a shared
content axis, with the code-heavy content concentrated in each run's own strided u0 (excluded
from every pairing because a single expert dominates its routing). This prices that unit purely
as an axis reference, regardless of its own concentration screen, and projects it out of `bias`
and every priced price vector before scoring, which is symmetric because projecting the prices
alone would leave `bias`'s own axis component correlating with nothing. It also prices the
rejected off1912 strided candidate's u0 as a second, independent axis definition and reports the
per-layer angle between the two, since the two agreeing is what licenses using this one at all.

Usage:
    uv run python scripts/run_triad.py
    uv run python scripts/run_triad.py --out .../triad_step500_corrected.csv --project-code-axis
"""

import argparse
import csv
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from moe_congestion_routing.game import lp
from moe_congestion_routing.metrics.probe_comparison import (
    UNIT_TOKENS,
    axis_angle_degrees,
    project_out,
)
from moe_congestion_routing.metrics.probe_series import probe_dump_path, read_dump
from moe_congestion_routing.metrics.triad import TriadRow, priced_unit, triad_rows

CROSSPROBE_DIR = Path("artifacts/exp1/crossprobe")
DEFAULT_OUT_CSV = "assets/results/price-recovery/triad_step500_three-pairings.csv"
RUNS = ("a", "b")

# (role, cell suffix, unit start/stop in tokens). "spread" needs both its units, whereas the
# other two assets contribute only the one designated unit.
UNIT_JOBS = (
    ("tail", "tail", 0, UNIT_TOKENS),
    ("strided", "strided", UNIT_TOKENS, 2 * UNIT_TOKENS),
    ("spread_u0", "spread_off1", 0, UNIT_TOKENS),
    ("spread_u1", "spread_off1", UNIT_TOKENS, 2 * UNIT_TOKENS),
)

# The two axis-defining units are never priced against the triad, so they are screened neither
# in solving nor in scoring: only their direction matters, and both are concentrated by
# construction. "strided_axis" is the axis carried across the trajectory, whereas
# "strided_axis_check" is the independent, rejected candidate it is cross-checked against at
# step 500.
AXIS_JOBS = (
    ("strided_axis", "strided", 0, UNIT_TOKENS),
    ("strided_axis_check", "strided_off1912", 0, UNIT_TOKENS),
)


def _solve_unit(job: tuple) -> tuple:
    """``(run, role, axis, duals_or_None)`` for one unit and layer, as a pool task.

    Module level and taking a path rather than a dump, because the pool is a ``spawn`` pool,
    which pickles both the callable and its arguments. ``screened`` is ``False`` for an
    axis-reference job, which is priced with the raw LP regardless of its own concentration.
    """
    run, role, path, axis, start, stop, topk, screened = job
    dump = read_dump(path)
    routing = dump.routing_map()[axis, start:stop, :]
    affinities = dump.affinities()[axis, start:stop, :]
    if screened:
        _, duals = priced_unit(routing, affinities, topk)
    else:
        duals = lp.solve(affinities, topk).capacity_duals
    return run, role, axis, duals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT_CSV)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--project-code-axis",
        action="store_true",
        help=(
            "project each run's own strided u0 (the code-heavy composition confound) out of "
            "bias and every priced price vector before scoring each cell"
        ),
    )
    args = parser.parse_args()

    cells = [("tail", "tail"), ("strided", "strided"), ("spread", "spread_off1")]
    if args.project_code_axis:
        # The axis-check role reads the rejected off1912 candidate, which exists only to define
        # a second, independent axis at step 500 and is never priced against the triad.
        cells.append(("strided_off1912", "strided_off1912"))

    dumps = {}
    for run in RUNS:
        for role, cell in cells:
            path = probe_dump_path(CROSSPROBE_DIR / f"{run}_{cell}", 0)
            dumps[(run, role)] = read_dump(path)

    for run in RUNS:
        bias = dumps[(run, "tail")].expert_bias()
        layers = dumps[(run, "tail")].layer_numbers
        for role in ("strided", "spread"):
            other = dumps[(run, role)]
            if other.layer_numbers != layers:
                raise ValueError(
                    f"run {run}: {role} cell's layers {other.layer_numbers} != {layers}"
                )
            if not np.array_equal(other.expert_bias(), bias):
                raise ValueError(
                    f"run {run}: {role} cell's expert_bias does not match the tail cell's, so "
                    "these are not probes of the same checkpoint"
                )

    # "spread_u0" and "spread_u1" both read the "spread" dump, and only their token slice differs.
    dump_role_of = {
        "tail": "tail",
        "strided": "strided",
        "spread_u0": "spread",
        "spread_u1": "spread",
        "strided_axis": "strided",
        "strided_axis_check": "strided_off1912",
    }
    jobs = []
    for run in RUNS:
        for unit_role, _cell, start, stop in UNIT_JOBS:
            dump = dumps[(run, dump_role_of[unit_role])]
            for axis in range(len(dump.layer_numbers)):
                jobs.append((run, unit_role, dump.path, axis, start, stop, dump.topk, True))
        if args.project_code_axis:
            for unit_role, _cell, start, stop in AXIS_JOBS:
                dump = dumps[(run, dump_role_of[unit_role])]
                for axis in range(len(dump.layer_numbers)):
                    jobs.append((run, unit_role, dump.path, axis, start, stop, dump.topk, False))

    print(f"solving {len(jobs)} LPs at n={UNIT_TOKENS}", file=sys.stderr)
    duals_by = {}
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        for i, (run, role, axis, duals) in enumerate(executor.map(_solve_unit, jobs), start=1):
            duals_by[(run, role, axis)] = duals
            print(f"  {i}/{len(jobs)}", file=sys.stderr, flush=True)

    rows: list[TriadRow] = []
    for run in RUNS:
        tail_dump = dumps[(run, "tail")]
        bias = tail_dump.expert_bias()
        for axis, layer in enumerate(tail_dump.layer_numbers):
            tail_u0 = duals_by[(run, "tail", axis)]
            strided_u1 = duals_by[(run, "strided", axis)]
            spread_u0 = duals_by[(run, "spread_u0", axis)]
            spread_u1 = duals_by[(run, "spread_u1", axis)]
            layer_bias = bias[axis]

            if args.project_code_axis:
                code_axis = duals_by[(run, "strided_axis", axis)]
                axis_check = duals_by[(run, "strided_axis_check", axis)]
                angle = axis_angle_degrees(code_axis, axis_check)
                print(
                    f"axis angle: run {run} layer {layer}: {angle:.2f} degrees "
                    "(strided u0 vs off1912 u0)",
                    file=sys.stderr,
                )
                layer_bias = project_out(layer_bias, code_axis)
                tail_u0, strided_u1, spread_u0, spread_u1 = (
                    None if duals is None else project_out(duals, code_axis)
                    for duals in (tail_u0, strided_u1, spread_u0, spread_u1)
                )

            for role, duals in (
                ("tail", tail_u0),
                ("strided", strided_u1),
                ("spread_u0", spread_u0),
                ("spread_u1", spread_u1),
            ):
                if duals is None:
                    print(f"refused: run {run} layer {layer} {role}", file=sys.stderr)
            rows.extend(
                triad_rows(
                    run,
                    layer,
                    layer_bias,
                    tail_u0,
                    strided_u1,
                    spread_u0,
                    spread_u1,
                    seed=1000 * layer + (0 if run == "a" else 500),
                )
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TriadRow._fields)
        for row in rows:
            writer.writerow(row)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Solve the tail x strided x spread triad on the two step-500 checkpoints' cross-probe cells.

A third, full-span asset was drawn expressly so the tail and strided assets' `kappa`, published
against each other in `crossprobe_2x2_matched-16384_step500.csv`, could be triangulated against a
third, more representative one. This reads the different probes of the A and B runs (step 500)
cells, prices the one designated 16,384-token unit each asset contributes, and writes every
pairwise and triad-identity row `metrics.triad` computes.

Usage:
    uv run python scripts/run_triad.py
"""

import csv
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from moe_congestion_routing.metrics.probe_comparison import UNIT_TOKENS
from moe_congestion_routing.metrics.probe_series import probe_dump_path, read_dump
from moe_congestion_routing.metrics.triad import TriadRow, priced_unit, triad_rows

CROSSPROBE_DIR = Path("artifacts/exp1/crossprobe")
OUT_CSV = Path("assets/results/price-recovery/triad_step500_three-pairings.csv")
RUNS = ("a", "b")

# (role, cell suffix, unit start/stop in tokens). "spread" needs both its units, whereas the
# other two assets contribute only the one designated unit.
UNIT_JOBS = (
    ("tail", "tail", 0, UNIT_TOKENS),
    ("strided", "strided", UNIT_TOKENS, 2 * UNIT_TOKENS),
    ("spread_u0", "spread_off1", 0, UNIT_TOKENS),
    ("spread_u1", "spread_off1", UNIT_TOKENS, 2 * UNIT_TOKENS),
)


def _solve_unit(job: tuple) -> tuple:
    """``(run, role, axis, duals_or_None)`` for one unit and layer, as a pool task.

    Module level and taking a path rather than a dump, because the pool is a ``spawn`` pool,
    which pickles both the callable and its arguments.
    """
    run, role, path, axis, start, stop, topk = job
    dump = read_dump(path)
    routing = dump.routing_map()[axis, start:stop, :]
    affinities = dump.affinities()[axis, start:stop, :]
    _, duals = priced_unit(routing, affinities, topk)
    return run, role, axis, duals


def main() -> None:
    dumps = {}
    for run in RUNS:
        for role, cell in (("tail", "tail"), ("strided", "strided"), ("spread", "spread_off1")):
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
    }
    jobs = []
    for run in RUNS:
        for unit_role, _cell, start, stop in UNIT_JOBS:
            dump = dumps[(run, dump_role_of[unit_role])]
            for axis in range(len(dump.layer_numbers)):
                jobs.append((run, unit_role, dump.path, axis, start, stop, dump.topk))

    print(f"solving {len(jobs)} LPs at n={UNIT_TOKENS}", file=sys.stderr)
    duals_by = {}
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
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
                    bias[axis],
                    tail_u0,
                    strided_u1,
                    spread_u0,
                    spread_u1,
                    seed=1000 * layer + (0 if run == "a" else 500),
                )
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TriadRow._fields)
        for row in rows:
            writer.writerow(row)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Solve the cross-probe cells into matched 16,384-token units and cache their prices.

The 2x2 exists to separate two things §6.1 could not: the probe instrument and run identity. Doing
that needs every comparison made at the same token count, because the LP's capacity is
``ceil(n*K/E)`` and a 32,768-token instance is a different problem rather than a bigger sample of
the same one. So the strided cells are cut in half and the tail cells are used whole, giving six
units of 16,384 tokens each at capacity 2048.

Each unit is ``(run, asset, half)``. Holding asset and half fixed and changing the run isolates the
run effect; holding run and half fixed and changing the asset isolates the instrument effect. The
per-expert dual vectors are cached, because the cross-asset price correlation -- two batches from
*different assets* on the *same* weights, which share no document pool and so inherit none of the
within-asset anticorrelation -- is computed from them and is the only clean test of the
independence assumption the halves violate by construction.

Usage:
    uv run python scripts/run_crossprobe_analysis.py
    uv run python scripts/run_crossprobe_analysis.py --refresh   # re-solve instead of using cache
"""

import argparse
import csv
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from moe_congestion_routing.game import lp
from moe_congestion_routing.game.compare import dual_agreement
from moe_congestion_routing.metrics.probe_series import read_dump

UNIT_TOKENS = 16384
CACHE = Path("artifacts/game/crossprobe_duals.npz")
OUT_CSV = Path("artifacts/game/crossprobe_units.csv")

# (unit key, cell, which 16,384-token slice). A tail cell holds exactly one unit; a strided cell
# holds two, and they are NOT interchangeable -- the strided first half is the batch §6.5 found
# anomalous, so pooling the two would average the finding away.
UNITS = (
    ("a/tail/h1", "a_tail", 0),
    ("b/tail/h1", "b_tail", 0),
    ("a/strided/h1", "a_strided", 0),
    ("a/strided/h2", "a_strided", 1),
    ("b/strided/h1", "b_strided", 0),
    ("b/strided/h2", "b_strided", 1),
)


def _solve(job: tuple) -> tuple:
    """``(key, layer, duals, bias, corr, spread)`` for one unit and layer, as a pool task.

    Module level and taking a path rather than a dump, because the pool is a ``spawn`` pool, which
    pickles both the callable and its arguments.
    """
    key, path, half, axis = job
    dump = read_dump(path)
    affinities = dump.affinities()[axis]
    slice_ = affinities[half * UNIT_TOKENS : (half + 1) * UNIT_TOKENS]
    duals = lp.solve(slice_, dump.topk).capacity_duals
    bias = dump.expert_bias()[axis]
    correlation, linf = dual_agreement(bias, duals)
    return (
        key,
        int(dump.layer_numbers[axis]),
        np.asarray(duals, dtype=np.float64),
        np.asarray(bias, dtype=np.float64),
        float(correlation),
        float(linf),
        float(duals.max() - duals.min()),
    )


def solve_all(jobs: list) -> dict:
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        out = {}
        for i, row in enumerate(executor.map(_solve, jobs), start=1):
            out[(row[0], row[1])] = row[2:]
            print(f"  {i}/{len(jobs)}", file=sys.stderr, flush=True)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="re-solve instead of using the cache"
    )
    args = parser.parse_args()

    jobs, missing = [], []
    for key, cell, half in UNITS:
        path = f"artifacts/exp1/crossprobe/{cell}/probes/iter_0000000.npz"
        if not Path(path).exists():
            missing.append(cell)
            continue
        for axis in range(8):
            jobs.append((key, path, half, axis))
    if missing:
        print(f"skipping absent cells: {sorted(set(missing))}", file=sys.stderr)
    if not jobs:
        raise SystemExit("no cross-probe dumps found")

    if CACHE.exists() and not args.refresh:
        print(f"using cached {CACHE} (pass --refresh to re-solve)")
        blob = np.load(CACHE, allow_pickle=False)
        solved = {}
        for name in blob.files:
            if not name.endswith("|duals"):
                continue
            key, layer = name.rsplit("|", 1)[0].rsplit("@", 1)
            solved[(key, int(layer))] = (
                blob[f"{key}@{layer}|duals"],
                blob[f"{key}@{layer}|bias"],
                float(blob[f"{key}@{layer}|scalars"][0]),
                float(blob[f"{key}@{layer}|scalars"][1]),
                float(blob[f"{key}@{layer}|scalars"][2]),
            )
    else:
        print(f"solving {len(jobs)} LPs at n={UNIT_TOKENS}", file=sys.stderr)
        solved = solve_all(jobs)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        flat = {}
        for (key, layer), (duals, bias, corr, linf, spread) in solved.items():
            flat[f"{key}@{layer}|duals"] = duals
            flat[f"{key}@{layer}|bias"] = bias
            flat[f"{key}@{layer}|scalars"] = np.array([corr, linf, spread])
        np.savez(CACHE, **flat)
        print(f"cached {CACHE}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["unit", "run", "asset", "half", "layer", "corr_bias_price", "linf", "price_spread"]
        )
        for (key, layer), (_, _, corr, linf, spread) in sorted(solved.items()):
            run, asset, half = key.split("/")
            writer.writerow([key, run, asset, half, layer, corr, linf, spread])
    print(f"wrote {OUT_CSV} ({len(solved)} rows)")


if __name__ == "__main__":
    main()

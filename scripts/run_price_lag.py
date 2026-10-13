#!/usr/bin/env python
"""Measure whether the stored ALF-LB bias lags past prices more than future ones.

`run_kappa_trajectory.py --dump-duals` already pays for the 336 LP solves this needs and writes
their duals, biases and per-side admissibility to one `.npz`. This script does no LP work at all:
per `(run, layer)` it takes that run's own longest evenly spaced admissible run of steps (dropping
any isolated point, since a lag shift indexes by dump position and only means what it claims when
every consecutive pair is one dump apart), then reports the asymmetry statistic `A(k) = c(+k) -
c(-k)` for `lag_dumps` in `-max_lag..max_lag`.

`A(k) > 0` at small positive `k` is evidence the bias trails the price; `A(k) ~ 0` is ambiguous
between no lag and a lag masked by the bias's own influence on the future prices it is scored
against, because `b(t)` shapes routing which shapes the affinities the router sees next. Both
readings are bounded further by the probe spacing: this can see a lag of 25 steps and cannot see
one of 5. Neither bound is optional when reading the output.

Usage:
    uv run python scripts/run_kappa_trajectory.py --out artifacts/game/kappa_trajectory.csv \\
        --dump-duals artifacts/game/kappa_trajectory_duals.npz
    uv run python scripts/run_price_lag.py --duals artifacts/game/kappa_trajectory_duals.npz
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from moe_congestion_routing.metrics.probe_comparison import (
    PriceLagRow,
    PriceLagStepRow,
    longest_admissible_run,
    price_lag_per_step_rows,
    price_lag_rows,
)

DEFAULT_SUMMARY_OUT = "assets/results/price-recovery/price-lag_both-runs_asymmetry.csv"
DEFAULT_PER_STEP_OUT = "assets/results/price-recovery/price-lag_both-runs_per-step.csv"


def _rows_for_run_and_layer(
    run: str,
    layer: int,
    steps: np.ndarray,
    admissible: np.ndarray,
    bias: np.ndarray,
    duals: np.ndarray,
    max_lag: int,
) -> tuple[list[PriceLagRow], list[PriceLagStepRow]]:
    """This `(run, layer)`'s own longest admissible run, then both row kinds over it."""
    run_steps = longest_admissible_run([int(s) for s in steps], [bool(a) for a in admissible])
    position_of_step = {int(s): i for i, s in enumerate(steps)}
    positions = [position_of_step[s] for s in run_steps]
    run_bias = bias[positions]
    run_duals = duals[positions]
    summary = price_lag_rows(run_steps, run_bias, run_duals, run=run, layer=layer, max_lag=max_lag)
    per_step = price_lag_per_step_rows(
        run_steps, run_bias, run_duals, run=run, layer=layer, max_lag=max_lag
    )
    return summary, per_step


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duals", required=True, help="the .npz from --dump-duals")
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--per-step-out", default=DEFAULT_PER_STEP_OUT)
    parser.add_argument("--max-lag", type=int, default=4)
    args = parser.parse_args()

    data = np.load(args.duals)
    steps, layers = data["steps"], data["layers"]

    by_run = {
        "A": ("admissible_a", "bias_a", "duals_a"),
        "B": ("admissible_b", "bias_b", "duals_b"),
    }
    summary_rows: list[PriceLagRow] = []
    per_step_rows: list[PriceLagStepRow] = []
    for layer in sorted({int(x) for x in layers}):
        mask = layers == layer
        order = np.argsort(steps[mask])
        layer_steps = steps[mask][order]
        for run, (admissible_key, bias_key, duals_key) in by_run.items():
            admissible = data[admissible_key][mask][order]
            bias = data[bias_key][mask][order]
            duals = data[duals_key][mask][order]
            summary, per_step = _rows_for_run_and_layer(
                run, layer, layer_steps, admissible, bias, duals, args.max_lag
            )
            summary_rows.extend(summary)
            per_step_rows.extend(per_step)

    summary_out = Path(args.summary_out)
    per_step_out = Path(args.per_step_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    per_step_out.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PriceLagRow._fields)
        for row in summary_rows:
            writer.writerow(row)
    with open(per_step_out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PriceLagStepRow._fields)
        for row in per_step_rows:
            writer.writerow(row)

    print(f"wrote {summary_out}: {len(summary_rows)} rows")
    print(f"wrote {per_step_out}: {len(per_step_rows)} rows")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Measure how much the stored ALF-LB bias moves between probe dumps, in units of its own step.

Megatron's update is `expert_bias += sign(load_error) * eta`, so every training step moves every
expert's bias by exactly `eta`. What varies is how often the sign flips. That fixes the reference
scale for a gap of `g` training steps: a memoryless expert whose sign flips at random accumulates
`sqrt(g) * eta`, and one drifting steadily accumulates `g * eta`. Probe dumps here are 25 steps
apart, so the two ends are 5 eta and 25 eta.

The point of measuring it is that a bias oscillating at its own step size cannot track a population
price more finely than that step size, which would put a floor under `kappa`. A floor that is
uniform across layers cannot, however, explain a `kappa` decay that is not.

**The single-number version is ill-posed, so the structure function is the reported object.** The
same series gives 3.5 to 8.1 eta depending on how much low-frequency movement a detrending window
calls drift rather than jitter, and quoting the window with the number papers over that rather than
fixing it. `J(D) = sd(b(t+D) - b(t))` as a curve in `D` has no such freedom, and its *shape* is the
diagnostic: band oscillation plateaus almost at once, a random walk grows as `sqrt(D)`, steady drift
grows as `D`. The fitted exponent `alpha` from `log J` against `log D` names which. If a single
number is ever needed downstream, pre-register it as `J` at one named `D`.

Usage:
    uv run python scripts/run_bias_jitter.py RUNDIR [RUNDIR ...] --out artifacts/game/jitter.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from moe_congestion_routing.metrics.probe_series import read_series

WINDOWS = (3, 5, 7)
# Lags in dumps. The series is 21 dumps, so lag 16 is the longest with enough pairs to average.
LAGS = (1, 2, 4, 8, 12, 16)


def rows_for_run(run_dir: str, eta: float, asset: str | None) -> list:
    series = read_series(run_dir, asset=asset)
    dumps = series.dumps
    steps = np.array([d.step for d in dumps])
    gaps = np.unique(np.diff(steps))
    bias = np.stack([d.expert_bias() for d in dumps])  # [T, L, E]

    out = []
    for axis, layer in enumerate(dumps[0].layer_numbers):
        b = bias[:, axis, :]
        diffs = np.diff(b, axis=0) / eta
        row = {
            "run": run_dir,
            "layer": int(layer),
            "step_gap": int(gaps[0]) if gaps.size == 1 else -1,
            "jitter_raw": float(np.std(diffs)),
            "lag1_autocorr": float(
                np.mean([np.corrcoef(diffs[:-1, e], diffs[1:, e])[0, 1] for e in range(b.shape[1])])
            ),
            "net_displacement": float(np.mean(np.abs(b[-1] - b[0]) / eta)),
        }
        for k in WINDOWS:
            kernel = np.ones(k) / k
            trend = np.stack(
                [np.convolve(diffs[:, e], kernel, mode="same") for e in range(b.shape[1])], 1
            )
            row[f"jitter_detrended_w{k}"] = float(np.std(diffs - trend))
        # The structure function, which is what the docstring argues should be reported instead of
        # any one of the detrended numbers above.
        for lag in LAGS:
            row[f"J_lag{lag}"] = float(np.std((b[lag:] - b[:-lag]) / eta))
        gap = int(gaps[0]) if gaps.size == 1 else 1
        logd = np.log([lag * gap for lag in LAGS])
        logj = np.log([row[f"J_lag{lag}"] for lag in LAGS])
        row["structure_alpha"] = float(np.polyfit(logd, logj, 1)[0])

        t = np.arange(diffs.shape[0])
        design = np.vstack([t, np.ones_like(t)]).T
        fit = design @ np.linalg.lstsq(design, diffs, rcond=None)[0]
        row["jitter_detrended_linear"] = float(np.std(diffs - fit))
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", metavar="RUNDIR")
    parser.add_argument("--eta", type=float, default=1.0e-3, help="moe_router_bias_update_rate")
    parser.add_argument(
        "--asset", help="dump directory stem, required when a run probed more than one"
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        rows.extend(rows_for_run(run_dir, args.eta, args.asset))

    gap = rows[0]["step_gap"]
    print(f"dump gap {gap} steps: memoryless floor {np.sqrt(gap):.1f} eta, pure drift {gap} eta\n")
    print(
        f"{'run':<12} {'L':>2} {'w=5':>6} {'linear':>7} "
        + " ".join(f"J({lag * gap}):".rjust(8) for lag in LAGS)
        + f" {'alpha':>6} {'reading':>14}"
    )
    for r in rows:
        alpha = r["structure_alpha"]
        reading = "plateau" if alpha < 0.2 else ("random walk" if alpha < 0.7 else "drift")
        print(
            f"{Path(r['run']).name:<12} {r['layer']:2d} {r['jitter_detrended_w5']:6.2f} "
            f"{r['jitter_detrended_linear']:7.2f} "
            + " ".join(f"{r[f'J_lag{lag}']:8.1f}" for lag in LAGS)
            + f" {alpha:6.2f} {reading:>14}"
        )
    print("\nA plateau in J would mean band oscillation. Growth means the bias is going somewhere.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

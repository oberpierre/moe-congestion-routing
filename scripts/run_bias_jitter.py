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

from moe_congestion_routing.metrics.probe_comparison import segment_autocorr
from moe_congestion_routing.metrics.probe_series import read_series

WINDOWS = (3, 5, 7)
# Lags in dumps. The series is 21 dumps, so lag 16 is the longest with enough pairs to average.
LAGS = (1, 2, 4, 8, 12, 16)

DEFAULT_AUTOCORR_OUT = "assets/results/price-recovery/bias-autocorr_both-runs_segmented.csv"
# The committed cross-asset kappa trajectory, joined by layer for the second registered prediction.
DEFAULT_KAPPA_CSV = "assets/results/price-recovery/kappa-trajectory_cross-asset_21-steps.csv"
# The registered window for kappa's decay rate: after the early rise and before the run ends.
KAPPA_SLOPE_STEP_LO = 100
KAPPA_SLOPE_STEP_HI = 500


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


def kappa_decay_rates(kappa_csv: str) -> dict[int, float]:
    """Per-layer slope of `kappa` against step over `[KAPPA_SLOPE_STEP_LO, KAPPA_SLOPE_STEP_HI]`.

    Read from the committed cross-asset trajectory rather than recomputed, so this and
    `run_kappa_trajectory.py`'s own LP solves never disagree. `kappa` is a joint statistic over
    both runs (`run_kappa_trajectory.py`'s `RUN_A`/`RUN_B`), so the rate is per layer only and is
    joined onto both runs' rows here. Refused (non-admissible) cells are skipped rather than fit,
    because their `kappa` is NaN.
    """
    by_layer: dict[int, list[tuple[int, float]]] = {}
    with open(kappa_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            if not (KAPPA_SLOPE_STEP_LO <= step <= KAPPA_SLOPE_STEP_HI):
                continue
            if row["admissible"] != "True":
                continue
            by_layer.setdefault(int(row["layer"]), []).append((step, float(row["kappa"])))
    return {
        layer: float(np.polyfit([s for s, _ in pairs], [k for _, k in pairs], 1)[0])
        for layer, pairs in by_layer.items()
    }


def autocorr_rows_for_run(
    run_dir: str,
    asset: str | None,
    segments: int,
    kappa_rates: dict[int, float],
) -> list[dict]:
    """One row per `(layer, segment)` of this run: `segment_autocorr` joined to `kappa_decay_rate`.

    Unlike `rows_for_run`'s pooled `lag1_autocorr`, this never divides by `eta`: an
    autocorrelation is scale-invariant, and the registered predictions are about its sign and
    level, not about a rate that the bf16 update path already distorts elsewhere.
    """
    series = read_series(run_dir, asset=asset)
    dumps = series.dumps
    steps = [d.step for d in dumps]
    bias = np.stack([d.expert_bias() for d in dumps])  # [T, L, E]

    out = []
    for axis, layer in enumerate(dumps[0].layer_numbers):
        diffs = np.diff(bias[:, axis, :], axis=0)
        for segment_index, start, n_diffs, autocorr in segment_autocorr(diffs, segments=segments):
            out.append(
                {
                    "run": run_dir,
                    "layer": int(layer),
                    "segments": segments,
                    "segment_index": segment_index,
                    "step_lo": steps[start],
                    "step_hi": steps[start + n_diffs],
                    "n_diffs": n_diffs,
                    "lag1_autocorr": autocorr,
                    "kappa_decay_rate": kappa_rates.get(int(layer), float("nan")),
                }
            )
    return out


def report_predictions(rows: list[dict]) -> None:
    """Print the two registered predictions, with their signs and nothing more: no verdict.

    Prediction 1 is the count of `(run, layer)` cells whose autocorrelation declines from the
    first segment to the last. Prediction 2 is each layer's late-segment level next to its
    `kappa` decay rate, so a reader can see whether the near-zero cells are the fast-decaying ones.
    """
    by_cell: dict[tuple[str, int], dict[int, float]] = {}
    for row in rows:
        by_cell.setdefault((row["run"], row["layer"]), {})[row["segment_index"]] = row[
            "lag1_autocorr"
        ]
    declined = sum(1 for levels in by_cell.values() if levels[max(levels)] < levels[min(levels)])
    print(
        f"\nprediction 1: {declined}/{len(by_cell)} cells decline "
        "from the first to the last segment"
    )

    last_segment = max(row["segment_index"] for row in rows)
    print("\nprediction 2: late-segment lag1_autocorr beside this layer's kappa decay rate")
    print(f"{'run':<12} {'L':>2} {'late autocorr':>14} {'kappa_decay_rate':>18}")
    for row in sorted(rows, key=lambda r: (r["run"], r["layer"])):
        if row["segment_index"] != last_segment:
            continue
        print(
            f"{Path(row['run']).name:<12} {row['layer']:2d} "
            f"{row['lag1_autocorr']:14.3f} {row['kappa_decay_rate']:18.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", metavar="RUNDIR")
    parser.add_argument("--eta", type=float, default=1.0e-3, help="moe_router_bias_update_rate")
    parser.add_argument(
        "--asset", help="dump directory stem, required when a run probed more than one"
    )
    parser.add_argument("--out")
    parser.add_argument(
        "--segments",
        type=int,
        default=2,
        help="time segments the Delta-b series is split into for segment_autocorr",
    )
    parser.add_argument("--autocorr-out", default=DEFAULT_AUTOCORR_OUT)
    parser.add_argument("--kappa-csv", default=DEFAULT_KAPPA_CSV)
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

    kappa_rates = kappa_decay_rates(args.kappa_csv)
    autocorr_rows = []
    for run_dir in args.run_dirs:
        autocorr_rows.extend(autocorr_rows_for_run(run_dir, args.asset, args.segments, kappa_rates))
    report_predictions(autocorr_rows)

    if args.autocorr_out:
        autocorr_out = Path(args.autocorr_out)
        autocorr_out.parent.mkdir(parents=True, exist_ok=True)
        with open(autocorr_out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(autocorr_rows[0]))
            writer.writeheader()
            writer.writerows(autocorr_rows)
        print(f"\nwrote {autocorr_out} ({len(autocorr_rows)} rows)")


if __name__ == "__main__":
    main()

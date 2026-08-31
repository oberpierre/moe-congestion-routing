#!/usr/bin/env python
"""Track `kappa`, the bias-versus-population-price correlation, across a training run.

`kappa` is measurable at step 500 from the cross-probe cells, where both batches come from the same
weights, and it is 0.82-0.85. Within one asset it is measurable at step 100 and dies of low `rho`
after step 200. Those are two different estimators, so the difference between 1.0 and 0.83 could be
a real decay of price internalization over training or an artifact of switching estimators. This
resolves that by running the cross-asset estimator at every step.

**The pairing, and its one assumption.** Run A probed the tail asset and run B the strided asset, so
at an intermediate step the only cross-asset pairing available takes one price from each *run*:
`p*(tail)` from run A and `p*(strided second half)` from run B, correlated against a single bias.
The two runs are not the same model. That contamination is calibrated at step 500, where the
cross-probe supplies both prices from one set of weights, so the clean and contaminated estimates
can be compared directly.

**Read that as one calibration point, not as a bound.** The tempting argument is that run divergence
is cumulative, so the step-500 contamination is an upper bound on earlier steps. It is not: the
run effect measured on the bias side runs 0.042 / 0.124 / 0.144 / 0.091 at steps 100 / 200 / 350 /
500, peaking in the middle. Anything this script reports at an intermediate step inherits an
uncalibrated contamination of that order.

Every unit is screened before it is priced, and a refused unit is emitted with `admissible = False`
and NaN statistics rather than dropped.

`--dump-duals PATH` additionally writes one `.npz`, for every requested `(step, layer)` cell,
carrying `duals_a`/`duals_b` (run A's tail / run B's strided-second-half capacity duals),
`bias_a`/`bias_b` (each run's own stored bias) and `admissible_a`/`admissible_b` from `screen_a`
and `screen_b` separately, so a later within-run analysis is not limited to the conjunction the
CSV's `admissible` column records. It does not change any value the CSV writes: the duals are a
side effect of the LP solves this script already pays for, or, when only one side of a cell's
conjunction is admissible, one extra solve for that side alone.

Usage:
    uv run python scripts/run_kappa_trajectory.py --out artifacts/game/kappa_trajectory.csv
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
    half_split_row,
    screen_batch,
)
from moe_congestion_routing.metrics.probe_series import probe_dump_path, read_dump, read_series

RUN_A = "artifacts/exp1/alflb/20260819-195028"  # tail asset, S=8
RUN_B = "artifacts/exp1/alflb/20260824-232033"  # strided asset, S=16

FIELDS = [
    "step",
    "layer",
    "admissible",
    "refused_reason",
    "bias_from",
    "load_ratio_tail",
    "load_ratio_strided",
    "rho",
    "corr_bias_tail",
    "corr_bias_strided",
    "kappa",
    "rho_boot_low",
    "rho_boot_high",
    "kappa_boot_low",
    "kappa_boot_high",
    "kappa_boot_undefined",
]


def _cell(job: tuple) -> tuple[dict, dict | None]:
    """One (step, layer). Screens both units first, and only then pays for two LP solves.

    Returns the CSV row unchanged by ``dump_duals`` and, only when it is set, a second dict of
    this cell's own duals/bias/admissibility for the price-lag `.npz`.
    """
    step, axis, resamples, seed, dump_duals = job
    a = read_dump(probe_dump_path(RUN_A, step))
    b = read_dump(probe_dump_path(RUN_B, step))
    layer = int(a.layer_numbers[axis])

    # Run A's tail dump is one unit. Run B's strided dump is twice the size, and its FIRST half is
    # the batch a single expert dominates, so the second half is the one paired here.
    tail_map = a.routing_map()[axis][:UNIT_TOKENS]
    strided_map = b.routing_map()[axis][UNIT_TOKENS:]
    screen_a = screen_batch(tail_map, a.topk)
    screen_b = screen_batch(strided_map, b.topk)

    row = dict.fromkeys(FIELDS, "")
    row.update(
        step=step,
        layer=layer,
        bias_from="A",
        load_ratio_tail=round(screen_a.max_load_over_balanced, 4),
        load_ratio_strided=round(screen_b.max_load_over_balanced, 4),
    )

    duals_payload = None
    if dump_duals:
        num_experts = a.num_experts
        # Solved per side of the conjunction rather than only when both pass, because a within-run
        # price-lag reader needs each run's own usable steps and the conjunction the CSV records
        # below discards a step whenever only the other run's screen fails.
        duals_a = (
            lp.solve(a.affinities()[axis][:UNIT_TOKENS], a.topk).capacity_duals
            if screen_a.admissible
            else np.full(num_experts, np.nan)
        )
        duals_b = (
            lp.solve(b.affinities()[axis][UNIT_TOKENS:], b.topk).capacity_duals
            if screen_b.admissible
            else np.full(num_experts, np.nan)
        )
        duals_payload = {
            "step": step,
            "layer": layer,
            "duals_a": duals_a,
            "duals_b": duals_b,
            "bias_a": a.expert_bias()[axis],
            "bias_b": b.expert_bias()[axis],
            "admissible_a": screen_a.admissible,
            "admissible_b": screen_b.admissible,
        }

    if not (screen_a.admissible and screen_b.admissible):
        reason = "; ".join(x for x in (screen_a.reason, screen_b.reason) if x)
        row.update(admissible=False, refused_reason=reason)
        for f in FIELDS[7:]:
            row[f] = float("nan")
        return row, duals_payload

    # Both sides passed, so the dump-duals solves above (when requested) are exactly the tail and
    # strided duals this branch needs, and reusing them pays for no LP solve twice.
    if dump_duals:
        duals_tail = duals_payload["duals_a"]
        duals_strided = duals_payload["duals_b"]
    else:
        duals_tail = lp.solve(a.affinities()[axis][:UNIT_TOKENS], a.topk).capacity_duals
        duals_strided = lp.solve(b.affinities()[axis][UNIT_TOKENS:], b.topk).capacity_duals
    stats = half_split_row(
        a.expert_bias()[axis],
        duals_tail,
        duals_strided,
        step=step,
        layer=layer,
        resamples=resamples,
        seed=seed + 1000 * step + layer,
    )
    row.update(
        admissible=True,
        refused_reason="",
        rho=stats.rho,
        corr_bias_tail=stats.corr_bias_a,
        corr_bias_strided=stats.corr_bias_b,
        kappa=stats.kappa,
        rho_boot_low=stats.rho_boot_low,
        rho_boot_high=stats.rho_boot_high,
        kappa_boot_low=stats.kappa_boot_low,
        kappa_boot_high=stats.kappa_boot_high,
        kappa_boot_undefined=stats.kappa_boot_undefined,
    )
    return row, duals_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=int, action="append", dest="steps")
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--dump-duals",
        default=None,
        help="also write duals_a/b, bias_a/b and admissible_a/b for every cell to this .npz",
    )
    args = parser.parse_args()

    steps_a = {d.step for d in read_series(RUN_A).dumps}
    steps_b = {d.step for d in read_series(RUN_B).dumps}
    steps = sorted(steps_a & steps_b)
    if args.steps:
        steps = [s for s in steps if s in set(args.steps)]
    jobs = [
        (s, axis, args.resamples, args.seed, bool(args.dump_duals))
        for s in steps
        for axis in range(8)
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = refused = 0
    duals_records: list[dict] = []
    with (
        open(out, "w", newline="") as handle,
        ProcessPoolExecutor(
            max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")
        ) as executor,
    ):
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        handle.flush()
        print(f"{len(jobs)} cells over {len(steps)} steps", file=sys.stderr, flush=True)
        for row, duals_payload in executor.map(_cell, jobs):
            writer.writerow(row)
            handle.flush()
            written += 1
            refused += 0 if row["admissible"] is True else 1
            if duals_payload is not None:
                duals_records.append(duals_payload)
            print(f"  {written}/{len(jobs)}", file=sys.stderr, flush=True)
    print(f"wrote {out}: {written} rows, {refused} refused by the screen")

    if args.dump_duals:
        dump_path = Path(args.dump_duals)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dump_path,
            steps=np.array([d["step"] for d in duals_records]),
            layers=np.array([d["layer"] for d in duals_records]),
            duals_a=np.stack([d["duals_a"] for d in duals_records]),
            duals_b=np.stack([d["duals_b"] for d in duals_records]),
            bias_a=np.stack([d["bias_a"] for d in duals_records]),
            bias_b=np.stack([d["bias_b"] for d in duals_records]),
            admissible_a=np.array([d["admissible_a"] for d in duals_records]),
            admissible_b=np.array([d["admissible_b"] for d in duals_records]),
        )
        print(f"wrote {dump_path}: {len(duals_records)} cells")


if __name__ == "__main__":
    main()

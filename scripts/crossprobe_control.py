#!/usr/bin/env python
"""Check the cross-probe procedure against dumps we already have, before reading its results.

Three tiers, because the right bar differs per quantity.

**Exact, for all four cells.** ``expert_bias`` is a model quantity that no probe asset touches, so
every cell's bias must equal its source run's stored bias at the same step, bit for bit: it is
loaded verbatim from the state dict and nothing between the load and the dump does arithmetic on
it. A failure here means the wrong checkpoint, or that ``finetune: true`` did not restore what the
procedure assumes. The asset sha is checked the same way, against the config naming the wrong npz.

**Self-consistent, for all four cells.** ``selection_conformance`` must report no untied
disagreement, or the dump cannot be scored offline at all.

**Within tolerance, for the two cells that re-measure an existing dump.** ``a_tail`` and
``b_strided`` repeat what their source run measured during training, so their affinities can be
compared. They are NOT bit-identical and must not be required to be: a bf16 forward pass reduces in
an order that depends on the kernels chosen, so running four GPUs where the original ran sixteen
leaves the first MoE layer exact and lets the difference accumulate with depth. Measured on a_tail:
layer 2 exact, layer 9 at 6.3e-2 worst-element, mean 7.2e-5, correlation 0.99996 -- and
``corr(b, p*)``, the quantity actually reported, agreeing to 1.2e-4.

So the gate is on ``corr(b, p*)`` and not on the bits. ``CORR_TOLERANCE`` sits about fifty times
below the smallest effect this analysis rests on and about ten times above the drift observed here,
which is the same shape of argument that fixed the conformance gate's own units defect.

Usage:
    uv run python scripts/crossprobe_control.py
    uv run python scripts/crossprobe_control.py --no-lp   # skip the LP tier (seconds, not minutes)
"""

import argparse
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from moe_congestion_routing.game import lp
from moe_congestion_routing.game.compare import dual_agreement
from moe_congestion_routing.metrics.probe_series import (
    probe_dump_path,
    read_dump,
    selection_conformance,
)

RUN_A = "artifacts/exp1/alflb/20260819-195028"
RUN_B = "artifacts/exp1/alflb/20260824-232033"

# cell -> (source run, the training dump it re-measures, or None when it is a new measurement)
CELLS = {
    "a_tail": (RUN_A, probe_dump_path(RUN_A, 500)),
    "a_strided": (RUN_A, None),
    "b_tail": (RUN_B, None),
    "b_strided": (RUN_B, probe_dump_path(RUN_B, 500)),
    "a_tail16": (RUN_A, None),
    "b_tail16": (RUN_B, None),
}

# Every reported effect here is 0.05 or larger, and the platform-to-platform agreement already
# accepted in this analysis is 6.5e-5. A correlation moving by less than this is not a difference
# any conclusion can see.
CORR_TOLERANCE = 1e-3


def _one_layer_correlation(job: tuple) -> tuple:
    """``(layer, corr(b_train, p*))`` for one layer, as a process-pool task.

    Module level and taking a path rather than a dump, because the pool is a ``spawn`` pool, which
    pickles both the callable and its arguments.
    """
    path, axis = job
    dump = read_dump(path)
    duals = lp.solve(dump.affinities()[axis], dump.topk).capacity_duals
    correlation, _ = dual_agreement(dump.expert_bias()[axis], duals)
    return dump.layer_numbers[axis], correlation


def _lp_correlations(path: str) -> dict:
    """``corr(b_train, p*)`` per layer for one dump, one LP solve per layer."""
    layers = len(read_dump(path).layer_numbers)
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        return dict(executor.map(_one_layer_correlation, [(path, a) for a in range(layers)]))


def check(cell: str, run_dir: str, reference: str | None, use_lp: bool) -> list:
    """Every failure for one cell, as strings. Empty means the cell is usable."""
    fresh_dir = f"artifacts/exp1/crossprobe/{cell}"
    try:
        fresh_path = probe_dump_path(fresh_dir, 0)
    except FileNotFoundError:
        return [f"missing {fresh_dir}/probes step 0"]
    fresh = read_dump(fresh_path)

    failures = []
    source = read_dump(probe_dump_path(run_dir, 500))
    if not np.array_equal(fresh.expert_bias(), source.expert_bias()):
        gap = float(np.abs(fresh.expert_bias() - source.expert_bias()).max())
        failures.append(
            f"expert_bias differs from {run_dir} step 500 by {gap:.3e}, so the "
            "checkpoint that loaded is not the one this cell names"
        )

    for row in selection_conformance(fresh):
        if row.untied_disagreements:
            failures.append(
                f"layer {row.layer} has {row.untied_disagreements} untied selection disagreements"
            )

    if reference is None:
        print(
            f"{cell}: {'PASS' if not failures else 'FAIL'}  (new measurement, no dump to "
            f"compare against; bias and conformance checked)"
        )
        return failures

    original = read_dump(reference)
    if fresh.token_sha256 != original.token_sha256:
        failures.append(
            f"probe asset sha {fresh.token_sha256[:16]} != {original.token_sha256[:16]}"
        )
        print(f"{cell}: FAIL  wrong asset")
        return failures

    gap = float(np.abs(fresh.affinities() - original.affinities()).max())
    if use_lp:
        got, want = _lp_correlations(fresh_path), _lp_correlations(reference)
        moved = {
            layer: abs(value - want[layer])
            for layer, value in got.items()
            if layer in want and abs(value - want[layer]) > CORR_TOLERANCE
        }
        for layer, delta in moved.items():
            failures.append(f"layer {layer}: corr(b, p*) moved {delta:.2e} > {CORR_TOLERANCE:.0e}")
        worst = max((abs(v - want[k]) for k, v in got.items() if k in want), default=float("nan"))
        detail = f"worst corr(b, p*) shift {worst:.2e} (tolerance {CORR_TOLERANCE:.0e})"
    else:
        detail = "LP tier skipped"

    print(
        f"{cell}: {'PASS' if not failures else 'FAIL'}  max|d affinity| {gap:.3e} "
        f"(bit-identity NOT required), {detail}"
    )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-lp",
        action="store_true",
        help="skip the LP tier, which is the only one that tests the reported quantity",
    )
    args = parser.parse_args()

    failures = {}
    for cell, (run_dir, reference) in CELLS.items():
        found = check(cell, run_dir, reference, use_lp=not args.no_lp)
        if found:
            failures[cell] = found

    if failures:
        print("", file=sys.stderr)
        for cell, found in failures.items():
            for line in found:
                print(f"{cell}: {line}", file=sys.stderr)
        raise SystemExit(1)
    print("\nControls pass. The 2x2 is comparable.")


if __name__ == "__main__":
    main()

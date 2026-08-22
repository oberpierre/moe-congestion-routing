#!/usr/bin/env python
"""Grid the synthetic ALF-LB-versus-LP comparison and write it to a CSV.

Builds a cross product of shapes, separations and seeds, then for each cell runs one annealed
comparison (the theorem's hypothesis test) and three deployed comparisons at the step sizes an
Both modes are needed because a fixed step size orbits a limit cycle rather than converging, so
only the annealed one tests the theorem's hypothesis.

Usage:
    uv run python scripts/run_game_comparison.py --out artifacts/game/alf_lb_vs_lp.csv
    uv run python scripts/run_game_comparison.py --out artifacts/game/quick.csv --quick
"""

import argparse
import csv
import dataclasses
import time
from collections import Counter
from pathlib import Path

from moe_congestion_routing.game.compare import Comparison, compare
from moe_congestion_routing.game.ensemble import Instance, affinities

# The annealed budget is 40000, not 20000, so the one large-shape cell that can settle
# (N=2048, E=64, K=8 at separation=2.0, seed=1) actually does: it settles at step 25582,
# past a 20000-step budget.
_SHAPES = [(512, 8, 2), (2048, 64, 8)]
_SEPARATIONS = [2.0, 0.2]
_SEEDS = [0, 1, 2]
_ANNEALED_ETA = 1e-2
_ANNEALED_STEPS = 40000
_DEPLOYED_ETAS = [1e-3, 1e-2, 1e-1]
_DEPLOYED_STEPS = 2000


def _grid(quick: bool) -> list[Instance]:
    shapes = [_SHAPES[0]] if quick else _SHAPES
    seeds = [_SEEDS[0]] if quick else _SEEDS
    return [
        Instance(n=n, e=e, k=k, separation=separation, seed=seed)
        for n, e, k in shapes
        for separation in _SEPARATIONS
        for seed in seeds
    ]


def _cells(quick: bool) -> list[tuple[Instance, str, float, int]]:
    """One (instance, mode, eta, steps) tuple per CSV row."""
    cells = []
    for inst in _grid(quick):
        cells.append((inst, "annealed", _ANNEALED_ETA, _ANNEALED_STEPS))
        for eta in _DEPLOYED_ETAS:
            cells.append((inst, "deployed", eta, _DEPLOYED_STEPS))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="only the (512, 8, 2) shape and seed 0, for a fast smoke run",
    )
    args = parser.parse_args()

    cells = _cells(args.quick)
    instance_fields = [f.name for f in dataclasses.fields(Instance)]
    header = instance_fields + list(Comparison._fields)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[Comparison] = []
    instances: list[Instance] = []
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (inst, mode, eta, steps) in enumerate(cells, start=1):
            start = time.perf_counter()
            a = affinities(inst)
            c = compare(a, inst.k, eta=eta, steps=steps, mode=mode)
            elapsed = time.perf_counter() - start
            print(
                f"[cell {i}/{len(cells)}] n={inst.n} e={inst.e} k={inst.k} "
                f"separation={inst.separation} seed={inst.seed} mode={mode} eta={eta} "
                f"tier={c.tier} done in {elapsed:.1f}s",
                flush=True,
            )
            writer.writerow(
                [getattr(inst, name) for name in instance_fields] + list(c._asdict().values())
            )
            rows.append(c)
            instances.append(inst)

    tier_counts = Counter(c.tier for c in rows)
    print(f"\ntier counts: {dict(tier_counts)}")
    for inst, c in zip(instances, rows, strict=True):
        if c.tier in ("settled", "tie_slack"):
            # gap_at_matched_cap is printed alongside max_load rather than gap_over_span
            # alone, because a gap without the realized max load next to it hides the
            # capacity violation that produced it.
            print(
                f"  [{c.tier}] n={inst.n} e={inst.e} k={inst.k} separation={inst.separation} "
                f"seed={inst.seed} mode={c.mode} eta={c.eta} max_load={c.max_load} "
                f"gap_at_matched_cap={c.gap_at_matched_cap:.6f} gap_over_span={c.gap_over_span:.4f}"
            )
            if c.tier == "settled":
                print(f"      dual_correlation={c.dual_correlation:.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Calibrate the softplus barrier's ``tau`` by comparing it against the capacity LP oracle.

For every ``(tau, asset, step, layer, unit)`` cell, prices the unit's affinities under the
softplus barrier (``losses/cost_families.py``'s ``'softplus_barrier'`` family, via the
incremental-arc oracle in ``game/incremental.py``) and compares the result against the
capacity-constrained LP ``game/lp.py`` already solves at ``cap = N*K/E`` on the same unit. As
``tau`` falls the two are expected to agree, because the barrier is the smooth relaxation of the
hard capacity and its marginal cost approaches the constraint's dual as the smoothing vanishes.
This script is the measurement that decides whether they do and over what range.

The LP oracle does not depend on ``tau``, so it is solved once per ``(asset, step, layer, unit)``
and reused across every ``tau`` in the sweep.

Usage:
    uv run python scripts/barrier_tau_sweep.py \\
        --run-dir artifacts/exp1/control/control-trunk \\
        --out artifacts/barrier_tau_sweep/control-trunk.csv
    uv run python scripts/barrier_tau_sweep.py --run-dir artifacts/exp1/control/control-trunk \\
        --out /tmp/probe.csv --asset standing_climbmix_small_16x2048 --layer 2 --step 0 \\
        --unit u0 --dry-run
"""

import argparse
import csv
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from moe_congestion_routing.game import lp
from moe_congestion_routing.game.incremental import solve_incremental
from moe_congestion_routing.losses.cost_families import marginal_cost
from moe_congestion_routing.metrics.phi_gap import arc_schedule_length
from moe_congestion_routing.metrics.probe_comparison import probe_units
from moe_congestion_routing.metrics.probe_series import read_dump

_DEFAULT_TAUS: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)


class Unit(NamedTuple):
    """One ``(asset, step, layer, unit)`` slice: which dump it came from and its token range."""

    dump_path: Path
    asset: str
    step: int
    layer: int
    unit: str
    start: int
    stop: int


class TauSweepRow(NamedTuple):
    tau: float
    asset: str
    step: int
    layer: int
    unit: str
    lam: float
    n: int
    e: int
    k: int
    cap: int
    num_arcs: int
    barrier_affinity: float
    barrier_max_load: int
    lp_affinity: float
    lp_max_load: int
    affinity_diff: float  # barrier - lp
    max_load_diff: int  # barrier - lp
    arc_growths: int
    dump_path: str


def enumerate_units(
    run_dir: Path,
    *,
    assets: list[str] | None,
    layers: list[int] | None,
    steps: list[int] | None,
    units: list[str] | None,
) -> list[Unit]:
    """Every ``(asset, step, layer, unit)`` this run's probe dumps can be sliced into.

    Reads only each dump's metadata, matching ``phi_gap_grid.enumerate_cells``'s own reason for
    doing so: listing every cell must not cost as much as solving a single one.
    """
    probes_dir = run_dir / "probes"
    if not probes_dir.is_dir():
        raise FileNotFoundError(f"no probes directory under {run_dir}")
    asset_dirs = sorted(p for p in probes_dir.iterdir() if p.is_dir())
    if not asset_dirs:
        raise FileNotFoundError(f"no per-asset probe directories under {probes_dir}")
    if assets is not None:
        wanted = set(assets)
        asset_dirs = [p for p in asset_dirs if p.name in wanted]

    result: list[Unit] = []
    for asset_dir in asset_dirs:
        for dump_path in sorted(asset_dir.glob("*.npz")):
            dump = read_dump(dump_path)
            if steps is not None and dump.step not in steps:
                continue
            dump_layers = [n for n in dump.layer_numbers if layers is None or n in layers]
            for unit_name, start, stop in probe_units(dump.meta["N"]):
                if units is not None and unit_name not in units:
                    continue
                for layer in dump_layers:
                    result.append(
                        Unit(
                            dump_path=dump_path,
                            asset=asset_dir.name,
                            step=dump.step,
                            layer=layer,
                            unit=unit_name,
                            start=start,
                            stop=stop,
                        )
                    )
    return result


def sweep_unit(unit: Unit, taus: tuple[float, ...], *, lam: float) -> list[TauSweepRow]:
    """One ``(asset, step, layer, unit)``'s row per ``tau``, the LP solved once and reused."""
    dump = read_dump(unit.dump_path)
    axis = dump.layer_numbers.index(unit.layer)
    scores = dump.router_scores()[axis]
    a = np.array(scores[unit.start : unit.stop])
    n, e = a.shape
    k = dump.topk
    balanced_load = n * k / e
    max_span = float((a.max(axis=1) - a.min(axis=1)).max())

    cap = n * k // e
    if cap * e != n * k:
        raise ValueError(
            f"{unit.dump_path} layer={unit.layer} unit={unit.unit}: n*k={n * k} is not divisible "
            f"by e={e}, so cap=N*K/E has rounding slack and the two oracles are not directly "
            "comparable"
        )
    lp_result = lp.solve(a, k, cap=cap)

    rows = []
    for tau in taus:
        num_arcs = arc_schedule_length(
            n, k, e, max_span, lam=lam, cost_family="softplus_barrier", tau=tau
        )
        arc_prices = marginal_cost(
            np.arange(1, num_arcs + 1),
            balanced_load,
            lam=lam,
            cost_family="softplus_barrier",
            tau=tau,
        )
        oracle = solve_incremental(a, k, arc_prices)
        barrier_max_load = int(oracle.loads.max())
        rows.append(
            TauSweepRow(
                tau=tau,
                asset=unit.asset,
                step=unit.step,
                layer=unit.layer,
                unit=unit.unit,
                lam=lam,
                n=n,
                e=e,
                k=k,
                cap=cap,
                num_arcs=num_arcs,
                barrier_affinity=oracle.affinity,
                barrier_max_load=barrier_max_load,
                lp_affinity=lp_result.objective,
                lp_max_load=lp_result.max_load,
                affinity_diff=oracle.affinity - lp_result.objective,
                max_load_diff=barrier_max_load - lp_result.max_load,
                arc_growths=oracle.arc_growths,
                dump_path=str(unit.dump_path),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="a training run directory")
    parser.add_argument("--out", required=True, type=Path, help="CSV path to write")
    parser.add_argument(
        "--tau",
        action="append",
        dest="taus",
        type=float,
        help=f"repeatable, defaults to {_DEFAULT_TAUS}",
    )
    parser.add_argument("--lam", type=float, default=1.0, help="lambda for the barrier price")
    parser.add_argument("--asset", action="append", dest="assets", help="repeatable")
    parser.add_argument("--layer", action="append", dest="layers", type=int, help="repeatable")
    parser.add_argument("--step", action="append", dest="steps", type=int, help="repeatable")
    parser.add_argument("--unit", action="append", dest="units", help="repeatable, e.g. u0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the (asset, step, layer, unit) cells that would be swept, and exit",
    )
    args = parser.parse_args()

    taus = tuple(args.taus) if args.taus else _DEFAULT_TAUS
    run_units = enumerate_units(
        args.run_dir, assets=args.assets, layers=args.layers, steps=args.steps, units=args.units
    )
    print(f"{len(run_units)} (asset, step, layer, unit) cells, {len(taus)} tau values each")

    if args.dry_run:
        for unit in run_units:
            print(f"  {unit.asset} step={unit.step} layer={unit.layer} unit={unit.unit}")
        return

    if args.out.exists():
        raise SystemExit(f"{args.out} exists, so this run would discard it. Pass a fresh --out.")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TauSweepRow._fields)
        for i, unit in enumerate(run_units):
            start = time.perf_counter()
            rows = sweep_unit(unit, taus, lam=args.lam)
            for row in rows:
                writer.writerow(row)
            handle.flush()
            elapsed = time.perf_counter() - start
            print(
                f"[{i + 1}/{len(run_units)}] {unit.asset} step={unit.step} layer={unit.layer} "
                f"unit={unit.unit} elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(f"wrote {len(run_units) * len(taus)} rows to {args.out}")


if __name__ == "__main__":
    main()

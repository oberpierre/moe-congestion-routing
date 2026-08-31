#!/usr/bin/env python
"""Score one probe dump's unit at one layer against the shared soft-potential reference game.

Prints the potential-gap row, once per declared reference cost, with the incremental-arc LP
oracle actually solved. This is a diagnostic driver over a single dump, not the sweep: the
per-arm CSV layout, the resume-on-restart read and the uniqueness assertion over
``(run_id, arm, asset, layer, step, cost_family, lam)`` belong to a separate grid driver over
many dumps, arms and steps.

Usage:
    uv run python scripts/run_phi_gap.py DUMP.npz --layer 2 --unit u0 \\
        --out artifacts/phi_gap/probe.csv
    uv run python scripts/run_phi_gap.py DUMP.npz --layer 2 --unit u0 --unit u1
"""

import argparse
import csv
import time
from pathlib import Path

from moe_congestion_routing.losses.cost_families import COST_FAMILIES
from moe_congestion_routing.metrics.phi_gap import PhiGapRow, phi_gap_rows
from moe_congestion_routing.metrics.probe_series import read_dump


def _write_csv(path: str, rows: list[PhiGapRow]) -> None:
    out_path = Path(path)
    # Refuse rather than truncate. Each row costs a minute of solving, and scoring a second unit
    # into the same path is the natural way to use this, so an overwrite would silently discard
    # work that cannot be recovered without paying for it again.
    if out_path.exists():
        raise SystemExit(f"{out_path} exists, so this run would discard it. Pass a fresh --out.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PhiGapRow._fields)
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dump", metavar="DUMP", help="path to one probe dump .npz file")
    parser.add_argument("--layer", type=int, required=True, help="Megatron layer number to score")
    parser.add_argument(
        "--unit",
        action="append",
        dest="units",
        required=True,
        help="a probe_units name such as u0, repeatable",
    )
    parser.add_argument("--lam", type=float, default=1.0, help="lambda for both reference costs")
    parser.add_argument(
        "--cost-family",
        action="append",
        dest="cost_families",
        choices=COST_FAMILIES,
        help="repeatable, defaults to both declared reference costs",
    )
    parser.add_argument("--out", help="CSV path, written in addition to the printed rows")
    args = parser.parse_args()

    cost_families = tuple(args.cost_families) if args.cost_families else COST_FAMILIES
    dump = read_dump(args.dump)

    all_rows: list[PhiGapRow] = []
    for unit in args.units:
        for cost_family in cost_families:
            start = time.perf_counter()
            rows = phi_gap_rows(dump, args.layer, unit, lam=args.lam, cost_families=(cost_family,))
            elapsed = time.perf_counter() - start
            row = rows[0]
            all_rows.append(row)
            print(
                f"unit={row.unit} layer={row.layer} step={row.step} cost={row.reference_cost} "
                f"lam={row.lam} admissible={row.admissible} "
                f"max_load/L={row.max_load_over_balanced:.3f} dead={row.dead_experts} "
                f"gap_per_token={row.gap_per_token:.6f} "
                f"affinity_shortfall={row.affinity_shortfall:.6f} "
                f"congestion_excess={row.congestion_excess:.6f} "
                f"gap_normalized={row.gap_normalized:.6f} normalizer={row.normalizer:.6f} "
                f"arc_growths={row.arc_growths} arcs_used_max={row.arcs_used_max} "
                f"max_fractional_deviation={row.max_fractional_deviation:.2e} "
                f"elapsed={elapsed:.1f}s"
            )

    if args.out:
        _write_csv(args.out, all_rows)
        print(f"wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

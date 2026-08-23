#!/usr/bin/env python
"""Turn one run's probe dumps into the two ALF-LB-versus-LP-price tables, per MoE layer.

Table 1 (verification) writes to ``--out-verification``: the annealed ALF-LB run on a dump's
own affinities, scored against the LP oracle. Table 2 (internalization) writes to
``--out-internalization``: the dump's own stored bias against that batch's LP capacity duals,
plus the resolvability gate that says whether the comparison means anything. They are two files
rather than one file with a "table" column, because they answer different questions and one of
them has no pass value.

Budget: at the real shape (N=16384, E=64, K=8) one LP solve is about 21s and the annealed
iterator is roughly 7 minutes per 20000 steps, so one 8-layer dump at the default
``--annealed-steps 40000`` budget is on the order of an hour and a half. Use ``--layers`` and
``--annealed-steps`` to shrink that for a quick check.

Usage:
    uv run python scripts/run_probe_comparison.py RUNDIR --bias-update-rate 1.0e-3 \\
        --out-verification artifacts/probe_cmp/verification.csv \\
        --out-internalization artifacts/probe_cmp/internalization.csv
"""

import argparse
import csv
from pathlib import Path

from moe_congestion_routing.game.compare import Comparison
from moe_congestion_routing.metrics.probe_comparison import (
    InternalizationRow,
    VerificationRow,
    internalization_rows,
    verification_rows,
)
from moe_congestion_routing.metrics.probe_series import read_series


def _write_verification_csv(path: str, rows: list[VerificationRow]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["step", "layer", "bias_update_rate", *Comparison._fields]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [row.step, row.layer, row.bias_update_rate, *row.comparison._asdict().values()]
            )


def _write_internalization_csv(path: str, rows: list[InternalizationRow]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(InternalizationRow._fields)
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", metavar="RUNDIR", help="a run directory holding probes/*.npz")
    parser.add_argument(
        "--bias-update-rate",
        type=float,
        required=True,
        help=(
            "moe_router_bias_update_rate the run was trained with. Required with no default "
            "because the dump does not record it and the resolvability gate divides by it"
        ),
    )
    parser.add_argument("--out-verification", required=True, help="Table 1 CSV output path")
    parser.add_argument("--out-internalization", required=True, help="Table 2 CSV output path")
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        dest="steps",
        help="a dump's iteration number, repeatable. Defaults to the series' last dump",
    )
    parser.add_argument(
        "--layers",
        type=int,
        action="append",
        dest="layers",
        help="a Megatron layer number, repeatable. Defaults to every layer in the dump",
    )
    parser.add_argument(
        "--annealed-steps",
        type=int,
        default=40000,
        help="Table 1's ALF-LB annealed step budget, the same one the synthetic grid uses",
    )
    args = parser.parse_args()

    series = read_series(args.run_dir)

    verification = verification_rows(
        series,
        bias_update_rate=args.bias_update_rate,
        annealed_steps=args.annealed_steps,
        steps=args.steps,
        layers=args.layers,
    )
    internalization = internalization_rows(
        series,
        bias_update_rate=args.bias_update_rate,
        steps=args.steps,
        layers=args.layers,
    )

    _write_verification_csv(args.out_verification, verification)
    _write_internalization_csv(args.out_internalization, internalization)

    print(f"wrote {len(verification)} verification rows to {args.out_verification}")
    print(f"wrote {len(internalization)} internalization rows to {args.out_internalization}")


if __name__ == "__main__":
    main()

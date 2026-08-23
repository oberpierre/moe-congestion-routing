#!/usr/bin/env python
"""Turn one or more run directories' ``probes/`` dumps into the router-saturation table.

Router saturation is the fraction of probe tokens whose selected expert set is unchanged
relative to a reference dump, per MoE layer, so 1.0 means routing on this batch has frozen.

Each positional argument is ``[ARM=]RUNDIR``: a leading ``ARM=`` labels that run's rows, a bare
``RUNDIR`` leaves ``arm`` empty. The arm is never inferred from the directory name, because there
is no launcher-written snapshot to read it from on the probe side, the way
``collect_eval_results.py`` reads one for evals.

Usage:
    uv run python scripts/collect_probe_metrics.py s2c=artifacts/probe_smoke/s2c_gpu_leg \\
        --allow-role dev
    uv run python scripts/collect_probe_metrics.py a=<run_a> b=<run_b> --out table.csv
"""

import argparse
import contextlib
import csv
import sys
from dataclasses import fields

from moe_congestion_routing.metrics.probe_series import (
    IncomparableProbes,
    SaturationRow,
    read_series,
    saturation_rows,
)


@contextlib.contextmanager
def _open_or_stdout(path: str | None):
    """``open(path, "w")`` when ``path`` is given, else ``sys.stdout`` (left open on exit)."""
    if path is None:
        yield sys.stdout
    else:
        with open(path, "w", newline="") as f:
            yield f


def _parse_positional(spec: str) -> tuple[str | None, str]:
    """Split ``[ARM=]RUNDIR`` into ``(arm, run_dir)``, ``arm`` is ``None`` with no ``=``."""
    arm, sep, run_dir = spec.partition("=")
    return (arm, run_dir) if sep else (None, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="one or more [ARM=]RUNDIR run directories to read probes/ from",
    )
    parser.add_argument(
        "--allow-role",
        action="append",
        default=None,
        help="role permitted to be scored, repeatable, defaulting to 'standing' alone",
    )
    parser.add_argument(
        "--reference-step",
        type=int,
        default=None,
        help="step to compare every dump against, defaulting to each series' last emitted step",
    )
    parser.add_argument("--out", default=None, help="write the CSV here instead of stdout")
    args = parser.parse_args()

    allow_roles = tuple(args.allow_role) if args.allow_role else ("standing",)

    try:
        series_list = [
            read_series(run_dir, arm=arm, allow_roles=allow_roles)
            for arm, run_dir in (_parse_positional(spec) for spec in args.run_dirs)
        ]
        rows = saturation_rows(series_list, reference_step=args.reference_step)
    except (IncomparableProbes, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    columns = [f.name for f in fields(SaturationRow)]
    with _open_or_stdout(args.out) as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: getattr(row, column) for column in columns})


if __name__ == "__main__":
    main()

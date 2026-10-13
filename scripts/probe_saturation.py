#!/usr/bin/env python
"""Turn one or more runs' probe dumps into the router-saturation table.

Per run this picks exactly one probe asset (the ``--asset`` decision: sigma' on the winners and
the logit scale are properties of the router, not of which tokens are probed, so a second asset
would only buy error bars) and streams one ``GateRow`` per MoE layer per dump to ``--out``,
holding at most one dump open at a time.

Usage:
    uv run python scripts/probe_saturation.py artifacts/exp1/control/control-trunk \\
        --out /tmp/sat_healthy.csv
    uv run python scripts/probe_saturation.py artifacts/exp1/control/control-trunk \\
        --asset standing_climbmix_small_strided_16x2048 --out /tmp/sat_strided.csv
"""

import argparse
import csv
import sys
from contextlib import contextmanager
from pathlib import Path

from moe_congestion_routing.metrics.probe_saturation import (
    GateRow,
    reduce_dump,
    select_asset_dir,
)


def _dump_iteration(path: Path) -> int:
    """The iteration number off the filename (``iter_%07d.npz``), so ``--every``/``--max-iter``
    can filter without opening the archive."""
    return int(path.stem.removeprefix("iter_"))


@contextmanager
def _open_or_stdout(path: str | None):
    """``open(path, "w")`` when ``path`` is given, else ``sys.stdout`` (left open on exit)."""
    if path is None:
        yield sys.stdout
    else:
        with open(path, "w", newline="") as f:
            yield f


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dirs", nargs="+", help="one or more run directories to read probes/ from"
    )
    parser.add_argument("--out", default=None, help="write the CSV here instead of stdout")
    parser.add_argument(
        "--asset",
        default=None,
        help="probe-asset directory stem to select, one asset per run "
        "(default: the run's only asset, an error if it has several)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="reduce every Nth dump in a run's asset, by ascending iteration (default: 1, all)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="skip dumps whose iteration exceeds this value (default: none, all)",
    )
    parser.add_argument("--sat", type=float, default=0.99, help="sigmoid saturation threshold")
    parser.add_argument(
        "--resp", type=float, default=0.01, help="sigmoid' responsiveness threshold"
    )
    args = parser.parse_args()

    try:
        with _open_or_stdout(args.out) as out:
            writer = csv.DictWriter(out, fieldnames=list(GateRow._fields))
            writer.writeheader()
            for run_dir in args.run_dirs:
                probes_dir = Path(run_dir) / "probes"
                asset_dir = select_asset_dir(probes_dir, args.asset)
                paths = sorted(asset_dir.glob("*.npz"), key=_dump_iteration)
                if args.max_iter is not None:
                    paths = [p for p in paths if _dump_iteration(p) <= args.max_iter]
                paths = paths[:: args.every]

                baseline = None
                for path in paths:
                    _iteration, _meta, rows, baseline = reduce_dump(
                        path, sat=args.sat, resp=args.resp, baseline=baseline
                    )
                    for row in rows:
                        writer.writerow(row._asdict())
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

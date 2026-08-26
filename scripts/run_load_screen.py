#!/usr/bin/env python
"""Screen a probe batch by how far the deployed router's load departs from balanced.

A price is only meaningful for a batch the router treats as a sample. On one of the four
cross-probe batches a single expert took 64% of the tokens, so the LP had to price it deeply
negative to push them off, and every correlation computed against the stored bias collapsed. That
batch was not detectable from any price-side statistic without first solving the LP.

``max deployed load / L``, with ``L = n*K/E``, separates it on every layer: 1.5-2.4 on the batches
that behave and 5.5 on the one that does not. It reads the routing map only, so it costs
milliseconds and no LP, and it is the check to run **before** trusting a price.

Usage:
    uv run python scripts/run_load_screen.py RUNDIR [RUNDIR ...] --out artifacts/game/screen.csv
    uv run python scripts/run_load_screen.py RUNDIR --halves     # score each half separately
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from moe_congestion_routing.metrics.probe_series import read_dump, read_series

# Above this, a batch's routing is concentrated enough that its equilibrium prices describe that
# concentration rather than the population. Chosen from the measured separation (every sound batch
# here reaches at most 2.44 and the anomalous one starts at 4.10 on its second layer), so it is a
# gap in observed data rather than a round number, and it is a warning rather than a refusal.
CONCENTRATION_WARN = 3.0


def rows_for_dump(path: str, halves: bool) -> list:
    dump = read_dump(path)
    routing = dump.routing_map()
    tokens = routing.shape[1]
    cuts = (
        [(0, tokens, "all")]
        if not halves
        else [
            (0, tokens // 2, "h1"),
            (tokens // 2, tokens, "h2"),
        ]
    )
    out = []
    for start, stop, name in cuts:
        n = stop - start
        balanced = n * dump.topk / routing.shape[2]
        for axis, layer in enumerate(dump.layer_numbers):
            load = routing[axis][start:stop].sum(axis=0)
            out.append(
                {
                    "dump": path,
                    "step": dump.step,
                    "part": name,
                    "layer": int(layer),
                    "tokens": n,
                    "balanced_load": balanced,
                    "max_load_over_balanced": float(load.max() / balanced),
                    "min_load_over_balanced": float(load.min() / balanced),
                    "argmax_expert": int(np.argmax(load)),
                    "dead_experts": int((load == 0).sum()),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", metavar="RUNDIR")
    parser.add_argument("--out", help="CSV path; prints a summary either way")
    parser.add_argument("--halves", action="store_true", help="score each half separately")
    parser.add_argument("--step", type=int, action="append", dest="steps")
    args = parser.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        series = read_series(run_dir)
        dumps = series.dumps
        if args.steps:
            dumps = [d for d in dumps if d.step in args.steps]
        for dump in dumps:
            rows.extend(rows_for_dump(str(dump.path), args.halves))

    flagged = 0
    for run_dir in args.run_dirs:
        for part in ("all", "h1", "h2"):
            sel = [r for r in rows if r["dump"].startswith(run_dir) and r["part"] == part]
            if not sel:
                continue
            worst = max(sel, key=lambda r: r["max_load_over_balanced"])
            hot = worst["max_load_over_balanced"] > CONCENTRATION_WARN
            mark = "  WARN concentrated" if hot else ""
            flagged += 1 if mark else 0
            print(
                f"{run_dir} [{part}] step {worst['step']}: worst max/L "
                f"{worst['max_load_over_balanced']:.2f} on layer {worst['layer']} "
                f"(expert {worst['argmax_expert']}), dead {worst['dead_experts']}{mark}"
            )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out} ({len(rows)} rows)")
    if flagged:
        print(
            f"\n{flagged} batch(es) above max/L {CONCENTRATION_WARN}: their prices describe that "
            "concentration, not the population.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Turns a phi-gap grid CSV into the tidy frontier table one arm's figure plots.

For every `(run_id, arm, asset, unit, layer, step, cost_family)` group, one row per lambda in
the expected grid, sorted ascending, carrying the group key plus `lam`, `affinity_shortfall`,
`congestion_excess`, `gap_normalized`, `normalizer` and `admissible`. Those five are already
columns of the input CSV, so no column is invented here. A group missing any expected lambda
raises by name rather than silently drawing a frontier through a hole.

Usage:
    uv run python scripts/run_phi_gap_frontier.py --in assets/results/phi-gap/control.csv \\
        --out assets/results/phi-gap/frontier_control_step0_layer2.csv
"""

import argparse
import csv
from pathlib import Path

GROUP_FIELDS = ("run_id", "arm", "asset", "unit", "layer", "step", "cost_family")
FRONTIER_FIELDS = GROUP_FIELDS + (
    "lam",
    "affinity_shortfall",
    "congestion_excess",
    "gap_normalized",
    "normalizer",
    "admissible",
)

# Pre-registered before any lambda other than 1.0 was scored, geometric in powers of two around
# the declared lambda=1.0 reference, so a sweep is checked against this fixed set rather than
# against whatever lambdas the input CSV happens to carry.
DECLARED_LAM_GRID: tuple[float, ...] = (
    0.0,
    0.03125,
    0.0625,
    0.125,
    0.1875,
    0.25,
    0.375,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)


def build_frontier(rows: list[dict], expect_lams: list[float]) -> list[dict]:
    """Group `rows` and emit one row per `(group, lam)`, sorted by group then ascending lam.

    Raises `ValueError` naming the group and the missing lambdas when a group's `"ok"` rows do
    not cover every value in `expect_lams`, because a curve drawn through a hole misstates its
    own shape and nothing downstream would notice.
    """
    groups: dict[tuple[str, ...], dict[float, dict]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row[field] for field in GROUP_FIELDS)
        groups.setdefault(key, {})[float(row["lam"])] = row

    out_rows: list[dict] = []
    for key in sorted(groups):
        by_lam = groups[key]
        key_dict = dict(zip(GROUP_FIELDS, key, strict=True))
        missing = [lam for lam in expect_lams if lam not in by_lam]
        if missing:
            raise ValueError(
                f"group {key_dict} is missing lam {missing} of the expected grid {expect_lams}, "
                "so its frontier would be drawn through a hole"
            )
        for lam in sorted(expect_lams):
            row = by_lam[lam]
            out_rows.append(
                dict(zip(GROUP_FIELDS, key, strict=True))
                | {
                    "lam": lam,
                    "affinity_shortfall": row["affinity_shortfall"],
                    "congestion_excess": row["congestion_excess"],
                    "gap_normalized": row["gap_normalized"],
                    "normalizer": row["normalizer"],
                    "admissible": row["admissible"],
                }
            )
            if lam == 0.0:
                label = " ".join(f"{field}={key_dict[field]}" for field in GROUP_FIELDS)
                print(f"  {label}: gap_normalized(lam=0)={row['gap_normalized']}")
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--in", dest="in_path", required=True, type=Path, help="a phi-gap CSV")
    parser.add_argument("--out", required=True, type=Path, help="the frontier CSV to write")
    parser.add_argument(
        "--expect-lam",
        action="append",
        dest="expect_lams",
        type=float,
        help="repeatable, defaults to the declared lambda grid",
    )
    args = parser.parse_args()

    expect_lams = list(args.expect_lams) if args.expect_lams else list(DECLARED_LAM_GRID)

    with args.in_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = build_frontier(rows, expect_lams)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FRONTIER_FIELDS)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

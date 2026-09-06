#!/usr/bin/env python
"""Rebuild the tail x strided x spread triad from the dual and bias stores alone, both the
uncorrected and `--project-code-axis` variants, and check each against its own committed CSV.

`run_triad.py` reads probe dumps and solves an LP for every unit. This reads only the committed
stores `scripts/price_probe_duals.py` produced from those same dumps, feeds their duals and bias
straight into `metrics.triad.trajectory_cells`, and never solves an LP: the point is that the
regression this compares against needs nothing but the committed CSVs, not a probe dump on disk
and not a few seconds of LP time per cell.

Four `(run, layer=8, tail-spread, ...)` rows in the uncorrected triad are expected to differ, in
`kappa` only: they sit below `half_split_row`'s `rho >= c_x * c_y` admissibility floor, which
post-dates the committed file, so a fresh computation refuses them to NaN where the committed file
still carries a number. They are reported by name and exit 0, whereas any other difference exits 1.

Usage:
    uv run python scripts/reconstruct_triad_from_store.py
"""

import argparse
import csv
import math
import sys
from collections.abc import Iterable
from pathlib import Path

from moe_congestion_routing.metrics.dual_store import read_bias_store, read_dual_store
from moe_congestion_routing.metrics.triad import TriadAssets, TriadRow, trajectory_cells

DEFAULT_DUALS = "assets/results/price-recovery/duals_crossprobe_step500.csv"
DEFAULT_BIAS = "assets/results/price-recovery/bias_crossprobe_step500.csv"
DEFAULT_TRIAD_CSV = "assets/results/price-recovery/triad_step500_three-pairings.csv"
DEFAULT_CORRECTED_TRIAD_CSV = "assets/results/price-recovery/triad_step500_corrected.csv"

# Fixed for this one committed corpus, matching run_triad.py's own CROSSPROBE_DIR/RUNS/UNIT_JOBS:
# the probe-batch stem `enumerate_dual_cells` derives an asset name from, for each of the three
# roles this triad needs. Not a general-purpose lookup, because this script rebuilds one fixed
# regression fixture rather than driving an arbitrary sweep.
ASSETS = TriadAssets(
    tail="standing_climbmix_small_16x2048",
    strided="standing_climbmix_small_strided_16x2048",
    spread="standing_climbmix_small_spread_off1_16x2048",
)

# The four uncorrected rows the admissibility floor in `half_split_row` (added after this file was
# published) refuses to NaN that the committed CSV still carries a number for. Hard-coded rather
# than detected, so a fifth row going stale fails loudly instead of being silently swallowed by a
# pattern match.
KNOWN_STALE_KEYS = frozenset(
    {
        ("a", 8, "tail-spread", "u0", "u0"),
        ("a", 8, "tail-spread", "u0", "u1"),
        ("b", 8, "tail-spread", "u0", "u0"),
        ("b", 8, "tail-spread", "u0", "u1"),
    }
)

# The corrected variant carries none of the uncorrected file's four stale rows: projecting the
# axis out moves `tail-spread` layer 8's rho and both bias correlations enough that it no longer
# sits below `half_split_row`'s admissibility floor, so its committed kappa matches exactly rather
# than needing the same exemption (measured when this script was rewritten to check both files).
KNOWN_STALE_KEYS_CORRECTED = frozenset()


def reconstruct_rows(
    duals_paths: list[Path], bias_paths: list[Path], *, project_code_axis: bool
) -> list[TriadRow]:
    """Every triad row the stores can reproduce, for one variant of the correction."""
    duals = read_dual_store(duals_paths)
    bias = read_bias_store(bias_paths)
    cells = trajectory_cells(
        duals, bias, assets=ASSETS, project_code_axis=project_code_axis, base_seed=0
    )
    return [row for cell in cells for row in cell.rows]


def _row_key(run: str, layer: str, pair: str, unit_x: str, unit_y: str) -> tuple:
    return (run, int(layer), pair, unit_x, unit_y)


def _read_committed(csv_path: Path) -> dict[tuple, dict[str, str]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return {
            _row_key(r["run"], r["layer"], r["pair"], r["unit_x"], r["unit_y"]): r for r in reader
        }


def _floats_agree(a: str, b: str) -> bool:
    fa, fb = float(a), float(b)
    if math.isnan(fa) and math.isnan(fb):
        return True
    return fa == fb


def compare(
    fresh_rows: Iterable[TriadRow],
    committed: dict[tuple, dict[str, str]],
    *,
    known_stale_keys: frozenset,
    label: str,
) -> int:
    """Print every mismatch and the known-stale rows by name, and return the exit code."""
    fresh = {
        _row_key(r.run, str(r.layer), r.pair, r.unit_x, r.unit_y): r._asdict() for r in fresh_rows
    }
    compare_fields = [
        f for f in TriadRow._fields if f not in ("run", "layer", "pair", "unit_x", "unit_y")
    ]

    ok = True
    stale_seen = []
    for key in sorted(set(fresh) | set(committed)):
        if key not in committed:
            print(f"MISMATCH [{label}] {key}: reconstructed but absent from the committed CSV")
            ok = False
            continue
        if key not in fresh:
            print(f"MISMATCH [{label}] {key}: in the committed CSV but not reconstructed")
            ok = False
            continue
        c_row, f_row = committed[key], fresh[key]
        diffs = [
            field for field in compare_fields if not _floats_agree(c_row[field], str(f_row[field]))
        ]
        if key in known_stale_keys:
            if diffs == ["kappa"]:
                stale_seen.append(key)
            else:
                print(f"MISMATCH [{label}] {key}: known-stale differs in {diffs}, expected kappa")
                ok = False
        elif diffs:
            print(f"MISMATCH [{label}] {key}: differs in {diffs}")
            ok = False

    for key in sorted(stale_seen):
        print(
            f"[{label}] known-stale (kappa refused to NaN by the current admissibility rule): {key}"
        )

    missing_stale = known_stale_keys - set(stale_seen)
    if missing_stale:
        print(f"MISMATCH [{label}]: expected stale rows not stale: {sorted(missing_stale)}")
        ok = False

    print(
        f"[{label}] {len(fresh)} rows reconstructed, {len(committed)} rows committed, "
        f"{len(stale_seen)} known-stale, {'0 other mismatches' if ok else 'mismatches above'}"
    )
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duals", action="append", type=Path, help="repeatable")
    parser.add_argument("--bias", action="append", type=Path, help="repeatable")
    parser.add_argument("--triad-csv", default=DEFAULT_TRIAD_CSV, type=Path)
    parser.add_argument("--corrected-triad-csv", default=DEFAULT_CORRECTED_TRIAD_CSV, type=Path)
    args = parser.parse_args()

    duals_paths = args.duals or [Path(DEFAULT_DUALS)]
    bias_paths = args.bias or [Path(DEFAULT_BIAS)]

    uncorrected_rows = reconstruct_rows(duals_paths, bias_paths, project_code_axis=False)
    uncorrected_committed = _read_committed(args.triad_csv)
    exit_code = compare(
        uncorrected_rows,
        uncorrected_committed,
        known_stale_keys=KNOWN_STALE_KEYS,
        label="uncorrected",
    )

    corrected_rows = reconstruct_rows(duals_paths, bias_paths, project_code_axis=True)
    corrected_committed = _read_committed(args.corrected_triad_csv)
    exit_code |= compare(
        corrected_rows,
        corrected_committed,
        known_stale_keys=KNOWN_STALE_KEYS_CORRECTED,
        label="corrected",
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

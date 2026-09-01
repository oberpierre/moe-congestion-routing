"""Drives the phi-gap grid over one run's probe dumps and writes that arm's CSV.

Enumerates every `(asset, layer, unit, cost_family)` cell a run's probe dumps can be scored on,
runs :func:`~moe_congestion_routing.metrics.phi_gap.phi_gap_rows` on each (serially or across
worker processes), and turns the results into :class:`GridRow` objects for a caller to write.
A cell costs a whole LP solve, tens of seconds to minutes, so this module never recomputes one
whose key already appears in the output file: that is the caller's job, using
:func:`existing_keys` before it ever calls :func:`run_grid`, because `run_grid` itself never
opens a file.

`Cell` pins one `cost_family` rather than a sequence, unlike `phi_gap_rows` itself, so that
`run_grid` can call `emit` exactly once per cell regardless of how many cost families a sweep
covers.
"""

import csv
import multiprocessing
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from moe_congestion_routing.metrics.phi_gap import PhiGapRow, phi_gap_rows
from moe_congestion_routing.metrics.probe_comparison import probe_units
from moe_congestion_routing.metrics.probe_series import read_dump


class Cell(NamedTuple):
    """One grid cell: which dump, which asset it came from, and which slice of it to score.

    `asset` is carried here rather than re-derived from `dump_path`, because the per-asset
    probe layout names the asset by its parent directory and a `Cell` should not need the
    caller to re-parse a path to know which arm-level column it belongs to.
    """

    dump_path: Path
    asset: str
    step: int
    layer: int
    unit: str
    cost_family: str
    lam: float


class GridRow(NamedTuple):
    """One finished cell: `run_id`/`arm`/`asset` are the driver's to supply because a probe
    dump does not know which run or arm produced it, matching what :func:`phi_gap_rows` itself
    emits.

    The coordinates are carried here rather than read off `row`, because `row` is `None` on a
    failed cell and a failure that cannot name its own cell is unactionable across thousands of
    them. A failed row is still not treated as done, so its key may later appear a second time
    with `status="ok"`, and a reader takes the `"ok"` row.
    """

    run_id: str
    arm: str
    asset: str
    unit: str
    layer: int
    step: int
    cost_family: str
    lam: float
    status: str  # "ok" | "failed"
    detail: str  # "" when ok, else the exception's message, single-line
    row: PhiGapRow | None


KEY_FIELDS = ("run_id", "arm", "asset", "unit", "layer", "step", "cost_family", "lam")

# Every column an arm's CSV carries, past the resume key. `cost_family` renames PhiGapRow's own
# `reference_cost` field to match `Cell.cost_family` and `KEY_FIELDS`, and every other PhiGapRow
# field is carried through unchanged. A failed row leaves all of these blank, because none of
# them exist without a PhiGapRow.
_PHI_GAP_DATA_FIELDS = (
    "affinity_space",
    "score_function",
    "admissible",
    "max_load_over_balanced",
    "dead_experts",
    "gap_per_token",
    "affinity_shortfall",
    "congestion_excess",
    "gap_normalized",
    "normalizer",
    "arc_growths",
    "arcs_used_max",
    "max_fractional_deviation",
    "token_sha256",
    "dump_path",
)


CSV_FIELDS = KEY_FIELDS + ("status", "detail") + _PHI_GAP_DATA_FIELDS


def enumerate_cells(
    run_dir: Path,
    *,
    lam: float = 1.0,
    cost_families: Sequence[str] = ("linear", "quadratic"),
    assets: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    steps: Sequence[int] | None = None,
) -> list[Cell]:
    """Every cell this run's `<run_dir>/probes/<asset>/iter_*.npz` dumps can be scored on.

    Reads only each dump's metadata (:func:`read_dump` never opens the array data), so listing
    thousands of cells costs one small `numpy.load` per dump rather than one LP solve. `assets`,
    `layers` and `steps` narrow the sweep to the named values when given, keeping their
    dump-native order otherwise.
    """
    probes_dir = Path(run_dir) / "probes"
    if not probes_dir.is_dir():
        raise FileNotFoundError(f"no probes directory under {run_dir}")
    asset_dirs = sorted(p for p in probes_dir.iterdir() if p.is_dir())
    if not asset_dirs:
        raise FileNotFoundError(f"no per-asset probe directories under {probes_dir}")
    if assets is not None:
        wanted = set(assets)
        asset_dirs = [p for p in asset_dirs if p.name in wanted]

    cells: list[Cell] = []
    for asset_dir in asset_dirs:
        for dump_path in sorted(asset_dir.glob("*.npz")):
            dump = read_dump(dump_path)
            if steps is not None and dump.step not in steps:
                continue
            dump_layers = [n for n in dump.layer_numbers if layers is None or n in layers]
            units = probe_units(dump.meta["N"])
            for layer in dump_layers:
                for unit_name, _start, _stop in units:
                    for cost_family in cost_families:
                        cells.append(
                            Cell(
                                dump_path=dump_path,
                                asset=asset_dir.name,
                                step=dump.step,
                                layer=layer,
                                unit=unit_name,
                                cost_family=cost_family,
                                lam=lam,
                            )
                        )
    return cells


def _default_solve(cell: Cell) -> list[PhiGapRow]:
    """The real oracle: read the dump fresh and score exactly the one cost family this cell
    pins. Module-level, not a closure, so a `spawn` worker can import and pickle it."""
    dump = read_dump(cell.dump_path)
    return phi_gap_rows(
        dump, cell.layer, cell.unit, lam=cell.lam, cost_families=(cell.cost_family,)
    )


def _first_line(message: str) -> str:
    """The exception message with any embedded newline collapsed, so a CSV cell built from it
    is a single logical line rather than one that visually breaks the row when opened by eye."""
    return message.splitlines()[0] if message else ""


def _run_cell(
    cell: Cell, solve: Callable[[Cell], list[PhiGapRow]]
) -> tuple[Cell, str, str, PhiGapRow | None]:
    """Run one cell, catching only the two exceptions :func:`phi_gap_rows` is documented to
    raise on a well-posed but adversarial unit.

    Anything else propagates, because those two are the failure modes an unattended sweep is
    expected to meet cell by cell, whereas any other exception more likely means every
    remaining cell will fail the same way and the sweep should stop rather than spend CPU-hours
    discovering that one cell at a time.
    """
    try:
        rows = solve(cell)
    except (AssertionError, ValueError) as exc:
        return cell, "failed", _first_line(str(exc)), None
    if len(rows) != 1:
        # A cell pins exactly one cost_family, so a solve returning anything else is `solve`
        # itself misbehaving rather than a property of the unit being scored, and stopping the
        # sweep on it is preferable to silently emitting the wrong number of rows per cell.
        raise ValueError(
            f"{cell.dump_path} layer={cell.layer} unit={cell.unit} cost_family="
            f"{cell.cost_family}: solve returned {len(rows)} rows, expected exactly 1"
        )
    row = rows[0]
    got = (row.unit, row.layer, row.step, row.reference_cost, row.lam)
    want = (cell.unit, cell.layer, cell.step, cell.cost_family, cell.lam)
    if got != want:
        # The CSV's key columns come from the cell, so a row describing a different coordinate
        # would be filed under the wrong key rather than caught by anything downstream.
        raise ValueError(f"{cell.dump_path}: solve returned coordinates {got} for cell {want}")
    return cell, "ok", "", row


def run_grid(
    cells: Sequence[Cell],
    *,
    run_id: str,
    arm: str,
    emit: Callable[[GridRow], None],
    workers: int = 1,
    solve: Callable[[Cell], list[PhiGapRow]] | None = None,
) -> None:
    """Run every cell and call `emit(row)` once per finished cell.

    Never opens a file and never prints, so a test drives it with a list-appending `emit` and a
    fake `solve` in milliseconds. `workers > 1` uses `spawn`, matching every other process pool
    in this repo, so a custom `solve` must be importable at module level rather than a closure.

    Propagates whatever a cell raised outside `(AssertionError, ValueError)`, or a
    `KeyboardInterrupt`, after cancelling whatever had not yet started, so a caller relying on
    this function returning to mean success still sees a failed or interrupted run as a failure.
    Rows already handed to `emit` before that point stay emitted.
    """
    solve = _default_solve if solve is None else solve

    def to_grid_row(result: tuple[Cell, str, str, PhiGapRow | None]) -> GridRow:
        cell, status, detail, row = result
        return GridRow(
            run_id=run_id,
            arm=arm,
            asset=cell.asset,
            unit=cell.unit,
            layer=cell.layer,
            step=cell.step,
            cost_family=cell.cost_family,
            lam=cell.lam,
            status=status,
            detail=detail,
            row=row,
        )

    if workers == 1:
        for cell in cells:
            emit(to_grid_row(_run_cell(cell, solve)))
        return

    # spawn re-execs a fresh interpreter per worker rather than forking this process, because
    # forking a process that already holds numpy/scipy state and their background threads can
    # deadlock a child that inherits a lock one of those threads held but no longer exists to
    # release.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = {executor.submit(_run_cell, cell, solve): cell for cell in cells}
        drained: set[Future] = set()
        try:
            for future in as_completed(futures):
                drained.add(future)
                emit(to_grid_row(future.result()))
        except BaseException:
            # Cancel what has not started, wait out what has, then emit every result that
            # finished without being consumed. At the planned parallelism one unexpected
            # exception would otherwise discard hundreds of finished solves worth minutes each,
            # which is the loss the resume exists to stop paying for twice.
            executor.shutdown(wait=True, cancel_futures=True)
            for future in futures:
                if future in drained or not future.done() or future.cancelled():
                    continue
                try:
                    result = future.result()
                except BaseException:
                    continue
                emit(to_grid_row(result))
            raise


def _key_tuple(
    run_id: str,
    arm: str,
    asset: str,
    unit: str,
    layer: int,
    step: int,
    cost_family: str,
    lam: float,
) -> tuple[str, ...]:
    """The `KEY_FIELDS` tuple, stringified to match what a round trip through the CSV produces.

    The one place a key is built. A caller filtering cells before solving and the writer filing
    the finished row must agree exactly, and two implementations of that would drift into either
    re-solving everything or skipping a cell forever.
    """
    return (run_id, arm, asset, unit, str(layer), str(step), cost_family, str(lam))


def candidate_key(cell: Cell, *, run_id: str, arm: str) -> tuple[str, ...]:
    """The key `cell` would land under if solved, for comparing against `existing_keys` before
    paying for the solve."""
    return _key_tuple(
        run_id, arm, cell.asset, cell.unit, cell.layer, cell.step, cell.cost_family, cell.lam
    )


def _row_key(grid_row: GridRow) -> tuple[str, ...]:
    """The key a finished `GridRow` is filed under, including a failed one, which carries its
    coordinates so it can name the cell it failed on."""
    return _key_tuple(
        grid_row.run_id,
        grid_row.arm,
        grid_row.asset,
        grid_row.unit,
        grid_row.layer,
        grid_row.step,
        grid_row.cost_family,
        grid_row.lam,
    )


# A row is the output of one LP solve, so a change to the oracle invalidates every row taken
# before it. There is no version column recording that, because the answer is to delete the CSV
# and re-run rather than to filter rows by when they were computed.
def _to_csv_row(grid_row: GridRow) -> dict[str, str]:
    """Flatten one `GridRow` into `CSV_FIELDS`.

    The key columns are always written, including on a failed row, because a failure that does
    not name its own cell cannot be investigated across thousands of them. Only the measurement
    columns are blank on a failure, since those are the ones that genuinely do not exist.
    """
    out = dict.fromkeys(CSV_FIELDS, "")
    out.update(dict(zip(KEY_FIELDS, _row_key(grid_row), strict=True)))
    out["status"] = grid_row.status
    out["detail"] = grid_row.detail
    row = grid_row.row
    if row is not None:
        for field in _PHI_GAP_DATA_FIELDS:
            out[field] = str(getattr(row, field))
    return out


def existing_keys(csv_path: Path) -> set[tuple[str, ...]]:
    """The `KEY_FIELDS` tuples already written to `csv_path`, as the raw strings a CSV round
    trip produces, which is also how :func:`append_rows` builds a key to compare against them.

    A `status="failed"` row is skipped, because treating it as "done" would permanently exclude
    a cell that never actually produced a measurement. It does carry its key columns, so that a
    reader can see which cell failed, which is why the status is what gates this and not a blank
    key. A row
    with a blank key field, from a file truncated mid-write, is skipped the same way rather than
    raising, so the cell it belongs to is recomputed rather than the whole sweep failing on a
    partial trailing line.
    """
    if not csv_path.exists():
        return set()
    keys: set[tuple[str, ...]] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return keys
        for raw in reader:
            # `status` sits after the key columns, so a row killed mid-write cannot read as
            # `"ok"`, and that is what keeps a partial trailing line out of the resume set. The
            # csv module has nothing left to raise on here, because `newline=""` closes the one
            # case it does raise for.
            if raw.get("status") != "ok":
                continue
            if any(raw.get(field) in (None, "") for field in KEY_FIELDS):
                continue
            keys.add(tuple(raw[field] for field in KEY_FIELDS))
    return keys


def append_rows(csv_path: Path, rows: Iterable[GridRow]) -> int:
    """Append `rows` to `csv_path`, writing the header only when the file does not yet exist.

    Every `"ok"` row's key is checked against `existing_keys(csv_path)` and against every other
    `"ok"` row already validated in this same call *before* anything is written, so a duplicate
    key raises `ValueError` and leaves the file exactly as it was. Once writing starts each row
    is flushed immediately, so a process killed mid-sweep leaves only whole rows on disk. A
    `"failed"` row is exempt from the duplicate check and always appended, because it is not
    counted as done and the same cell is expected to be attempted again.
    """
    rows = list(rows)
    seen = existing_keys(csv_path)
    for grid_row in rows:
        if grid_row.row is None:
            continue
        key = _row_key(grid_row)
        if key in seen:
            raise ValueError(f"duplicate phi-gap row key {key!r} for {csv_path}")
        seen.add(key)

    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
            f.flush()
        for grid_row in rows:
            writer.writerow(_to_csv_row(grid_row))
            f.flush()
            written += 1
    return written

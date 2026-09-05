"""Prices one run's probe dumps into a resumable dual-vector store, once per LP, and owns the
expert-bias store beside it.

Mirrors :mod:`phi_gap_grid`'s resume shape deliberately, so the two drivers cannot drift apart in
behaviour: cells are cheap to enumerate from a dump's metadata alone, a cell costs one LP solve,
rows are appended as each cell finishes, and a caller reads the CSV's existing keys before ever
calling :func:`price_cells` so a resumed sweep pays for nothing it already has. Unlike the phi-gap
grid, a cell here is priced exactly once regardless of how many downstream questions (the corrected
and uncorrected kappa variants, any future correlation) read the resulting dual vector, because the
LP time lives entirely in this one pass and reading a stored vector back costs nothing.

`price_cells` calls `emit` once per :class:`DualRow` and, at most once per `(layer, step)` this
run's dumps agree on, once per :class:`BiasRow`. It opens no file and prints nothing except when a
second exception is raised while draining a crash: that one is printed rather than swallowed,
because swallowing it would make the drain look clean while it silently lost the mismatch.
"""

import csv
import multiprocessing
import sys
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np

from moe_congestion_routing.game import lp
from moe_congestion_routing.metrics.probe_comparison import probe_units, screen_batch
from moe_congestion_routing.metrics.probe_series import IncomparableProbes, read_dump


class DualCell(NamedTuple):
    """One grid cell: which dump, which asset it came from, and which 16,384-token unit of it.

    `asset` is carried here rather than re-derived from `dump_path`, matching `phi_gap_grid.Cell`,
    because the per-asset probe layout names the asset by its parent directory and a legacy flat
    dump names it by the probe batch file it was drawn from: either way the caller that already
    resolved it should not have to re-derive it from a path at every use.
    """

    dump_path: Path
    asset: str
    unit: str
    layer: int
    step: int


DUAL_KEY_FIELDS = ("run_id", "asset", "unit", "layer", "step")
BIAS_KEY_FIELDS = ("run_id", "layer", "step")

# The screen fields a refused unit still carries, and the two identifying columns a resume never
# needs but a human investigating a row does. `score_function` is a data column rather than a key
# field, matching phi_gap_grid.py's own row: it lets a reader refuse to pool a sigmoid run's rows
# with a softmax run's, without meaning a second cell exists at the same key.
_DUAL_DATA_FIELDS = (
    "status",
    "detail",
    "score_function",
    "admissible",
    "max_load_over_balanced",
    "dead_experts",
    "token_sha256",
    "dump_path",
)
# `asset` is a data column, not part of BIAS_KEY_FIELDS, because the bias is a property of the
# weights rather than of any one asset. It is carried so a cross-process mismatch can name which
# two assets disagreed, which is the whole point of comparing rather than dropping a known key.
_BIAS_DATA_FIELDS = ("status", "detail", "asset", "token_sha256")


def dual_fields(num_experts: int) -> tuple[str, ...]:
    """The dual store's CSV header at this run's own expert count.

    Not a module-level constant, because `num_experts` is a property of the dump being priced
    rather than something this module may assume, so a caller must read it off the run before it
    can know how many `dual_i` columns a row has.
    """
    return DUAL_KEY_FIELDS + _DUAL_DATA_FIELDS + tuple(f"dual_{i}" for i in range(num_experts))


def bias_fields(num_experts: int) -> tuple[str, ...]:
    """The bias store's CSV header at this run's own expert count, see :func:`dual_fields`."""
    return BIAS_KEY_FIELDS + _BIAS_DATA_FIELDS + tuple(f"bias_{i}" for i in range(num_experts))


class DualRow(NamedTuple):
    """One finished dual cell, `"ok"` (priced or refused) or `"failed"`.

    `admissible`/`max_load_over_balanced`/`dead_experts`/`token_sha256`/`dump_path` are `None` or
    `""` on a failed row along with `duals`, which is NaN-filled instead of empty even on failure
    and even on a refusal, so every row in one run's CSV carries the same number of dual columns.
    """

    run_id: str
    asset: str
    unit: str
    layer: int
    step: int
    status: str  # "ok" | "failed"
    detail: str
    admissible: bool | None
    max_load_over_balanced: float | None
    dead_experts: int | None
    token_sha256: str
    dump_path: str
    duals: tuple[float, ...]
    # Trails with a default so a caller that built a DualRow before this field
    # existed keeps constructing one without naming it.
    score_function: str = ""


class BiasRow(NamedTuple):
    """One `(run_id, layer, step)`'s stored `expert_bias`, a property of the weights rather than
    of any one asset, so `BIAS_KEY_FIELDS` carries no `asset` or `unit` column. `asset` is still
    carried as data, naming whichever cell's dump this particular reading came from, so a
    cross-process mismatch against a stored row can say which two assets disagreed."""

    run_id: str
    layer: int
    step: int
    status: str
    detail: str
    token_sha256: str
    bias: tuple[float, ...]
    asset: str = ""


class DualSolveResult(NamedTuple):
    """What one cell's screen-then-price step produces, before it is wrapped into a `DualRow`."""

    admissible: bool
    max_load_over_balanced: float
    dead_experts: int
    duals: np.ndarray  # NaN-filled [E] when `admissible` is False


def _key_tuple(run_id: str, asset: str, unit: str, layer: int, step: int) -> tuple[str, ...]:
    """The one place a dual key is built, so a caller filtering cells before pricing and the row
    a finished cell lands under can never drift apart the way two independent implementations
    would risk."""
    return (run_id, asset, unit, str(layer), str(step))


def dual_key(cell: DualCell, *, run_id: str) -> tuple[str, ...]:
    """The key `cell` would land under if priced, for comparing against `existing_dual_keys`
    before a caller pays for the solve."""
    return _key_tuple(run_id, cell.asset, cell.unit, cell.layer, cell.step)


def _dual_row_key(row: DualRow) -> tuple[str, ...]:
    return _key_tuple(row.run_id, row.asset, row.unit, row.layer, row.step)


def bias_key(run_id: str, layer: int, step: int) -> tuple[str, str, str]:
    """The key a `(run_id, layer, step)` bias reading lands under, the one place it is built so a
    caller filtering cells before a bias-only read and the row a finished read lands under cannot
    drift apart."""
    return (run_id, str(layer), str(step))


def _bias_row_key(row: BiasRow) -> tuple[str, str, str]:
    return bias_key(row.run_id, row.layer, row.step)


def _unit_bounds(n_tokens: int, unit: str) -> tuple[int, int]:
    for name, start, stop in probe_units(n_tokens):
        if name == unit:
            return start, stop
    available = [name for name, _, _ in probe_units(n_tokens)]
    raise ValueError(f"unit {unit!r} is not among {available!r} for {n_tokens} tokens")


def _cells_for_dir(
    dump_dir: Path, asset: str, layers: Sequence[int] | None, steps: Sequence[int] | None
) -> list[DualCell]:
    """Every `(unit, layer, step)` cell this one directory's dumps can be priced on."""
    cells: list[DualCell] = []
    for dump_path in sorted(dump_dir.glob("*.npz")):
        dump = read_dump(dump_path)
        if steps is not None and dump.step not in steps:
            continue
        dump_layers = [n for n in dump.layer_numbers if layers is None or n in layers]
        units = probe_units(dump.meta["N"])
        for layer in dump_layers:
            for unit_name, _start, _stop in units:
                cells.append(
                    DualCell(
                        dump_path=dump_path,
                        asset=asset,
                        unit=unit_name,
                        layer=layer,
                        step=dump.step,
                    )
                )
    return cells


def enumerate_dual_cells(
    run_dir: Path,
    *,
    assets: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    steps: Sequence[int] | None = None,
) -> list[DualCell]:
    """Every cell this run's `<run_dir>/probes/` dumps can be priced on.

    Accepts both probe layouts, the same rule `probe_series.read_series` resolves a run's
    `probes/` directory by: subdirectories mean the per-asset layout, one per asset named by its
    directory, whereas loose `.npz` files mean the legacy flat layout, which holds exactly one
    asset, named the way `read_series` itself validates a requested asset against a flat
    directory, by the stem of the dump's own `moe_probe_batch` metadata field rather than by any
    path segment, since a flat directory has no per-asset directory to name it with.

    Reads only each dump's metadata, so listing thousands of cells costs one small `numpy.load`
    per dump rather than one LP solve. `assets`, `layers` and `steps` narrow the sweep to the
    named values when given.
    """
    probes_dir = Path(run_dir) / "probes"
    if not probes_dir.is_dir():
        raise FileNotFoundError(f"no probes directory under {run_dir}")
    subdirs = sorted(p for p in probes_dir.iterdir() if p.is_dir())
    flat_paths = sorted(probes_dir.glob("*.npz"))
    if subdirs and flat_paths:
        raise IncomparableProbes(
            f"{probes_dir}: holds both loose dumps and asset subdirectories, which no writer "
            "produces, so this is a hand-edited directory and which layout is authoritative "
            "cannot be guessed"
        )
    if subdirs:
        asset_dirs = subdirs
        if assets is not None:
            wanted = set(assets)
            asset_dirs = [p for p in asset_dirs if p.name in wanted]
        cells: list[DualCell] = []
        for asset_dir in asset_dirs:
            cells.extend(_cells_for_dir(asset_dir, asset_dir.name, layers, steps))
        return cells
    if not flat_paths:
        raise FileNotFoundError(f"no probe dumps found under {probes_dir}")
    asset_name = Path(read_dump(flat_paths[0]).meta["moe_probe_batch"]).stem
    if assets is not None and asset_name not in set(assets):
        return []
    return _cells_for_dir(probes_dir, asset_name, layers, steps)


def _default_solve(cell: DualCell) -> DualSolveResult:
    """The real oracle: read the dump fresh, screen the unit and, only if admissible, price it.

    Prices `router_scores()`, matching `phi_gap.py` rather than `probe_comparison.py`'s
    sigmoid-only `affinities()`, so a cell from any arm's dump can be priced regardless of score
    function: a duals row is meaningful on its own as the router's own valuation, and only a
    later bias correlation needs the sigmoid space ALF-LB's bias was actually added to. On a
    sigmoid dump the two are bit-identical, so this changes nothing for the crossprobe cells the
    committed step-500 triad was priced from.

    Reads the dump fresh from `cell.dump_path` rather than closing over one, matching
    `phi_gap_grid._default_solve`, so a `spawn` worker needs to pickle only the cell.
    """
    dump = read_dump(cell.dump_path)
    axis_index = dump.layer_numbers.index(cell.layer)
    start, stop = _unit_bounds(dump.meta["N"], cell.unit)
    routing = dump.routing_map()[axis_index, start:stop, :]
    screen = screen_batch(routing, dump.topk)
    if not screen.admissible:
        duals = np.full(dump.num_experts, np.nan)
    else:
        scores = dump.router_scores()[axis_index, start:stop, :]
        duals = lp.solve(scores, dump.topk).capacity_duals
    return DualSolveResult(
        screen.admissible, screen.max_load_over_balanced, screen.dead_experts, duals
    )


def _first_line(message: str) -> str:
    return message.splitlines()[0] if message else ""


class _CellResult(NamedTuple):
    """What one worker call returns to the main process: the dual solve (or its failure) and,
    independently, this cell's bias reading (or `None` when this run keeps none), so a bias-read
    failure never marks the dual price as failed and vice versa."""

    cell: DualCell
    status: str
    detail: str
    solve_result: DualSolveResult | None
    num_experts: int
    token_sha256: str
    score_function: str
    bias: np.ndarray | None


def _price_cell(cell: DualCell, solve: Callable[[DualCell], DualSolveResult]) -> _CellResult:
    """Run one cell: price it (catching only the well-posed-but-adversarial failure modes
    `screen_batch`/`lp.solve` are documented to raise) and, separately, read its bias.

    A dump with no stored `expert_bias` raises `IncomparableProbes` for every cell drawn from it,
    which is caught here alone so the dual price above is unaffected by an arm simply keeping no
    bias. Anything else, from either step, propagates: a sweep meeting an exception it does not
    recognize should stop rather than spend CPU-hours discovering every remaining cell fails the
    same way.

    `dump.layer_numbers.index(cell.layer)` is inside its own try, matching `phi_gap_grid`, so an
    enumerator bug naming a layer this dump does not have is a failed row rather than an
    exception that aborts every remaining cell in the sweep.
    """
    dump = read_dump(cell.dump_path)
    num_experts = dump.num_experts
    try:
        axis_index = dump.layer_numbers.index(cell.layer)
    except ValueError as exc:
        return _CellResult(
            cell,
            "failed",
            _first_line(str(exc)),
            None,
            num_experts,
            dump.token_sha256,
            dump.score_function,
            None,
        )
    bias = None
    try:
        bias = dump.expert_bias()[axis_index]
    except IncomparableProbes:
        bias = None
    try:
        result = solve(cell)
    except (AssertionError, ValueError) as exc:
        return _CellResult(
            cell,
            "failed",
            _first_line(str(exc)),
            None,
            num_experts,
            dump.token_sha256,
            dump.score_function,
            bias,
        )
    return _CellResult(
        cell, "ok", "", result, num_experts, dump.token_sha256, dump.score_function, bias
    )


def price_cells(
    cells: Sequence[DualCell],
    *,
    run_id: str,
    emit: Callable[[DualRow | BiasRow], None],
    workers: int = 1,
    solve: Callable[[DualCell], DualSolveResult] | None = None,
) -> None:
    """Price every cell and call `emit(row)` once per finished `DualRow`, plus at most once per
    `(layer, step)` this call's cells cover for `BiasRow`.

    A `(layer, step)`'s bias is emitted from whichever cell reaches it first, because the bias is
    a property of the weights rather than of the asset that happened to probe them, and every
    later cell at the same `(layer, step)` is checked against it rather than re-emitted: when the
    two agree it is silently skipped, and when they differ it raises `ValueError`, because that
    means two dumps claiming the same step came from different weights. A run whose dumps carry
    no `expert_bias` emits no `BiasRow` at all and is not a failure.

    Opens no file itself and prints only from inside the crash-drain guard below, so a test
    drives it with a list-appending `emit` and a fake `solve` in milliseconds. `workers > 1` uses
    `spawn`, matching every other process pool in this repo, so a custom `solve` must be
    importable at module level rather than a closure.

    Propagates whatever a cell raised outside `(AssertionError, ValueError)`, a bias mismatch, or
    a `KeyboardInterrupt`, after cancelling whatever had not yet started, so a caller relying on
    this function returning to mean success still sees a failed or interrupted run as a failure.
    Rows already handed to `emit` before that point stay emitted.
    """
    solve = _default_solve if solve is None else solve
    bias_seen: dict[tuple[int, int], tuple[str, np.ndarray]] = {}

    def handle(result: _CellResult) -> None:
        cell = result.cell
        if result.status == "ok":
            sr = result.solve_result
            assert sr is not None
            row = DualRow(
                run_id=run_id,
                asset=cell.asset,
                unit=cell.unit,
                layer=cell.layer,
                step=cell.step,
                status="ok",
                detail="",
                admissible=sr.admissible,
                max_load_over_balanced=sr.max_load_over_balanced,
                dead_experts=sr.dead_experts,
                token_sha256=result.token_sha256,
                dump_path=str(cell.dump_path),
                duals=tuple(float(x) for x in sr.duals),
                score_function=result.score_function,
            )
        else:
            row = DualRow(
                run_id=run_id,
                asset=cell.asset,
                unit=cell.unit,
                layer=cell.layer,
                step=cell.step,
                status="failed",
                detail=result.detail,
                admissible=None,
                max_load_over_balanced=None,
                dead_experts=None,
                # The dump was read successfully even though the price failed, so both of these
                # are known and blanking them would throw away a diagnostic that cost nothing.
                token_sha256=result.token_sha256,
                dump_path=str(cell.dump_path),
                duals=tuple(float("nan") for _ in range(result.num_experts)),
                score_function=result.score_function,
            )
        emit(row)

        if result.bias is None:
            return
        key = (cell.layer, cell.step)
        if key in bias_seen:
            prev_asset, prev_bias = bias_seen[key]
            if not np.array_equal(prev_bias, result.bias):
                raise ValueError(
                    f"run {run_id!r} layer {cell.layer} step {cell.step}: bias from asset "
                    f"{cell.asset!r} does not match asset {prev_asset!r}'s bias at the same "
                    "step, so these dumps are not probes of the same weights"
                )
            return
        bias_seen[key] = (cell.asset, np.asarray(result.bias))
        emit(
            BiasRow(
                run_id=run_id,
                layer=cell.layer,
                step=cell.step,
                status="ok",
                detail="",
                token_sha256=result.token_sha256,
                bias=tuple(float(x) for x in result.bias),
                asset=cell.asset,
            )
        )

    if workers == 1:
        for cell in cells:
            handle(_price_cell(cell, solve))
        return

    # spawn re-execs a fresh interpreter per worker rather than forking this process, matching
    # phi_gap_grid.run_grid, because forking a process already holding numpy/scipy state and its
    # background threads can deadlock a child that inherits a lock nothing remains to release.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = {executor.submit(_price_cell, cell, solve): cell for cell in cells}
        drained: set[Future] = set()
        try:
            for future in as_completed(futures):
                drained.add(future)
                handle(future.result())
        except BaseException:
            # Cancel what has not started, wait out what has, then emit every result that
            # finished without being consumed, so one exception does not discard finished solves
            # worth minutes each.
            executor.shutdown(wait=True, cancel_futures=True)
            for future in futures:
                if future in drained or not future.done() or future.cancelled():
                    continue
                try:
                    result = future.result()
                except BaseException:
                    continue
                try:
                    handle(result)
                except BaseException as drain_exc:
                    # handle() raises only on a bias mismatch, so this is a second, unrelated
                    # cell disagreeing with a stored bias while the drain is recovering from the
                    # first exception. Printing and continuing keeps the drain going, whereas
                    # letting it propagate here would replace the original exception and stop
                    # the loop partway through, discarding every finished cell after it.
                    print(f"error while draining a finished cell: {drain_exc}", file=sys.stderr)
            raise


def _to_csv_row(row: DualRow, fields: tuple[str, ...]) -> dict[str, str]:
    out = dict.fromkeys(fields, "")
    out["run_id"] = row.run_id
    out["asset"] = row.asset
    out["unit"] = row.unit
    out["layer"] = str(row.layer)
    out["step"] = str(row.step)
    out["status"] = row.status
    out["detail"] = row.detail
    # Known regardless of status, because the dump was already read before a solve could fail,
    # so a failed row still names which dump and score space it came from.
    out["score_function"] = row.score_function
    out["token_sha256"] = row.token_sha256
    out["dump_path"] = row.dump_path
    if row.admissible is not None:
        out["admissible"] = str(row.admissible)
        out["max_load_over_balanced"] = str(row.max_load_over_balanced)
        out["dead_experts"] = str(row.dead_experts)
    for i, value in enumerate(row.duals):
        out[f"dual_{i}"] = str(value)
    return out


def _existing_header(csv_path: Path) -> tuple[str, ...] | None:
    """The header row already on `csv_path`, or `None` when the file does not exist or was
    killed before its header was ever flushed."""
    if not csv_path.exists():
        return None
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            return tuple(next(reader))
        except StopIteration:
            return None


def existing_dual_keys(csv_path: Path) -> set[tuple[str, ...]]:
    """The `DUAL_KEY_FIELDS` tuples already written to `csv_path` as `"ok"`, as the raw strings a
    CSV round trip produces, matching `phi_gap_grid.existing_keys` exactly: a `"failed"` row is
    skipped so its cell is retried, and a row with a blank key field, from a file truncated
    mid-write, is skipped the same way rather than raising.
    """
    if not csv_path.exists():
        return set()
    keys: set[tuple[str, ...]] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return keys
        for raw in reader:
            if raw.get("status") != "ok":
                continue
            if any(raw.get(field) in (None, "") for field in DUAL_KEY_FIELDS):
                continue
            keys.add(tuple(raw[field] for field in DUAL_KEY_FIELDS))
    return keys


def append_dual_rows(csv_path: Path, rows: Iterable[DualRow]) -> int:
    """Append `rows` to `csv_path`, writing the header only when the file does not yet exist.

    Every `"ok"` row's key is checked against `existing_dual_keys(csv_path)` and against every
    other `"ok"` row already validated in this same call before anything is written, so a
    duplicate key raises `ValueError` and leaves the file exactly as it was. A `"failed"` row is
    exempt, matching `phi_gap_grid.append_rows`, because it is not counted as done. The header
    width is taken from the rows themselves rather than assumed, since it is a property of this
    run's own expert count, and is checked against a header already on disk before anything is
    written: two calls disagreeing on expert count would otherwise interleave rows nothing can
    read back correctly.
    """
    rows = list(rows)
    if not rows:
        return 0
    num_experts = len(rows[0].duals)
    for row in rows:
        if len(row.duals) != num_experts:
            raise ValueError(
                f"{csv_path}: rows disagree on expert count, {len(row.duals)} vs {num_experts}"
            )
    fields = dual_fields(num_experts)

    existing_header = _existing_header(csv_path)
    if existing_header is not None and existing_header != fields:
        raise ValueError(
            f"{csv_path}: existing header has {len(existing_header)} columns, this call's rows "
            f"would write {len(fields)}, so they disagree on expert count or column set"
        )

    seen = existing_dual_keys(csv_path)
    for row in rows:
        if row.status != "ok":
            continue
        key = _dual_row_key(row)
        if key in seen:
            raise ValueError(f"duplicate dual row key {key!r} for {csv_path}")
        seen.add(key)

    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
            f.flush()
        for row in rows:
            writer.writerow(_to_csv_row(row, fields))
            f.flush()
            written += 1
    return written


# ---------------------------------------------------------------------------------------------
# The bias store: `existing_bias_keys`/`append_bias_rows` mirror the dual pair above, plus a
# comparison against an already-stored key rather than a silent drop, because dropping it unread
# is what kept the cross-asset agreement check from ever firing under the per-asset split.
# ---------------------------------------------------------------------------------------------


def _to_bias_csv_row(row: BiasRow, fields: tuple[str, ...]) -> dict[str, str]:
    out = dict.fromkeys(fields, "")
    out["run_id"] = row.run_id
    out["layer"] = str(row.layer)
    out["step"] = str(row.step)
    out["status"] = row.status
    out["detail"] = row.detail
    out["asset"] = row.asset
    out["token_sha256"] = row.token_sha256
    for i, value in enumerate(row.bias):
        out[f"bias_{i}"] = str(value)
    return out


def existing_bias_keys(csv_path: Path) -> set[tuple[str, str, str]]:
    """The `BIAS_KEY_FIELDS` tuples already written to `csv_path` as `"ok"`, matching
    `existing_dual_keys` exactly: a `"failed"` row or a row with a blank key field, from a file
    truncated mid-write, is skipped rather than counted as present.
    """
    if not csv_path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return keys
        for raw in reader:
            if raw.get("status") != "ok":
                continue
            if any(raw.get(field) in (None, "") for field in BIAS_KEY_FIELDS):
                continue
            keys.add(tuple(raw[field] for field in BIAS_KEY_FIELDS))
    return keys


def _read_ok_bias_rows(
    csv_path: Path,
) -> dict[tuple[str, str, str], tuple[str, tuple[float, ...]]]:
    """Every `"ok"` row already on `csv_path`, keyed by `BIAS_KEY_FIELDS`, as `(asset, bias)` so
    :func:`append_bias_rows` can compare an incoming row against what a key already holds rather
    than dropping it unread."""
    if not csv_path.exists():
        return {}
    rows: dict[tuple[str, str, str], tuple[str, tuple[float, ...]]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            return rows
        bias_cols = sorted(
            (name for name in fieldnames if name.startswith("bias_")),
            key=lambda name: int(name.removeprefix("bias_")),
        )
        for raw in reader:
            if raw.get("status") != "ok":
                continue
            if any(raw.get(field) in (None, "") for field in BIAS_KEY_FIELDS):
                continue
            try:
                bias = tuple(float(raw[col]) for col in bias_cols)
            except (KeyError, ValueError):
                continue  # a truncated trailing line, matching existing_bias_keys's tolerance
            key = tuple(raw[field] for field in BIAS_KEY_FIELDS)
            rows[key] = (raw.get("asset", ""), bias)
    return rows


def append_bias_rows(csv_path: Path, rows: Iterable[BiasRow]) -> int:
    """Append `rows` to `csv_path`, writing the header only when the file does not yet exist.

    An `"ok"` row whose key is already present is not written a second time, but its values are
    read back and compared first rather than dropped unread: agreement is silently skipped, and
    disagreement raises `ValueError` naming the step and both assets.
    """
    rows = list(rows)
    if not rows:
        return 0
    num_experts = len(rows[0].bias)
    for row in rows:
        if len(row.bias) != num_experts:
            raise ValueError(
                f"{csv_path}: rows disagree on expert count, {len(row.bias)} vs {num_experts}"
            )
    fields = bias_fields(num_experts)

    existing_header = _existing_header(csv_path)
    if existing_header is not None and existing_header != fields:
        raise ValueError(
            f"{csv_path}: existing header has {len(existing_header)} columns, this call's rows "
            f"would write {len(fields)}, so they disagree on expert count or column set"
        )

    known = _read_ok_bias_rows(csv_path)
    to_write = []
    for row in rows:
        if row.status != "ok":
            to_write.append(row)
            continue
        key = _bias_row_key(row)
        incoming = tuple(float(x) for x in row.bias)
        if key in known:
            prev_asset, prev_bias = known[key]
            if prev_bias != incoming:
                raise ValueError(
                    f"bias mismatch at run {row.run_id!r} layer {row.layer} step {row.step}: "
                    f"asset {row.asset!r} disagrees with already-stored asset {prev_asset!r}, "
                    "so these are not probes of the same weights"
                )
            continue  # matches what this key already holds, so nothing new needs writing
        known[key] = (row.asset, incoming)
        to_write.append(row)

    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
            f.flush()
        for row in to_write:
            writer.writerow(_to_bias_csv_row(row, fields))
            f.flush()
            written += 1
    return written


def price_bias_only_cells(
    cells: Sequence[DualCell],
    *,
    run_id: str,
    emit: Callable[[BiasRow], None],
) -> None:
    """Fill each cell's bias row without solving an LP: read its dump, take `expert_bias`, emit.

    For a cell whose dual is already stored and only its bias is missing, so a bias store lost or
    truncated after a complete pricing pass is rebuilt from the dumps still on disk rather than by
    re-paying for every solve. Serial only, unlike :func:`price_cells`: reading one dump's bias
    costs a fraction of a second, so a `spawn` pool's per-worker startup would dominate rather
    than pay for itself. A run whose dumps carry no `expert_bias` emits nothing here, matching
    `price_cells`, and is not a failure. Deduplicates and compares within this call exactly as
    `price_cells` does, catching two cells at the same `(layer, step)` from different assets in
    the *same* call, whereas the cross-invocation case is `append_bias_rows`'s job.
    """
    bias_seen: dict[tuple[int, int], tuple[str, np.ndarray]] = {}
    for cell in cells:
        dump = read_dump(cell.dump_path)
        try:
            axis_index = dump.layer_numbers.index(cell.layer)
        except ValueError:
            continue  # not among this dump's layers, so this cell has nothing to contribute
        try:
            bias = dump.expert_bias()[axis_index]
        except IncomparableProbes:
            continue  # this run keeps no bias, matching price_cells's own tolerance of that
        key = (cell.layer, cell.step)
        if key in bias_seen:
            prev_asset, prev_bias = bias_seen[key]
            if not np.array_equal(prev_bias, bias):
                raise ValueError(
                    f"run {run_id!r} layer {cell.layer} step {cell.step}: bias from asset "
                    f"{cell.asset!r} does not match asset {prev_asset!r}'s bias at the same "
                    "step, so these dumps are not probes of the same weights"
                )
            continue
        bias_seen[key] = (cell.asset, np.asarray(bias))
        emit(
            BiasRow(
                run_id=run_id,
                layer=cell.layer,
                step=cell.step,
                status="ok",
                detail="",
                token_sha256=dump.token_sha256,
                bias=tuple(float(x) for x in bias),
                asset=cell.asset,
            )
        )

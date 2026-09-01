import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

from moe_congestion_routing.metrics.phi_gap import PhiGapRow
from moe_congestion_routing.metrics.phi_gap_grid import (
    CSV_FIELDS,
    KEY_FIELDS,
    Cell,
    GridRow,
    append_rows,
    candidate_key,
    enumerate_cells,
    existing_keys,
    run_grid,
)

# ---------------------------------------------------------------------------------------------
# Test fixtures: module-level so a `spawn` worker can import and pickle them, and torch-free so
# the module they exercise stays that way too.
# ---------------------------------------------------------------------------------------------


def _write_dump(probes_dir, *, step, num_layers=2, n_tokens=8, num_experts=4, topk=1):
    """A metadata-only-relevant synthetic dump: enough for `enumerate_cells`, which never reads
    `routing_map` or `logits`, unlike `phi_gap_test.py`'s fixture which those rows do need."""
    probes_dir.mkdir(parents=True, exist_ok=True)
    logits = numpy.zeros((num_layers, n_tokens, num_experts), dtype=numpy.float32)
    routing_map = numpy.zeros((num_layers, n_tokens, num_experts), dtype=bool)
    packed = numpy.packbits(routing_map, axis=-1, bitorder="big")
    meta = {
        "moe_router_score_function": "softmax",
        "has_expert_bias": False,
        "iteration": step,
        "token_sha256": "cafe" * 16,
        "role": "standing",
        "moe_probe_coarse_interval": 25,
        "layer_numbers": list(range(2, 2 + num_layers)),
        "E": num_experts,
        "K": topk,
        "N": n_tokens,
        "moe_probe_batch": "assets/probe/default_asset.npz",
    }
    path = probes_dir / f"iter_{step:07d}.npz"
    numpy.savez(path, routing_map=packed, logits=logits, metadata=numpy.array(json.dumps(meta)))
    return path


def _fake_row(cell: Cell) -> PhiGapRow:
    """A `PhiGapRow` cheap enough to build in microseconds."""
    return PhiGapRow(
        unit=cell.unit,
        layer=cell.layer,
        step=cell.step,
        reference_cost=cell.cost_family,
        lam=cell.lam,
        affinity_space="score",
        score_function="softmax",
        admissible=True,
        max_load_over_balanced=1.0,
        dead_experts=0,
        gap_per_token=0.5,
        affinity_shortfall=0.1,
        congestion_excess=0.4,
        gap_normalized=0.2,
        normalizer=1.0,
        arc_growths=0,
        arcs_used_max=10,
        max_fractional_deviation=0.0,
        token_sha256="deadbeef",
        dump_path=str(cell.dump_path),
    )


def _raise_on_second(cell: Cell) -> list[PhiGapRow]:
    """Module level, not a closure, so a spawn worker could pickle it if one ever ran it."""
    if cell.unit == "u1":
        raise AssertionError("u1 is cursed")
    return [_fake_row(cell)]


_SPAWN_WITNESS = "unset-in-parent"


def _slow_or_poison(cell: Cell) -> list[PhiGapRow]:
    """Raises immediately on the poison cell and dawdles on the rest, so the poison result is
    drained while the others are still in flight. Module level so a spawn worker can import it."""
    import time

    if cell.unit == "poison":
        raise TypeError("boom")
    time.sleep(1.0)
    return [_fake_row(cell)]


def _report_witness(cell: Cell) -> list[PhiGapRow]:
    """Reports whether the worker inherited a global the parent set after import. Under spawn it
    cannot, because the child re-imports this module. Module level so spawn can import it."""
    row = _fake_row(cell)
    return [row._replace(token_sha256=_SPAWN_WITNESS)]


def _wrong_coordinates(cell: Cell) -> list[PhiGapRow]:
    """Returns a row describing a different unit than the cell asked for."""
    return [_fake_row(cell)._replace(unit="somewhere-else")]


def _ok_solve(cell: Cell) -> list[PhiGapRow]:
    return [_fake_row(cell)]


class _CountingSolve:
    """Wraps `_ok_solve` and counts calls, so a resume test can assert zero solves happened
    rather than only that the output file did not grow."""

    def __init__(self):
        self.calls = 0

    def __call__(self, cell: Cell) -> list[PhiGapRow]:
        self.calls += 1
        return _ok_solve(cell)


class _PoisonOnNth:
    """Raises `exc_factory(cell)` on the `n`-th call (1-indexed) and succeeds otherwise."""

    def __init__(self, n: int, exc_factory):
        self.n = n
        self.calls = 0
        self.exc_factory = exc_factory

    def __call__(self, cell: Cell) -> list[PhiGapRow]:
        self.calls += 1
        if self.calls == self.n:
            raise self.exc_factory(cell)
        return _ok_solve(cell)


def _cell(step: int, *, asset="asset0", layer=2, unit="u0", cost_family="linear", lam=1.0) -> Cell:
    return Cell(
        dump_path=Path(f"iter_{step:07d}.npz"),
        asset=asset,
        step=step,
        layer=layer,
        unit=unit,
        cost_family=cost_family,
        lam=lam,
    )


# ---------------------------------------------------------------------------------------------
# Torch stays out of the import graph.
# ---------------------------------------------------------------------------------------------


def test_no_torch_import():
    script = (
        "import sys; "
        "import moe_congestion_routing.metrics.phi_gap_grid; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


# ---------------------------------------------------------------------------------------------
# enumerate_cells
# ---------------------------------------------------------------------------------------------


def test_enumerate_cells_is_the_full_cross_product(tmp_path, monkeypatch):
    from moe_congestion_routing.metrics import probe_comparison

    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 4)
    run_dir = tmp_path / "run"
    _write_dump(run_dir / "probes" / "asset0", step=0, num_layers=2, n_tokens=8)

    cells = enumerate_cells(run_dir)
    # 1 asset x 2 layers x 2 units (n_tokens=8 / UNIT_TOKENS=4) x 2 cost families (the default).
    assert len(cells) == 1 * 2 * 2 * 2
    assert {c.asset for c in cells} == {"asset0"}
    assert {c.layer for c in cells} == {2, 3}
    assert {c.unit for c in cells} == {"u0", "u1"}
    assert {c.cost_family for c in cells} == {"linear", "quadratic"}
    assert all(c.lam == 1.0 for c in cells)


def test_enumerate_cells_filters_by_asset_layer_step(tmp_path, monkeypatch):
    from moe_congestion_routing.metrics import probe_comparison

    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 4)
    run_dir = tmp_path / "run"
    _write_dump(run_dir / "probes" / "asset0", step=0, num_layers=2, n_tokens=4)
    _write_dump(run_dir / "probes" / "asset0", step=25, num_layers=2, n_tokens=4)
    _write_dump(run_dir / "probes" / "asset1", step=0, num_layers=2, n_tokens=4)

    cells = enumerate_cells(
        run_dir,
        cost_families=("linear",),
        assets=["asset0"],
        layers=[3],
        steps=[25],
    )
    assert len(cells) == 1
    (cell,) = cells
    assert cell.asset == "asset0"
    assert cell.layer == 3
    assert cell.unit == "u0"
    assert cell.cost_family == "linear"
    assert cell.dump_path.name == "iter_0000025.npz"


def test_enumerate_cells_raises_on_missing_probes_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        enumerate_cells(tmp_path / "no-such-run")


# ---------------------------------------------------------------------------------------------
# run_grid: the mechanics, exercised with a fake solve so every test here runs in milliseconds.
# ---------------------------------------------------------------------------------------------


def test_run_grid_emits_one_row_per_cell_serially():
    cells = [_cell(0, unit="u0"), _cell(0, unit="u1"), _cell(25, unit="u0")]
    collected: list[GridRow] = []
    run_grid(cells, run_id="run-a", arm="control", emit=collected.append, solve=_ok_solve)
    assert len(collected) == 3
    assert all(row.status == "ok" for row in collected)
    assert all(row.run_id == "run-a" and row.arm == "control" for row in collected)


def test_a_failed_cell_is_a_row_and_the_grid_finishes():
    cells = [_cell(0, unit="u0"), _cell(0, unit="u1"), _cell(25, unit="u0")]
    solve = _PoisonOnNth(2, lambda cell: AssertionError(f"bad cell {cell.unit}"))
    collected: list[GridRow] = []
    run_grid(cells, run_id="run-a", arm="control", emit=collected.append, solve=solve)

    assert len(collected) == 3
    statuses = [row.status for row in collected]
    assert statuses.count("ok") == 2
    assert statuses.count("failed") == 1
    failed = next(row for row in collected if row.status == "failed")
    assert failed.row is None
    assert "bad cell u1" in failed.detail


def test_a_crash_propagates_and_keeps_earlier_rows():
    cells = [_cell(0, unit="u0"), _cell(0, unit="u1"), _cell(25, unit="u0")]
    solve = _PoisonOnNth(2, lambda cell: RuntimeError(f"boom on {cell.unit}"))
    collected: list[GridRow] = []
    with pytest.raises(RuntimeError, match="boom on u1"):
        run_grid(cells, run_id="run-a", arm="control", emit=collected.append, solve=solve)
    assert len(collected) == 1
    assert collected[0].status == "ok"


def test_parallel_equals_serial_once_sorted_by_key():
    cells = [
        _cell(step, unit=u, layer=layer)
        for step in (0, 25)
        for u in ("u0", "u1")
        for layer in (2, 3)
    ]

    serial: list[GridRow] = []
    run_grid(cells, run_id="run-a", arm="control", emit=serial.append, workers=1, solve=_ok_solve)

    parallel: list[GridRow] = []
    run_grid(cells, run_id="run-a", arm="control", emit=parallel.append, workers=2, solve=_ok_solve)

    def key(row: GridRow):
        r = row.row
        return (row.asset, r.unit, r.layer, r.step, r.reference_cost)

    assert sorted(serial, key=key) == sorted(parallel, key=key)


# ---------------------------------------------------------------------------------------------
# append_rows / existing_keys: the CSV side of resume.
# ---------------------------------------------------------------------------------------------


def _grid_row(cell: Cell, *, run_id="run-a", arm="control") -> GridRow:
    return GridRow(
        run_id=run_id,
        arm=arm,
        asset=cell.asset,
        unit=cell.unit,
        layer=cell.layer,
        step=cell.step,
        cost_family=cell.cost_family,
        lam=cell.lam,
        status="ok",
        detail="",
        row=_fake_row(cell),
    )


def test_header_is_written_once_and_a_resumed_append_does_not_repeat_it(tmp_path):
    csv_path = tmp_path / "control.csv"
    append_rows(csv_path, [_grid_row(_cell(0))])
    append_rows(csv_path, [_grid_row(_cell(25))])

    with csv_path.open() as f:
        lines = f.readlines()
    assert lines[0].strip() == ",".join(CSV_FIELDS)
    header_lines = [line for line in lines if line.strip() == ",".join(CSV_FIELDS)]
    assert len(header_lines) == 1
    assert len(lines) == 3  # header + 2 data rows


def test_duplicate_key_raises_and_leaves_the_file_unchanged(tmp_path):
    csv_path = tmp_path / "control.csv"
    append_rows(csv_path, [_grid_row(_cell(0))])
    before = csv_path.read_bytes()

    with pytest.raises(ValueError, match="duplicate"):
        append_rows(csv_path, [_grid_row(_cell(0))])

    assert csv_path.read_bytes() == before


def test_existing_keys_round_trips_a_written_row(tmp_path):
    csv_path = tmp_path / "control.csv"
    cell = _cell(0, asset="asset0", layer=2, unit="u0", cost_family="linear", lam=1.0)
    append_rows(csv_path, [_grid_row(cell, run_id="run-a", arm="control")])

    keys = existing_keys(csv_path)
    assert keys == {("run-a", "control", "asset0", "u0", "2", "0", "linear", "1.0")}


def test_a_failed_row_is_never_treated_as_an_existing_key(tmp_path):
    csv_path = tmp_path / "control.csv"
    cell = _cell(0, asset="asset0")
    failed = GridRow(
        run_id="run-a",
        arm="control",
        asset=cell.asset,
        unit=cell.unit,
        layer=cell.layer,
        step=cell.step,
        cost_family=cell.cost_family,
        lam=cell.lam,
        status="failed",
        detail="boom",
        row=None,
    )
    append_rows(csv_path, [failed])
    assert existing_keys(csv_path) == set()

    # The failed cell's own (real) key is therefore still free to be written afterwards.
    append_rows(csv_path, [_grid_row(_cell(0, asset="asset0"))])
    assert len(existing_keys(csv_path)) == 1


def test_resume_performs_zero_solves_and_appends_zero_rows(tmp_path):
    csv_path = tmp_path / "control.csv"
    cells = [_cell(0, unit="u0"), _cell(0, unit="u1"), _cell(25, unit="u0")]

    solve_1 = _CountingSolve()
    run_grid(
        cells,
        run_id="run-a",
        arm="control",
        emit=lambda row: append_rows(csv_path, [row]),
        solve=solve_1,
    )
    assert solve_1.calls == 3
    with csv_path.open() as f:
        rows_after_first = sum(1 for _ in csv.DictReader(f))
    assert rows_after_first == 3

    keys = existing_keys(csv_path)
    remaining = [c for c in cells if candidate_key(c, run_id="run-a", arm="control") not in keys]
    assert remaining == []

    solve_2 = _CountingSolve()
    run_grid(
        remaining,
        run_id="run-a",
        arm="control",
        emit=lambda row: append_rows(csv_path, [row]),
        solve=solve_2,
    )
    assert solve_2.calls == 0
    with csv_path.open() as f:
        rows_after_second = sum(1 for _ in csv.DictReader(f))
    assert rows_after_second == 3


def test_partial_trailing_line_is_dropped_not_raised(tmp_path):
    csv_path = tmp_path / "control.csv"
    append_rows(csv_path, [_grid_row(_cell(0))])
    append_rows(csv_path, [_grid_row(_cell(25))])

    # Simulate a process killed mid-`writerow`: cut the last line off inside its own key
    # columns, which sit first in `CSV_FIELDS`, rather than merely shortening a trailing
    # stat column that the key does not depend on.
    data = csv_path.read_bytes()
    last_newline = data.rindex(b"\n", 0, len(data) - 1)
    truncated = data[: last_newline + 1] + data[last_newline + 1 : last_newline + 15]
    csv_path.write_bytes(truncated)

    keys = existing_keys(csv_path)  # must not raise
    assert ("run-a", "control", "asset0", "u0", "2", "0", "linear", "1.0") in keys
    # The truncated second row's key is incomplete and therefore absent, so that cell would be
    # recomputed on a resumed run rather than being wrongly treated as already done.
    assert len(keys) == 1


def test_key_fields_order_matches_the_declared_contract():
    assert KEY_FIELDS == ("run_id", "arm", "asset", "unit", "layer", "step", "cost_family", "lam")


def test_a_failed_row_records_which_cell_failed(tmp_path):
    # A sweep runs thousands of cells unattended and phi_gap_rows raises without naming its own
    # cell, so a failure whose key columns are blank cannot be investigated without re-running
    # the whole grid to find it again.
    csv_path = tmp_path / "control.csv"
    cells = [_cell(0, unit="u0"), _cell(0, unit="u1")]
    rows: list[GridRow] = []
    run_grid(
        cells,
        run_id="run-a",
        arm="control",
        emit=rows.append,
        solve=_raise_on_second,
    )
    append_rows(csv_path, rows)

    failed = [r for r in rows if r.status == "failed"]
    assert len(failed) == 1
    assert (failed[0].unit, failed[0].layer, failed[0].step) == ("u1", 2, 0)
    assert failed[0].cost_family == "linear"

    written = list(csv.DictReader(csv_path.open()))
    failed_line = [r for r in written if r["status"] == "failed"][0]
    for field in KEY_FIELDS:
        assert failed_line[field] != "", f"{field} is blank on a failed row"
    assert failed_line["unit"] == "u1"
    assert failed_line["detail"] == "u1 is cursed"
    # The measurement columns are the only ones that genuinely do not exist on a failure.
    assert failed_line["gap_per_token"] == ""


def test_a_parallel_crash_emits_every_cell_that_finished(tmp_path):
    # A crash discarding in-flight completed work is the whole loss the resume exists to avoid,
    # and at the planned worker count one unexpected exception would throw away hundreds of
    # solves worth minutes each. The serial path cannot show this, because it never has more
    # than one cell in flight.
    cells = [_cell(0, layer=i, unit="poison" if i == 0 else "u0") for i in range(5)]
    got: list[GridRow] = []
    with pytest.raises(TypeError, match="boom"):
        run_grid(cells, run_id="r", arm="c", emit=got.append, workers=4, solve=_slow_or_poison)

    assert sorted(row.layer for row in got) == [1, 2, 3, 4]


def test_a_crash_leaves_whole_rows_on_disk(tmp_path):
    # The done-means says the rows survive on disk, and asserting on an in-memory list would
    # also pass if nothing were ever written or if a row were half-written.
    csv_path = tmp_path / "control.csv"
    cells = [_cell(step) for step in (0, 25, 50)]
    with pytest.raises(RuntimeError):
        run_grid(
            cells,
            run_id="run-a",
            arm="control",
            emit=lambda row: append_rows(csv_path, [row]),
            solve=_PoisonOnNth(2, lambda cell: RuntimeError("stop")),
        )

    written = list(csv.DictReader(csv_path.open()))
    assert [r["step"] for r in written] == ["0"]
    assert written[0]["gap_per_token"] != ""


def test_csv_fields_carry_every_phi_gap_row_field(tmp_path):
    # CSV_FIELDS is a hand-written list beside PhiGapRow rather than derived from it, so a field
    # added to the row would otherwise be dropped from every sweep silently, and one deleted
    # here would take a provenance column with it.
    folded_into_the_key = {"unit", "layer", "step", "reference_cost", "lam"}
    for field in PhiGapRow._fields:
        if field in folded_into_the_key:
            continue
        assert field in CSV_FIELDS, f"{field} would be dropped from every grid CSV"


def test_the_worker_pool_starts_with_spawn(tmp_path):
    # Only spawn re-imports this module in the child, so a global the parent set after import is
    # not inherited. Under fork the child would see the parent's value.
    global _SPAWN_WITNESS
    _SPAWN_WITNESS = "set-in-parent"
    try:
        got: list[GridRow] = []
        run_grid([_cell(0)], run_id="r", arm="c", emit=got.append, workers=2, solve=_report_witness)
    finally:
        _SPAWN_WITNESS = "unset-in-parent"
    assert got[0].row is not None
    assert got[0].row.token_sha256 == "unset-in-parent"


def test_a_truncated_quoted_field_is_skipped_not_raised(tmp_path):
    # A detail message routinely contains a comma, so csv quotes it, and a kill mid-write can cut
    # inside those quotes. That is the case the csv.Error handler exists for, and the unquoted
    # truncation test cannot reach it.
    csv_path = tmp_path / "control.csv"
    cell = _cell(0)
    failed = GridRow(
        run_id="run-a",
        arm="control",
        asset=cell.asset,
        unit=cell.unit,
        layer=cell.layer,
        step=cell.step,
        cost_family=cell.cost_family,
        lam=cell.lam,
        status="failed",
        detail="boom, and then more boom",
        row=None,
    )
    append_rows(csv_path, [_grid_row(_cell(25))])
    append_rows(csv_path, [failed])
    raw = csv_path.read_text()
    assert '"boom, and then more boom"' in raw
    csv_path.write_text(raw[: raw.rindex("boom, and") + 5])

    assert existing_keys(csv_path) == {
        ("run-a", "control", "asset0", "u0", "2", "25", "linear", "1.0")
    }


def test_a_non_default_lam_round_trips_through_the_key(tmp_path):
    csv_path = tmp_path / "control.csv"
    cell = _cell(0, lam=0.25)
    append_rows(csv_path, [_grid_row(cell)])
    assert existing_keys(csv_path) == {
        ("run-a", "control", "asset0", "u0", "2", "0", "linear", "0.25")
    }


def test_a_row_describing_another_cell_raises(tmp_path):
    # The CSV's key columns are the cell's claim rather than the measurement's, so a row echoing
    # a different coordinate would be filed under the wrong key instead of caught downstream.
    with pytest.raises(ValueError, match="coordinates"):
        run_grid([_cell(0)], run_id="r", arm="c", emit=lambda row: None, solve=_wrong_coordinates)

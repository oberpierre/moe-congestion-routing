import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

from moe_congestion_routing.metrics.dual_store import (
    BIAS_KEY_FIELDS,
    DUAL_KEY_FIELDS,
    BiasRow,
    DualCell,
    DualRow,
    DualSolveResult,
    append_bias_rows,
    append_dual_rows,
    bias_fields,
    bias_key,
    dual_fields,
    dual_key,
    enumerate_dual_cells,
    existing_bias_keys,
    existing_dual_keys,
    price_bias_only_cells,
    price_cells,
)
from moe_congestion_routing.metrics.probe_series import IncomparableProbes

# ---------------------------------------------------------------------------------------------
# Test fixtures: module-level so a `spawn` worker can import and pickle them, torch-free so the
# module they exercise stays that way too.
# ---------------------------------------------------------------------------------------------


def _write_dump(
    dump_dir,
    *,
    step,
    num_layers=2,
    n_tokens=8,
    num_experts=4,
    topk=1,
    has_expert_bias=True,
    bias_value=0.0,
    moe_probe_batch="assets/probe/default_asset.npz",
):
    """A metadata-only-relevant synthetic dump, matching `phi_gap_grid_test`'s fixture, plus a
    real `expert_bias` array: `price_cells` always reads bias for real, unlike `solve`, which a
    test replaces."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    logits = numpy.zeros((num_layers, n_tokens, num_experts), dtype=numpy.float32)
    routing_map = numpy.zeros((num_layers, n_tokens, num_experts), dtype=bool)
    packed = numpy.packbits(routing_map, axis=-1, bitorder="big")
    meta = {
        "moe_router_score_function": "softmax",
        "has_expert_bias": has_expert_bias,
        "iteration": step,
        "token_sha256": "cafe" * 16,
        "role": "standing",
        "moe_probe_coarse_interval": 25,
        "layer_numbers": list(range(2, 2 + num_layers)),
        "E": num_experts,
        "K": topk,
        "N": n_tokens,
        "moe_probe_batch": moe_probe_batch,
    }
    arrays = {"routing_map": packed, "logits": logits, "metadata": numpy.array(json.dumps(meta))}
    if has_expert_bias:
        arrays["expert_bias"] = numpy.full(
            (num_layers, num_experts), bias_value, dtype=numpy.float32
        )
    path = dump_dir / f"iter_{step:07d}.npz"
    numpy.savez(path, **arrays)
    return path


def _cell(step: int, *, asset="asset0", layer=2, unit="u0", dump_path=None) -> DualCell:
    return DualCell(
        dump_path=dump_path or Path(f"iter_{step:07d}.npz"),
        asset=asset,
        unit=unit,
        layer=layer,
        step=step,
    )


def _ok_solve(cell: DualCell) -> DualSolveResult:
    return DualSolveResult(True, 1.0, 0, numpy.array([0.1, 0.2, 0.3, 0.4]))


def _refused_solve(cell: DualCell) -> DualSolveResult:
    return DualSolveResult(False, 5.0, 1, numpy.full(4, numpy.nan))


class _CountingSolve:
    """Wraps `_ok_solve` and counts calls, so a resume test can assert zero solves happened
    rather than only that the output file did not grow."""

    def __init__(self):
        self.calls = 0

    def __call__(self, cell: DualCell) -> DualSolveResult:
        self.calls += 1
        return _ok_solve(cell)


class _PoisonOnNth:
    """Raises `exc_factory(cell)` on the `n`-th call (1-indexed) and succeeds otherwise."""

    def __init__(self, n: int, exc_factory):
        self.n = n
        self.calls = 0
        self.exc_factory = exc_factory

    def __call__(self, cell: DualCell) -> DualSolveResult:
        self.calls += 1
        if self.calls == self.n:
            raise self.exc_factory(cell)
        return _ok_solve(cell)


_SPAWN_WITNESS = "unset-in-parent"


def _report_witness(cell: DualCell) -> DualSolveResult:
    """Reports whether the worker inherited a global the parent set after import, by carrying it
    in `max_load_over_balanced` since `DualSolveResult` has no text field. Under spawn it cannot,
    because the child re-imports this module."""
    marker = 1.0 if _SPAWN_WITNESS == "unset-in-parent" else 99.0
    return DualSolveResult(True, marker, 0, numpy.zeros(4))


def _slow_or_poison(cell: DualCell) -> DualSolveResult:
    """Raises immediately on the poison cell and dawdles on the rest, so the poison result is
    drained while the others are still in flight. Module level so a spawn worker can import it."""
    import time

    if cell.unit == "poison":
        raise TypeError("boom")
    time.sleep(1.0)
    return _ok_solve(cell)


# ---------------------------------------------------------------------------------------------
# Torch stays out of the import graph.
# ---------------------------------------------------------------------------------------------


def test_no_torch_import():
    script = (
        "import sys; "
        "import moe_congestion_routing.metrics.dual_store; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


# ---------------------------------------------------------------------------------------------
# enumerate_dual_cells
# ---------------------------------------------------------------------------------------------


def test_enumerate_dual_cells_is_the_full_cross_product(tmp_path, monkeypatch):
    from moe_congestion_routing.metrics import probe_comparison

    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 4)
    run_dir = tmp_path / "run"
    _write_dump(run_dir / "probes" / "asset0", step=0, num_layers=2, n_tokens=8)

    cells = enumerate_dual_cells(run_dir)
    # 1 asset x 2 layers x 2 units (n_tokens=8 / UNIT_TOKENS=4).
    assert len(cells) == 1 * 2 * 2
    assert {c.asset for c in cells} == {"asset0"}
    assert {c.layer for c in cells} == {2, 3}
    assert {c.unit for c in cells} == {"u0", "u1"}


def test_enumerate_dual_cells_filters_by_asset_layer_step(tmp_path, monkeypatch):
    from moe_congestion_routing.metrics import probe_comparison

    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 4)
    run_dir = tmp_path / "run"
    _write_dump(run_dir / "probes" / "asset0", step=0, num_layers=2, n_tokens=4)
    _write_dump(run_dir / "probes" / "asset0", step=25, num_layers=2, n_tokens=4)
    _write_dump(run_dir / "probes" / "asset1", step=0, num_layers=2, n_tokens=4)

    cells = enumerate_dual_cells(run_dir, assets=["asset0"], layers=[3], steps=[25])
    assert len(cells) == 1
    (cell,) = cells
    assert cell.asset == "asset0"
    assert cell.layer == 3
    assert cell.unit == "u0"
    assert cell.dump_path.name == "iter_0000025.npz"


def test_enumerate_dual_cells_raises_on_missing_probes_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        enumerate_dual_cells(tmp_path / "no-such-run")


def test_enumerate_dual_cells_reads_the_flat_layout(tmp_path, monkeypatch):
    from moe_congestion_routing.metrics import probe_comparison

    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 4)
    run_dir = tmp_path / "run"
    _write_dump(
        run_dir / "probes",
        step=0,
        num_layers=1,
        n_tokens=4,
        moe_probe_batch="/scratch/assets/probe/standing_climbmix_small_16x2048.npz",
    )

    cells = enumerate_dual_cells(run_dir)
    assert len(cells) == 1
    assert cells[0].asset == "standing_climbmix_small_16x2048"


def test_enumerate_dual_cells_raises_on_mixed_layout(tmp_path):
    run_dir = tmp_path / "run"
    _write_dump(run_dir / "probes", step=0, num_layers=1, n_tokens=4)
    _write_dump(run_dir / "probes" / "asset0", step=0, num_layers=1, n_tokens=4)
    with pytest.raises(IncomparableProbes):
        enumerate_dual_cells(run_dir)


# ---------------------------------------------------------------------------------------------
# price_cells: the mechanics, exercised with a fake solve so every test here runs in milliseconds.
# ---------------------------------------------------------------------------------------------


def test_price_cells_emits_one_row_per_cell_serially(tmp_path):
    dump_path = _write_dump(tmp_path / "probes" / "asset0", step=0, has_expert_bias=False)
    cells = [_cell(0, dump_path=dump_path, unit="u0"), _cell(0, dump_path=dump_path, unit="u1")]
    collected: list = []
    price_cells(cells, run_id="run-a", emit=collected.append, solve=_ok_solve)
    dual_rows = [r for r in collected if isinstance(r, DualRow)]
    assert len(dual_rows) == 2
    assert all(row.status == "ok" for row in dual_rows)
    assert all(row.run_id == "run-a" for row in dual_rows)


def test_a_refused_unit_is_a_stored_row_with_nan_duals(tmp_path):
    dump_path = _write_dump(tmp_path / "probes" / "asset0", step=0, has_expert_bias=False)
    cells = [_cell(0, dump_path=dump_path, unit="u0")]
    collected: list[DualRow] = []
    price_cells(cells, run_id="run-a", emit=collected.append, solve=_refused_solve)
    (row,) = collected
    assert row.status == "ok"
    assert row.admissible is False
    assert row.max_load_over_balanced == 5.0
    assert row.dead_experts == 1
    assert all(v != v for v in row.duals)  # NaN


def test_a_failed_cell_is_a_row_and_the_pass_continues(tmp_path):
    dump_path = _write_dump(tmp_path / "probes" / "asset0", step=0, has_expert_bias=False)
    cells = [_cell(0, dump_path=dump_path, unit="u0"), _cell(0, dump_path=dump_path, unit="u1")]
    solve = _PoisonOnNth(2, lambda cell: AssertionError(f"bad cell {cell.unit}"))
    collected: list[DualRow] = []
    price_cells(cells, run_id="run-a", emit=collected.append, solve=solve)

    assert len(collected) == 2
    statuses = [row.status for row in collected]
    assert statuses.count("ok") == 1
    assert statuses.count("failed") == 1
    failed = next(row for row in collected if row.status == "failed")
    assert "bad cell u1" in failed.detail
    assert (
        failed.asset == "asset0" and failed.unit == "u1" and failed.layer == 2 and failed.step == 0
    )
    assert failed.admissible is None
    assert all(v != v for v in failed.duals)


def test_a_crash_propagates_and_keeps_earlier_rows(tmp_path):
    dump_path = _write_dump(tmp_path / "probes" / "asset0", step=0, has_expert_bias=False)
    cells = [_cell(0, dump_path=dump_path, unit="u0"), _cell(0, dump_path=dump_path, unit="u1")]
    solve = _PoisonOnNth(2, lambda cell: RuntimeError(f"boom on {cell.unit}"))
    collected: list[DualRow] = []
    with pytest.raises(RuntimeError, match="boom on u1"):
        price_cells(cells, run_id="run-a", emit=collected.append, solve=solve)
    assert len(collected) == 1
    assert collected[0].status == "ok"


def test_parallel_equals_serial_once_sorted_by_key(tmp_path):
    dump_path = _write_dump(
        tmp_path / "probes" / "asset0", step=0, num_layers=2, has_expert_bias=False
    )
    cells = [
        _cell(0, dump_path=dump_path, unit=u, layer=layer) for u in ("u0", "u1") for layer in (2, 3)
    ]

    serial: list[DualRow] = []
    price_cells(cells, run_id="run-a", emit=serial.append, workers=1, solve=_ok_solve)

    parallel: list[DualRow] = []
    price_cells(cells, run_id="run-a", emit=parallel.append, workers=2, solve=_ok_solve)

    def key(row: DualRow):
        return (row.asset, row.unit, row.layer, row.step)

    assert sorted(serial, key=key) == sorted(parallel, key=key)


def test_the_worker_pool_starts_with_spawn(tmp_path):
    # Only spawn re-imports this module in the child, so a global the parent set after import is
    # not inherited. Under fork the child would see the parent's value.
    import moe_congestion_routing.metrics.dual_store_test as this_module

    dump_path = _write_dump(tmp_path / "probes" / "asset0", step=0, has_expert_bias=False)
    this_module._SPAWN_WITNESS = "set-in-parent"
    try:
        got: list[DualRow] = []
        price_cells(
            [_cell(0, dump_path=dump_path)],
            run_id="r",
            emit=got.append,
            workers=2,
            solve=_report_witness,
        )
    finally:
        this_module._SPAWN_WITNESS = "unset-in-parent"
    assert got[0].max_load_over_balanced == 1.0


# ---------------------------------------------------------------------------------------------
# price_cells: the bias store, written in the same pass.
# ---------------------------------------------------------------------------------------------


def test_a_run_with_no_bias_writes_no_bias_rows_and_does_not_fail(tmp_path):
    dump_path = _write_dump(tmp_path / "probes" / "asset0", step=0, has_expert_bias=False)
    collected: list = []
    price_cells(
        [_cell(0, dump_path=dump_path)], run_id="run-a", emit=collected.append, solve=_ok_solve
    )
    assert len(collected) == 1
    assert isinstance(collected[0], DualRow)


def test_bias_is_emitted_once_per_layer_step_from_first_asset(tmp_path):
    dump_a = _write_dump(
        tmp_path / "probes" / "asset_a", step=0, num_layers=1, num_experts=4, bias_value=1.0
    )
    dump_b = _write_dump(
        tmp_path / "probes" / "asset_b", step=0, num_layers=1, num_experts=4, bias_value=1.0
    )
    cells = [
        _cell(0, dump_path=dump_a, asset="asset_a"),
        _cell(0, dump_path=dump_b, asset="asset_b"),
    ]
    collected: list = []
    price_cells(cells, run_id="run-a", emit=collected.append, solve=_ok_solve)
    bias_rows = [r for r in collected if isinstance(r, BiasRow)]
    assert len(bias_rows) == 1
    assert bias_rows[0].bias == (1.0, 1.0, 1.0, 1.0)


def test_a_bias_mismatch_across_assets_at_one_step_raises(tmp_path):
    dump_a = _write_dump(
        tmp_path / "probes" / "asset_a", step=0, num_layers=1, num_experts=4, bias_value=1.0
    )
    dump_b = _write_dump(
        tmp_path / "probes" / "asset_b", step=0, num_layers=1, num_experts=4, bias_value=2.0
    )
    cells = [
        _cell(0, dump_path=dump_a, asset="asset_a"),
        _cell(0, dump_path=dump_b, asset="asset_b"),
    ]
    collected: list = []
    with pytest.raises(ValueError, match="does not match"):
        price_cells(cells, run_id="run-a", emit=collected.append, solve=_ok_solve)
    # The first cell's dual row and bias row are kept, matching a crash keeping its earlier rows.
    assert any(isinstance(r, DualRow) for r in collected)
    assert any(isinstance(r, BiasRow) for r in collected)


# ---------------------------------------------------------------------------------------------
# append_dual_rows / existing_dual_keys: the CSV side of resume.
# ---------------------------------------------------------------------------------------------


def _dual_row(cell: DualCell, *, run_id="run-a", num_experts=4) -> DualRow:
    return DualRow(
        run_id=run_id,
        asset=cell.asset,
        unit=cell.unit,
        layer=cell.layer,
        step=cell.step,
        status="ok",
        detail="",
        admissible=True,
        max_load_over_balanced=1.0,
        dead_experts=0,
        token_sha256="deadbeef",
        dump_path=str(cell.dump_path),
        duals=tuple(float(i) for i in range(num_experts)),
    )


def test_header_is_written_once_and_a_resumed_append_does_not_repeat_it(tmp_path):
    csv_path = tmp_path / "duals.csv"
    append_dual_rows(csv_path, [_dual_row(_cell(0))])
    append_dual_rows(csv_path, [_dual_row(_cell(25))])

    with csv_path.open() as f:
        lines = f.readlines()
    assert lines[0].strip() == ",".join(dual_fields(4))
    header_lines = [line for line in lines if line.strip() == ",".join(dual_fields(4))]
    assert len(header_lines) == 1
    assert len(lines) == 3  # header + 2 data rows


def test_duplicate_key_raises_and_leaves_the_file_unchanged(tmp_path):
    csv_path = tmp_path / "duals.csv"
    append_dual_rows(csv_path, [_dual_row(_cell(0))])
    before = csv_path.read_bytes()

    with pytest.raises(ValueError, match="duplicate"):
        append_dual_rows(csv_path, [_dual_row(_cell(0))])

    assert csv_path.read_bytes() == before


def test_existing_dual_keys_round_trips_a_written_row(tmp_path):
    csv_path = tmp_path / "duals.csv"
    cell = _cell(0, asset="asset0", layer=2, unit="u0")
    append_dual_rows(csv_path, [_dual_row(cell, run_id="run-a")])

    keys = existing_dual_keys(csv_path)
    assert keys == {("run-a", "asset0", "u0", "2", "0")}


def test_a_failed_row_is_never_treated_as_an_existing_key(tmp_path):
    csv_path = tmp_path / "duals.csv"
    cell = _cell(0, asset="asset0")
    failed = DualRow(
        run_id="run-a",
        asset=cell.asset,
        unit=cell.unit,
        layer=cell.layer,
        step=cell.step,
        status="failed",
        detail="boom",
        admissible=None,
        max_load_over_balanced=None,
        dead_experts=None,
        token_sha256="",
        dump_path="",
        duals=tuple(float("nan") for _ in range(4)),
    )
    append_dual_rows(csv_path, [failed])
    assert existing_dual_keys(csv_path) == set()

    append_dual_rows(csv_path, [_dual_row(_cell(0, asset="asset0"))])
    assert len(existing_dual_keys(csv_path)) == 1


def test_resume_performs_zero_solves_and_appends_zero_rows(tmp_path):
    dump_path = _write_dump(tmp_path / "probes" / "asset0", step=0, has_expert_bias=False)
    csv_path = tmp_path / "duals.csv"
    cells = [_cell(0, dump_path=dump_path, unit="u0"), _cell(0, dump_path=dump_path, unit="u1")]

    solve_1 = _CountingSolve()
    price_cells(
        cells,
        run_id="run-a",
        emit=lambda row: append_dual_rows(csv_path, [row]) if isinstance(row, DualRow) else None,
        solve=solve_1,
    )
    assert solve_1.calls == 2
    with csv_path.open() as f:
        rows_after_first = sum(1 for _ in csv.DictReader(f))
    assert rows_after_first == 2

    keys = existing_dual_keys(csv_path)
    remaining = [c for c in cells if dual_key(c, run_id="run-a") not in keys]
    assert remaining == []

    solve_2 = _CountingSolve()
    price_cells(
        remaining,
        run_id="run-a",
        emit=lambda row: append_dual_rows(csv_path, [row]) if isinstance(row, DualRow) else None,
        solve=solve_2,
    )
    assert solve_2.calls == 0
    with csv_path.open() as f:
        rows_after_second = sum(1 for _ in csv.DictReader(f))
    assert rows_after_second == 2


def test_partial_trailing_line_is_dropped_not_raised(tmp_path):
    csv_path = tmp_path / "duals.csv"
    append_dual_rows(csv_path, [_dual_row(_cell(0))])
    append_dual_rows(csv_path, [_dual_row(_cell(25))])

    data = csv_path.read_bytes()
    last_newline = data.rindex(b"\n", 0, len(data) - 1)
    truncated = data[: last_newline + 1] + data[last_newline + 1 : last_newline + 15]
    csv_path.write_bytes(truncated)

    keys = existing_dual_keys(csv_path)  # must not raise
    assert ("run-a", "asset0", "u0", "2", "0") in keys
    assert len(keys) == 1


def test_key_fields_order_matches_the_declared_contract():
    assert DUAL_KEY_FIELDS == ("run_id", "asset", "unit", "layer", "step")
    assert BIAS_KEY_FIELDS == ("run_id", "layer", "step")


def test_dual_fields_and_bias_fields_scale_with_expert_count():
    assert dual_fields(4)[-4:] == ("dual_0", "dual_1", "dual_2", "dual_3")
    assert dual_fields(64)[-1] == "dual_63"
    assert bias_fields(4)[-4:] == ("bias_0", "bias_1", "bias_2", "bias_3")


def test_a_truncated_quoted_field_is_skipped_not_raised(tmp_path):
    csv_path = tmp_path / "duals.csv"
    cell = _cell(0)
    failed = DualRow(
        run_id="run-a",
        asset=cell.asset,
        unit=cell.unit,
        layer=cell.layer,
        step=cell.step,
        status="failed",
        detail="boom, and then more boom",
        admissible=None,
        max_load_over_balanced=None,
        dead_experts=None,
        token_sha256="",
        dump_path="",
        duals=tuple(float("nan") for _ in range(4)),
    )
    append_dual_rows(csv_path, [_dual_row(_cell(25))])
    append_dual_rows(csv_path, [failed])
    raw = csv_path.read_text()
    assert '"boom, and then more boom"' in raw
    csv_path.write_text(raw[: raw.rindex("boom, and") + 5])

    assert existing_dual_keys(csv_path) == {("run-a", "asset0", "u0", "2", "25")}


def test_a_parallel_crash_emits_every_cell_that_finished(tmp_path):
    dump_path = _write_dump(
        tmp_path / "probes" / "asset0", step=0, num_layers=5, has_expert_bias=False
    )
    cells = [
        _cell(0, dump_path=dump_path, layer=2 + i, unit="poison" if i == 0 else "u0")
        for i in range(5)
    ]
    got: list[DualRow] = []
    with pytest.raises(TypeError, match="boom"):
        price_cells(cells, run_id="r", emit=got.append, workers=4, solve=_slow_or_poison)

    assert sorted(row.layer for row in got) == [3, 4, 5, 6]


def test_a_crash_leaves_whole_rows_on_disk(tmp_path):
    dump_path = _write_dump(tmp_path / "probes" / "asset0", step=0, has_expert_bias=False)
    csv_path = tmp_path / "duals.csv"
    cells = [_cell(step, dump_path=dump_path) for step in (0, 25, 50)]
    with pytest.raises(RuntimeError):
        price_cells(
            cells,
            run_id="run-a",
            emit=lambda row: (
                append_dual_rows(csv_path, [row]) if isinstance(row, DualRow) else None
            ),
            solve=_PoisonOnNth(2, lambda cell: RuntimeError("stop")),
        )

    written = list(csv.DictReader(csv_path.open()))
    assert [r["step"] for r in written] == ["0"]
    assert written[0]["dual_0"] != ""


# ---------------------------------------------------------------------------------------------
# score_function: read from the dump, carried through failure, into the CSV.
# ---------------------------------------------------------------------------------------------


def test_score_function_is_read_from_the_dump(tmp_path):
    dump_path = _write_dump(tmp_path / "probes" / "asset0", step=0, has_expert_bias=False)
    collected: list = []
    price_cells(
        [_cell(0, dump_path=dump_path, unit="u0")],
        run_id="run-a",
        emit=collected.append,
        solve=_ok_solve,
    )
    (row,) = collected
    assert row.score_function == "softmax"  # _write_dump's own moe_router_score_function


def test_a_failed_row_still_carries_token_sha256_and_dump_path(tmp_path):
    dump_path = _write_dump(tmp_path / "probes" / "asset0", step=0, has_expert_bias=False)
    cell = _cell(0, dump_path=dump_path, unit="u0")
    collected: list = []
    price_cells(
        [cell],
        run_id="run-a",
        emit=collected.append,
        solve=_PoisonOnNth(1, lambda cell: AssertionError("boom")),
    )
    (row,) = collected
    assert row.status == "failed"
    assert row.token_sha256 == "cafe" * 16
    assert row.dump_path == str(dump_path)
    assert row.score_function == "softmax"


def test_a_bad_layer_is_a_failed_row_not_an_abort(tmp_path):
    dump_path = _write_dump(tmp_path / "probes" / "asset0", step=0, num_layers=1)
    cell = _cell(0, dump_path=dump_path, layer=999, unit="u0")  # not among this dump's layers
    collected: list = []
    price_cells([cell], run_id="run-a", emit=collected.append, solve=_ok_solve)
    (row,) = collected
    assert row.status == "failed"


# ---------------------------------------------------------------------------------------------
# The drain guard: a second exception raised while handling a finished cell must not replace
# the crash the drain exists to preserve.
# ---------------------------------------------------------------------------------------------


def test_the_drain_survives_a_raising_handler_and_reraises_the_original(tmp_path):
    dump_a = _write_dump(
        tmp_path / "probes" / "asset_a", step=0, num_layers=1, num_experts=4, bias_value=1.0
    )
    dump_b = _write_dump(
        tmp_path / "probes" / "asset_b", step=0, num_layers=1, num_experts=4, bias_value=2.0
    )
    dump_c = _write_dump(
        tmp_path / "probes" / "asset_c", step=0, num_layers=2, num_experts=4, bias_value=1.0
    )
    cells = [
        _cell(0, dump_path=dump_a, asset="asset_a", layer=2, unit="poison"),
        _cell(0, dump_path=dump_a, asset="asset_a", layer=2, unit="u0"),
        _cell(0, dump_path=dump_b, asset="asset_b", layer=2, unit="u0"),
        _cell(0, dump_path=dump_c, asset="asset_c", layer=3, unit="u0"),
    ]
    got: list = []
    with pytest.raises(TypeError, match="boom"):
        price_cells(cells, run_id="r", emit=got.append, workers=4, solve=_slow_or_poison)

    dual_rows = [r for r in got if isinstance(r, DualRow)]
    bias_rows = [r for r in got if isinstance(r, BiasRow)]
    # asset_a's dual row is emitted before its bias mismatch with asset_b is discovered, matching
    # a crash keeping every row already handed to `emit`. asset_b's bias row never exists, because
    # `handle()` raises before reaching that `emit` call, and that raise is the one caught and
    # printed by the drain guard rather than the one that propagates.
    assert {(r.asset, r.layer) for r in dual_rows} == {
        ("asset_a", 2),
        ("asset_b", 2),
        ("asset_c", 3),
    }
    assert {(r.asset, r.layer) for r in bias_rows} == {("asset_a", 2), ("asset_c", 3)}


# ---------------------------------------------------------------------------------------------
# The bias store: existing_bias_keys / append_bias_rows, its own resume and cross-invocation
# comparison, and price_bias_only_cells filling a lost or partial store with no LP solve.
# ---------------------------------------------------------------------------------------------


def _bias_row(
    *, run_id="run-a", layer=2, step=0, asset="asset0", num_experts=4, value=1.0
) -> BiasRow:
    return BiasRow(
        run_id=run_id,
        layer=layer,
        step=step,
        status="ok",
        detail="",
        token_sha256="deadbeef",
        bias=tuple(float(value) for _ in range(num_experts)),
        asset=asset,
    )


def test_existing_bias_keys_round_trips_a_written_row(tmp_path):
    csv_path = tmp_path / "bias.csv"
    append_bias_rows(csv_path, [_bias_row(run_id="run-a", layer=2, step=0)])
    assert existing_bias_keys(csv_path) == {("run-a", "2", "0")}


def test_a_bias_row_matching_an_existing_key_is_silently_skipped(tmp_path):
    csv_path = tmp_path / "bias.csv"
    append_bias_rows(csv_path, [_bias_row(asset="asset_a", value=1.0)])
    written = append_bias_rows(csv_path, [_bias_row(asset="asset_b", value=1.0)])
    assert written == 0
    assert len(existing_bias_keys(csv_path)) == 1


def test_a_disagreeing_bias_across_separate_invocations_raises(tmp_path):
    # Two separate append_bias_rows calls against the same file, matching the per-asset split
    # where each asset is its own script invocation, which is the case the old in-process-only
    # check could never see.
    csv_path = tmp_path / "bias.csv"
    append_bias_rows(csv_path, [_bias_row(asset="asset_a", value=1.0)])
    with pytest.raises(ValueError, match="asset_a") as excinfo:
        append_bias_rows(csv_path, [_bias_row(asset="asset_b", value=2.0)])
    assert "asset_b" in str(excinfo.value)


def test_mismatched_header_width_raises_on_dual_append(tmp_path):
    csv_path = tmp_path / "duals.csv"
    append_dual_rows(csv_path, [_dual_row(_cell(0), num_experts=4)])
    with pytest.raises(ValueError, match="header"):
        append_dual_rows(csv_path, [_dual_row(_cell(25), num_experts=8)])


def test_mismatched_header_width_raises_on_bias_append(tmp_path):
    csv_path = tmp_path / "bias.csv"
    append_bias_rows(csv_path, [_bias_row(num_experts=4)])
    with pytest.raises(ValueError, match="header"):
        append_bias_rows(csv_path, [_bias_row(step=25, num_experts=8)])


def test_a_lost_bias_store_is_rebuilt_without_a_solve(tmp_path):
    dump_path = _write_dump(
        tmp_path / "probes" / "asset0", step=0, num_layers=1, num_experts=4, bias_value=1.0
    )
    duals_csv = tmp_path / "duals.csv"
    bias_csv = tmp_path / "bias.csv"
    cells = [_cell(0, dump_path=dump_path, unit="u0"), _cell(0, dump_path=dump_path, unit="u1")]

    def emit(row):
        if isinstance(row, DualRow):
            append_dual_rows(duals_csv, [row])
        else:
            append_bias_rows(bias_csv, [row])

    counting_solve = _CountingSolve()
    price_cells(cells, run_id="run-a", emit=emit, solve=counting_solve)
    assert counting_solve.calls == 2
    assert len(existing_bias_keys(bias_csv)) == 1

    bias_csv.unlink()  # the bias store is lost, the dual store is not

    done_duals = existing_dual_keys(duals_csv)
    done_bias = existing_bias_keys(bias_csv)
    bias_only = [
        c
        for c in cells
        if dual_key(c, run_id="run-a") in done_duals
        and bias_key("run-a", c.layer, c.step) not in done_bias
    ]
    assert len(bias_only) == 2  # both cells' duals are present, neither's bias is

    price_bias_only_cells(bias_only, run_id="run-a", emit=emit)
    assert counting_solve.calls == 2  # unchanged: the bias store came back with no new solve
    assert len(existing_bias_keys(bias_csv)) == 1


def test_a_partial_bias_store_is_completed_without_duplicating_existing_rows(tmp_path):
    dump_a = _write_dump(
        tmp_path / "probes" / "asset_a", step=0, num_layers=2, num_experts=4, bias_value=1.0
    )
    dump_b = _write_dump(
        tmp_path / "probes" / "asset_b", step=0, num_layers=2, num_experts=4, bias_value=1.0
    )
    duals_csv = tmp_path / "duals.csv"
    bias_csv = tmp_path / "bias.csv"
    cells = [_cell(0, dump_path=dump_a, asset="asset_a", layer=layer) for layer in (2, 3)] + [
        _cell(0, dump_path=dump_b, asset="asset_b", layer=layer) for layer in (2, 3)
    ]

    def emit(row):
        if isinstance(row, DualRow):
            append_dual_rows(duals_csv, [row])
        else:
            append_bias_rows(bias_csv, [row])

    price_cells(cells, run_id="run-a", emit=emit, solve=_ok_solve)
    assert existing_bias_keys(bias_csv) == {("run-a", "2", "0"), ("run-a", "3", "0")}

    # Truncate the store down to layer 2's row only, as a partial/interrupted store would leave.
    surviving = [raw for raw in csv.DictReader(bias_csv.open()) if raw["layer"] == "2"]
    header = bias_fields(4)
    with bias_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(surviving)

    done_duals = existing_dual_keys(duals_csv)
    done_bias = existing_bias_keys(bias_csv)
    assert done_bias == {("run-a", "2", "0")}
    bias_only = [
        c
        for c in cells
        if dual_key(c, run_id="run-a") in done_duals
        and bias_key("run-a", c.layer, c.step) not in done_bias
    ]
    assert {c.layer for c in bias_only} == {3}

    price_bias_only_cells(bias_only, run_id="run-a", emit=emit)
    assert existing_bias_keys(bias_csv) == {("run-a", "2", "0"), ("run-a", "3", "0")}


def test_a_refused_row_is_skipped_by_the_resume(tmp_path):
    dump_path = _write_dump(tmp_path / "probes" / "asset0", step=0, has_expert_bias=False)
    csv_path = tmp_path / "duals.csv"
    cell = _cell(0, dump_path=dump_path, unit="u0")
    price_cells(
        [cell],
        run_id="run-a",
        emit=lambda row: append_dual_rows(csv_path, [row]) if isinstance(row, DualRow) else None,
        solve=_refused_solve,
    )
    keys = existing_dual_keys(csv_path)
    remaining = [c for c in [cell] if dual_key(c, run_id="run-a") not in keys]
    assert remaining == []

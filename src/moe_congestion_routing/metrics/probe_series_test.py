import json

import numpy
import pytest

from moe_congestion_routing.metrics.probe_series import (
    IncomparableProbes,
    ProbeDump,
    ProbeSeries,
    SaturationRow,
    read_dump,
    read_series,
    saturation_rows,
)
from moe_congestion_routing.metrics.router_probe import ROUTING_MAP_BITORDER


def _write_dump(
    probes_dir,
    *,
    step,
    routing_map,
    token_sha256="cafe" * 16,
    role="standing",
    coarse_interval=6,
    layer_numbers=None,
    topk=2,
):
    """A synthetic ``.npz`` dump with the metadata keys ``probe_series.py`` reads.

    ``routing_map`` is ``[L, N, E]`` bool in unpacked form, packed here the same way
    ``router_probe.py`` packs it, so a round trip through this helper exercises the real bit
    layout rather than a stand-in.
    """
    probes_dir.mkdir(parents=True, exist_ok=True)
    num_layers, _num_tokens, num_experts = routing_map.shape
    if layer_numbers is None:
        layer_numbers = list(range(2, 2 + num_layers))
    packed = numpy.packbits(routing_map, axis=-1, bitorder=ROUTING_MAP_BITORDER)
    meta = {
        "iteration": step,
        "token_sha256": token_sha256,
        "role": role,
        "moe_probe_coarse_interval": coarse_interval,
        "layer_numbers": layer_numbers,
        "E": num_experts,
        "K": topk,
    }
    path = probes_dir / f"iter_{step:07d}.npz"
    numpy.savez(path, routing_map=packed, metadata=numpy.array(json.dumps(meta)))
    return path


def _one_hot_map(selections: list[list[int]], num_experts: int) -> numpy.ndarray:
    """``[N, E]`` bool from a per-token list of selected expert indices."""
    out = numpy.zeros((len(selections), num_experts), dtype=bool)
    for token, selected in enumerate(selections):
        for expert in selected:
            out[token, expert] = True
    return out


# --- read_dump / ProbeDump -------------------------------------------------------------------


def test_read_dump_round_trips_metadata_and_routing_map(tmp_path):
    routing_map = numpy.stack(
        [_one_hot_map([[0, 1], [2, 3]], 4), _one_hot_map([[1, 2], [0, 3]], 4)], axis=0
    )
    path = _write_dump(
        tmp_path / "probes", step=6, routing_map=routing_map, layer_numbers=[5, 7], topk=2
    )
    dump = read_dump(path)
    assert dump.step == 6
    assert dump.token_sha256 == "cafe" * 16
    assert dump.role == "standing"
    assert dump.coarse_interval == 6
    assert dump.layer_numbers == (5, 7)
    assert dump.num_experts == 4
    assert dump.topk == 2
    numpy.testing.assert_array_equal(dump.routing_map(), routing_map)


# --- read_series -------------------------------------------------------------------------------


def test_read_series_sorts_ascending_by_step(tmp_path):
    probes_dir = tmp_path / "probes"
    routing_map = numpy.stack([_one_hot_map([[0, 1]], 4)], axis=0)
    _write_dump(probes_dir, step=12, routing_map=routing_map)
    _write_dump(probes_dir, step=0, routing_map=routing_map)
    _write_dump(probes_dir, step=6, routing_map=routing_map)
    series = read_series(tmp_path)
    assert [dump.step for dump in series.dumps] == [0, 6, 12]


def test_read_series_raises_file_not_found_when_probes_dir_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match=str(tmp_path / "probes")):
        read_series(tmp_path)


def test_read_series_raises_file_not_found_when_probes_dir_is_empty(tmp_path):
    (tmp_path / "probes").mkdir()
    with pytest.raises(FileNotFoundError, match=str(tmp_path / "probes")):
        read_series(tmp_path)


def test_read_series_refuses_a_non_standing_role_by_default(tmp_path):
    routing_map = numpy.stack([_one_hot_map([[0, 1]], 4)], axis=0)
    _write_dump(tmp_path / "probes", step=0, routing_map=routing_map, role="dev")
    with pytest.raises(IncomparableProbes, match="dev"):
        read_series(tmp_path)


def test_read_series_allows_a_role_named_in_allow_roles(tmp_path):
    routing_map = numpy.stack([_one_hot_map([[0, 1]], 4)], axis=0)
    _write_dump(tmp_path / "probes", step=0, routing_map=routing_map, role="dev")
    series = read_series(tmp_path, allow_roles=("dev",))
    assert series.role == "dev"


def test_read_series_refuses_two_dumps_with_different_token_sha256(tmp_path):
    routing_map = numpy.stack([_one_hot_map([[0, 1]], 4)], axis=0)
    probes_dir = tmp_path / "probes"
    _write_dump(probes_dir, step=0, routing_map=routing_map, token_sha256="a" * 64)
    _write_dump(probes_dir, step=6, routing_map=routing_map, token_sha256="b" * 64)
    with pytest.raises(IncomparableProbes, match="token_sha256"):
        read_series(tmp_path)


def test_read_series_records_the_given_arm(tmp_path):
    routing_map = numpy.stack([_one_hot_map([[0, 1]], 4)], axis=0)
    _write_dump(tmp_path / "probes", step=0, routing_map=routing_map)
    series = read_series(tmp_path, arm="switch")
    assert series.arm == "switch"


# --- saturation_rows: single series -------------------------------------------------------------


def test_saturation_rows_reference_dump_agrees_with_itself_on_every_layer(tmp_path):
    routing_map = numpy.stack(
        [_one_hot_map([[0, 1], [1, 2]], 4), _one_hot_map([[2, 3], [0, 1]], 4)], axis=0
    )
    _write_dump(tmp_path / "probes", step=6, routing_map=routing_map, layer_numbers=[2, 5])
    series = read_series(tmp_path)
    rows = saturation_rows([series])
    assert len(rows) == 2  # one row per layer
    assert all(row.agreement == 1.0 for row in rows)
    assert all(row.reference_step == 6 for row in rows)
    assert {row.layer for row in rows} == {2, 5}


def test_saturation_rows_default_reference_is_the_last_dump_in_a_single_series(tmp_path):
    probes_dir = tmp_path / "probes"
    ref_map = numpy.stack([_one_hot_map([[0, 1], [2, 3]], 4)], axis=0)
    # step 0 disagrees with the reference on token 1's selection.
    early_map = numpy.stack([_one_hot_map([[0, 1], [1, 2]], 4)], axis=0)
    _write_dump(probes_dir, step=0, routing_map=early_map)
    _write_dump(probes_dir, step=6, routing_map=ref_map)
    series = read_series(tmp_path)
    rows = {row.step: row for row in saturation_rows([series])}
    assert rows[6].reference_step == 6
    assert rows[6].agreement == 1.0
    assert rows[0].reference_step == 6
    assert rows[0].agreement == pytest.approx(0.5)  # 1 of 2 tokens agrees


def test_saturation_rows_layer_is_the_megatron_layer_number_not_the_axis_index(tmp_path):
    routing_map = numpy.stack([_one_hot_map([[0, 1]], 4), _one_hot_map([[1, 2]], 4)], axis=0)
    _write_dump(tmp_path / "probes", step=0, routing_map=routing_map, layer_numbers=[5, 9])
    series = read_series(tmp_path)
    rows = saturation_rows([series])
    assert {row.layer for row in rows} == {5, 9}
    assert 0 not in {row.layer for row in rows}
    assert 1 not in {row.layer for row in rows}


def test_saturation_rows_explicit_reference_step_is_used_and_recorded(tmp_path):
    probes_dir = tmp_path / "probes"
    map_a = numpy.stack([_one_hot_map([[0, 1]], 4)], axis=0)
    map_b = numpy.stack([_one_hot_map([[2, 3]], 4)], axis=0)
    _write_dump(probes_dir, step=0, routing_map=map_a)
    _write_dump(probes_dir, step=6, routing_map=map_b)
    series = read_series(tmp_path)
    rows = {row.step: row for row in saturation_rows([series], reference_step=0)}
    assert rows[0].reference_step == 0
    assert rows[0].agreement == 1.0
    assert rows[6].reference_step == 0
    assert rows[6].agreement == 0.0


def test_saturation_rows_raises_value_error_when_reference_step_is_absent(tmp_path):
    routing_map = numpy.stack([_one_hot_map([[0, 1]], 4)], axis=0)
    _write_dump(tmp_path / "probes", step=0, routing_map=routing_map)
    _write_dump(tmp_path / "probes", step=6, routing_map=routing_map)
    series = read_series(tmp_path)
    with pytest.raises(ValueError, match=r"\[0, 6\]"):
        saturation_rows([series], reference_step=99)


def test_saturation_row_carries_provenance_fields(tmp_path):
    routing_map = numpy.stack([_one_hot_map([[0, 1]], 4)], axis=0)
    _write_dump(
        tmp_path / "probes",
        step=0,
        routing_map=routing_map,
        token_sha256="ab" * 32,
        coarse_interval=6,
    )
    series = read_series(tmp_path, arm="switch")
    row = saturation_rows([series])[0]
    assert isinstance(row, SaturationRow)
    assert row.run_dir == str(tmp_path)
    assert row.arm == "switch"
    assert row.role == "standing"
    assert row.token_sha256 == "ab" * 32
    assert row.coarse_interval == 6
    assert row.num_tokens == 1


# --- saturation_rows: multiple series ------------------------------------------------------------


def _series_with(tmp_path, name, *, steps, token_sha256, coarse_interval, num_experts=4):
    probes_dir = tmp_path / name / "probes"
    for step in steps:
        routing_map = numpy.stack([_one_hot_map([[0, 1]], num_experts)], axis=0)
        _write_dump(
            probes_dir,
            step=step,
            routing_map=routing_map,
            token_sha256=token_sha256,
            coarse_interval=coarse_interval,
        )
    return read_series(tmp_path / name)


def test_saturation_rows_refuses_two_series_with_different_token_sha256(tmp_path):
    a = _series_with(tmp_path, "a", steps=[0, 6], token_sha256="a" * 64, coarse_interval=6)
    b = _series_with(tmp_path, "b", steps=[0, 6], token_sha256="b" * 64, coarse_interval=6)
    with pytest.raises(IncomparableProbes, match="token_sha256"):
        saturation_rows([a, b])


def test_saturation_rows_refuses_two_series_with_different_coarse_interval(tmp_path):
    a = _series_with(tmp_path, "a", steps=[0, 6], token_sha256="a" * 64, coarse_interval=6)
    b = _series_with(tmp_path, "b", steps=[0, 6], token_sha256="a" * 64, coarse_interval=3)
    with pytest.raises(IncomparableProbes, match="coarse_interval"):
        saturation_rows([a, b])


def test_saturation_rows_restricts_cross_run_tables_to_the_coarse_grid(tmp_path):
    # Dumps at {0, 2, 4, 6, 12} against coarse_interval=6: only {0, 6, 12} land on the grid.
    a = _series_with(
        tmp_path, "a", steps=[0, 2, 4, 6, 12], token_sha256="a" * 64, coarse_interval=6
    )
    b = _series_with(
        tmp_path, "b", steps=[0, 2, 4, 6, 12], token_sha256="a" * 64, coarse_interval=6
    )
    rows = saturation_rows([a, b])
    assert {row.step for row in rows} == {0, 6, 12}
    assert len(rows) == 3 * 1 * 2  # 3 steps x 1 layer x 2 series


def test_saturation_rows_single_series_is_not_restricted_to_the_coarse_grid(tmp_path):
    a = _series_with(
        tmp_path, "a", steps=[0, 2, 4, 6, 12], token_sha256="a" * 64, coarse_interval=6
    )
    rows = saturation_rows([a])
    assert {row.step for row in rows} == {0, 2, 4, 6, 12}


def test_saturation_rows_of_an_empty_series_sequence_is_empty():
    assert saturation_rows([]) == []


# --- ProbeSeries / ProbeDump construction is exercised via read_series/read_dump above; this
# checks the dataclasses themselves stay plain data (no surprise defaults) since callers build
# them directly in a couple of later specs' fixtures.


def test_probe_series_properties_delegate_to_the_first_dump(tmp_path):
    routing_map = numpy.stack([_one_hot_map([[0, 1]], 4)], axis=0)
    _write_dump(tmp_path / "probes", step=0, routing_map=routing_map, role="standing")
    dump = read_dump(tmp_path / "probes" / "iter_0000000.npz")
    series = ProbeSeries(run_dir=tmp_path, arm=None, dumps=(dump,))
    assert series.token_sha256 == dump.token_sha256
    assert series.role == dump.role
    assert series.coarse_interval == dump.coarse_interval
    assert isinstance(dump, ProbeDump)

import json

import numpy
import pytest

from moe_congestion_routing.game.alflb import top_k_map
from moe_congestion_routing.metrics.probe_series import (
    IncomparableProbes,
    ProbeDump,
    ProbeSeries,
    SaturationRow,
    read_dump,
    read_series,
    saturation_rows,
    selection_conformance,
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
    logits=None,
    expert_bias=None,
    score_function="sigmoid",
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
    arrays = {"routing_map": packed}
    if logits is not None:
        arrays["logits"] = numpy.asarray(logits, dtype=numpy.float32)
    if expert_bias is not None:
        arrays["expert_bias"] = numpy.asarray(expert_bias, dtype=numpy.float32)
    meta = {
        "moe_router_score_function": score_function,
        "has_expert_bias": expert_bias is not None,
        "iteration": step,
        "token_sha256": token_sha256,
        "role": role,
        "moe_probe_coarse_interval": coarse_interval,
        "layer_numbers": layer_numbers,
        "E": num_experts,
        "K": topk,
    }
    path = probes_dir / f"iter_{step:07d}.npz"
    numpy.savez(path, **arrays, metadata=numpy.array(json.dumps(meta)))
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


# ---------------------------------------------------------------------------------------------
# Affinities, the expert bias, and whether the offline replica selects what the model selected.
# ---------------------------------------------------------------------------------------------


def _conformance_dump(tmp_path, *, stored_selections, **kwargs):
    """A one-layer dump whose logits give token 1 an exact tie between experts 0 and 1.

    Token 0 and token 2 have a clear winner, so any disagreement they show is a real one, and
    token 1's two leading scores are equal in float32, so which of them wins is decided by a tie
    rule rather than by the affinities.
    """
    logits = numpy.array(
        [[[3.0, 1.0, 0.5, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.5, 1.0, 3.0]]],
        dtype=numpy.float32,
    )
    routing_map = _one_hot_map(stored_selections, 4)[numpy.newaxis, :, :]
    return _write_dump(
        tmp_path / "probes",
        step=0,
        routing_map=routing_map,
        topk=1,
        logits=logits,
        expert_bias=numpy.zeros((1, 4), dtype=numpy.float32),
        **kwargs,
    )


def test_affinities_are_the_float32_sigmoid_of_the_stored_logits(tmp_path):
    path = _conformance_dump(tmp_path, stored_selections=[[0], [0], [3]])
    dump = read_dump(path)

    logits = dump.logits()
    expected = 1.0 / (1.0 + numpy.exp(-logits.astype(numpy.float32)))
    # Equality rather than a tolerance, because the point of evaluating in float32 is that the
    # replica sees exactly the numbers the model saw.
    assert numpy.array_equal(dump.affinities(), expected.astype(numpy.float64))


def test_affinities_refuse_a_dump_that_was_not_scored_with_sigmoid(tmp_path):
    path = _conformance_dump(tmp_path, stored_selections=[[0], [0], [3]], score_function="softmax")
    with pytest.raises(IncomparableProbes, match="softmax"):
        read_dump(path).affinities()


def test_expert_bias_refuses_a_run_that_carried_none(tmp_path):
    routing_map = _one_hot_map([[0], [1]], 4)[numpy.newaxis, :, :]
    path = _write_dump(tmp_path / "probes", step=0, routing_map=routing_map, topk=1)
    with pytest.raises(IncomparableProbes, match="no expert_bias"):
        read_dump(path).expert_bias()


def test_conformance_is_clean_when_the_stored_map_is_the_replica_s_own_choice(tmp_path):
    # Expert 0 wins token 1 under the lowest-index tie rule, so this is the map a conforming
    # run would have stored.
    path = _conformance_dump(tmp_path, stored_selections=[[0], [0], [3]])

    (row,) = selection_conformance(read_dump(path))

    assert row.disagreeing_tokens == 0
    assert row.untied_disagreements == 0
    assert row.exact_ties == 1


def test_conformance_separates_a_tied_disagreement_from_a_real_one(tmp_path):
    # Token 1 is stored as expert 1, which the tie rule is entitled to differ on. Token 2 is
    # stored as expert 0 against a clear winner of expert 3, which nothing excuses.
    path = _conformance_dump(tmp_path, stored_selections=[[0], [1], [0]])

    (row,) = selection_conformance(read_dump(path))

    assert row.disagreeing_tokens == 2
    # The one that matters: the tied token is excused and the clear-winner token is not.
    assert row.untied_disagreements == 1
    assert row.exact_ties == 1


def test_top_k_survives_the_ulp_disagreement_between_numpy_and_torch(tmp_path):
    """The property `affinities` actually needs, measured rather than assumed.

    Its docstring does not claim bit-identity with the model's sigmoid, because there is none.
    What it claims is that the selection is insensitive to the difference, so assert exactly
    that, on logits spread widely enough that ULP-level disagreement is common.
    """
    torch = pytest.importorskip("torch")
    rng = numpy.random.default_rng(0)
    logits = rng.standard_normal((1, 4096, 16)).astype(numpy.float32) * 4.0
    routing_map = numpy.zeros((1, 4096, 16), dtype=bool)
    path = _write_dump(
        tmp_path / "probes",
        step=0,
        routing_map=routing_map,
        topk=4,
        logits=logits,
        expert_bias=numpy.zeros((1, 16), dtype=numpy.float32),
    )

    ours = read_dump(path).affinities()[0].astype(numpy.float32)
    theirs = torch.sigmoid(torch.from_numpy(logits[0])).numpy()

    # The premise of the test: the two really do disagree, so the assertion below is not vacuous.
    assert not numpy.array_equal(ours, theirs)
    assert numpy.abs(ours.view(numpy.int32) - theirs.view(numpy.int32)).max() <= 4
    ours_selected = numpy.sort(top_k_map(ours, 4), axis=1)
    theirs_selected = numpy.sort(top_k_map(theirs, 4), axis=1)
    assert numpy.array_equal(ours_selected, theirs_selected)

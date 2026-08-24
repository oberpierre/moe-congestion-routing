import json

import numpy
import pytest

from moe_congestion_routing.game.alflb import top_k_map
from moe_congestion_routing.metrics.probe_comparison import (
    DUAL_SPREAD_GATE,
    gated_dual_agreement,
    internalization_rows,
    part_indices,
    price_stability_rows_for_dump,
    verification_rows,
)
from moe_congestion_routing.metrics.probe_dump_format import ROUTING_MAP_BITORDER
from moe_congestion_routing.metrics.probe_series import IncomparableProbes, read_series


def _sigmoid(z: numpy.ndarray) -> numpy.ndarray:
    return 1.0 / (1.0 + numpy.exp(-z.astype(numpy.float64)))


def _write_dump(
    probes_dir,
    *,
    step,
    logits,
    expert_bias=None,
    topk,
    score_function="sigmoid",
    routing_map=None,
    layer_numbers=None,
    token_sha256="cafe" * 16,
    role="standing",
    coarse_interval=6,
    probe_seqs=1,
):
    """A synthetic probe dump whose routing map conforms to top-K of its own scores by default.

    Passing ``routing_map`` overrides that default so a test can construct a dump whose stored
    selection disagrees with what the offline replica would compute, which is what the
    conformance refusal test needs.
    """
    probes_dir.mkdir(parents=True, exist_ok=True)
    logits = numpy.asarray(logits, dtype=numpy.float32)
    num_layers, num_tokens, num_experts = logits.shape
    if layer_numbers is None:
        layer_numbers = list(range(2, 2 + num_layers))
    if routing_map is None:
        scores = _sigmoid(logits) if score_function == "sigmoid" else logits.astype(numpy.float64)
        bias_for_map = (
            numpy.zeros((num_layers, num_experts))
            if expert_bias is None
            else numpy.asarray(expert_bias)
        )
        routing_map = numpy.zeros((num_layers, num_tokens, num_experts), dtype=bool)
        for layer in range(num_layers):
            layer_scores = scores[layer] + bias_for_map[layer]
            numpy.put_along_axis(routing_map[layer], top_k_map(layer_scores, topk), True, axis=1)
    packed = numpy.packbits(routing_map, axis=-1, bitorder=ROUTING_MAP_BITORDER)
    meta = {
        "moe_router_score_function": score_function,
        "has_expert_bias": expert_bias is not None,
        "iteration": step,
        "token_sha256": token_sha256,
        "role": role,
        "moe_probe_coarse_interval": coarse_interval,
        "moe_probe_seqs": probe_seqs,
        "layer_numbers": layer_numbers,
        "E": num_experts,
        "K": topk,
    }
    arrays = {"routing_map": packed, "logits": logits}
    if expert_bias is not None:
        arrays["expert_bias"] = numpy.asarray(expert_bias, dtype=numpy.float32)
    path = probes_dir / f"iter_{step:07d}.npz"
    numpy.savez(path, **arrays, metadata=numpy.array(json.dumps(meta)))
    return path


def _affinity_logits(num_layers, num_tokens, num_experts, seed, separation=3.0):
    rng = numpy.random.default_rng(seed)
    return (separation * rng.standard_normal((num_layers, num_tokens, num_experts))).astype(
        numpy.float32
    )


# --- the resolvability gate, on constructed duals ----------------------------------------------


def test_gate_masks_correlation_below_the_ratio_but_not_at_or_above_it():
    bias = numpy.array([1.0, 3.0])
    # Two-point vectors correlate at exactly +-1, so the value above the gate is a fixed
    # point this test can assert on rather than an approximation of an arbitrary number.
    below = numpy.array([0.0, 4.0])  # spread / eta = 4.0, below DUAL_SPREAD_GATE = 8.0
    above = numpy.array([0.0, 10.0])  # spread / eta = 10.0, above DUAL_SPREAD_GATE = 8.0

    spread_below, corr_below, _ = gated_dual_agreement(bias, below, bias_update_rate=1.0)
    spread_above, corr_above, _ = gated_dual_agreement(bias, above, bias_update_rate=1.0)

    assert spread_below == pytest.approx(4.0)
    assert numpy.isnan(corr_below)
    assert spread_above == pytest.approx(10.0)
    assert corr_above == pytest.approx(1.0)


def test_the_gate_admits_a_ratio_of_exactly_eight():
    # The boundary itself, which 4.0 and 10.0 straddle without pinning. The rule is "refuse
    # below 8", so exactly 8 is admitted, and without this a `<` weakened to `<=` passes.
    bias = numpy.array([1.0, 3.0])
    at_gate = numpy.array([0.0, 8.0])

    spread, correlation, _ = gated_dual_agreement(bias, at_gate, bias_update_rate=1.0)

    assert spread == pytest.approx(DUAL_SPREAD_GATE)
    assert not numpy.isnan(correlation)
    assert correlation == pytest.approx(1.0)


def test_gate_constant_is_eight():
    # Pinned so the boundary test above stays meaningful if the constant ever moves: the two
    # constructed ratios (4.0, 10.0) must keep straddling it.
    assert DUAL_SPREAD_GATE == 8.0


# --- the three refusals --------------------------------------------------------------------


def test_refuses_a_non_sigmoid_dump(tmp_path):
    probes_dir = tmp_path / "probes"
    logits = _affinity_logits(1, 6, 3, seed=0)
    _write_dump(
        probes_dir,
        step=10,
        logits=logits,
        expert_bias=numpy.zeros((1, 3)),
        topk=1,
        score_function="softmax",
    )
    series = read_series(tmp_path)
    with pytest.raises(IncomparableProbes, match="score function is 'softmax'"):
        verification_rows(series, bias_update_rate=1e-3)


def test_refuses_a_dump_with_no_expert_bias(tmp_path):
    probes_dir = tmp_path / "probes"
    logits = _affinity_logits(1, 6, 3, seed=0)
    _write_dump(probes_dir, step=10, logits=logits, expert_bias=None, topk=1)
    series = read_series(tmp_path)
    with pytest.raises(IncomparableProbes, match="has no expert_bias"):
        internalization_rows(series, bias_update_rate=1e-3)


def test_refuses_a_dump_whose_selection_conformance_is_not_clean(tmp_path):
    probes_dir = tmp_path / "probes"
    # Well-separated logits so the deliberately wrong entry below is an untied disagreement
    # rather than a coincidental tie the conformance check is required to tolerate.
    logits = numpy.array([[[5.0, -5.0, -5.0], [-5.0, 5.0, -5.0], [-5.0, -5.0, 5.0]]])
    bias = numpy.zeros((1, 3))
    wrong_map = numpy.zeros((1, 3, 3), dtype=bool)
    wrong_map[0, 0, 0] = True  # token 0's true top-1 is expert 0
    wrong_map[0, 1, 1] = True  # token 1's true top-1 is expert 1
    wrong_map[0, 2, 0] = True  # token 2's true top-1 is expert 2, but the stored map says 0
    _write_dump(
        probes_dir,
        step=10,
        logits=logits,
        expert_bias=bias,
        topk=1,
        routing_map=wrong_map,
        layer_numbers=[7],
    )
    series = read_series(tmp_path)
    with pytest.raises(IncomparableProbes, match=r"layer 7 has 1 untied selection disagreement"):
        verification_rows(series, bias_update_rate=1e-3)
    with pytest.raises(IncomparableProbes, match=r"layer 7 has 1 untied selection disagreement"):
        internalization_rows(series, bias_update_rate=1e-3)


# --- row selection and shape --------------------------------------------------------------------


def test_defaults_to_the_series_last_step_and_every_layer(tmp_path):
    probes_dir = tmp_path / "probes"
    _write_dump(
        probes_dir,
        step=10,
        logits=_affinity_logits(2, 6, 3, seed=1),
        expert_bias=numpy.zeros((2, 3)),
        topk=1,
        layer_numbers=[4, 5],
    )
    _write_dump(
        probes_dir,
        step=20,
        logits=_affinity_logits(2, 6, 3, seed=2),
        expert_bias=numpy.zeros((2, 3)),
        topk=1,
        layer_numbers=[4, 5],
    )
    series = read_series(tmp_path)

    verification = verification_rows(series, bias_update_rate=1e-2, annealed_steps=25)
    internalization = internalization_rows(series, bias_update_rate=1e-2)

    assert [(row.step, row.layer) for row in verification] == [(20, 4), (20, 5)]
    assert [(row.step, row.layer) for row in internalization] == [(20, 4), (20, 5)]
    assert all(row.bias_update_rate == pytest.approx(1e-2) for row in verification)
    assert all(row.comparison.eta == pytest.approx(1e-2) for row in verification)


def test_explicit_step_and_layers_narrow_the_rows(tmp_path):
    probes_dir = tmp_path / "probes"
    _write_dump(
        probes_dir,
        step=10,
        logits=_affinity_logits(2, 6, 3, seed=1),
        expert_bias=numpy.zeros((2, 3)),
        topk=1,
        layer_numbers=[4, 5],
    )
    _write_dump(
        probes_dir,
        step=20,
        logits=_affinity_logits(2, 6, 3, seed=2),
        expert_bias=numpy.zeros((2, 3)),
        topk=1,
        layer_numbers=[4, 5],
    )
    series = read_series(tmp_path)

    verification = verification_rows(
        series, bias_update_rate=1e-2, annealed_steps=25, steps=[10], layers=[5]
    )

    assert [(row.step, row.layer) for row in verification] == [(10, 5)]


def test_unknown_step_raises(tmp_path):
    probes_dir = tmp_path / "probes"
    _write_dump(
        probes_dir,
        step=10,
        logits=_affinity_logits(1, 6, 3, seed=1),
        expert_bias=numpy.zeros((1, 3)),
        topk=1,
    )
    series = read_series(tmp_path)
    with pytest.raises(ValueError, match="steps \\[99\\]"):
        verification_rows(series, bias_update_rate=1e-2, steps=[99])


def test_unknown_layer_raises(tmp_path):
    probes_dir = tmp_path / "probes"
    _write_dump(
        probes_dir,
        step=10,
        logits=_affinity_logits(1, 6, 3, seed=1),
        expert_bias=numpy.zeros((1, 3)),
        topk=1,
        layer_numbers=[4],
    )
    series = read_series(tmp_path)
    with pytest.raises(ValueError, match="layers \\[99\\]"):
        verification_rows(series, bias_update_rate=1e-2, layers=[99])


def test_internalization_rows_actually_apply_the_gate(tmp_path):
    """The wiring, not the branch. `gated_dual_agreement` is tested directly above, but nothing
    checked that `internalization_rows` calls it, so dropping the call went uncaught.

    A huge bias update rate drives the ratio under the gate whatever the duals are, which makes
    this a property of the wiring rather than of a hand-tuned affinity matrix.
    """
    # Five of eight tokens want expert 0 while its capacity is two, so that constraint binds and
    # its shadow price separates from the rest. A dump whose duals are all equal has zero spread
    # and would sit under the gate whatever the rate is, proving nothing about the wiring.
    crowded = [[4.0, 0.0, 0.0, 0.0]] * 5
    spread_out = [[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    logits = numpy.array([crowded + spread_out], dtype=numpy.float32)
    # A varying bias, because a constant one has no variance and correlates as NaN whatever the
    # gate does, which would make the masked case below indistinguishable from the unmasked one.
    bias = numpy.array([[-0.03, 0.01, 0.0, 0.02]], dtype=numpy.float32)
    _write_dump(tmp_path / "probes", step=0, logits=logits, expert_bias=bias, topk=1)
    series = read_series(tmp_path)

    resolvable = internalization_rows(series, bias_update_rate=1e-9)
    swamped = internalization_rows(series, bias_update_rate=1e9)

    # Same duals both times, so only the rate moved the row across the gate.
    assert resolvable[0].dual_spread_over_eta > DUAL_SPREAD_GATE
    assert not numpy.isnan(resolvable[0].dual_correlation)
    assert swamped[0].dual_spread_over_eta < DUAL_SPREAD_GATE
    assert numpy.isnan(swamped[0].dual_correlation)
    # The ratio is reported either way, because a masked correlation without it says nothing.
    assert swamped[0].dual_spread_over_eta > 0.0


# --- how a batch is cut into parts ---------------------------------------------------------


@pytest.mark.parametrize("split,seed", [("sequence", None), ("stride", None), ("random", 0)])
def test_part_indices_are_disjoint_and_cover_the_batch(split, seed):
    parts = part_indices(num_tokens=16, num_sequences=4, num_parts=4, split=split, seed=seed)

    assert [len(p) for p in parts] == [4, 4, 4, 4]
    pooled = numpy.concatenate(parts)
    numpy.testing.assert_array_equal(numpy.sort(pooled), numpy.arange(16))


def test_sequence_split_cuts_on_sequence_boundaries_and_stride_does_not():
    # 4 sequences of 4 tokens, rows sequence-major, so sequence s owns rows 4s..4s+3.
    sequence = part_indices(num_tokens=16, num_sequences=4, num_parts=2, split="sequence")
    stride = part_indices(num_tokens=16, num_sequences=4, num_parts=2, split="stride")

    numpy.testing.assert_array_equal(sequence[0], numpy.arange(0, 8))
    numpy.testing.assert_array_equal(sequence[1], numpy.arange(8, 16))
    # The stride part draws from every sequence, which is the whole reason both modes exist.
    numpy.testing.assert_array_equal(stride[0], numpy.arange(0, 16, 2))
    assert len({int(i) // 4 for i in stride[0]}) == 4


def test_sequence_split_refuses_a_part_that_would_straddle_a_sequence():
    # 3 sequences cannot be cut in two without splitting one of them down the middle.
    with pytest.raises(ValueError, match="whole number of sequences"):
        part_indices(num_tokens=12, num_sequences=3, num_parts=2, split="sequence")


def test_part_indices_refuses_fewer_than_two_parts_and_an_unknown_split():
    with pytest.raises(ValueError, match="at least 2"):
        part_indices(num_tokens=16, num_sequences=4, num_parts=1, split="stride")
    with pytest.raises(ValueError, match="split must be one of"):
        part_indices(num_tokens=16, num_sequences=4, num_parts=2, split="shuffled")


# --- the price-stability row ----------------------------------------------------------------


def test_price_stability_reports_pairwise_prices_without_the_gate_masking_them(tmp_path):
    """The pairwise columns must survive a bias update rate that masks the bias columns.

    Neither side of a pairwise comparison is a stored bias, so no eta orbit is involved and the
    resolvability gate must not reach them. Without this, dropping the gate's scope went uncaught.
    """
    # Five of eight tokens want expert 0 against a capacity of two, so that constraint binds and
    # its shadow price separates, which is what makes a correlation between parts meaningful.
    crowded = [[4.0, 0.0, 0.0, 0.0]] * 5
    spread_out = [[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    logits = numpy.array([crowded + spread_out], dtype=numpy.float32)
    bias = numpy.array([[-0.03, 0.01, 0.0, 0.02]], dtype=numpy.float32)
    _write_dump(tmp_path / "probes", step=0, logits=logits, expert_bias=bias, topk=1)
    dump = read_series(tmp_path).dumps[-1]

    swamped = price_stability_rows_for_dump(
        dump, bias_update_rate=1e9, num_parts=2, split="stride"
    )[0]

    assert numpy.isnan(swamped.bias_correlation_full)
    assert numpy.isnan(swamped.mean_bias_correlation_parts)
    assert not numpy.isnan(swamped.mean_pairwise_correlation)
    assert swamped.part_tokens == 4
    assert swamped.num_parts == 2 and swamped.split == "stride"


def test_price_stability_refuses_a_dump_it_cannot_reproduce(tmp_path):
    logits = numpy.array([[[5.0, -5.0, -5.0], [-5.0, 5.0, -5.0], [-5.0, -5.0, 5.0]]])
    wrong_map = numpy.zeros((1, 3, 3), dtype=bool)
    wrong_map[0, 0, 0] = True
    wrong_map[0, 1, 1] = True
    wrong_map[0, 2, 0] = True  # token 2's true top-1 is expert 2
    _write_dump(
        tmp_path / "probes",
        step=10,
        logits=logits,
        expert_bias=numpy.zeros((1, 3)),
        topk=1,
        routing_map=wrong_map,
        layer_numbers=[7],
    )
    dump = read_series(tmp_path).dumps[-1]
    with pytest.raises(IncomparableProbes, match=r"layer 7 has 1 untied selection disagreement"):
        price_stability_rows_for_dump(dump, bias_update_rate=1e-3, split="stride")


def test_random_split_is_reproducible_from_its_seed_and_moves_with_it():
    same_a = part_indices(num_tokens=64, num_sequences=4, num_parts=2, split="random", seed=7)
    same_b = part_indices(num_tokens=64, num_sequences=4, num_parts=2, split="random", seed=7)
    other = part_indices(num_tokens=64, num_sequences=4, num_parts=2, split="random", seed=8)

    numpy.testing.assert_array_equal(same_a[0], same_b[0])
    assert not numpy.array_equal(same_a[0], other[0])


def test_random_split_draws_from_every_sequence_unlike_the_sequence_split():
    # 4 sequences of 16 tokens. The point of the control is that composition is held fixed, so
    # each part must carry all four sequences rather than a subset of them.
    parts = part_indices(num_tokens=64, num_sequences=4, num_parts=2, split="random", seed=0)

    for part in parts:
        assert {int(i) // 16 for i in part} == {0, 1, 2, 3}


def test_a_seed_is_required_for_random_and_refused_for_the_others():
    """Both directions, because either silence would lose the reproducibility this exists for:
    a missing seed makes the cut unrepeatable, and an accepted-but-ignored seed makes a row
    claim a seed decided its cut when nothing did."""
    with pytest.raises(ValueError, match="requires a seed"):
        part_indices(num_tokens=16, num_sequences=4, num_parts=2, split="random")
    for deterministic in ("sequence", "stride"):
        with pytest.raises(ValueError, match="takes no seed"):
            part_indices(num_tokens=16, num_sequences=4, num_parts=2, split=deterministic, seed=0)


def test_four_parts_give_six_pairs_and_a_spread_where_two_give_one_pair_and_zero(tmp_path):
    crowded = [[4.0, 0.0, 0.0, 0.0]] * 5
    spread_out = [[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    logits = numpy.array([crowded + spread_out], dtype=numpy.float32)
    bias = numpy.array([[-0.03, 0.01, 0.0, 0.02]], dtype=numpy.float32)
    _write_dump(tmp_path / "probes", step=0, logits=logits, expert_bias=bias, topk=1)
    dump = read_series(tmp_path).dumps[-1]

    two = price_stability_rows_for_dump(
        dump, bias_update_rate=1e-3, num_parts=2, split="random", split_seed=0
    )[0]
    four = price_stability_rows_for_dump(
        dump, bias_update_rate=1e-3, num_parts=4, split="random", split_seed=0
    )[0]

    assert (two.num_pairs, two.part_tokens) == (1, 4)
    assert (four.num_pairs, four.part_tokens) == (6, 2)
    # One pair has no spread to report, which is exactly why four parts are worth the same money.
    assert two.stdev_pairwise_correlation == 0.0
    assert two.split_seed == 0 and four.split_seed == 0

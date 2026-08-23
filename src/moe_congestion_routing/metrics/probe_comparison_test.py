import json

import numpy
import pytest

from moe_congestion_routing.game.alflb import top_k_map
from moe_congestion_routing.metrics.probe_comparison import (
    DUAL_SPREAD_GATE,
    gated_dual_agreement,
    internalization_rows,
    verification_rows,
)
from moe_congestion_routing.metrics.probe_series import IncomparableProbes, read_series
from moe_congestion_routing.metrics.router_probe import ROUTING_MAP_BITORDER


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

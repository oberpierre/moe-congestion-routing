import json
import math
import subprocess
import sys

import numpy
import pytest

from moe_congestion_routing.game.incremental import solve_incremental
from moe_congestion_routing.losses.cost_families import marginal_cost
from moe_congestion_routing.metrics import probe_comparison
from moe_congestion_routing.metrics.phi_gap import (
    PhiGapRow,
    _balanced_assignment,
    arc_schedule_length,
    phi_gap_rows,
)
from moe_congestion_routing.metrics.probe_dump_format import ROUTING_MAP_BITORDER
from moe_congestion_routing.metrics.probe_series import read_dump


def test_no_torch_import():
    # This module composes game/incremental.py, losses/cost_families.py and metrics/probe_series.py,
    # all torch-free, and the offline analysis path depends on that property staying true here too.
    script = (
        "import sys; "
        "import moe_congestion_routing.metrics.phi_gap; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def _write_dump(probes_dir, *, step, logits, routing_map, topk, score_function="sigmoid"):
    """A one-layer synthetic dump with an explicit realized ``routing_map``.

    Unlike ``probe_comparison_test.py``'s helper, this never derives ``routing_map`` from the
    logits, because every test here needs to control the realized assignment independently of
    what top-K of the affinities would have picked.
    """
    probes_dir.mkdir(parents=True, exist_ok=True)
    logits = numpy.asarray(logits, dtype=numpy.float32)
    num_layers, _num_tokens, num_experts = logits.shape
    packed = numpy.packbits(routing_map, axis=-1, bitorder=ROUTING_MAP_BITORDER)
    meta = {
        "moe_router_score_function": score_function,
        "has_expert_bias": False,
        "iteration": step,
        "token_sha256": "cafe" * 16,
        "role": "standing",
        "moe_probe_coarse_interval": 6,
        "layer_numbers": list(range(2, 2 + num_layers)),
        "E": num_experts,
        "K": topk,
        "moe_probe_batch": "assets/probe/default_asset.npz",
    }
    path = probes_dir / f"iter_{step:07d}.npz"
    numpy.savez(path, routing_map=packed, logits=logits, metadata=numpy.array(json.dumps(meta)))
    return path


def test_balanced_assignment_is_exactly_balanced_at_the_real_shape():
    # The affinity-blind baseline is deterministic and seedless, and it is only exactly
    # balanced when E divides n*K, which the real probe shape does: 16384*8/64 = 2048 exactly.
    n, e, k = 16384, 64, 8
    experts = _balanced_assignment(n, k, e)
    loads = numpy.bincount(experts.ravel(), minlength=e)
    assert numpy.all(loads == n * k / e)


def test_gap_normalized_is_zero_at_the_oracle_and_one_at_the_baseline(tmp_path, monkeypatch):
    # A unit this small is not the reported instrument, only a probe_units cut this test controls
    # so the oracle solve stays instant, monkeypatching the module attribute probe_units itself
    # reads at call time rather than the value phi_gap.py captured at import time.
    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 4)

    n, e, k = 4, 2, 1
    logits = numpy.log(numpy.array([[0.95, 0.05]] * n) / numpy.array([[0.05, 0.95]] * n))
    logits = logits[numpy.newaxis, :, :].astype(numpy.float32)

    balanced_load = n * k / e
    num_arcs = 2 * math.ceil(n * k / e)

    # Read once to get router_scores() in the exact float32-then-widened form phi_gap_rows will
    # also see, so the oracle computed here and the one phi_gap_rows recomputes agree bit-for-bit.
    # The routing_map here is never scored, only the logits are, so its content is arbitrary.
    probe_path = _write_dump(
        tmp_path / "probes_a",
        step=0,
        logits=logits,
        routing_map=numpy.zeros((1, n, e), dtype=bool),
        topk=k,
    )
    a = read_dump(probe_path).router_scores()[0]
    arc_prices = marginal_cost(
        numpy.arange(1, num_arcs + 1), balanced_load, lam=1.0, cost_family="linear"
    )
    oracle = solve_incremental(a, k, arc_prices)

    oracle_dump_path = _write_dump(
        tmp_path / "probes_oracle",
        step=0,
        logits=logits,
        routing_map=oracle.x[numpy.newaxis],
        topk=k,
    )
    oracle_rows = phi_gap_rows(
        read_dump(oracle_dump_path), layer=2, unit="u0", lam=1.0, cost_families=("linear",)
    )
    assert oracle_rows[0].gap_normalized == pytest.approx(0.0, abs=1e-6)

    baseline_experts = _balanced_assignment(n, k, e)
    baseline_map = numpy.zeros((1, n, e), dtype=bool)
    for token in range(n):
        for expert in baseline_experts[token]:
            baseline_map[0, token, expert] = True
    baseline_dump_path = _write_dump(
        tmp_path / "probes_baseline", step=0, logits=logits, routing_map=baseline_map, topk=k
    )
    baseline_rows = phi_gap_rows(
        read_dump(baseline_dump_path), layer=2, unit="u0", lam=1.0, cost_families=("linear",)
    )
    assert baseline_rows[0].gap_normalized == pytest.approx(1.0, abs=1e-6)


def test_affinity_shortfall_can_be_negative_while_the_sum_stays_nonnegative(tmp_path, monkeypatch):
    # A router that always argmaxes its own affinity, ignoring congestion, beats the balanced
    # oracle on raw affinity, so affinity_shortfall is negative here by construction. The guard
    # this proves absent would have rejected that as invalid, when it is the control's whole
    # story: it buys affinity with congestion, and only the sum is bounded below.
    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 4)

    n, e, k = 4, 2, 1
    logits = numpy.log(numpy.array([[0.95, 0.05]] * n) / numpy.array([[0.05, 0.95]] * n))
    logits = logits[numpy.newaxis, :, :].astype(numpy.float32)

    concentrated = numpy.zeros((1, n, e), dtype=bool)
    concentrated[0, :, 0] = True
    path = _write_dump(tmp_path / "probes", step=0, logits=logits, routing_map=concentrated, topk=k)

    rows = phi_gap_rows(read_dump(path), layer=2, unit="u0", lam=1.0, cost_families=("linear",))
    row = rows[0]
    assert row.affinity_shortfall < 0
    assert row.gap_per_token >= 0


def test_a_refused_screen_still_emits_a_full_row(tmp_path, monkeypatch):
    # A concentrated unit's gap is a real measurement of a router far from equilibrium, not an
    # artifact to withhold, so admissible=False must still carry finite statistics rather than
    # NaN.
    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 8)

    n, e, k = 8, 4, 1
    rng = numpy.random.default_rng(0)
    scores = rng.uniform(0.05, 0.95, size=(n, e))
    logits = numpy.log(scores / (1.0 - scores))[numpy.newaxis, :, :].astype(numpy.float32)

    concentrated = numpy.zeros((1, n, e), dtype=bool)
    concentrated[0, :, 0] = True
    path = _write_dump(tmp_path / "probes", step=0, logits=logits, routing_map=concentrated, topk=k)

    rows = phi_gap_rows(
        read_dump(path), layer=2, unit="u0", lam=1.0, cost_families=("linear", "quadratic")
    )
    assert len(rows) == 2
    for row in rows:
        assert row.admissible is False
        assert row.dead_experts == 3
        for value in (
            row.max_load_over_balanced,
            row.gap_per_token,
            row.affinity_shortfall,
            row.congestion_excess,
            row.normalizer,
            row.max_fractional_deviation,
            row.gap_normalized,
        ):
            assert math.isfinite(value)


def test_phi_gap_rows_shape_and_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 4)

    n, e, k = 4, 2, 1
    rng = numpy.random.default_rng(1)
    scores = rng.uniform(0.05, 0.95, size=(n, e))
    logits = numpy.log(scores / (1.0 - scores))[numpy.newaxis, :, :].astype(numpy.float32)

    from moe_congestion_routing.game.alflb import top_k_map

    routing_map = numpy.zeros((1, n, e), dtype=bool)
    numpy.put_along_axis(routing_map[0], top_k_map(scores, k), True, axis=1)
    path = _write_dump(tmp_path / "probes", step=7, logits=logits, routing_map=routing_map, topk=k)

    rows = phi_gap_rows(
        read_dump(path), layer=2, unit="u0", lam=2.0, cost_families=("linear", "quadratic")
    )
    assert [row.reference_cost for row in rows] == ["linear", "quadratic"]
    for row in rows:
        assert isinstance(row, PhiGapRow)
        assert row.unit == "u0"
        assert row.layer == 2
        assert row.step == 7
        assert row.lam == 2.0
        assert row.affinity_space == "score"
        assert row.score_function == "sigmoid"
        assert row.token_sha256 == "cafe" * 16
        assert row.dump_path == str(path)
        assert row.arc_growths == 0


def test_the_unit_is_scored_not_the_whole_dump(tmp_path, monkeypatch):
    # The silent failure this guards: taking n from the dump rather than the unit doubles the
    # balanced load, so every price, the oracle and the normalizer describe a different game while
    # every field still looks entirely reasonable. Scoring u0 of a two-unit dump must equal
    # scoring a dump holding only those tokens.
    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 4)

    n, e, k = 8, 2, 1
    rng = numpy.random.default_rng(3)
    scores = rng.uniform(0.05, 0.95, size=(n, e))
    logits = numpy.log(scores / (1.0 - scores))[numpy.newaxis, :, :].astype(numpy.float32)
    routing_map = numpy.zeros((1, n, e), dtype=bool)
    routing_map[0, :, 0] = True
    routing_map[0, ::2, 0] = False
    routing_map[0, ::2, 1] = True

    both = _write_dump(
        tmp_path / "two" / "probes", step=0, logits=logits, routing_map=routing_map, topk=k
    )
    first = _write_dump(
        tmp_path / "one" / "probes",
        step=0,
        logits=logits[:, :4],
        routing_map=routing_map[:, :4],
        topk=k,
    )

    from_two = phi_gap_rows(read_dump(both), layer=2, unit="u0", cost_families=("linear",))[0]
    from_one = phi_gap_rows(read_dump(first), layer=2, unit="u0", cost_families=("linear",))[0]
    assert from_two.gap_per_token == pytest.approx(from_one.gap_per_token)
    assert from_two.normalizer == pytest.approx(from_one.normalizer)
    assert from_two.max_load_over_balanced == pytest.approx(from_one.max_load_over_balanced)
    # u1 is a different game, so a slice that silently scored the whole dump would make these agree.
    second = phi_gap_rows(read_dump(both), layer=2, unit="u1", cost_families=("linear",))[0]
    assert second.gap_per_token != pytest.approx(from_two.gap_per_token)


def test_gap_normalized_is_not_clipped_above_one(tmp_path, monkeypatch):
    # A router worse than ignoring affinity entirely reads above 1, and that is the finding rather
    # than an error. Clipping would erase it, and the real control arm at step 0 sits near 5.9.
    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 8)

    n, e, k = 8, 4, 1
    rng = numpy.random.default_rng(5)
    scores = rng.uniform(0.05, 0.95, size=(n, e))
    logits = numpy.log(scores / (1.0 - scores))[numpy.newaxis, :, :].astype(numpy.float32)
    concentrated = numpy.zeros((1, n, e), dtype=bool)
    concentrated[0, :, 0] = True
    path = _write_dump(tmp_path / "probes", step=0, logits=logits, routing_map=concentrated, topk=k)

    row = phi_gap_rows(read_dump(path), layer=2, unit="u0", cost_families=("linear",))[0]
    assert row.gap_normalized > 1.0


def test_lam_reaches_the_prices_rather_than_only_the_column(tmp_path, monkeypatch):
    # lam scales every arc price, so a row echoing lam while pricing the oracle at the default is
    # wrong in exactly the way the lambda sweep cannot afford, and the sweep is the primary figure.
    # Comparing gaps is too weak to catch it, because discrete_potential takes lam too and the gap
    # moves either way. On an instance whose optimum is the same at both lam the whole congestion
    # side scales exactly, so the ratio is 2 when lam reaches the prices and affine with a nonzero
    # intercept when it does not.
    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 8)

    n, e, k = 8, 4, 1
    rng = numpy.random.default_rng(7)
    scores = rng.uniform(0.05, 0.95, size=(n, e))
    logits = numpy.log(scores / (1.0 - scores))[numpy.newaxis, :, :].astype(numpy.float32)
    concentrated = numpy.zeros((1, n, e), dtype=bool)
    concentrated[0, :, 0] = True
    path = _write_dump(tmp_path / "probes", step=0, logits=logits, routing_map=concentrated, topk=k)

    dump = read_dump(path)
    one = phi_gap_rows(dump, layer=2, unit="u0", lam=1.0, cost_families=("linear",))[0]
    two = phi_gap_rows(dump, layer=2, unit="u0", lam=2.0, cost_families=("linear",))[0]
    # Equal shortfalls are the precondition: the optimum is the same assignment at both lam, so
    # the congestion side is the only thing lam can move.
    assert two.affinity_shortfall == pytest.approx(one.affinity_shortfall)
    assert two.congestion_excess == pytest.approx(2.0 * one.congestion_excess, rel=1e-9)


def test_balanced_assignment_uses_the_declared_formula(tmp_path):
    # Any balanced permutation passes a load check, whereas the normalizer of every row in the
    # thesis is defined by this exact map, and its period E/gcd(K,E) is what the evidence for
    # using a fixed draw rather than a sampled one was measured on.
    expected = numpy.array([[0, 1], [2, 3], [0, 1], [2, 3]])
    assert numpy.array_equal(_balanced_assignment(4, 2, 4), expected)
    assert numpy.array_equal(_balanced_assignment(3, 1, 2), numpy.array([[0], [1], [0]]))


def test_phi_gap_rows_scores_a_softmax_dump(tmp_path, monkeypatch):
    # Eleven of the twelve arms are softmax, so the composition they will all use needs its own
    # case: a sigmoid-only affinity accessor here would raise, and one that ignored the dump's
    # score function would silently score a different affinity.
    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 8)

    n, e, k = 8, 4, 2
    rng = numpy.random.default_rng(11)
    logits = rng.normal(size=(1, n, e)).astype(numpy.float32)
    routing_map = numpy.zeros((1, n, e), dtype=bool)
    routing_map[0, :, :k] = True
    path = _write_dump(
        tmp_path / "probes",
        step=0,
        logits=logits,
        routing_map=routing_map,
        topk=k,
        score_function="softmax",
    )

    row = phi_gap_rows(read_dump(path), layer=2, unit="u0", cost_families=("linear",))[0]
    assert row.score_function == "softmax"
    assert row.affinity_space == "score"
    assert math.isfinite(row.gap_per_token) and row.gap_per_token >= 0.0


# n=16384, k=8, e=64, max_span=0.202505 is the real shape measured on the control arm's step-0
# layer-2 unit u0. For lam >= 0.25 the feasibility floor governs in both families and the schedule
# is the constant 4096 the committed rows were solved at, so this table is what stops the sweep
# silently redefining that budget.
_SCHEDULE_TABLE = [
    (8.0, 4096, 4096),
    (4.0, 4096, 4096),
    (2.0, 4096, 4096),
    (1.0, 4096, 4096),
    (0.5, 4096, 4096),
    (0.25, 4096, 4096),
    (0.125, 6636, 5214),
]


@pytest.mark.parametrize("lam,expected_linear,expected_quadratic", _SCHEDULE_TABLE)
def test_arc_schedule_length_reproduces_the_measured_table(
    lam, expected_linear, expected_quadratic
):
    n, k, e, max_span = 16384, 8, 64, 0.202505
    assert arc_schedule_length(n, k, e, max_span, lam=lam, cost_family="linear") == expected_linear
    assert (
        arc_schedule_length(n, k, e, max_span, lam=lam, cost_family="quadratic")
        == expected_quadratic
    )


def test_lam_zero_equals_the_solver_on_a_small_instance(tmp_path, monkeypatch):
    # phi_gap_rows(lam=0) must not call solve_incremental at all, so this proves the closed form
    # agrees with what an explicit all-zero-price solve of the same instance would have returned.
    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 8)

    n, e, k = 8, 4, 1
    rng = numpy.random.default_rng(13)
    scores = rng.uniform(0.05, 0.95, size=(n, e))
    logits = numpy.log(scores / (1.0 - scores))[numpy.newaxis, :, :].astype(numpy.float32)
    routing_map = numpy.zeros((1, n, e), dtype=bool)
    routing_map[0, :, 0] = True
    path = _write_dump(tmp_path / "probes", step=0, logits=logits, routing_map=routing_map, topk=k)

    dump = read_dump(path)
    row = phi_gap_rows(dump, layer=2, unit="u0", lam=0.0, cost_families=("linear",))[0]

    a = dump.router_scores()[0]
    # n arcs at price 0 seats any assignment, including the one every expert would need if every
    # token concentrated on it, so this schedule never saturates.
    oracle = solve_incremental(a, k, numpy.zeros(n))
    affinity_realized = float(a[routing_map[0]].sum())
    phi_star = oracle.affinity - oracle.congestion
    gap_per_token = (phi_star - affinity_realized) / n

    baseline_experts = _balanced_assignment(n, k, e)
    tokens = numpy.arange(n)[:, None]
    affinity_baseline = float(a[tokens, baseline_experts].sum())
    normalizer = phi_star - affinity_baseline

    assert row.gap_per_token == pytest.approx(gap_per_token, abs=1e-9)
    assert row.gap_normalized == pytest.approx(gap_per_token * n / normalizer, abs=1e-9)


def test_lam_zero_is_zero_at_topk_and_positive_off_it(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 8)

    from moe_congestion_routing.game.alflb import top_k_map

    n, e, k = 8, 4, 2
    rng = numpy.random.default_rng(17)
    logits = rng.normal(size=(1, n, e)).astype(numpy.float32)

    # Read once to get router_scores() in the exact form phi_gap_rows will also see, the same
    # trick test_gap_normalized_is_zero_at_the_oracle_and_one_at_the_baseline uses above.
    probe_path = _write_dump(
        tmp_path / "probes_a",
        step=0,
        logits=logits,
        routing_map=numpy.zeros((1, n, e), dtype=bool),
        topk=k,
    )
    a = read_dump(probe_path).router_scores()[0]

    topk_idx = top_k_map(a, k)
    topk_map = numpy.zeros((1, n, e), dtype=bool)
    numpy.put_along_axis(topk_map[0], topk_idx, True, axis=1)
    topk_path = _write_dump(
        tmp_path / "probes_topk", step=0, logits=logits, routing_map=topk_map, topk=k
    )
    topk_row = phi_gap_rows(
        read_dump(topk_path), layer=2, unit="u0", lam=0.0, cost_families=("linear",)
    )[0]
    assert topk_row.gap_normalized == pytest.approx(0.0, abs=1e-12)

    # Move token 0 off its top-K set onto an expert it did not select, so the assignment is no
    # longer top-K anywhere else being touched.
    perturbed_map = topk_map.copy()
    selected = numpy.flatnonzero(perturbed_map[0, 0])
    unselected = numpy.flatnonzero(~perturbed_map[0, 0])
    perturbed_map[0, 0, selected[0]] = False
    perturbed_map[0, 0, unselected[0]] = True
    perturbed_path = _write_dump(
        tmp_path / "probes_perturbed", step=0, logits=logits, routing_map=perturbed_map, topk=k
    )
    perturbed_row = phi_gap_rows(
        read_dump(perturbed_path), layer=2, unit="u0", lam=0.0, cost_families=("linear",)
    )[0]
    assert perturbed_row.gap_normalized > 0


def test_lam_negative_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_comparison, "UNIT_TOKENS", 4)

    n, e, k = 4, 2, 1
    logits = numpy.zeros((1, n, e), dtype=numpy.float32)
    routing_map = numpy.zeros((1, n, e), dtype=bool)
    routing_map[0, :, 0] = True
    path = _write_dump(tmp_path / "probes", step=0, logits=logits, routing_map=routing_map, topk=k)

    with pytest.raises(ValueError, match="lam"):
        phi_gap_rows(read_dump(path), layer=2, unit="u0", lam=-1.0)

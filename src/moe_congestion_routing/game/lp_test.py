import subprocess
import sys

import numpy as np
import pytest

from moe_congestion_routing.game.ensemble import N512_E8_K2_SEP2_SEED1, affinities
from moe_congestion_routing.game.lp import default_cap, solve
from moe_congestion_routing.game.reference import enumerate_optimal


def test_no_torch_import():
    # game/ is torch-free so it can run on a login node and in an analysis environment.
    script = (
        "import sys; "
        "import moe_congestion_routing.game.lp; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("seed", range(5))
def test_agrees_with_enumeration_on_objective(seed):
    rng = np.random.default_rng(seed)
    a = rng.random((6, 3))
    result = solve(a, k=2, cap=4)
    reference_objective, _ = enumerate_optimal(a, k=2, cap=4)
    assert result.objective == pytest.approx(reference_objective, abs=1e-9)


def test_integral_on_maximally_degenerate_instance():
    a = np.full((6, 3), 0.5)
    result = solve(a, k=2, cap=4)
    assert result.max_fractional_deviation == 0.0


def test_integral_on_two_level_instance():
    a = np.zeros((6, 3))
    a[:, 0] = 1.0  # one expert strictly preferred by every token, the rest tied at 0.
    result = solve(a, k=2, cap=4)
    assert result.max_fractional_deviation == 0.0


def test_k_greater_than_e_raises():
    a = np.random.default_rng(0).random((5, 2))
    with pytest.raises(ValueError, match="k=3"):
        solve(a, k=3)


def test_capacity_too_small_raises():
    a = np.random.default_rng(0).random((5, 3))
    with pytest.raises(ValueError, match="sum\\(cap\\)"):
        solve(a, k=2, cap=1)


def test_returned_assignment_satisfies_both_constraints():
    a = np.random.default_rng(2).random((10, 4))
    result = solve(a, k=2, cap=6)
    assert np.all(result.x.sum(axis=1) == 2)
    assert np.all(result.x.sum(axis=0) <= 6)


def test_every_expert_is_exactly_full_on_a_divisible_instance():
    # When cap * E == N * k the loads must sum to N * k and none may exceed cap, so counting alone
    # forces every expert to sit exactly at cap. Nothing about the affinities can change that.
    a = np.random.default_rng(11).random((30, 5))
    result = solve(a, k=2, cap=12)
    assert result.divisible
    assert result.max_load == 12
    assert set(result.x.sum(axis=0).tolist()) == {12}


def test_an_expert_with_spare_capacity_prices_at_zero():
    # Complementary slackness: a constraint that is not binding cannot carry a shadow price. The
    # converse does not hold, since a binding expert may also price at zero when the optimum is
    # degenerate, so only this direction is asserted.
    a = np.random.default_rng(7).random((37, 5))
    result = solve(a, k=2, cap=16)
    slack = result.x.sum(axis=0) < result.cap
    assert slack.any()
    np.testing.assert_allclose(result.capacity_duals[slack], 0.0)


def test_duals_shapes_and_capacity_dual_sign_on_divisible_instance():
    n, e, k = 512, 8, 2
    a = np.random.default_rng(1).random((n, e))
    result = solve(a, k=k)
    assert result.divisible
    assert result.capacity_duals.shape == (e,)
    assert result.token_duals.shape == (n,)
    assert np.all(result.capacity_duals <= 0)


def test_divisible_true_when_cap_times_e_equals_n_times_k():
    n, e, k = 6, 3, 2
    cap = default_cap(n, k, e)
    result = solve(np.random.default_rng(3).random((n, e)), k=k, cap=cap)
    assert result.divisible


def test_divisible_false_when_default_cap_rounds_up():
    n, e, k = 5, 3, 2
    cap = default_cap(n, k, e)
    assert cap * e != n * k
    result = solve(np.random.default_rng(3).random((n, e)), k=k, cap=cap)
    assert not result.divisible


def test_per_expert_cap_vector_stays_integral_and_scores_no_higher_than_the_uniform_cap():
    # A per-expert vector hands the oracle less slack than the uniform cap that bounds it
    # (uniform capacity at every expert, not only the loaded ones), so its objective can only be
    # lower or equal, never higher.
    from moe_congestion_routing.game.alflb import iterate

    n, e, k = N512_E8_K2_SEP2_SEED1.n, N512_E8_K2_SEP2_SEED1.e, N512_E8_K2_SEP2_SEED1.k
    a = affinities(N512_E8_K2_SEP2_SEED1)
    cap0 = default_cap(n, k, e)
    alf = iterate(a, k=k, eta=1e-2, steps=2000, mode="deployed")
    load_e = alf.cycle_worst.x.sum(axis=0)
    per_expert_cap = np.maximum(load_e, cap0)

    per_expert = solve(a, k, cap=per_expert_cap)
    uniform = solve(a, k, cap=int(per_expert_cap.max()))

    assert per_expert.max_fractional_deviation == 0.0
    assert per_expert.objective <= uniform.objective
    assert per_expert.objective == pytest.approx(876.1654466091404)
    assert uniform.objective == pytest.approx(876.1905057465385)

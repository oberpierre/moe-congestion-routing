import subprocess
import sys

import numpy as np
import pytest

from moe_congestion_routing.game.incremental import solve_incremental
from moe_congestion_routing.game.lp import solve
from moe_congestion_routing.game.reference import enumerate_soft_optimal


def test_no_torch_import():
    # game/ is torch-free so it can run on a login node and in an analysis environment.
    script = (
        "import sys; "
        "import moe_congestion_routing.game.incremental; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def _tie_count(a: np.ndarray, k: int, arc_prices: np.ndarray, best_objective: float) -> int:
    """Count assignments within a small tolerance of ``best_objective``, for uniqueness checks.

    Test-only brute force, separate from ``enumerate_soft_optimal``, so a bug in the production
    reference cannot also hide the tie it would need to disclaim.
    """
    import itertools
    import math

    n, e = a.shape
    arc_prices = np.asarray(arc_prices, dtype=np.float64)
    prices_full = (
        np.broadcast_to(arc_prices, (e, arc_prices.shape[0]))
        if arc_prices.ndim == 1
        else arc_prices
    )
    combos = list(itertools.combinations(range(e), k))
    masks = np.zeros((len(combos), e), dtype=bool)
    for idx, combo in enumerate(combos):
        masks[idx, list(combo)] = True

    count = 0
    for combo_indices in itertools.product(range(len(combos)), repeat=n):
        x = masks[list(combo_indices)]
        loads = x.sum(axis=0)
        if np.any(loads > prices_full.shape[1]):
            continue
        congestion = sum(prices_full[expert, : loads[expert]].sum() for expert in range(e))
        objective = float((a * x).sum()) - congestion
        if math.isclose(objective, best_objective, abs_tol=1e-9):
            count += 1
    return count


@pytest.mark.parametrize("n,e,k", [(6, 3, 2), (5, 3, 2), (7, 3, 2), (6, 4, 2)])
@pytest.mark.parametrize("seed", range(3))
def test_agrees_with_enumeration_flat_and_ramped_prices(n, e, k, seed):
    rng = np.random.default_rng(seed * 100 + n + e + k)
    a = rng.random((n, e))
    # The ramp tops out below the instance's own max_span, the bound the production truncation
    # uses, so the schedule never saturates and this test measures agreement rather than the
    # separate saturation-refusal behaviour covered below.
    max_span = float(np.max(a.max(axis=1) - a.min(axis=1)))
    for arc_prices in (np.zeros(n + 1), np.linspace(0.0, 0.9 * max_span, n + 1)):
        result = solve_incremental(a, k, arc_prices)
        reference_objective, reference_x = enumerate_soft_optimal(a, k, arc_prices)
        assert result.objective == pytest.approx(reference_objective, abs=1e-6)
        if _tie_count(a, k, arc_prices, reference_objective) == 1:
            assert np.array_equal(result.x, reference_x)


def test_flat_zero_prices_reproduce_unconstrained_top_k():
    # Lambda = 0 frontier endpoint: with no congestion cost each token independently takes its own
    # highest-affinity K experts, so this needs no solver at all and is the cheapest cross-check.
    rng = np.random.default_rng(0)
    n, e, k = 6, 5, 2
    a = rng.random((n, e))
    result = solve_incremental(a, k, np.zeros(n + 1))
    for i in range(n):
        expected = set(np.argsort(-a[i])[:k].tolist())
        got = set(np.flatnonzero(result.x[i]).tolist())
        assert got == expected
    # A zero-length schedule of n + 1 arcs is far more than any expert could ever fill, so this is
    # also the headroom case: the saturation refusal must not fire.
    assert np.all(result.arcs_used < result.arcs_available)


def test_steep_prices_converge_to_the_hard_capacity_lp():
    # As prices past L become large relative to the affinity span, the incremental optimum must
    # converge to game/lp.py's hard-capacity optimum: this pins the two oracles against each other
    # at the frontier's other (lambda -> infinity) endpoint.
    rng = np.random.default_rng(3)
    n, e, k, cap = 8, 4, 1, 2  # divisible: cap * e == n * k, matching lp.py's own convention
    a = rng.random((n, e))
    arc_prices = np.array([0.0] * cap + [1e6] * 3)
    result = solve_incremental(a, k, arc_prices)
    lp_result = solve(a, k, cap=cap)
    assert result.congestion == pytest.approx(0.0)
    assert result.objective == pytest.approx(lp_result.objective, abs=1e-6)


def test_truncation_refusal_fires_when_the_optimum_saturates_its_schedule():
    # cap-like divisible instance (2 * 3 == 6 * 1): counting alone forces every expert to exactly
    # 2 units regardless of preference, so a 2-arc schedule is saturated by every expert.
    rng = np.random.default_rng(5)
    a = rng.random((6, 3))
    with pytest.raises(ValueError, match="saturated"):
        solve_incremental(a, k=1, arc_prices=np.zeros(2))


def test_truncation_refusal_does_not_fire_with_headroom():
    rng = np.random.default_rng(5)
    a = rng.random((6, 3))
    result = solve_incremental(a, k=1, arc_prices=np.zeros(7))
    assert np.all(result.arcs_used < result.arcs_available)


def test_monotonicity_assertion_fires_on_decreasing_prices():
    a = np.random.default_rng(1).random((4, 3))
    with pytest.raises(ValueError, match="non-decreasing"):
        solve_incremental(a, k=1, arc_prices=np.array([1.0, 0.5, 0.2]))


def test_k_greater_than_e_raises():
    a = np.random.default_rng(0).random((5, 2))
    with pytest.raises(ValueError, match="k=3"):
        solve_incremental(a, k=3, arc_prices=np.zeros(5))


def test_per_expert_price_schedule_shape():
    rng = np.random.default_rng(9)
    a = rng.random((5, 3))
    arc_prices = np.tile(np.linspace(0.0, 2.0, 6), (3, 1))
    result = solve_incremental(a, k=1, arc_prices=arc_prices)
    assert result.x.sum(axis=1).tolist() == [1] * 5


def test_wrong_expert_row_count_raises():
    a = np.random.default_rng(2).random((5, 3))
    with pytest.raises(ValueError, match="rows"):
        solve_incremental(a, k=1, arc_prices=np.zeros((2, 6)))

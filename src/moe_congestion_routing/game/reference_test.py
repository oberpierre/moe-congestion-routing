import numpy as np
import pytest

from moe_congestion_routing.game.reference import enumerate_optimal, enumerate_soft_optimal


def test_hand_verified_small_instance():
    # Two tokens, cap=1 forces distinct experts, so the optimum is the two largest
    # affinities that do not collide: token 0 -> expert 1 (0.9), token 1 -> expert 0 (0.8).
    a = np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.3]])
    objective, x = enumerate_optimal(a, k=1, cap=1)
    assert objective == pytest.approx(1.7)
    assert np.array_equal(x, np.array([[False, True, False], [True, False, False]]))


def test_hand_verified_small_instance_combinations():
    a = np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.3], [0.4, 0.6, 0.7]])
    objective, x = enumerate_optimal(a, k=2, cap=2)
    assert objective == pytest.approx(3.5)
    assert np.array_equal(
        x, np.array([[False, True, True], [True, True, False], [True, False, True]])
    )


def test_respects_capacity():
    objective, x = enumerate_optimal(np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.3]]), k=1, cap=1)
    assert np.all(x.sum(axis=0) <= 1)
    assert np.all(x.sum(axis=1) == 1)


def test_raises_when_search_space_too_large():
    # C(2, 1)**30 ~= 1.07e9, well past the 10**7 guard, and cheap to check because the
    # guard fires from math.comb before any enumeration happens.
    a = np.zeros((30, 2))
    with pytest.raises(ValueError, match="10000000"):
        enumerate_optimal(a, k=1, cap=30)


def test_soft_optimal_hand_verified_small_instance():
    # Zero prices reduce to unconstrained top-1: each token takes its own best expert, so this is
    # the same instance as the hard-capacity hand-verified test with cap raised out of the way.
    a = np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.3]])
    objective, x = enumerate_soft_optimal(a, k=1, arc_prices=np.zeros(2))
    assert objective == pytest.approx(1.7)
    assert np.array_equal(x, np.array([[False, True, False], [True, False, False]]))


def test_soft_optimal_charges_congestion():
    # Both tokens prefer expert 0. A price schedule that charges 1.0 for the second unit and
    # nothing for the first makes splitting across experts strictly better than concentrating.
    a = np.array([[0.9, 0.1], [0.9, 0.1]])
    objective, x = enumerate_soft_optimal(a, k=1, arc_prices=np.array([0.0, 1.0]))
    assert objective == pytest.approx(1.0)
    assert x.sum(axis=0).tolist() == [1, 1]


def test_soft_optimal_skips_combinations_that_exceed_the_arc_schedule():
    # With only one arc supplied per expert, any combination loading an expert past 1 unit is not
    # priceable, so the optimum is forced to spread even though concentrating would score higher.
    a = np.array([[0.9, 0.1], [0.9, 0.1]])
    objective, x = enumerate_soft_optimal(a, k=1, arc_prices=np.array([0.0]))
    assert x.sum(axis=0).tolist() == [1, 1]
    assert objective == pytest.approx(1.0)


def test_soft_optimal_raises_when_search_space_too_large():
    a = np.zeros((30, 2))
    with pytest.raises(ValueError, match="10000000"):
        enumerate_soft_optimal(a, k=1, arc_prices=np.zeros(30))

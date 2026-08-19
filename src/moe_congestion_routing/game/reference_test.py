import numpy as np
import pytest

from moe_congestion_routing.game.reference import enumerate_optimal


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

"""Brute-force solver for the LP to certify `game.lp.solve` against the ground truth.

Not to be used as actual LP solver, therefore raises for large instances."""

import itertools
import math

import numpy as np

MAX_ENUMERATED_ASSIGNMENTS = 10**7


def enumerate_optimal(a: np.ndarray, k: int, cap: int) -> tuple[float, np.ndarray]:
    """Brute-force the capacity-constrained top-K assignment problem `game.lp.solve` also solves.

    Raises ``ValueError`` when ``C(E, K)**N`` exceeds ``MAX_ENUMERATED_ASSIGNMENTS``, so this can
    never be mistaken for something callable at scale.
    """
    a = np.asarray(a, dtype=np.float64)
    n, e = a.shape
    num_combos = math.comb(e, k)
    total = num_combos**n
    if total > MAX_ENUMERATED_ASSIGNMENTS:
        raise ValueError(
            f"enumerate_optimal: C(E,K)**N = C({e},{k})**{n} = {total} exceeds "
            f"{MAX_ENUMERATED_ASSIGNMENTS}; instance too large to enumerate"
        )

    combos = list(itertools.combinations(range(e), k))
    masks = np.zeros((num_combos, e), dtype=bool)
    for idx, combo in enumerate(combos):
        masks[idx, list(combo)] = True

    best_objective = -np.inf
    best_x = None
    for combo_indices in itertools.product(range(num_combos), repeat=n):
        x = masks[list(combo_indices)]
        if np.any(x.sum(axis=0) > cap):
            continue
        objective = float((a * x).sum())
        if objective > best_objective:
            best_objective = objective
            best_x = x.copy()

    if best_x is None:
        raise ValueError(f"enumerate_optimal: no assignment satisfies cap={cap}")
    return best_objective, best_x

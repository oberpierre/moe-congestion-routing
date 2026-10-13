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


def enumerate_soft_optimal(
    a: np.ndarray, k: int, arc_prices: np.ndarray
) -> tuple[float, np.ndarray]:
    """Brute-force the arc-priced assignment problem `game.incremental.solve_incremental` solves.

    Same enumeration and the same ``MAX_ENUMERATED_ASSIGNMENTS`` guard as ``enumerate_optimal``.
    There is no hard capacity here, so every combination is feasible as long as no expert's
    realized load exceeds how many prices ``arc_prices`` supplies for it, whereas a combination
    that would need more is skipped rather than priced. The oracle instead lengthens its own
    schedule and retries, so supply ``arc_prices`` long enough to seat any load before comparing
    the two, or this one silently enumerates a smaller feasible set than the oracle searches.
    """
    a = np.asarray(a, dtype=np.float64)
    n, e = a.shape
    arc_prices = np.asarray(arc_prices, dtype=np.float64)
    prices_full = (
        np.broadcast_to(arc_prices, (e, arc_prices.shape[0]))
        if arc_prices.ndim == 1
        else arc_prices
    )
    num_combos = math.comb(e, k)
    total = num_combos**n
    if total > MAX_ENUMERATED_ASSIGNMENTS:
        raise ValueError(
            f"enumerate_soft_optimal: C(E,K)**N = C({e},{k})**{n} = {total} exceeds "
            f"{MAX_ENUMERATED_ASSIGNMENTS}. Instance too large to enumerate"
        )

    combos = list(itertools.combinations(range(e), k))
    masks = np.zeros((num_combos, e), dtype=bool)
    for idx, combo in enumerate(combos):
        masks[idx, list(combo)] = True

    best_objective = -np.inf
    best_x = None
    for combo_indices in itertools.product(range(num_combos), repeat=n):
        x = masks[list(combo_indices)]
        loads = x.sum(axis=0)
        if np.any(loads > prices_full.shape[1]):
            continue  # some expert's load needs more arcs than were supplied, so skip it
        congestion = sum(prices_full[expert, : loads[expert]].sum() for expert in range(e))
        objective = float((a * x).sum()) - congestion
        if objective > best_objective:
            best_objective = objective
            best_x = x.copy()

    if best_x is None:
        raise ValueError("enumerate_soft_optimal: no assignment fits within arc_prices' length")
    return best_objective, best_x

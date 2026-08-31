"""Incremental-arc LP oracle for the soft (uncapacitated, convex-congestion) assignment game.

Given fixed affinities and a per-expert schedule of arc prices, maximizes affinity minus the
congestion cost the assignment buys. Unlike ``lp.py`` there is no hard capacity: an expert can
take any load, but each additional unit costs more than the last. Names no congestion cost family
and computes no cost itself. The caller decides which family and which lambda produced
``arc_prices`` and hands this module a plain array.

Each expert ``e`` gets unit-capacity arcs ``j = 1..J_e`` priced ``arc_prices[e, j-1]``. Because
prices are non-decreasing in ``j``, the cheapest arcs fill first at any optimum with no explicit
ordering constraint needed. Due to total unimodularity, the solution is already integral, same
as ``lp.py``'s relaxation integral.
"""

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


class IncrementalResult(NamedTuple):
    x: np.ndarray  # bool [N, E], exactly K True per row
    affinity: float  # (a * x).sum() at x
    congestion: float  # the arc cost the assignment bought
    objective: float  # affinity - congestion
    loads: np.ndarray  # int [E], x.sum(axis=0)
    arcs_used: np.ndarray  # int [E], how many arcs each expert filled
    arcs_available: np.ndarray  # int [E], J_e after truncation
    max_fractional_deviation: float  # max|. - round(.)| over x and y before rounding


def solve_incremental(a: np.ndarray, k: int, arc_prices: np.ndarray) -> IncrementalResult:
    """Solve the incremental-arc assignment LP and return it with its diagnostics.

    ``arc_prices`` is either a shared ``[J]`` schedule broadcast to every expert or a per-expert
    ``[E, J]`` schedule. Raises ``ValueError`` when it is not non-decreasing along ``j``, and
    raises when any expert's rounded optimum uses every arc it was given — a silently truncated
    optimum would be a wrong number that looks like a right one, so extend that expert's schedule
    before trusting the result.
    """
    a = np.asarray(a, dtype=np.float64)
    n, e = a.shape
    if k > e:
        raise ValueError(f"infeasible: k={k} > e={e}, cannot choose {k} experts out of {e}")

    arc_prices = np.asarray(arc_prices, dtype=np.float64)
    if arc_prices.ndim == 1:
        prices_full = np.broadcast_to(arc_prices, (e, arc_prices.shape[0])).copy()
    elif arc_prices.ndim == 2:
        if arc_prices.shape[0] != e:
            raise ValueError(
                f"arc_prices has {arc_prices.shape[0]} rows, expected one per expert ({e})"
            )
        prices_full = arc_prices.copy()
    else:
        raise ValueError(f"arc_prices must be 1-D [J] or 2-D [E, J], got shape {arc_prices.shape}")

    j_full = prices_full.shape[1]
    if j_full == 0:
        raise ValueError("arc_prices must supply at least one arc")
    if not np.all(np.diff(prices_full, axis=1) >= 0):
        raise ValueError(
            "arc_prices must be non-decreasing along j: the integrality argument depends on "
            "cheaper arcs filling before costlier ones, with no explicit ordering constraint"
        )

    # No token will ever pay more than the largest affinity gain it could realize by switching
    # experts, so an arc priced above that bound can never belong to an optimal assignment. Cutting
    # the schedule there off shrinks the LP without changing its answer, and the first such arc is
    # kept as a sentinel: an optimum that reaches it has broken the bound, so treat that as a
    # signal to raise rather than trust the truncation.
    max_span = float(np.max(a.max(axis=1) - a.min(axis=1)))
    j_e = np.empty(e, dtype=np.int64)
    for expert in range(e):
        exceeds = np.flatnonzero(prices_full[expert] > max_span)
        j_e[expert] = int(exceeds[0]) + 1 if exceeds.size else j_full

    num_x = n * e
    offsets = np.concatenate(([0], np.cumsum(j_e)))
    num_y = int(offsets[-1])

    # Flatten x[i, e] to column index i*e + expert, matching a.ravel()'s row-major order, exactly
    # as lp.py does, so the objective vector and the constraint columns index the same variable.
    columns = np.arange(num_x)
    token_of = columns // e
    expert_of = columns % e

    token_rows = sp.csr_matrix((np.ones(num_x), (token_of, columns)), shape=(n, num_x))
    token_block = sp.hstack([token_rows, sp.csr_matrix((n, num_y))], format="csr")
    b_token = np.full(n, k, dtype=np.float64)

    # Per-expert flow conservation: load in from tokens equals arcs filled out, which is what lets
    # the cheapest arcs fill first without an explicit ordering constraint on y.
    x_expert_rows = sp.csr_matrix((np.ones(num_x), (expert_of, columns)), shape=(e, num_x))
    y_expert_of = np.repeat(np.arange(e), j_e)
    y_expert_rows = sp.csr_matrix(
        (-np.ones(num_y), (y_expert_of, np.arange(num_y))), shape=(e, num_y)
    )
    expert_block = sp.hstack([x_expert_rows, y_expert_rows], format="csr")
    b_expert = np.zeros(e, dtype=np.float64)

    a_eq = sp.vstack([token_block, expert_block], format="csr")
    b_eq = np.concatenate([b_token, b_expert])

    y_prices = np.concatenate([prices_full[expert, : j_e[expert]] for expert in range(e)])
    c = np.concatenate([-a.ravel(), y_prices])

    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=(0, 1), method="highs-ds")
    if not res.success:
        raise RuntimeError(f"linprog did not succeed: {res.message}")

    x_lp = res.x[:num_x].reshape(n, e)
    y_lp = res.x[num_x:]

    max_fractional_deviation = float(
        max(
            np.max(np.abs(x_lp - np.round(x_lp))),
            np.max(np.abs(y_lp - np.round(y_lp))) if num_y else 0.0,
        )
    )
    # Insurance against a future HiGHS presolve or method change, not a fix for an observed
    # defect: no fractional result was seen while writing this, including under deliberate ties.
    assert max_fractional_deviation <= 1e-6, (
        f"non-integral LP solution: max_fractional_deviation={max_fractional_deviation}"
    )

    x = np.round(x_lp).astype(bool)
    y = np.round(y_lp).astype(bool)

    experts_per_token = x.sum(axis=1)
    bad_tokens = np.flatnonzero(experts_per_token != k)
    if bad_tokens.size:
        raise ValueError(
            f"rounded assignment violates the top-k constraint: tokens {bad_tokens.tolist()} "
            f"have counts {experts_per_token[bad_tokens].tolist()}, expected k={k}"
        )

    loads = x.sum(axis=0)
    arcs_used = np.array(
        [int(y[offsets[expert] : offsets[expert + 1]].sum()) for expert in range(e)]
    )
    saturated = np.flatnonzero(arcs_used == j_e)
    if saturated.size:
        raise ValueError(
            f"incremental oracle saturated its arc budget for experts {saturated.tolist()}: "
            f"arcs_used == arcs_available == {j_e[saturated].tolist()}. The true optimum may "
            "need more arcs than arc_prices supplied for these experts; extend their schedule."
        )

    congestion = float(y_prices[y].sum())
    affinity = float((a * x).sum())

    return IncrementalResult(
        x=x,
        affinity=affinity,
        congestion=congestion,
        objective=affinity - congestion,
        loads=loads.astype(np.int64),
        arcs_used=arcs_used.astype(np.int64),
        arcs_available=j_e,
        max_fractional_deviation=max_fractional_deviation,
    )

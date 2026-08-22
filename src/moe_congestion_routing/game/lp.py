"""Exact oracle for the capacity-constrained top-K assignment problem.

Given per-token affinities, pick exactly ``k`` experts for every token without loading any expert
past ``cap``, maximizing total affinity. Every variable sits in exactly one token row and exactly
one expert row, so the constraint matrix is a bipartite incidence matrix and is totally unimodular.
That means the LP relaxation already has integral vertices: this is the exact integer answer, not a
relaxation of it.

Ties are broken by HiGHS, not by the lowest-index rule the rest of this package uses, so two
equally optimal assignments can name different experts. When comparing assignments against this
oracle, skip the tied tokens. Find them with ``tie_margins(a + capacity_duals, k)``, because the
LP ranks by that sum rather than by ``a`` alone.

``method="highs-ds"`` (dual simplex) is pinned for the *duals*, A simplex solve stops at a vertex,
which is both integral and tied to a single basis, so its dual is the well-defined basic one.
Interior-point can stop anywhere inside an optimal face. The capacity dual is the shadow price
ALF-LB's expert bias is conjectured to equal, so we need the integral vertex duals for comparison.
"""

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


class LpResult(NamedTuple):
    x: np.ndarray  # bool [N, E], exactly K True per row
    objective: float  # (a * x).sum() at x
    cap: int | np.ndarray  # the capacity actually solved with, as given: a scalar or an [E] vector
    max_load: int  # x.sum(axis=0).max()
    max_fractional_deviation: float  # max|x_lp - round(x_lp)| before rounding
    capacity_duals: np.ndarray  # float64 [E], res.ineqlin.marginals, sign as reported
    token_duals: np.ndarray  # float64 [N], res.eqlin.marginals
    divisible: bool  # cap * E == N * K, so every capacity row binds


def default_cap(n: int, k: int, e: int) -> int:
    """The smallest per-expert capacity under which an assignment can exist.

    The ``n * k`` unit assignments spread over ``e`` experts need ``ceil(n * k / e)`` each, and it
    is written as ``-(-x // y)`` because ``//`` floors: negating, flooring and negating back rounds
    up, whereas ``math.ceil(n * k / e)`` would round a float and can be off by one on large inputs.
    """
    return -(-n * k // e)


def solve(a: np.ndarray, k: int, cap: int | np.ndarray | None = None) -> LpResult:
    """Solve the capacity-constrained top-K assignment LP and return it with its duals.

    ``cap`` is either a scalar applied to every expert or an integer ``[E]`` vector of
    per-expert capacities. Raises ``ValueError`` when the instance is infeasible by counting
    alone (``k > e``, or ``sum(cap) < n * k``).
    """
    a = np.asarray(a, dtype=np.float64)
    n, e = a.shape
    if k > e:
        raise ValueError(f"infeasible: k={k} > e={e}, cannot choose {k} experts out of {e}")
    if cap is None:
        cap = default_cap(n, k, e)
    # Broadcast a scalar to a uniform [E] vector so the rest of the solve is one code path
    # regardless of whether the caller passed a single capacity or a per-expert one.
    cap_vec = np.broadcast_to(np.asarray(cap, dtype=np.float64), (e,))
    total_cap = float(cap_vec.sum())
    if total_cap < n * k:
        raise ValueError(f"infeasible: sum(cap)={total_cap} < n*k={n * k}")

    num_vars = n * e
    # Flatten x[i, e] to column index i*e + expert, matching a.ravel()'s row-major order so
    # the objective vector and the constraint columns index the same variable.
    columns = np.arange(num_vars)
    token_of = columns // e
    expert_of = columns % e

    a_eq = sp.csr_matrix(
        (np.ones(num_vars), (token_of, columns)),
        shape=(n, num_vars),
    )
    b_eq = np.full(n, k, dtype=np.float64)

    a_ub = sp.csr_matrix(
        (np.ones(num_vars), (expert_of, columns)),
        shape=(e, num_vars),
    )
    b_ub = cap_vec

    c = -a.ravel()
    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0, 1),
        method="highs-ds",
    )
    if not res.success:
        raise RuntimeError(f"linprog did not succeed: {res.message}")

    x_lp = res.x.reshape(n, e)
    max_fractional_deviation = float(np.max(np.abs(x_lp - np.round(x_lp))))
    # Insurance against a future HiGHS presolve or method change, not a fix for an observed
    # defect: no fractional result was seen in planning, including under deliberate ties.
    assert max_fractional_deviation <= 1e-6, (
        f"non-integral LP solution: max_fractional_deviation={max_fractional_deviation}"
    )

    x = np.round(x_lp).astype(bool)

    # axis=1 sums across experts for one token, so this is the top-k constraint.
    experts_per_token = x.sum(axis=1)
    bad_tokens = np.flatnonzero(experts_per_token != k)
    if bad_tokens.size:
        raise ValueError(
            f"rounded assignment violates the top-k constraint: tokens {bad_tokens.tolist()} "
            f"have counts {experts_per_token[bad_tokens].tolist()}, expected k={k}"
        )

    # axis=0 sums across tokens for one expert, so this is that expert's load. Compared against
    # cap_vec rather than the raw cap so this check is elementwise for a per-expert vector too.
    load_per_expert = x.sum(axis=0)
    overloaded = np.flatnonzero(load_per_expert > cap_vec)
    if overloaded.size:
        raise ValueError(
            f"rounded assignment violates the capacity constraint: experts {overloaded.tolist()} "
            f"have loads {load_per_expert[overloaded].tolist()}, exceeding "
            f"cap={cap_vec[overloaded].tolist()}"
        )

    return LpResult(
        x=x,
        objective=float((a * x).sum()),
        cap=cap,
        max_load=int(load_per_expert.max()),
        max_fractional_deviation=max_fractional_deviation,
        capacity_duals=np.asarray(res.ineqlin.marginals, dtype=np.float64),
        token_duals=np.asarray(res.eqlin.marginals, dtype=np.float64),
        divisible=(total_cap == n * k),
    )

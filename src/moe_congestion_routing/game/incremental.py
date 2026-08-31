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
    arcs_available: np.ndarray  # int [E], J_e after truncation and any growth
    max_fractional_deviation: float  # max|. - round(.)| over x and y before rounding
    arc_growths: int  # how many times J_e was doubled before the solve held


def solve_incremental(a: np.ndarray, k: int, arc_prices: np.ndarray) -> IncrementalResult:
    """Solve the incremental-arc assignment LP and return it with its diagnostics.

    ``arc_prices`` is either a shared ``[J]`` schedule broadcast to every expert or a per-expert
    ``[E, J]`` schedule. Raises ``ValueError`` when it is not non-decreasing along ``j``. When the
    schedule starts too short to seat the batch, or an optimum saturates it, ``J_e`` doubles and
    the solve retries, up to a cap of ``N`` where doubling stops helping and the refusal becomes a
    genuine one, and a caller-supplied ``arc_prices`` too short for what a retry wants raises
    rather than being extrapolated past what was given.
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
    # experts, so an arc priced above that bound can never belong to an optimal assignment. This
    # bound alone can undershoot the batch's own demand, since a token that must go somewhere is
    # not the token the bound reasons about, so it is only the first of two floors on J_e below.
    max_span = float(np.max(a.max(axis=1) - a.min(axis=1)))
    j_span = np.empty(e, dtype=np.int64)
    for expert in range(e):
        exceeds = np.flatnonzero(prices_full[expert] > max_span)
        j_span[expert] = int(exceeds[0]) + 1 if exceeds.size else j_full

    # Sigma_e x_ie = k is an equality, so every token must be seated somewhere, and a schedule
    # shorter than the batch cannot seat it at all.
    feasibility_floor = int(np.ceil(n * k / e))
    j_e = np.maximum(j_span, feasibility_floor)
    # Total capacity equal to total demand leaves no aggregate slack, so pigeonhole pins every
    # expert to its cap and the schedule saturates whatever the prices are. Start one doubling up
    # where the caller's array allows it, rather than spending a whole solve to learn that.
    # Only ever upward: capping at j_full must not pull j_e back below the floor, or the
    # too-short check below would stop firing on a schedule that genuinely cannot seat the batch.
    if int(j_e.sum()) <= n * k:
        j_e = np.maximum(j_e, np.minimum(2 * j_e, j_full))
    if np.any(j_e > j_full):
        needed = int(j_e.max())
        raise ValueError(
            f"arc_prices supplies {j_full} arcs per expert but the feasibility floor and span "
            f"bound together need {needed}, so extend arc_prices before calling solve_incremental"
        )

    def _grow(j_e: np.ndarray, reason: str) -> np.ndarray:
        # Shared by the infeasible and saturated branches below: both face the same two terminal
        # conditions, the hard cap N where growing further can never help, and a caller-supplied
        # schedule too short to grow into, which is not this oracle's cost to invent.
        if np.all(j_e >= n):
            raise ValueError(
                f"incremental oracle still {reason} with every J_e at the cap J_e = N = {n}: "
                "this is a failure of the instance itself, not of the arc provisioning."
            )
        candidate = np.minimum(j_e * 2, n)
        if np.any(candidate > j_full):
            needed = int(candidate.max())
            raise ValueError(
                f"incremental oracle {reason} and the retry wants {needed} arcs per expert, but "
                f"arc_prices supplies only {j_full}, so extend arc_prices to at least {needed}"
            )
        return candidate

    arc_growths = 0
    while True:
        num_x = n * e
        offsets = np.concatenate(([0], np.cumsum(j_e)))
        num_y = int(offsets[-1])

        # Flatten x[i, e] to column index i*e + expert, matching a.ravel()'s row-major order,
        # exactly as lp.py does, so the objective vector and the constraint columns index the
        # same variable.
        columns = np.arange(num_x)
        token_of = columns // e
        expert_of = columns % e

        token_rows = sp.csr_matrix((np.ones(num_x), (token_of, columns)), shape=(n, num_x))
        token_block = sp.hstack([token_rows, sp.csr_matrix((n, num_y))], format="csr")
        b_token = np.full(n, k, dtype=np.float64)

        # Per-expert flow conservation: load in from tokens equals arcs filled out, which is what
        # lets the cheapest arcs fill first without an explicit ordering constraint on y.
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
            # Infeasible here means the schedule was too short to seat the batch, not that the
            # instance is unsolvable, so this is not an error until growth has nowhere left to go.
            j_e = _grow(j_e, f"got an infeasible linprog result ({res.message})")
            arc_growths += 1
            continue

        x_lp = res.x[:num_x].reshape(n, e)
        y_lp = res.x[num_x:]

        max_fractional_deviation = float(
            max(
                np.max(np.abs(x_lp - np.round(x_lp))),
                np.max(np.abs(y_lp - np.round(y_lp))) if num_y else 0.0,
            )
        )
        # Insurance against a future HiGHS presolve or method change, not a fix for an observed
        # defect: no fractional result was seen while writing this, including under deliberate
        # ties.
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
            # Before the cap this means the schedule was provisioned too short to see the true
            # optimum, so it grows and retries. At the cap J_e = N it means the instance itself
            # needs more than one arc per token per expert can ever supply, which cannot happen,
            # so it is a genuine failure of the instance rather than of the provisioning.
            j_e = _grow(j_e, f"saturated its arc budget for experts {saturated.tolist()}")
            arc_growths += 1
            continue

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
            arc_growths=arc_growths,
        )

"""Scores one probe unit's realized routing against the shared soft-potential reference game.

Every arm, whatever congestion cost it was trained under or none, is graded against the same
reference game so the ranking means what its name claims: the soft potential
``Phi_c(x) = affinity(x) - congestion_c(x)``, maximized by
:func:`game.incremental.solve_incremental` over the incremental-arc LP relaxation, which is
integral. This module reads one probe dump's unit, builds the three assignments the row compares
(the realized routing, the oracle, and the affinity-blind exactly balanced baseline), and reports
the gap between the realized potential and the oracle's, raw and normalized.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from moe_congestion_routing.game.incremental import solve_incremental
from moe_congestion_routing.losses.cost_families import (
    discrete_potential,
    first_arc_above_price,
    marginal_cost,
)
from moe_congestion_routing.metrics.probe_comparison import probe_units, screen_batch
from moe_congestion_routing.metrics.probe_series import ProbeDump


class PhiGapRow(NamedTuple):
    """One ``(unit, layer, step, reference_cost)`` reading of the potential gap against the
    shared reference game, both raw per token and normalized against the affinity-blind baseline.
    """

    unit: str
    layer: int
    step: int
    reference_cost: str
    lam: float
    affinity_space: str
    score_function: str
    admissible: bool
    max_load_over_balanced: float
    dead_experts: int
    gap_per_token: float
    affinity_shortfall: float
    congestion_excess: float
    gap_normalized: float
    normalizer: float
    arc_growths: int
    arcs_used_max: int
    max_fractional_deviation: float
    token_sha256: str
    dump_path: str


def _layer_axis(dump: ProbeDump, layer: int) -> int:
    """The index into ``routing_map``'s/``logits``'s first axis that this layer number owns."""
    layer_numbers = dump.layer_numbers
    if layer not in layer_numbers:
        raise ValueError(f"{dump.path}: layer {layer} not among this dump's layers {layer_numbers}")
    return layer_numbers.index(layer)


def _unit_bounds(n_tokens: int, unit: str) -> tuple[int, int]:
    """The ``[start, stop)`` token range ``unit`` names, from ``probe_units``' own cut."""
    units = probe_units(n_tokens)
    for name, start, stop in units:
        if name == unit:
            return start, stop
    available = [name for name, _, _ in units]
    raise ValueError(f"unit {unit!r} is not among {available!r} for {n_tokens} tokens")


def _balanced_assignment(n: int, k: int, e: int) -> np.ndarray:
    """``[n, k]`` int expert indices: token ``i`` gets ``{(i*K + m) mod E : m = 0..K-1}``.

    Deterministic and seedless, so the baseline it defines reproduces the same value forever
    rather than depending on when it was drawn. Exactly balanced whenever ``E`` divides ``n*K``.
    """
    i = np.arange(n)[:, None]
    m = np.arange(k)[None, :]
    return (i * k + m) % e


def arc_schedule_length(
    n: int, k: int, e: int, max_span: float, *, lam: float, cost_family: str
) -> int:
    """The arc budget ``J`` every expert's schedule is truncated to at this ``lam``.

    ``min(n, 2 * max(J_span, ceil(n*k/e)))``, where ``J_span`` is the smallest arc index whose
    price exceeds the unit's own largest per-token affinity span. The feasibility floor
    ``ceil(n*k/e)`` alone leaves no aggregate slack, so pigeonhole saturates the first solve
    whatever the prices are, and doubling it starts one growth ahead where a caller-supplied
    schedule allows it.
    """
    balanced_load = n * k / e
    j_span = first_arc_above_price(max_span, balanced_load, lam=lam, cost_family=cost_family)
    # Integer ceiling of n*k/e without float rounding, matching solve_incremental's own floor.
    feasibility_floor = -(-n * k // e)
    return min(n, 2 * max(j_span, feasibility_floor))


def phi_gap_rows(
    dump: ProbeDump,
    layer: int,
    unit: str,
    *,
    lam: float = 1.0,
    cost_families: Sequence[str] = ("linear", "quadratic"),
) -> list[PhiGapRow]:
    """The potential-gap row, once per ``cost_families`` entry, for one dump/layer/unit.

    The affinity is the dump's own :meth:`~ProbeDump.router_scores`, whatever score function the
    dump used, whereas the realized loads come from ``routing_map`` alone with no bias added
    back in: the game scores what the router valued, and ALF-LB's bias is a balancing mechanism
    acting on that value rather than a term in it. ``admissible`` is always reported as a column,
    never used to skip the row, because a refused unit's gap is still a real reading of a router
    far from equilibrium rather than an artifact of the screen.
    """
    axis = _layer_axis(dump, layer)
    scores = dump.router_scores()[axis]
    routing = dump.routing_map()[axis]

    start, stop = _unit_bounds(scores.shape[0], unit)
    # Copy the unit out and drop the base, because router_scores widens every layer to
    # float64 and a view would hold all of them alive for the whole solve, which runs for a
    # minute or more and is what sets a parallel grid's memory ceiling.
    a = np.array(scores[start:stop])
    realized = np.array(routing[start:stop])
    del scores, routing

    n, e = a.shape
    k = dump.topk
    balanced_load = n * k / e
    # Fixed at exactly this length, not tuned: a shorter schedule starves the feasibility floor
    # into growing the arc budget mid-solve, whereas a longer one spends solve time nothing here
    # needs.
    num_arcs = 2 * math.ceil(n * k / e)

    screen = screen_batch(realized, k)

    loads_realized = realized.sum(axis=0)
    affinity_realized = float(a[realized].sum())

    baseline_experts = _balanced_assignment(n, k, e)
    loads_baseline = np.bincount(baseline_experts.ravel(), minlength=e).astype(np.int64)
    tokens = np.arange(n)[:, None]
    affinity_baseline = float(a[tokens, baseline_experts].sum())

    rows = []
    for cost_family in cost_families:
        arc_prices = marginal_cost(
            np.arange(1, num_arcs + 1), balanced_load, lam=lam, cost_family=cost_family
        )
        oracle = solve_incremental(a, k, arc_prices)

        congestion_realized = discrete_potential(
            loads_realized, balanced_load, lam=lam, cost_family=cost_family
        )
        congestion_baseline = discrete_potential(
            loads_baseline, balanced_load, lam=lam, cost_family=cost_family
        )

        phi_realized = affinity_realized - congestion_realized
        phi_star = oracle.affinity - oracle.congestion
        phi_baseline = affinity_baseline - congestion_baseline

        gap_per_token = (phi_star - phi_realized) / n
        # phi_star is a maximum over every feasible assignment including the realized one, so
        # the true gap can never be negative. A tiny negative reading is LP solve tolerance, not
        # a defect, but a reading negative by more than that means the oracle under-optimized.
        assert gap_per_token >= -1e-6, (
            f"gap_per_token {gap_per_token} < 0 beyond float slack: the oracle scored below the "
            "realized assignment it is supposed to dominate"
        )

        affinity_shortfall = (oracle.affinity - affinity_realized) / n
        congestion_excess = (congestion_realized - oracle.congestion) / n
        decomposition_sum = affinity_shortfall + congestion_excess
        assert abs(decomposition_sum - gap_per_token) <= 1e-9 * max(abs(gap_per_token), 1.0), (
            f"affinity_shortfall + congestion_excess ({decomposition_sum}) does not reproduce "
            f"gap_per_token ({gap_per_token})"
        )

        normalizer = phi_star - phi_baseline
        # Matches the guard game/compare.py already uses for an analogous span-in-the-denominator
        # ratio: a non-strict threshold relative to the numerator's own scale, so a denominator
        # that is float noise around zero reports NaN instead of an arbitrarily large ratio.
        if abs(normalizer) <= 1e-12 * abs(phi_star):
            gap_normalized = float("nan")
        else:
            gap_normalized = (phi_star - phi_realized) / normalizer

        rows.append(
            PhiGapRow(
                unit=unit,
                layer=layer,
                step=dump.step,
                reference_cost=cost_family,
                lam=lam,
                # Constant here by construction, and carried anyway so a collector joining
                # these rows to SwapGap rows can refuse a mismatch rather than silently
                # comparing a score-space gap against a logit-space one.
                affinity_space="score",
                score_function=dump.score_function,
                admissible=screen.admissible,
                max_load_over_balanced=screen.max_load_over_balanced,
                dead_experts=screen.dead_experts,
                gap_per_token=gap_per_token,
                affinity_shortfall=affinity_shortfall,
                congestion_excess=congestion_excess,
                gap_normalized=gap_normalized,
                normalizer=normalizer,
                arc_growths=oracle.arc_growths,
                arcs_used_max=int(oracle.arcs_used.max()),
                max_fractional_deviation=oracle.max_fractional_deviation,
                token_sha256=dump.token_sha256,
                dump_path=str(dump.path),
            )
        )
    return rows

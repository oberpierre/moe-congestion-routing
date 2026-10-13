"""Compare a converged (or not) ALF-LB run against the LP oracle on one affinity matrix.

Ties :mod:`alflb` and :mod:`lp` together into a single row of evidence for or against
the ALF-LB convergence to NE theorem. The theorem is conditional on the dual converging and
on no ties, so :func:`compare` reports which of those held (``tier``, ``settled_at``,
tie margins) alongside the gap, rather than collapsing everything into one pass/fail number.
"""

from typing import NamedTuple

import numpy as np

from moe_congestion_routing.game import alflb, lp

# Used only by `tied_token_mask`, whose job is to catch the oracle ranking's *exact* ties, which
# optimal capacity duals manufacture on essentially every instance. On a real dump all 61 tokens it
# selects have margin exactly zero, so the constant is doing no near-tie work and no ULP conversion
# would change the answer. The tier gate that once read it is gone: see `classify_tier`.
TIE_TOLERANCE = 1e-9


class Comparison(NamedTuple):
    # The instance and the run configuration.
    n: int
    e: int
    k: int
    mode: str  # "annealed" | "deployed"
    eta: float
    steps: int  # the budget

    # From alflb: the run's trajectory, its convergence, its designated basis and (deployed
    # only) its cycle. Everything in this group comes from ONE basis point, named by `basis`.
    steps_run: int  # AlfResult.steps
    settled_at: int | None
    basis: str  # "trajectory_closest" (annealed, or deployed if it settled) | "cycle_worst"
    basis_step: int  # the step of whichever BalancePoint the basis names
    objective: float
    max_load: int
    overflow: float  # max_load / balanced_load - 1
    cycle_length: int | None  # deployed only
    band_width: float  # deployed only
    cycle_objective_mean: float  # deployed only

    # From lp: the capacity-constrained oracle and the unconstrained "vanilla" baseline.
    divisible: bool
    default_cap: int  # lp.default_cap(n, k, e)
    lp_objective: float  # the oracle at default_cap
    vanilla_objective: float  # (a * x).sum() at top_k_map(a, k), no bias, no capacity
    vanilla_max_load: int  # its max load

    # From the comparison itself: gaps, ties, agreement and the tier verdict.
    balanced_load: float  # n * k / e
    oracle_max_load: int  # the oracle's own realized max load; equals default_cap
    # whenever the instance is divisible, and is the only place a non-divisible
    # oracle's slack shows up
    span: float  # vanilla_objective - lp_objective, the price of capacity
    gap_at_default_cap: float  # lp_objective - objective, NaN iff max_load > default_cap
    matched_cap: int  # max(max_load, default_cap)
    matched_lp_objective: float  # oracle at matched_cap, NaN iff matched_lp_unconstrained
    gap_at_matched_cap: float  # NaN iff matched_lp_unconstrained
    gap_at_per_expert_cap: float  # re-solve at maximum(load_e, default_cap), NaN likewise
    gap_over_span: float  # gap_at_matched_cap / span, THE headline
    matched_lp_unconstrained: bool  # matched_cap >= vanilla_max_load, voids the two matched gaps
    # The cycle's other end, deployed only: scored at the SAME matched_cap as the basis point
    # above so the two gaps are comparable and the whole family is recoverable by subtraction.
    cycle_best_step: int | None
    cycle_best_max_load: int | None
    cycle_best_objective: float
    cycle_best_gap_at_matched_cap: float  # matched_lp_objective - cycle_best_objective
    cycle_best_gap_over_span: float
    set_agreement: float  # fraction of tokens whose expert set equals the oracle's
    untied_set_agreement: float  # the same over untied tokens only
    tied_tokens: int  # pooled over both rankings
    median_tie_margin: float  # trajectory ranking (a + bias) alone
    min_tie_margin: float  # trajectory ranking (a + bias) alone
    # The same two margins counted in float steps of the width `a` was passed in, which for a
    # router dump is the width the model routes in. So the question they answer is "would a
    # router working in this arithmetic see these two scores as distinct", not "did our own
    # arithmetic distinguish them": the simulated bias is accumulated in float64 whatever `a`
    # was, so the ranking these are taken from is always a float64 quantity.
    median_tie_margin_ulp: float
    min_tie_margin_ulp: float
    oracle_min_margin: float  # oracle ranking (a + capacity_duals) alone, an instance property
    oracle_exact_ties: int  # count of exactly-zero margins on the oracle ranking
    excess_tokens: int  # sum(max(load_e - default_cap, 0)), the residual size
    tier: str  # "settled" | "unconverged", exactly `settled_at is not None`
    dual_correlation: float  # NaN if not divisible
    dual_linf: float  # NaN if not divisible


def dual_agreement(bias: np.ndarray, capacity_duals: np.ndarray) -> tuple[float, float]:
    """Pearson correlation and L-infinity distance between two mean-centered vectors.

    Both a converged ALF-LB bias and the LP's capacity duals are fixed only up to an additive
    constant when every capacity binds, so comparing the raw vectors fails for a reason that has
    nothing to do with the theorem. Centering removes that freedom.
    """
    bias = np.asarray(bias, dtype=np.float64)
    duals = np.asarray(capacity_duals, dtype=np.float64)
    bias_c = bias - bias.mean()
    duals_c = duals - duals.mean()
    # A vector with no variance correlates with nothing, and numpy reports that by dividing by a
    # zero standard deviation, so it returns the right NaN behind a RuntimeWarning. Returning the
    # NaN directly keeps the answer and drops the warning, which matters because this case is
    # reached on every real series: ALF-LB's stored bias is identically zero before its first
    # update, so eight warnings per run trained the reader to scroll past a warning from here.
    if bias_c.any() and duals_c.any():
        correlation = float(np.corrcoef(bias_c, duals_c)[0, 1])
    else:
        correlation = float("nan")
    linf = float(np.max(np.abs(bias_c - duals_c)))
    return correlation, linf


def token_set_agreement(x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
    """Row-wise equality of two bool ``[N, E]`` assignments."""
    return np.all(x_a == x_b, axis=1)


def tied_token_mask(*rankings: np.ndarray, k: int, tol: float = TIE_TOLERANCE) -> np.ndarray:
    """Bool ``[N]``, True for a token tied (margin <= tol) on any of the given rankings.

    A token can be decided by the tie rule under one ranking and not the other, so a token
    counts as tied if either does, because undercounting either ranking alone understates how
    much of the comparison the lowest-index tie-break, rather than the affinities, decided.
    """
    mask = None
    for y in rankings:
        row_tied = alflb.tie_margins(y, k) <= tol
        mask = row_tied if mask is None else (mask | row_tied)
    return mask


def classify_tier(settled_at: int | None) -> str:
    """Sort a row into settled / unconverged from the mechanism, not the gap.

    There was a third tier, ``tie_slack``, for a row that failed to settle while its minimum tie
    margin sat under an absolute tolerance, on the theory that the theorem's no-ties assumption
    had failed there. It is gone because the verdict was decided by a constant with no defensible
    value: on the step-500 dump the absolute rule called two of eight layers tie-limited, while
    the same eight margins expressed in float32 steps put seven of eight below one step. A binary
    label that moves that far with the unit says less than the number it was computed from, so
    ``min_tie_margin_ulp`` is reported and no tier is derived from it.
    """
    return "settled" if settled_at is not None else "unconverged"


def _vanilla_stats(a: np.ndarray, k: int) -> tuple[float, int]:
    """Objective and max load of the unbiased, uncapacitated top-K assignment."""
    idx = alflb.top_k_map(a, k)
    objective = float(np.take_along_axis(a, idx, axis=1).sum())
    load = np.bincount(idx.ravel(), minlength=a.shape[1])
    return objective, int(load.max())


def compare(a: np.ndarray, k: int, *, eta: float, steps: int, mode: str = "annealed") -> Comparison:
    # The float width `a` arrives in, kept before the widening below, because an absolute tie
    # margin is not interpretable on its own: the same number is 69 float32 steps near affinity
    # 0.0002 and 0.017 of one step near 0.8. A caller holding float32 router affinities in a
    # float64 array must narrow them back first, or its margins are counted in the wrong unit.
    score_width = np.asarray(a).dtype
    a = np.asarray(a, dtype=np.float64)
    n, e = a.shape
    balanced_load = n * k / e
    cap0 = lp.default_cap(n, k, e)

    oracle = lp.solve(a, k, cap=cap0)
    lp_objective = oracle.objective

    vanilla_objective, vanilla_max_load = _vanilla_stats(a, k)
    span = vanilla_objective - lp_objective

    result = alflb.iterate(a, k, eta=eta, steps=steps, mode=mode)

    if mode == "annealed":
        basis = "trajectory_closest"
        basis_point = result.closest_approach
    elif mode == "deployed":
        if result.cycle_worst is not None:
            basis = "cycle_worst"
            basis_point = result.cycle_worst
        else:
            # No cycle was detected, which happens when the run settles: the settle check
            # runs before the next iteration's cycle hash is taken, so a fixed point exits
            # with no cycle and no cycle_worst. Its closest approach IS the fixed point, so
            # this is a valid row rather than an error that would abort a grid mid-write.
            basis = "trajectory_closest"
            basis_point = result.closest_approach
    else:
        raise ValueError(f"mode must be 'annealed' or 'deployed', got {mode!r}")

    basis_step = basis_point.step
    objective = basis_point.objective
    max_load = basis_point.max_load
    x_basis = basis_point.x
    bias_basis = basis_point.bias
    load_e = x_basis.sum(axis=0)

    overflow = max_load / balanced_load - 1.0

    # ALF-LB is dropless, so an infeasible iterate scores above the constrained optimum: the raw
    # difference there measures the constraint violation, not any distance from optimality.
    gap_at_default_cap = lp_objective - objective if max_load <= cap0 else float("nan")

    matched_cap = max(max_load, cap0)
    matched_lp_unconstrained = matched_cap >= vanilla_max_load

    if matched_lp_unconstrained:
        # The matched LP is unconstrained here, so nothing about balancing is left to measure.
        # gap_at_default_cap is untouched by this guard: it depends only on its own cap and
        # stays exactly computable, including the degenerate case where it is exactly 0.0.
        matched_lp_objective = float("nan")
        gap_at_matched_cap = float("nan")
        gap_at_per_expert_cap = float("nan")
    else:
        matched_result = oracle if matched_cap == cap0 else lp.solve(a, k, cap=matched_cap)
        matched_lp_objective = matched_result.objective
        gap_at_matched_cap = matched_lp_objective - objective

        per_expert_cap = np.maximum(load_e, cap0)
        per_expert_result = lp.solve(a, k, cap=per_expert_cap)
        gap_at_per_expert_cap = per_expert_result.objective - objective

    # span itself needs a second, separate guard: when the unconstrained optimum is already
    # feasible, span is the same objective computed two ways and is float noise of either
    # sign (measured -8.881784e-16 on a degenerate instance), so a non-strict threshold here
    # is a division shield, not only a semantic one. Weakening it to a strict inequality
    # raises ZeroDivisionError on exactly that instance.
    span_negligible = span <= 1e-12 * abs(vanilla_objective)
    if matched_lp_unconstrained or span_negligible:
        gap_over_span = float("nan")
    else:
        gap_over_span = gap_at_matched_cap / span

    # cycle_best is the orbit's other end: chosen structurally by the same (max_load,
    # objective) key closest_approach uses, restricted to the cycle, never by the gap
    # itself. It is scored at the SAME matched_cap as the basis point above so the gap
    # family stays recoverable by subtraction rather than confounding a tighter ceiling
    # with a better phase.
    if result.cycle_best is not None:
        cycle_best_step = result.cycle_best.step
        cycle_best_max_load = result.cycle_best.max_load
        cycle_best_objective = result.cycle_best.objective
        if matched_lp_unconstrained:
            cycle_best_gap_at_matched_cap = float("nan")
        else:
            cycle_best_gap_at_matched_cap = matched_lp_objective - cycle_best_objective
        if matched_lp_unconstrained or span_negligible:
            cycle_best_gap_over_span = float("nan")
        else:
            cycle_best_gap_over_span = cycle_best_gap_at_matched_cap / span
    else:
        cycle_best_step = None
        cycle_best_max_load = None
        cycle_best_objective = float("nan")
        cycle_best_gap_at_matched_cap = float("nan")
        cycle_best_gap_over_span = float("nan")

    excess_tokens = int(np.sum(np.maximum(load_e - cap0, 0)))

    alf_ranking = a + bias_basis
    oracle_ranking = a + oracle.capacity_duals

    # Computed once each: alf_margins feeds both the pooled tie mask and the reported margins
    # below, so a second alflb.tie_margins(alf_ranking, k) call is not needed to build either.
    alf_margins = alflb.tie_margins(alf_ranking, k)
    oracle_margins = alflb.tie_margins(oracle_ranking, k)

    # tied_tokens and untied_set_agreement pool both rankings, because agreement is a
    # statement about both objects at once. The reported margins below take alf_margins alone,
    # because pooling would make their minimum identically zero: optimal capacity duals
    # manufacture ties on the oracle side on essentially every instance.
    tied_mask = (alf_margins <= TIE_TOLERANCE) | (oracle_margins <= TIE_TOLERANCE)
    tied_tokens = int(tied_mask.sum())

    median_tie_margin = float(np.median(alf_margins))
    min_tie_margin = float(np.min(alf_margins))

    # Divide each row's margin by one float step at that row's own k-th score, because floating
    # point spacing doubles every binade, so the same absolute margin is a different number of
    # distinguishable values depending on where in the range it sits. Only the trajectory ranking
    # is converted: the oracle ranking carries float64 LP duals, and `oracle_exact_ties` already
    # counts its ties exactly, which needs no unit at all.
    kth_score = np.sort(alf_ranking, axis=1)[:, ::-1][:, k - 1]
    step = np.spacing(kth_score.astype(score_width)).astype(np.float64)
    alf_margins_ulp = alf_margins / step
    median_tie_margin_ulp = float(np.median(alf_margins_ulp))
    min_tie_margin_ulp = float(np.min(alf_margins_ulp))

    # The oracle-side zeros are not noise to discard: they measure degeneracy of the
    # instance's optimum, a property of the LP rather than of this run's convergence.
    oracle_min_margin = float(np.min(oracle_margins))
    oracle_exact_ties = int(np.sum(oracle_margins == 0.0))

    agree_mask = token_set_agreement(x_basis, oracle.x)
    set_agreement = float(agree_mask.mean())
    untied_mask = ~tied_mask
    untied_set_agreement = (
        float(agree_mask[untied_mask].mean()) if untied_mask.any() else float("nan")
    )

    if oracle.divisible:
        dual_correlation, dual_linf = dual_agreement(bias_basis, oracle.capacity_duals)
    else:
        dual_correlation, dual_linf = float("nan"), float("nan")

    tier = classify_tier(result.settled_at)

    if mode == "deployed":
        cycle_length = result.cycle_length
        band_width = result.band_width
        cycle_objective_mean = result.cycle_objective_mean
    else:
        cycle_length = None
        band_width = float("nan")
        cycle_objective_mean = float("nan")

    return Comparison(
        n=n,
        e=e,
        k=k,
        mode=mode,
        eta=eta,
        steps=steps,
        steps_run=result.steps,
        settled_at=result.settled_at,
        basis=basis,
        basis_step=basis_step,
        objective=objective,
        max_load=max_load,
        overflow=overflow,
        cycle_length=cycle_length,
        band_width=band_width,
        cycle_objective_mean=cycle_objective_mean,
        divisible=oracle.divisible,
        oracle_max_load=oracle.max_load,
        default_cap=cap0,
        lp_objective=lp_objective,
        vanilla_objective=vanilla_objective,
        vanilla_max_load=vanilla_max_load,
        balanced_load=balanced_load,
        span=span,
        gap_at_default_cap=gap_at_default_cap,
        matched_cap=matched_cap,
        matched_lp_objective=matched_lp_objective,
        gap_at_matched_cap=gap_at_matched_cap,
        gap_at_per_expert_cap=gap_at_per_expert_cap,
        gap_over_span=gap_over_span,
        matched_lp_unconstrained=matched_lp_unconstrained,
        cycle_best_step=cycle_best_step,
        cycle_best_max_load=cycle_best_max_load,
        cycle_best_objective=cycle_best_objective,
        cycle_best_gap_at_matched_cap=cycle_best_gap_at_matched_cap,
        cycle_best_gap_over_span=cycle_best_gap_over_span,
        set_agreement=set_agreement,
        untied_set_agreement=untied_set_agreement,
        tied_tokens=tied_tokens,
        median_tie_margin=median_tie_margin,
        min_tie_margin=min_tie_margin,
        median_tie_margin_ulp=median_tie_margin_ulp,
        min_tie_margin_ulp=min_tie_margin_ulp,
        oracle_min_margin=oracle_min_margin,
        oracle_exact_ties=oracle_exact_ties,
        excess_tokens=excess_tokens,
        tier=tier,
        dual_correlation=dual_correlation,
        dual_linf=dual_linf,
    )

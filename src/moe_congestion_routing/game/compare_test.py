import subprocess
import sys

import numpy as np
import pytest

from moe_congestion_routing.game.compare import (
    TIE_TOLERANCE,
    classify_tier,
    compare,
    dual_agreement,
    tied_token_mask,
    token_set_agreement,
)
from moe_congestion_routing.game.ensemble import (
    N512_E8_K2_SEP2_SEED1,
    Instance,
    affinities,
)
from moe_congestion_routing.game.lp import solve


def test_importing_compare_does_not_pull_in_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import moe_congestion_routing.game.compare, sys; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------------------------
# The theorem test: annealed ALF-LB reproduces the LP optimum on a well-separated instance.
# ---------------------------------------------------------------------------------------------


def test_annealed_alf_lb_reaches_the_lp_optimum_on_a_well_separated_instance():
    a = affinities(N512_E8_K2_SEP2_SEED1)
    c = compare(a, 2, eta=1e-2, steps=2000, mode="annealed")

    assert c.settled_at is not None
    assert c.settled_at < 2000
    assert c.basis_step == c.settled_at
    assert c.set_agreement == 1.0
    assert c.overflow == 0.0
    assert c.gap_at_matched_cap == 0.0


# ---------------------------------------------------------------------------------------------
# The deployed objective is the worst phase's, not the cycle mean.
# ---------------------------------------------------------------------------------------------


def test_deployed_objective_is_the_worst_phase_not_the_cycle_mean():
    a = affinities(N512_E8_K2_SEP2_SEED1)
    c = compare(a, 2, eta=1e-3, steps=2000, mode="deployed")

    assert c.objective == pytest.approx(875.990285, abs=1e-6)
    assert c.cycle_objective_mean == pytest.approx(876.015190, abs=1e-6)
    assert c.basis == "cycle_worst"
    assert c.basis_step == 17


def test_oracle_max_load_equals_default_cap_on_a_divisible_instance():
    """The oracle balances exactly when it can, so this column only earns its place off-grid.

    Every shape the grid runs is divisible, which is why the oracle's own max load carries no
    information there. It is reported so a non-divisible instance cannot hide the oracle's slack.
    """
    divisible = compare(affinities(Instance(512, 8, 2, 2.0, 0)), 2, eta=1e-2, steps=2000)
    assert divisible.divisible
    assert divisible.oracle_max_load == divisible.default_cap == 128

    # n*k/e = 10/3 here, so no assignment balances and the oracle stops below its ceiling.
    rng = np.random.default_rng(3)
    non_divisible = compare(rng.random((5, 3)), 2, eta=1e-2, steps=200)
    assert not non_divisible.divisible
    assert non_divisible.oracle_max_load <= non_divisible.default_cap


# ---------------------------------------------------------------------------------------------
# The feasibility identity: on a divisible instance, feasible and settled are the same event.
# ---------------------------------------------------------------------------------------------


def test_max_load_equals_default_cap_iff_settled_on_a_divisible_instance():
    a = affinities(N512_E8_K2_SEP2_SEED1)

    settled = compare(a, 2, eta=1e-2, steps=2000, mode="annealed")
    assert settled.settled_at is not None
    assert settled.max_load == settled.default_cap

    unsettled = compare(a, 2, eta=1e-2, steps=5, mode="annealed")
    assert unsettled.settled_at is None
    assert unsettled.max_load != unsettled.default_cap


# ---------------------------------------------------------------------------------------------
# An unsettled annealed row is monotone in the budget, not independent of it.
# ---------------------------------------------------------------------------------------------


def test_unsettled_annealed_row_is_monotone_in_the_budget():
    # separation=0.02 does not settle within any budget used here, so both calls stay
    # unconverged and only the monotonicity claim, not equality, is checked.
    inst = Instance(n=512, e=8, k=2, separation=0.02, seed=1)
    a = affinities(inst)
    small = compare(a, 2, eta=1e-2, steps=2000, mode="annealed")
    large = compare(a, 2, eta=1e-2, steps=4000, mode="annealed")

    assert small.settled_at is None
    assert large.settled_at is None
    assert large.max_load <= small.max_load
    assert large.basis_step >= small.basis_step


# ---------------------------------------------------------------------------------------------
# classify_tier is pure: no run needed, every constant below was measured.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "settled_at, expected",
    [(None, "unconverged"), (0, "settled"), (100, "settled")],
)
def test_classify_tier_reads_only_whether_the_run_settled(settled_at, expected):
    # `0` is here because a run that settles on its very first step is falsy, so an
    # `if settled_at:` would call it unconverged.
    assert classify_tier(settled_at) == expected


# ---------------------------------------------------------------------------------------------
# The vacuity guard fires on two of three deployed step sizes on the reference instance.
# ---------------------------------------------------------------------------------------------


def test_vacuity_guard_fires_above_the_shipped_step_size():
    a = affinities(N512_E8_K2_SEP2_SEED1)
    c_1e3 = compare(a, 2, eta=1e-3, steps=2000, mode="deployed")
    c_1e2 = compare(a, 2, eta=1e-2, steps=2000, mode="deployed")
    c_1e1 = compare(a, 2, eta=1e-1, steps=2000, mode="deployed")

    assert c_1e3.vanilla_max_load == 135
    assert c_1e3.max_load == 130
    assert not c_1e3.matched_lp_unconstrained
    assert np.isfinite(c_1e3.gap_at_matched_cap)

    assert c_1e2.max_load == 137
    assert c_1e2.matched_lp_unconstrained
    assert np.isnan(c_1e2.gap_at_default_cap)
    assert np.isnan(c_1e2.gap_at_matched_cap)
    assert np.isnan(c_1e2.gap_at_per_expert_cap)
    assert np.isnan(c_1e2.gap_over_span)

    assert c_1e1.max_load == 181
    assert c_1e1.matched_lp_unconstrained
    assert np.isnan(c_1e1.gap_over_span)


# ---------------------------------------------------------------------------------------------
# gap_over_span is the headline, whereas the oracle-relative figure is not.
# ---------------------------------------------------------------------------------------------


def test_gap_over_span_is_the_headline_not_the_oracle_relative_figure():
    a = affinities(N512_E8_K2_SEP2_SEED1)
    c = compare(a, 2, eta=1e-3, steps=2000, mode="deployed")

    assert c.gap_over_span == pytest.approx(0.9852, abs=1e-3)
    assert c.gap_at_matched_cap / c.lp_objective == pytest.approx(1.91e-04, abs=1e-6)


# ---------------------------------------------------------------------------------------------
# The tie-tolerance guard: no margin falls in the empty band, and the tolerance counts more.
# ---------------------------------------------------------------------------------------------


def test_tie_tolerance_guard_and_tied_tokens_count_more_than_exact_zero():
    inst = Instance(n=2048, e=64, k=8, separation=2.0, seed=1)
    a = affinities(inst)
    oracle = solve(a, 8)

    from moe_congestion_routing.game.alflb import tie_margins

    margins = tie_margins(a + oracle.capacity_duals, 8)
    assert not np.any((margins > 1e-15) & (margins <= 1e-9))

    exact_zero = int(np.sum(margins == 0))
    tied = tied_token_mask(a + oracle.capacity_duals, k=8)
    assert int(tied.sum()) > exact_zero


# ---------------------------------------------------------------------------------------------
# dual_agreement is unchanged by a constant shift on either input: the gauge convention.
# ---------------------------------------------------------------------------------------------


def test_dual_agreement_is_invariant_to_an_additive_constant():
    rng = np.random.default_rng(0)
    bias = rng.standard_normal(8)
    duals = rng.standard_normal(8)

    baseline = dual_agreement(bias, duals)
    shifted_bias = dual_agreement(bias + 7.0, duals)
    shifted_duals = dual_agreement(bias, duals - 3.0)

    np.testing.assert_allclose(baseline, shifted_bias)
    np.testing.assert_allclose(baseline, shifted_duals)


# ---------------------------------------------------------------------------------------------
# gap_at_default_cap is NaN exactly when the basis is infeasible for the default cap.
# ---------------------------------------------------------------------------------------------


def test_gap_at_default_cap_nan_exactly_when_infeasible():
    a = affinities(N512_E8_K2_SEP2_SEED1)

    infeasible = compare(a, 2, eta=1e-2, steps=5, mode="annealed")
    assert infeasible.max_load > infeasible.default_cap
    assert np.isnan(infeasible.gap_at_default_cap)

    feasible = compare(a, 2, eta=1e-2, steps=2000, mode="annealed")
    assert feasible.max_load <= feasible.default_cap
    assert np.isfinite(feasible.gap_at_default_cap)


# ---------------------------------------------------------------------------------------------
# compare returns NaN for both dual fields on a non-divisible instance.
# ---------------------------------------------------------------------------------------------


def test_dual_fields_are_nan_on_a_non_divisible_instance():
    inst = Instance(n=5, e=3, k=2, separation=2.0, seed=0)
    a = affinities(inst)
    c = compare(a, 2, eta=1e-2, steps=200, mode="annealed")

    assert not c.divisible
    assert np.isnan(c.dual_correlation)
    assert np.isnan(c.dual_linf)


# ---------------------------------------------------------------------------------------------
# token_set_agreement and tied_token_mask, as plain functions.
# ---------------------------------------------------------------------------------------------


def test_token_set_agreement_row_wise_equality():
    x_a = np.array([[True, False], [False, True]])
    x_b = np.array([[True, False], [True, False]])
    np.testing.assert_array_equal(token_set_agreement(x_a, x_b), [True, False])


def test_tied_token_mask_unions_across_rankings():
    # Row 0 is tied on the first ranking only, row 1 on the second only, row 2 on neither.
    y1 = np.array([[1.0, 1.0, 0.0], [1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
    y2 = np.array([[1.0, 0.5, 0.0], [1.0, 1.0, 0.0], [1.0, 0.5, 0.0]])
    mask = tied_token_mask(y1, y2, k=1, tol=TIE_TOLERANCE + 0.4)
    np.testing.assert_array_equal(mask, [True, True, False])


# ---------------------------------------------------------------------------------------------
# The reported margins read the trajectory ranking alone: pooling with the oracle ranking makes
# the minimum identically zero, because optimal capacity duals manufacture ties on the oracle
# side on essentially every instance, so a pooled minimum would carry no information at all.
# ---------------------------------------------------------------------------------------------


def test_reported_margins_read_the_trajectory_ranking_not_the_pooled_minimum():
    inst = Instance(n=512, e=8, k=2, separation=2.0, seed=0)
    a = affinities(inst)
    c = compare(a, 2, eta=1e-2, steps=2000, mode="annealed")

    assert c.min_tie_margin > 0
    assert c.oracle_min_margin == 0.0
    assert c.oracle_exact_ties == 1


# ---------------------------------------------------------------------------------------------
# Tie margins in units of the arithmetic that produced the scores.
# ---------------------------------------------------------------------------------------------


def test_margin_in_ulp_tracks_the_width_the_caller_passed():
    """The same values in float32 and in float64 give the same absolute margin, and a ULP count
    differing by the ratio of the two widths' precision.

    This is the whole point of the column: an absolute margin cannot say whether a decision was
    close, because how many representable values fit inside it depends on the arithmetic.
    """
    inst = Instance(n=512, e=8, k=2, separation=2.0, seed=0)
    a64 = affinities(inst)
    a32 = a64.astype(np.float32).astype(np.float64).astype(np.float32)

    c64 = compare(a64, 2, eta=1e-2, steps=200, mode="annealed")
    c32 = compare(a32, 2, eta=1e-2, steps=200, mode="annealed")

    # float64 carries 29 more mantissa bits than float32, so a fixed gap spans about 2**29 times
    # as many representable values there. Asserted as a wide band rather than a point, because the
    # two runs rank slightly different values and the minimum is taken over different rows.
    assert c64.min_tie_margin_ulp / c32.min_tie_margin_ulp > 1e6
    assert c32.min_tie_margin_ulp > 1.0
    assert c32.median_tie_margin_ulp > c32.min_tie_margin_ulp


def test_margin_in_ulp_is_the_absolute_margin_divided_by_a_float32_step():
    """Pins the unit, not merely that the column moves with the margin.

    The k-th score is not reported, so the check brackets it: the ratio must lie between the
    margin divided by the largest float32 step anywhere in the score range and the margin divided
    by the smallest. The bracket is four orders of magnitude wide and still fails any conversion
    that used float64 steps, no conversion at all, or a step from the wrong quantity.
    """
    inst = Instance(n=256, e=8, k=2, separation=2.0, seed=3)
    a = affinities(inst).astype(np.float32)
    c = compare(a, 2, eta=1e-2, steps=200, mode="annealed")

    # Scores are affinity plus a bias of magnitude at most eta * steps, so the reachable range is
    # bounded by the affinity range widened by that, and the step is monotone in the magnitude.
    reach = 1e-2 * 200
    coarsest = float(np.spacing(np.float32(abs(a).max() + reach)))
    finest = float(np.spacing(np.float32(abs(a).min())))

    assert c.min_tie_margin > 0
    assert c.min_tie_margin / coarsest <= c.min_tie_margin_ulp <= c.min_tie_margin / finest


# ---------------------------------------------------------------------------------------------
# gap_at_default_cap is exactly computable and reported even when the run is already optimal,
# whereas gap_over_span is voided by the span guard on the same row rather than by the vacuity
# guard: this instance's matched LP is constrained, so only the division shield applies.
# ---------------------------------------------------------------------------------------------


def test_degenerate_row_reports_its_gap_and_voids_only_the_span():
    inst = Instance(n=8, e=4, k=1, separation=5.0, seed=18)
    a = affinities(inst)
    c = compare(a, 1, eta=1e-2, steps=2000, mode="annealed")

    assert c.tier == "settled"
    assert c.gap_at_default_cap == 0.0
    assert isinstance(c.gap_at_default_cap, float)
    assert np.isnan(c.gap_over_span)


# ---------------------------------------------------------------------------------------------
# A deployed run that settles exits with no cycle detected, because the settle check runs
# before the next iteration's cycle hash is taken. That is a valid row, not a ValueError.
# ---------------------------------------------------------------------------------------------


def test_deployed_run_that_settles_produces_a_row_not_an_error():
    a = np.array(
        [
            [0.9, 0.8, 0.1, 0.0],
            [0.9, 0.8, 0.1, 0.0],
            [0.0, 0.1, 0.8, 0.9],
            [0.0, 0.1, 0.8, 0.9],
        ]
    )
    c = compare(a, 2, eta=0.1, steps=5, mode="deployed")

    assert c.tier == "settled"
    assert c.basis == "trajectory_closest"
    assert c.cycle_best_step is None
    assert c.cycle_best_max_load is None
    assert np.isnan(c.cycle_best_objective)
    assert np.isnan(c.cycle_best_gap_at_matched_cap)
    assert np.isnan(c.cycle_best_gap_over_span)


# ---------------------------------------------------------------------------------------------
# Both cycle ends land at the values the spec measured, at the SAME matched capacity, and the
# gap family is recoverable by subtraction from the unprefixed columns alone.
# ---------------------------------------------------------------------------------------------


def test_cycle_best_and_cycle_worst_at_the_shared_yardstick_separation_2():
    a = affinities(Instance(n=512, e=8, k=2, separation=2.0, seed=0))
    c = compare(a, 2, eta=1e-3, steps=2000, mode="deployed")

    assert c.matched_lp_objective == pytest.approx(877.104203, abs=1e-6)
    assert c.span == pytest.approx(0.344926, abs=1e-6)

    assert c.basis == "cycle_worst"
    assert c.basis_step == 20
    assert c.max_load == 130
    assert c.objective == pytest.approx(876.948953, abs=1e-6)
    assert c.gap_at_matched_cap == pytest.approx(0.155251, abs=1e-6)
    assert c.gap_over_span == pytest.approx(0.450099, abs=1e-6)

    assert c.cycle_best_step == 22
    assert c.cycle_best_max_load == 129
    assert c.cycle_best_objective == pytest.approx(876.972394, abs=1e-6)
    assert c.cycle_best_gap_at_matched_cap == pytest.approx(0.131810, abs=1e-6)
    assert c.cycle_best_gap_over_span == pytest.approx(0.382140, abs=1e-6)

    assert c.cycle_best_gap_at_matched_cap == pytest.approx(
        c.gap_at_matched_cap - (c.cycle_best_objective - c.objective), abs=1e-9
    )


def test_cycle_best_and_cycle_worst_at_the_shared_yardstick_separation_0_2():
    # A second instance so the identity above is not a fixture coincidence.
    a = affinities(Instance(n=512, e=8, k=2, separation=0.2, seed=0))
    c = compare(a, 2, eta=1e-3, steps=2000, mode="deployed")

    assert c.matched_lp_objective == pytest.approx(569.187007, abs=1e-6)
    assert c.span == pytest.approx(0.059842, abs=1e-6)

    assert c.basis_step == 6
    assert c.max_load == 131
    assert c.objective == pytest.approx(569.134450, abs=1e-6)
    assert c.gap_over_span == pytest.approx(0.878268, abs=1e-6)

    assert c.cycle_best_step == 8
    assert c.cycle_best_max_load == 130
    assert c.cycle_best_objective == pytest.approx(569.144318, abs=1e-6)
    assert c.cycle_best_gap_over_span == pytest.approx(0.713353, abs=1e-6)

    assert c.cycle_best_gap_at_matched_cap == pytest.approx(
        c.gap_at_matched_cap - (c.cycle_best_objective - c.objective), abs=1e-9
    )


# ---------------------------------------------------------------------------------------------
# An annealed row has no cycle, so cycle_best_* stay unset even though the run's basis is
# trajectory_closest for a different reason than the settled-deployed case above.
# ---------------------------------------------------------------------------------------------


def test_annealed_row_has_no_cycle_best():
    a = affinities(N512_E8_K2_SEP2_SEED1)
    c = compare(a, 2, eta=1e-2, steps=2000, mode="annealed")

    assert c.basis == "trajectory_closest"
    assert c.cycle_best_step is None
    assert c.cycle_best_max_load is None
    assert np.isnan(c.cycle_best_objective)
    assert np.isnan(c.cycle_best_gap_at_matched_cap)
    assert np.isnan(c.cycle_best_gap_over_span)

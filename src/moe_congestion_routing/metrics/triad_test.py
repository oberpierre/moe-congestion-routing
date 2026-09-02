import math

import numpy
import pytest

from moe_congestion_routing.metrics import triad
from moe_congestion_routing.metrics.probe_comparison import project_out
from moe_congestion_routing.metrics.triad import priced_unit, triad_rows


def _numeric_fields_are_nan(row) -> bool:
    return all(
        math.isnan(value)
        for value in (
            row.rho,
            row.c_x,
            row.c_y,
            row.kappa,
            row.kappa_boot_low,
            row.kappa_boot_high,
            row.kappa_boot_undefined,
        )
    )


def _by_pair_and_units(rows, pair, unit_x, unit_y):
    for row in rows:
        if row.pair == pair and row.unit_x == unit_x and row.unit_y == unit_y:
            return row
    raise AssertionError(f"no row for {pair!r} {unit_x!r} {unit_y!r}")


def _identity(rows, asset):
    return _by_pair_and_units(rows, f"identity:{asset}", triad.PRIMARY_UNIT[asset], "")


def test_three_kappa_agree_and_the_identity_matches_when_prices_share_one_population():
    """`p_bar` plus independent per-asset noise: every pairwise `kappa` and every identity
    `kappa` must recover the true value, because that is exactly the model the ratio and
    the triad identity are both derived under.

    `bias` carries its own small independent deviation from `p_bar` rather than being `p_bar`
    exactly, because `kappa_true == 1` sits exactly on the `rho >= c_x * c_y` refusal boundary
    (`kappa == 1` means `rho == c_x * c_y` by the same identity that defines the refusal), so a
    point estimate built at that exact true value is refused on about half of all seeds by
    sampling noise alone. `kappa_true = corr(p_bar, p_bar + 0.3 * noise) ~ 0.958` here.
    """
    rng = numpy.random.default_rng(0)
    n = 64
    p_bar = rng.normal(size=n)
    bias = p_bar + 0.3 * rng.normal(size=n)

    duals_tail = p_bar + 0.3 * rng.normal(size=n)
    duals_strided = p_bar + 0.4 * rng.normal(size=n)
    duals_spread_u0 = p_bar + 0.35 * rng.normal(size=n)
    duals_spread_u1 = p_bar + 0.5 * rng.normal(size=n)

    rows = triad_rows(
        "test",
        5,
        bias,
        duals_tail,
        duals_strided,
        duals_spread_u0,
        duals_spread_u1,
        resamples=500,
        seed=1,
    )
    assert len(rows) == 8

    ts = _by_pair_and_units(rows, "tail-strided", "u0", "u1")
    tp = _by_pair_and_units(rows, "tail-spread", "u0", "u0")
    sp = _by_pair_and_units(rows, "strided-spread", "u1", "u0")
    for row in (ts, tp, sp):
        assert row.kappa == pytest.approx(0.958, abs=0.1)
        assert row.kappa_boot_low < row.kappa < row.kappa_boot_high

    for asset in ("tail", "strided", "spread"):
        identity = _identity(rows, asset)
        assert identity.kappa == pytest.approx(0.958, abs=0.15)
        assert identity.rho == pytest.approx(1.0, abs=0.3)  # this is s_x, not a pairwise rho


def test_a_pairwise_pattern_no_single_population_explains_breaks_the_identity():
    """Build three prices whose pairwise correlations cannot come from one common `p_bar`:
    tail correlates positively with both strided and spread through a shared factor `a`, while
    an independent factor `b` pulls strided and spread apart, making that one pairing
    negative. No rank-1 `s_x` triple reproduces all three signs at once, so every `s_x` must
    come back NaN. The two tail-anchored pairwise `kappa` look fine in isolation, which is
    exactly why the identity, which reads all three pairings together, is needed.
    """
    rng = numpy.random.default_rng(2)
    n = 64
    a = rng.normal(size=n)
    b = 2.0 * rng.normal(size=n)  # dominates strided/spread's shared component

    bias = a
    duals_tail = a
    duals_strided = a - b
    duals_spread_u0 = a + b
    duals_spread_u1 = a + b

    rows = triad_rows(
        "test",
        5,
        bias,
        duals_tail,
        duals_strided,
        duals_spread_u0,
        duals_spread_u1,
        resamples=500,
        seed=3,
    )

    ts = _by_pair_and_units(rows, "tail-strided", "u0", "u1")
    tp = _by_pair_and_units(rows, "tail-spread", "u0", "u0")
    sp = _by_pair_and_units(rows, "strided-spread", "u1", "u0")
    assert ts.rho > 0 and tp.rho > 0 and sp.rho < 0
    # tail-strided and tail-spread each look internally consistent in isolation...
    assert not math.isnan(ts.kappa)
    assert not math.isnan(tp.kappa)
    # ...but strided-spread cannot be, and the identity catches all three at once.
    assert math.isnan(sp.kappa)
    assert math.isnan(_identity(rows, "tail").kappa)
    assert math.isnan(_identity(rows, "strided").kappa)
    assert math.isnan(_identity(rows, "spread").kappa)


def test_a_refused_unit_emits_nan_rows_rather_than_being_dropped():
    """`priced_unit` returning ``None`` for one asset must NaN every row that needed it and
    leave every row that did not untouched, per the "refuse loudly, never drop" rule.
    """
    rng = numpy.random.default_rng(4)
    n = 64
    p_bar = rng.normal(size=n)
    bias = p_bar
    duals_strided = p_bar + 0.3 * rng.normal(size=n)
    duals_spread_u0 = p_bar + 0.3 * rng.normal(size=n)
    duals_spread_u1 = p_bar + 0.3 * rng.normal(size=n)

    rows = triad_rows(
        "test",
        5,
        bias,
        None,
        duals_strided,
        duals_spread_u0,
        duals_spread_u1,
        resamples=200,
        seed=5,
    )
    assert len(rows) == 8

    for pair, unit_x, unit_y in [
        ("tail-strided", "u0", "u1"),
        ("tail-spread", "u0", "u0"),
        ("tail-spread", "u0", "u1"),
    ]:
        assert _numeric_fields_are_nan(_by_pair_and_units(rows, pair, unit_x, unit_y))
    for asset in ("tail", "strided", "spread"):
        assert _numeric_fields_are_nan(_identity(rows, asset))

    # strided-spread never touches the refused tail unit, so both its rows stay finite.
    sp = _by_pair_and_units(rows, "strided-spread", "u1", "u0")
    sp_robust = _by_pair_and_units(rows, "strided-spread", "u1", "u1")
    assert not math.isnan(sp.rho)
    assert not math.isnan(sp_robust.rho)


def test_spread_robustness_rows_are_distinguishable_by_unit_y_not_by_pair():
    rows = triad_rows(
        "test",
        5,
        numpy.zeros(4),
        numpy.array([1.0, 2.0, 3.0, 4.0]),
        numpy.array([4.0, 3.0, 2.0, 1.0]),
        numpy.array([1.0, 3.0, 2.0, 4.0]),
        numpy.array([2.0, 4.0, 1.0, 3.0]),
        resamples=50,
        seed=6,
    )
    primary = _by_pair_and_units(rows, "tail-spread", "u0", "u0")
    robust = _by_pair_and_units(rows, "tail-spread", "u0", "u1")
    assert primary.pair == robust.pair == "tail-spread"
    assert primary.unit_y != robust.unit_y


def test_priced_unit_refuses_a_dead_expert_and_returns_none():
    tokens, experts, topk = 8, 4, 2
    routing_map = numpy.zeros((tokens, experts), dtype=bool)
    for i in range(tokens):
        routing_map[i, i % 3] = True
        routing_map[i, (i + 1) % 3] = True
    affinities = numpy.random.default_rng(7).normal(size=(tokens, experts))

    screen, duals = priced_unit(routing_map, affinities, topk)
    assert not screen.admissible
    assert duals is None


def test_priced_unit_prices_a_balanced_batch():
    tokens, experts, topk = 8, 4, 2
    routing_map = numpy.zeros((tokens, experts), dtype=bool)
    for i in range(tokens):
        routing_map[i, i % experts] = True
        routing_map[i, (i + 1) % experts] = True
    affinities = numpy.random.default_rng(8).normal(size=(tokens, experts))

    screen, duals = priced_unit(routing_map, affinities, topk)
    assert screen.admissible
    assert duals is not None
    assert duals.shape == (experts,)


def test_projecting_the_composition_axis_out_reconciles_the_triad():
    """The composition-axis correction, end to end, on one constructed instance.

    Three prices are built as ``p_bar + alpha * axis + independent noise``, with ``alpha``
    differing per asset the way tail/strided/spread's code-marker counts differ in the real data
    (strided's `alpha` sits between tail's and spread's), plus a bias carrying its own axis
    component the way the stored bias does. Before projection the three pairwise `kappa` disagree
    and the strided identity's `s` exceeds 1, both of which the real, uncorrected triad shows.
    Projecting `axis` out of every price *and* out of `bias` must reconcile all three `kappa` and
    bring every `s` back to `<= 1`. Projecting it out of the prices only, leaving `bias` raw, must
    fail to reconcile them, because that asymmetry is exactly what the correction forbids: it
    would leave bias's own axis component correlating with nothing.
    """
    rng = numpy.random.default_rng(0)
    n = 64
    p_bar = rng.normal(size=n)
    axis = rng.normal(size=n)  # the composition confound, independent of p_bar
    noise = 0.15

    # bias's own independent deviation (beyond its axis component) keeps kappa_true comfortably
    # below 1 after correction too, because kappa_true == 1 sits exactly on the rho >= c_x * c_y
    # refusal boundary and a point estimate built there is refused by sampling noise about half
    # the time.
    bias = p_bar + 1.0 * axis + 0.2 * rng.normal(size=n)
    tail = p_bar + 2.0 * axis + noise * rng.normal(size=n)
    strided = p_bar + 1.0 * axis + noise * rng.normal(size=n)  # between tail and spread on axis
    spread = p_bar + 0.0 * axis + noise * rng.normal(size=n)

    def _kappas_and_max_s(rows):
        ts = _by_pair_and_units(rows, "tail-strided", "u0", "u1")
        tp = _by_pair_and_units(rows, "tail-spread", "u0", "u0")
        sp = _by_pair_and_units(rows, "strided-spread", "u1", "u0")
        max_s = max(_identity(rows, a).rho for a in ("tail", "strided", "spread"))
        return (ts.kappa, tp.kappa, sp.kappa), max_s

    before_rows = triad_rows("t", 1, bias, tail, strided, spread, spread, resamples=500, seed=1)
    before_kappas, before_max_s = _kappas_and_max_s(before_rows)
    # "Disagree" includes a refusal: one pairing violating rho >= c_x * c_y (kappa > 1, caught by
    # the parameter-free refusal and reported as NaN) is the sharpest form of three kappa
    # disagreeing.
    assert not all(k == pytest.approx(1.0, abs=0.1) for k in before_kappas)
    assert before_max_s > 1.0  # s_strided exceeds 1

    projected_bias = project_out(bias, axis)
    projected_tail = project_out(tail, axis)
    projected_strided = project_out(strided, axis)
    projected_spread = project_out(spread, axis)

    after_rows = triad_rows(
        "t",
        1,
        projected_bias,
        projected_tail,
        projected_strided,
        projected_spread,
        projected_spread,
        resamples=500,
        seed=2,
    )
    after_kappas, after_max_s = _kappas_and_max_s(after_rows)
    assert all(k == pytest.approx(0.976, abs=0.05) for k in after_kappas)  # three agreeing kappa
    assert after_max_s <= 1.0

    # Projected prices, raw bias: the model's shared "shape" no longer matches between the two
    # sides, so kappa is depressed well below the true value even though the three still agree
    # with each other, which is exactly why agreement alone is not the test.
    asym_rows = triad_rows(
        "t",
        1,
        bias,
        projected_tail,
        projected_strided,
        projected_spread,
        projected_spread,
        resamples=500,
        seed=3,
    )
    asym_kappas, _ = _kappas_and_max_s(asym_rows)
    assert not all(k == pytest.approx(0.976, abs=0.05) for k in asym_kappas)

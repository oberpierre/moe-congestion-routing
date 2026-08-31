import math

import numpy
import pytest

from moe_congestion_routing.metrics import triad
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
    """
    rng = numpy.random.default_rng(0)
    n = 64
    p_bar = rng.normal(size=n)
    bias = p_bar  # kappa = 1 by construction

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
        assert row.kappa == pytest.approx(1.0, abs=0.25)
        assert row.kappa_boot_low < row.kappa < row.kappa_boot_high

    for asset in ("tail", "strided", "spread"):
        identity = _identity(rows, asset)
        assert identity.kappa == pytest.approx(1.0, abs=0.25)
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

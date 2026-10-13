import math

import numpy
import pytest

from moe_congestion_routing.metrics import triad
from moe_congestion_routing.metrics.dual_store import DualEntry
from moe_congestion_routing.metrics.probe_comparison import project_out
from moe_congestion_routing.metrics.triad import (
    TriadAssets,
    priced_unit,
    trajectory_cells,
    triad_rows,
)


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


# ---------------------------------------------------------------------------------------------
# trajectory_cells: fixtures are plain dicts standing in for `dual_store.read_dual_store` /
# `read_bias_store`'s own return shape, so no store file is ever written here.
# ---------------------------------------------------------------------------------------------

_ASSETS = TriadAssets(tail="tail_asset", strided="strided_asset", spread="spread_asset")


def _cell_duals(
    run_id,
    layer,
    step,
    *,
    tail=None,
    strided=None,
    spread0=None,
    spread1=None,
    axis=None,
    inadmissible=(),
):
    """The subset of one cell's five possible dual-store keys the caller actually supplies,
    matching how a real store often has some units refused or simply unpriced. Each supplied
    vector becomes a `DualEntry` with `admissible=True` unless its role name (``"tail"``,
    ``"strided"``, ``"spread0"``, ``"spread1"`` or ``"axis"``) is listed in `inadmissible`, which
    matches a screen-refused-but-still-priced row: the store holds a real vector either way.
    """
    out = {}
    if tail is not None:
        out[(run_id, _ASSETS.tail, "u0", layer, step)] = DualEntry(
            tail, admissible="tail" not in inadmissible
        )
    if strided is not None:
        out[(run_id, _ASSETS.strided, "u1", layer, step)] = DualEntry(
            strided, admissible="strided" not in inadmissible
        )
    if spread0 is not None:
        out[(run_id, _ASSETS.spread, "u0", layer, step)] = DualEntry(
            spread0, admissible="spread0" not in inadmissible
        )
    if spread1 is not None:
        out[(run_id, _ASSETS.spread, "u1", layer, step)] = DualEntry(
            spread1, admissible="spread1" not in inadmissible
        )
    if axis is not None:
        out[(run_id, _ASSETS.strided, "u0", layer, step)] = DualEntry(
            axis, admissible="axis" not in inadmissible
        )
    return out


class _CountingMapping(dict):
    """Counts `.get()` calls, standing in for an already-parsed store so a test can show one
    parsed mapping serves both variants rather than being reparsed per variant."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_calls = 0

    def get(self, key, default=None):
        self.get_calls += 1
        return super().get(key, default)


def test_the_seed_rule_reduces_to_run_triad_pys_formula_at_step_zero():
    rng = numpy.random.default_rng(0)
    n = 8
    vectors_by_run = {run_id: tuple(rng.normal(size=n) for _ in range(5)) for run_id in ("a", "b")}
    duals: dict = {}
    bias: dict = {}
    for run_id, (bias_vec, tail, strided, spread0, spread1) in vectors_by_run.items():
        duals.update(
            _cell_duals(run_id, 3, 0, tail=tail, strided=strided, spread0=spread0, spread1=spread1)
        )
        bias[(run_id, 3, 0)] = bias_vec

    cells = trajectory_cells(duals, bias, assets=_ASSETS, project_code_axis=False, resamples=50)
    assert {c.run_id for c in cells} == {"a", "b"}
    for cell in cells:
        run_index = 0 if cell.run_id == "a" else 500
        bias_vec, tail, strided, spread0, spread1 = vectors_by_run[cell.run_id]
        seed = 1000 * 3 + run_index
        expected = tuple(
            triad_rows(
                cell.run_id, 3, bias_vec, tail, strided, spread0, spread1, resamples=50, seed=seed
            )
        )
        # NaN fields (an undefined kappa) make plain tuple equality fail even when every field
        # agrees, since NaN != NaN, so each field is compared with NaN treated as equal to NaN.
        for got, want in zip(cell.rows, expected, strict=True):
            for field in got._fields:
                g, w = getattr(got, field), getattr(want, field)
                assert g == w or (isinstance(g, float) and math.isnan(g) and math.isnan(w))


def test_a_missing_bias_row_refuses_its_whole_step_with_no_row_emitted():
    n = 8
    vec = numpy.arange(n, dtype=float) + 1.0
    duals = _cell_duals("a", 3, 0, tail=vec, strided=vec, spread0=vec, spread1=vec)
    cells = trajectory_cells(duals, {}, assets=_ASSETS, project_code_axis=False, resamples=10)
    assert cells == []


def test_refused_units_names_the_pairing_role_missing_a_price():
    n = 8
    vec = numpy.arange(n, dtype=float) + 1.0
    duals = _cell_duals("a", 3, 0, strided=vec, spread0=vec, spread1=vec)  # tail missing
    bias = {("a", 3, 0): vec}
    [cell] = trajectory_cells(duals, bias, assets=_ASSETS, project_code_axis=False, resamples=10)
    assert cell.refused_units == "tail"


def test_a_pairing_units_admissible_false_refuses_it_exactly_like_a_missing_row():
    """The two refusal kinds a pairing role can hit, `admissible=False` and no row at all, must be
    indistinguishable to `trajectory_cells`: this is the half of the rule that stops a screen
    refusal from being silently used as though it had passed, which is what the axis's own
    exemption (the sibling assertion below) must NOT do for a pairing role.
    """
    rng = numpy.random.default_rng(2)
    n = 8
    bias_vec, tail, strided, spread0, spread1 = (rng.normal(size=n) for _ in range(5))
    bias = {("a", 3, 0): bias_vec}

    refused = _cell_duals(
        "a",
        3,
        0,
        tail=tail,
        strided=strided,
        spread0=spread0,
        spread1=spread1,
        inadmissible=("strided",),
    )
    missing = _cell_duals("a", 3, 0, tail=tail, spread0=spread0, spread1=spread1)  # strided absent

    [refused_cell] = trajectory_cells(
        refused, bias, assets=_ASSETS, project_code_axis=False, resamples=50
    )
    [missing_cell] = trajectory_cells(
        missing, bias, assets=_ASSETS, project_code_axis=False, resamples=50
    )
    assert refused_cell.refused_units == "strided"
    # NaN fields (an undefined rho/kappa on every row that touched the refused unit) make plain
    # tuple equality fail even when every field agrees, since NaN != NaN.
    for got, want in zip(refused_cell.rows, missing_cell.rows, strict=True):
        for field in got._fields:
            g, w = getattr(got, field), getattr(want, field)
            assert g == w or (isinstance(g, float) and math.isnan(g) and math.isnan(w))


def test_the_axis_is_used_despite_having_come_from_an_inadmissible_row():
    """The store no longer nulls a screen-refused row, so the axis key can carry a real vector
    even though the row it was read from was `admissible=False`. `trajectory_cells` must use it
    regardless, which is the entire reason a refused cell is priced instead of NaN-filled."""
    rng = numpy.random.default_rng(1)
    n = 8
    bias_vec, tail, strided, spread0, spread1, axis = (rng.normal(size=n) for _ in range(6))
    duals = _cell_duals(
        "a",
        3,
        0,
        tail=tail,
        strided=strided,
        spread0=spread0,
        spread1=spread1,
        axis=axis,
        inadmissible=("axis",),
    )
    bias = {("a", 3, 0): bias_vec}

    uncorrected = trajectory_cells(
        duals, bias, assets=_ASSETS, project_code_axis=False, resamples=50
    )
    corrected = trajectory_cells(duals, bias, assets=_ASSETS, project_code_axis=True, resamples=50)
    assert len(uncorrected) == 1 and len(corrected) == 1
    assert corrected[0].refused_units == ""  # the axis's own refusal never reaches refused_units
    assert corrected[0].rows != uncorrected[0].rows


def test_project_code_axis_raises_when_every_cell_is_missing_its_axis():
    n = 8
    vec = numpy.arange(n, dtype=float) + 1.0
    duals = _cell_duals("a", 3, 0, tail=vec, strided=vec, spread0=vec, spread1=vec)  # no axis
    bias = {("a", 3, 0): vec}
    with pytest.raises(ValueError, match="axis"):
        trajectory_cells(duals, bias, assets=_ASSETS, project_code_axis=True, resamples=10)


def test_one_cell_missing_its_axis_is_refused_not_every_cell():
    n = 8
    vec = numpy.arange(n, dtype=float) + 1.0
    axis = vec[::-1] + 1.0
    duals = {
        **_cell_duals("a", 3, 0, tail=vec, strided=vec, spread0=vec, spread1=vec),  # no axis
        **_cell_duals("a", 4, 0, tail=vec, strided=vec, spread0=vec, spread1=vec, axis=axis),
    }
    bias = {("a", 3, 0): vec, ("a", 4, 0): vec}
    cells = trajectory_cells(duals, bias, assets=_ASSETS, project_code_axis=True, resamples=10)
    assert {c.layer for c in cells} == {4}  # layer 3's axis-less cell is dropped, not raised on


def test_two_variants_come_from_one_already_parsed_store():
    n = 8
    vec = numpy.arange(n, dtype=float) + 1.0
    duals = _CountingMapping(
        _cell_duals("a", 3, 0, tail=vec, strided=vec, spread0=vec, spread1=vec, axis=vec[::-1])
    )
    bias = {("a", 3, 0): vec}

    uncorrected = trajectory_cells(
        duals, bias, assets=_ASSETS, project_code_axis=False, resamples=10
    )
    calls_after_first = duals.get_calls
    corrected = trajectory_cells(duals, bias, assets=_ASSETS, project_code_axis=True, resamples=10)

    assert len(uncorrected) == 1 and len(corrected) == 1
    # The corrected pass costs exactly one lookup more than the uncorrected one, which is the
    # axis and nothing else. `> calls_after_first` would be true of any second call under any
    # implementation, so it constrained nothing.
    assert calls_after_first == 4
    assert duals.get_calls - calls_after_first == 5
    # And the two variants are genuinely different, so neither was silently skipped.
    assert uncorrected[0].projected is False and corrected[0].projected is True
    assert uncorrected[0].rows[0].kappa != corrected[0].rows[0].kappa


# ---------------------------------------------------------------------------------------------
# The driver. Testing `scripts/run_triad_trajectory.py`.
# ---------------------------------------------------------------------------------------------


def _write_trajectory_stores(tmp_path, num_experts=8):
    """A minimal dual and bias store holding one full cell: the four pairing units and the axis."""
    import csv as _csv

    from moe_congestion_routing.metrics.dual_store import bias_fields, dual_fields

    assets = TriadAssets(tail="tail-asset", strided="strided-asset", spread="spread-asset")
    units = [
        (assets.tail, "u0"),
        (assets.strided, "u1"),
        (assets.spread, "u0"),
        (assets.spread, "u1"),
        (assets.strided, "u0"),
    ]
    duals_path = tmp_path / "d.csv"
    fields = dual_fields(num_experts)
    with duals_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (asset, unit) in enumerate(units):
            row = dict.fromkeys(fields, "")
            row.update(
                run_id="a",
                asset=asset,
                unit=unit,
                layer=3,
                step=0,
                status="ok",
                detail="",
                admissible="True",
                max_load_over_balanced="1.0",
                dead_experts="0",
                token_sha256="t",
                dump_path="d.npz",
                score_function="sigmoid",
            )
            for e in range(num_experts):
                row[f"dual_{e}"] = str(float((e + 1) * (i + 1) % 7) + 0.5 * e)
            w.writerow(row)

    bias_path = tmp_path / "b.csv"
    bfields = bias_fields(num_experts)
    with bias_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=bfields)
        w.writeheader()
        row = dict.fromkeys(bfields, "")
        row.update(
            run_id="a", layer=3, step=0, status="ok", detail="", asset=assets.tail, token_sha256="t"
        )
        for e in range(num_experts):
            row[f"bias_{e}"] = str(0.25 * e)
        w.writerow(row)
    return duals_path, bias_path, assets


def test_run_triad_trajectory_header_has_no_duplicate_column(tmp_path):
    """`TriadRow` carries its own `run` and `layer`, which always duplicate the cell's, so
    emitting the whole row after the cell columns put `layer` in the header twice. `csv.DictReader`
    collapses a repeated name onto one key and pandas renames it, so both silently drop a value.
    """
    import csv as _csv
    import subprocess
    import sys

    duals_path, bias_path, assets = _write_trajectory_stores(tmp_path)
    out = tmp_path / "traj.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_triad_trajectory.py",
            "--duals",
            str(duals_path),
            "--bias",
            str(bias_path),
            "--out",
            str(out),
            "--resamples",
            "10",
            "--asset-tail",
            assets.tail,
            "--asset-strided",
            assets.strided,
            "--asset-spread",
            assets.spread,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    with out.open(newline="") as f:
        header = next(_csv.reader(f))
    assert len(header) == len(set(header)), f"duplicate column in {header}"

    with out.open(newline="") as f:
        rows = list(_csv.DictReader(f))
    # Every written value survives the round trip a keyed reader makes, which is the property a
    # duplicate name breaks without raising.
    assert all(len(r) == len(header) for r in rows)
    assert {r["projected"] for r in rows} == {"True", "False"}

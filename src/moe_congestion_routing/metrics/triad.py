"""The tail x strided x spread triad at one set of weights: three pairwise `kappa` and a
third, price-only route to the same quantity.

The `kappa = sqrt(c_A * c_B / rho)` estimator needs two batches and gives one number. With three
assets on one checkpoint there are three pairings, so `kappa` can be triangulated three ways,
and a fourth: the pairwise `rho` alone determine each asset's `s_x = corr(p*(x), p_bar)` up to
the same shrinkage algebra, without ever touching the bias. Agreement across all of that is the
test of the shared-`p_bar` assumption every `kappa` published so far already depends on.

This module never opens a file, never runs the LP and never prints. It takes prices a caller
already screened and solved (`None` for a unit `screen_batch` refused) and returns
:class:`TriadRow`. The LP wrapper it does own, :func:`priced_unit`, is a thin, stateless
convenience so a caller does not have to import both `game.lp` and `probe_comparison` to get
one screened price.
"""

import math
from typing import NamedTuple

import numpy as np

from moe_congestion_routing.game import lp
from moe_congestion_routing.metrics.probe_comparison import (
    BatchScreen,
    HalfSplitRow,
    half_split_row,
    screen_batch,
)

# Which 16,384-token unit each asset contributes to the triad, fixed here rather than left for
# a caller to choose: the tail cell has only "u0", the strided cell's "u0" is the code half that
# refuses the screen at every step so "u1" is the one that prices, and the spread cell's "u0" is
# primary with "u1" emitted separately as a robustness check.
PRIMARY_UNIT = {"tail": "u0", "strided": "u1", "spread": "u0"}
SPREAD_ROBUSTNESS_UNIT = "u1"


class TriadRow(NamedTuple):
    """One pairwise or identity comparison, at one `(run, layer)`.

    ``pair`` is one of ``"tail-strided"``, ``"tail-spread"``, ``"strided-spread"`` for a
    pairwise row, or ``"identity:<asset>"`` for the triad-identity route, which reuses this
    same shape rather than adding a second row type: ``rho`` there holds the identity's
    `s_x`, ``c_x`` the asset's own bias correlation, ``c_y`` and the bootstrap columns unused.
    """

    run: str
    layer: int
    pair: str
    unit_x: str
    unit_y: str
    rho: float  # corr(p*_x, p*_y), or s_x for an identity row
    c_x: float  # corr(b, p*_x)
    c_y: float
    kappa: float  # sqrt(c_x * c_y / rho), NaN when the radicand is non-positive
    kappa_boot_low: float
    kappa_boot_high: float
    # A fraction, not a count: unlike HalfSplitRow's kappa_boot_undefined, a triad row's five
    # pairings need not share one resample budget, so the comparable quantity is the rate.
    kappa_boot_undefined: float


def priced_unit(
    routing_map: np.ndarray, affinities: np.ndarray, topk: int
) -> tuple[BatchScreen, np.ndarray | None]:
    """Screen one unit and, only if admissible, solve its LP price.

    ``routing_map`` and ``affinities`` are already the ``[tokens, experts]`` slice of exactly
    the unit being priced, because the screen and the price must see the same
    tokens. Returns ``duals is None`` exactly when the screen refused, so a caller never prices
    a batch neither of them called sound.
    """
    screen = screen_batch(routing_map, topk)
    if not screen.admissible:
        return screen, None
    return screen, lp.solve(affinities, topk).capacity_duals


def _nan_row(run: str, layer: int, pair: str, unit_x: str, unit_y: str) -> TriadRow:
    nan = float("nan")
    return TriadRow(run, layer, pair, unit_x, unit_y, nan, nan, nan, nan, nan, nan, nan)


def _pairwise_row(
    run: str,
    layer: int,
    pair: str,
    bias: np.ndarray,
    unit_x: str,
    duals_x: np.ndarray | None,
    unit_y: str,
    duals_y: np.ndarray | None,
    *,
    resamples: int,
    seed: int,
) -> TriadRow:
    """One pairing, or a NaN row when either side's unit was refused.

    Delegates the whole comparison to :func:`half_split_row`, because ``kappa`` here is the
    same joint-bootstrapped ratio already defined for two batches against one bias, and a
    triad pairing is exactly that with a different pair of batches.
    """
    if duals_x is None or duals_y is None:
        return _nan_row(run, layer, pair, unit_x, unit_y)
    row: HalfSplitRow = half_split_row(
        bias, duals_x, duals_y, step=layer, layer=layer, resamples=resamples, seed=seed
    )
    undefined_fraction = row.kappa_boot_undefined / row.boot_resamples
    return TriadRow(
        run=run,
        layer=layer,
        pair=pair,
        unit_x=unit_x,
        unit_y=unit_y,
        rho=row.rho,
        c_x=row.corr_bias_a,
        c_y=row.corr_bias_b,
        kappa=row.kappa,
        kappa_boot_low=row.kappa_boot_low,
        kappa_boot_high=row.kappa_boot_high,
        kappa_boot_undefined=undefined_fraction,
    )


def _s_and_kappa(rho_xy: float, rho_xz: float, rho_yz: float, c_x: float) -> tuple[float, float]:
    """One asset's triad-identity shrinkage ``s_x`` and the ``kappa`` it implies.

    Undefined by the same rule as ``half_split_row``'s ``kappa``: the denominator and the
    numerator's product must both be positive, else the radicand is not a real shrinkage and
    the row reports NaN rather than a manufactured value. A NaN input propagates through the
    same comparisons (always false), so a refused pairing needs no separate check here.
    """
    num = rho_xy * rho_xz
    s = math.sqrt(num / rho_yz) if rho_yz > 0 and num > 0 else float("nan")
    kappa = c_x / s if s == s and s != 0 else float("nan")
    return s, kappa


def _identity_rows(
    run: str, layer: int, ts: TriadRow, tp: TriadRow, sp: TriadRow
) -> list[TriadRow]:
    """The triad identity's three rows, one per asset, from the three *primary* pairwise rows.

    Reads ``rho`` and the bias correlations off ``ts``/``tp``/``sp`` rather than recomputing
    them, so a refused unit's NaN reaches here exactly once and propagates by arithmetic alone.
    """
    rho_ts, rho_tp, rho_sp = ts.rho, tp.rho, sp.rho
    c_tail, c_strided, c_spread = ts.c_x, ts.c_y, tp.c_y
    nan = float("nan")

    s_tail, k_tail = _s_and_kappa(rho_ts, rho_tp, rho_sp, c_tail)
    s_strided, k_strided = _s_and_kappa(rho_ts, rho_sp, rho_tp, c_strided)
    s_spread, k_spread = _s_and_kappa(rho_tp, rho_sp, rho_ts, c_spread)

    return [
        TriadRow(
            run,
            layer,
            "identity:tail",
            PRIMARY_UNIT["tail"],
            "",
            s_tail,
            c_tail,
            nan,
            k_tail,
            nan,
            nan,
            nan,
        ),
        TriadRow(
            run,
            layer,
            "identity:strided",
            PRIMARY_UNIT["strided"],
            "",
            s_strided,
            c_strided,
            nan,
            k_strided,
            nan,
            nan,
            nan,
        ),
        TriadRow(
            run,
            layer,
            "identity:spread",
            PRIMARY_UNIT["spread"],
            "",
            s_spread,
            c_spread,
            nan,
            k_spread,
            nan,
            nan,
            nan,
        ),
    ]


def triad_rows(
    run: str,
    layer: int,
    bias: np.ndarray,
    tail_u0: np.ndarray | None,
    strided_u1: np.ndarray | None,
    spread_u0: np.ndarray | None,
    spread_u1: np.ndarray | None,
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> list[TriadRow]:
    """All eight rows for one ``(run, layer)``: three primary pairings, the spread-robustness
    variant of the two pairings that touch it, and the three-asset triad identity.

    Each argument is a priced unit's capacity duals, or ``None`` when :func:`priced_unit`
    refused it. ``bias`` is the one stored ``expert_bias`` vector all three assets share,
    because they are three probes of the *same* checkpoint, never three different ones.
    """
    ts = _pairwise_row(
        run,
        layer,
        "tail-strided",
        bias,
        "u0",
        tail_u0,
        "u1",
        strided_u1,
        resamples=resamples,
        seed=seed,
    )
    tp = _pairwise_row(
        run,
        layer,
        "tail-spread",
        bias,
        "u0",
        tail_u0,
        "u0",
        spread_u0,
        resamples=resamples,
        seed=seed + 1,
    )
    tp_robust = _pairwise_row(
        run,
        layer,
        "tail-spread",
        bias,
        "u0",
        tail_u0,
        SPREAD_ROBUSTNESS_UNIT,
        spread_u1,
        resamples=resamples,
        seed=seed + 2,
    )
    sp = _pairwise_row(
        run,
        layer,
        "strided-spread",
        bias,
        "u1",
        strided_u1,
        "u0",
        spread_u0,
        resamples=resamples,
        seed=seed + 3,
    )
    sp_robust = _pairwise_row(
        run,
        layer,
        "strided-spread",
        bias,
        "u1",
        strided_u1,
        SPREAD_ROBUSTNESS_UNIT,
        spread_u1,
        resamples=resamples,
        seed=seed + 4,
    )
    identity = _identity_rows(run, layer, ts, tp, sp)
    return [ts, tp, tp_robust, sp, sp_robust, *identity]

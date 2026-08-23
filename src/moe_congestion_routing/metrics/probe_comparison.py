"""Turns probe dumps into the two per-layer tables the ALF-LB-versus-LP-price question needs.

Table 1 (verification) runs the annealed ALF-LB iteration on a dump's own affinities and scores
it against the LP oracle, exactly as the synthetic grid does: it has a pass value, because it asks
whether the mechanism reaches the theorem's fixed point on this instance. Table 2
(internalization) compares the dump's own stored bias, a stochastic average over training
batches, against this one batch's LP capacity duals: it has no pass value, because the two track
different quantities and an equality test between them is vacuously false. The two questions stay
two functions and, in the script, two files.

This module never opens a file and never prints. It consumes a :class:`ProbeSeries` the caller
already read and returns rows the caller writes out.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from moe_congestion_routing.game import lp
from moe_congestion_routing.game.compare import Comparison, compare, dual_agreement
from moe_congestion_routing.metrics.probe_series import (
    IncomparableProbes,
    ProbeDump,
    ProbeSeries,
    selection_conformance,
)

# Below this ratio the stored bias's position within its own +/-eta orbit spans more than a
# quarter of the dual spread, so a correlation against one batch's duals would report the phase
# of a limit cycle rather than a property of the bias. The constant rests on that argument
# alone. The synthetic ensemble runs continuously from 4.22 to 29.43 at the shipped bias rate,
# so moving this threshold changes which instances are refused rather than being free.
DUAL_SPREAD_GATE = 8.0


class VerificationRow(NamedTuple):
    """One (step, layer) of Table 1: the annealed run against this dump's own affinities."""

    step: int
    layer: int
    bias_update_rate: float
    comparison: Comparison


class InternalizationRow(NamedTuple):
    """One (step, layer) of Table 2: the dump's stored bias against this batch's LP duals."""

    step: int
    layer: int
    bias_update_rate: float
    dual_spread_over_eta: float
    dual_correlation: float
    dual_linf: float


def _check_conformance(dump: ProbeDump) -> None:
    """Refuse a dump the offline replica cannot honestly reproduce the routing of.

    A layer with ``untied_disagreements > 0`` means top-K of this dump's own affinities plus
    bias disagrees with the stored routing map on tokens that were not decided by a tie, which
    means the replica is scoring a router other than the one that actually ran.
    """
    for row in selection_conformance(dump):
        if row.untied_disagreements > 0:
            raise IncomparableProbes(
                f"{dump.path}: layer {row.layer} has {row.untied_disagreements} untied "
                "selection disagreements between the stored routing map and top-K of this "
                "dump's own affinities plus bias, so its routing cannot be honestly "
                "reproduced offline"
            )


def _select_dumps(series: ProbeSeries, steps: Sequence[int] | None) -> list[ProbeDump]:
    """Every requested step's dump, or just the series' last dump when none was requested."""
    if steps is None:
        return [series.dumps[-1]]
    by_step = {dump.step: dump for dump in series.dumps}
    missing = [step for step in steps if step not in by_step]
    if missing:
        available = [dump.step for dump in series.dumps]
        raise ValueError(
            f"{series.run_dir}: steps {missing} are not among this series' available steps "
            f"{available}"
        )
    return [by_step[step] for step in steps]


def _select_layers(dump: ProbeDump, layers: Sequence[int] | None) -> list[tuple[int, int]]:
    """``(axis_index, layer_number)`` pairs for the requested layers, or every layer in the dump."""
    if layers is None:
        return list(enumerate(dump.layer_numbers))
    index_of = {layer: axis_index for axis_index, layer in enumerate(dump.layer_numbers)}
    missing = [layer for layer in layers if layer not in index_of]
    if missing:
        raise ValueError(
            f"{dump.path}: layers {missing} are not among this dump's layer_numbers "
            f"{dump.layer_numbers}"
        )
    return [(index_of[layer], layer) for layer in layers]


def gated_dual_agreement(
    bias: np.ndarray, p_star: np.ndarray, bias_update_rate: float
) -> tuple[float, float, float]:
    """Table 2's one comparison: ``(dual_spread_over_eta, dual_correlation, dual_linf)``.

    The ratio is always returned because it is diagnostic on its own. The correlation comes
    back NaN below :data:`DUAL_SPREAD_GATE`, because below that ratio the stored bias's
    position within its own eta orbit is not pinned well enough for a correlation to mean
    anything, so quoting one would report orbit noise as a finding.
    """
    spread = float(p_star.max() - p_star.min()) / bias_update_rate
    correlation, linf = dual_agreement(bias, p_star)
    if spread < DUAL_SPREAD_GATE:
        correlation = float("nan")
    return spread, correlation, linf


def verification_rows(
    series: ProbeSeries,
    *,
    bias_update_rate: float,
    annealed_steps: int = 40000,
    steps: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
) -> list[VerificationRow]:
    """Table 1, per (step, layer): the annealed ALF-LB run scored against the LP oracle.

    Refuses (``IncomparableProbes``) a dump whose selection conformance is not clean, a
    non-sigmoid dump, or one with no stored bias, because any of those means the affinities or
    bias this compares against do not describe the router that actually ran.
    """
    rows = []
    for dump in _select_dumps(series, steps):
        _check_conformance(dump)
        affinities = dump.affinities()
        for axis_index, layer_number in _select_layers(dump, layers):
            comparison = compare(
                affinities[axis_index],
                dump.topk,
                eta=bias_update_rate,
                steps=annealed_steps,
                mode="annealed",
            )
            rows.append(VerificationRow(dump.step, layer_number, bias_update_rate, comparison))
    return rows


def internalization_rows(
    series: ProbeSeries,
    *,
    bias_update_rate: float,
    steps: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
) -> list[InternalizationRow]:
    """Table 2, per (step, layer): the dump's own stored bias against this batch's LP duals.

    Refuses under the same three conditions as :func:`verification_rows`. There is no pass
    value here, so this only ever reports a measurement and its resolvability gate, never a
    verdict.
    """
    rows = []
    for dump in _select_dumps(series, steps):
        _check_conformance(dump)
        affinities = dump.affinities()
        bias = dump.expert_bias()
        for axis_index, layer_number in _select_layers(dump, layers):
            oracle = lp.solve(affinities[axis_index], dump.topk)
            spread, correlation, linf = gated_dual_agreement(
                bias[axis_index], oracle.capacity_duals, bias_update_rate
            )
            rows.append(
                InternalizationRow(
                    dump.step, layer_number, bias_update_rate, spread, correlation, linf
                )
            )
    return rows

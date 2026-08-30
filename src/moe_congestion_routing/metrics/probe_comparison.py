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

# The reported unit, in tokens. This is a hard rule rather than a convention, because a
# coarser unit dilutes a concentrated sub-batch below `CONCENTRATION_LIMIT`, which is calibrated
# only at this `n`, so the two constants must never drift apart. Single-sourced here and imported
# by every script that slices a dump rather than re-declared at each call site.
UNIT_TOKENS = 16384


def probe_units(n_tokens: int) -> list[tuple[str, int, int]]:
    """The ``[("u0", 0, 16384), ("u1", 16384, 32768), ...]`` cut of ``n_tokens``.

    Raises when ``n_tokens`` is not a positive multiple of :data:`UNIT_TOKENS`, because a partial
    unit is not the instrument any screen or price here was calibrated on. Units are named
    ``u<i>`` rather than ``all``/``h1``/``h2``, because a name that says "half" is only true at
    ``n_tokens == 2 * UNIT_TOKENS`` and misnaming a tail unit as a half is exactly the defect that
    reached committed output.
    """
    if n_tokens <= 0 or n_tokens % UNIT_TOKENS != 0:
        raise ValueError(
            f"n_tokens {n_tokens} is not a positive multiple of UNIT_TOKENS {UNIT_TOKENS}"
        )
    num_units = n_tokens // UNIT_TOKENS
    return [(f"u{i}", i * UNIT_TOKENS, (i + 1) * UNIT_TOKENS) for i in range(num_units)]


# Below this ratio the stored bias's position within its own +/-eta orbit spans more than a
# quarter of the dual spread, so a correlation against one batch's duals would report the phase
# of a limit cycle rather than a property of the bias. The constant rests on that argument
# alone. The synthetic ensemble runs continuously from 4.22 to 29.43 at the shipped bias rate,
# so moving this threshold changes which instances are refused rather than being free.
DUAL_SPREAD_GATE = 8.0

# A price describes a batch the router treated as a sample. Above this ratio of the busiest expert's
# deployed load to the balanced load L = n*K/E, it does not: on one probe half a single expert took
# 64% of the tokens, so the LP priced it deeply negative to shed them and every correlation against
# the stored bias collapsed. Calibrated on 344 screened units rather than on the failing case, the
# observed separation is 3.26 (worst sound unit) against 4.04 (mildest distorted one), so 3.5
# sits in the gap. It also refuses exactly the pre-balancing steps of both runs, which is why
# there is no separate warmup constant.
CONCENTRATION_LIMIT = 3.5


class VerificationRow(NamedTuple):
    """One (step, layer) of Table 1: the annealed run against this dump's own affinities."""

    step: int
    layer: int
    bias_update_rate: float
    comparison: Comparison


class InternalizationRow(NamedTuple):
    """One (step, layer) of Table 2: the dump's stored bias against this batch's LP duals.

    The correlation column is ``bias_price_correlation`` and NOT ``dual_correlation``, which
    :class:`~moe_congestion_routing.game.compare.Comparison` already uses for the annealed
    replica's bias against the same batch's oracle duals. Both land in CSVs, so one shared header
    would let two different quantities be pooled with nothing failing.
    """

    step: int
    layer: int
    bias_update_rate: float
    dual_spread_over_eta: float
    bias_price_correlation: float
    bias_price_linf: float


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
    """Table 2's one comparison, as
    ``(dual_spread_over_eta, bias_price_correlation, bias_price_linf)``.

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
                # Narrowed back to the width the model computed in, which is lossless because
                # `affinities()` widened float32 values, so `compare` reports its tie margins in
                # units of one float32 step rather than of the float64 they are carried in.
                affinities[axis_index].astype(np.float32),
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


# Three ways to cut a batch into parts, which measure different things.
#
# "sequence" takes contiguous blocks of whole sequences, the analogue of forming a smaller
# training batch, so its parts share no document and it carries composition as well as sample size.
#
# "random" takes a seeded permutation, so each part is an unbiased subsample of the same documents
# and only sample size is left. This is the sampling control.
#
# "stride" takes every m-th row. Rows are sequence-major and a sequence length is even, so a stride
# of two selects by position parity, which puts almost every token next to its own neighbour in the
# other part. Text is locally coherent, so those parts are far more alike than two independent
# draws would be, which means stride *understates* sampling noise rather than measuring it. It is
# kept because the gap between it and "random" is how much that pairing flatters the number, and
# not because it is the control it looks like.
SPLIT_MODES = ("sequence", "stride", "random")


class PriceStabilityRow(NamedTuple):
    """One (step, layer): how much this batch's equilibrium prices depend on the batch."""

    step: int
    layer: int
    split: str
    split_seed: int | None  # empty for the deterministic splits, which take no seed
    num_parts: int
    part_tokens: int
    num_pairs: int
    mean_pairwise_correlation: float
    min_pairwise_correlation: float
    # Population standard deviation over the pairs, so a single pair reports 0.0 rather than NaN.
    # It is the only within-cell error bar here, and it needs num_parts > 2 to say anything.
    stdev_pairwise_correlation: float
    mean_pairwise_linf: float
    bias_correlation_full: float
    mean_bias_correlation_parts: float


def part_indices(
    num_tokens: int, num_sequences: int, num_parts: int, split: str, seed: int | None = None
) -> list[np.ndarray]:
    """Disjoint token-row index arrays covering the batch, one per part."""
    if split not in SPLIT_MODES:
        raise ValueError(f"split must be one of {SPLIT_MODES}, got {split!r}")
    if num_parts < 2:
        raise ValueError(f"num_parts must be at least 2 to have a pair to compare, got {num_parts}")
    if split == "random":
        if seed is None:
            raise ValueError("split 'random' requires a seed, so the cut can be reproduced")
        if num_tokens % num_parts != 0:
            raise ValueError(f"num_parts {num_parts} does not divide num_tokens {num_tokens}")
        shuffled = np.random.default_rng(seed).permutation(num_tokens)
        return list(np.split(shuffled, num_parts))
    if seed is not None:
        raise ValueError(f"split {split!r} is deterministic and takes no seed, got seed={seed}")
    if split == "stride":
        if num_tokens % num_parts != 0:
            raise ValueError(f"num_parts {num_parts} does not divide num_tokens {num_tokens}")
        return [np.arange(part, num_tokens, num_parts) for part in range(num_parts)]
    # Cutting on a sequence boundary needs the sequence count to divide, not the token count:
    # a part holding half of one sequence and half of another is not a smaller batch, it is a
    # truncation, and position within a sequence is not exchangeable the way sequences are.
    if num_sequences % num_parts != 0:
        raise ValueError(
            f"split 'sequence' needs num_parts {num_parts} to divide num_sequences "
            f"{num_sequences}, because a part must be a whole number of sequences"
        )
    block = num_tokens // num_parts
    return [np.arange(part * block, (part + 1) * block) for part in range(num_parts)]


def price_stability_rows_for_dump(
    dump: ProbeDump,
    *,
    bias_update_rate: float,
    num_parts: int = 2,
    split: str = "sequence",
    split_seed: int | None = None,
    layers: Sequence[int] | None = None,
) -> list[PriceStabilityRow]:
    """Table 3, per layer of one dump: the equilibrium price against itself on sub-batches.

    Table 2 asks whether the stored bias matches this batch's price. It cannot tell a bias that
    has stopped tracking the price from a price that has become batch-specific, because both
    lower the same correlation. This asks the second question on its own, by comparing the price
    of one part of the batch against the price of another at the same model, so no bias enters
    the pairwise columns and no convergence assumption is made anywhere.
    """
    _check_conformance(dump)
    affinities = dump.affinities()
    bias = dump.expert_bias()
    indices = part_indices(affinities.shape[1], dump.num_sequences, num_parts, split, split_seed)

    rows = []
    for axis_index, layer_number in _select_layers(dump, layers):
        layer_a = affinities[axis_index]
        layer_bias = bias[axis_index]

        full = lp.solve(layer_a, dump.topk)
        _, bias_correlation_full, _ = gated_dual_agreement(
            layer_bias, full.capacity_duals, bias_update_rate
        )

        part_duals = [lp.solve(layer_a[idx], dump.topk).capacity_duals for idx in indices]
        # The bias comparisons keep the resolvability gate, because they read the stored bias and
        # so inherit its eta orbit. The pairwise comparisons below do not, because neither side of
        # them is a bias and no orbit is involved.
        part_bias_correlations = [
            gated_dual_agreement(layer_bias, duals, bias_update_rate)[1] for duals in part_duals
        ]

        pairwise = [
            dual_agreement(part_duals[i], part_duals[j])
            for i in range(num_parts)
            for j in range(i + 1, num_parts)
        ]
        correlations = [c for c, _ in pairwise]
        linfs = [linf for _, linf in pairwise]

        rows.append(
            PriceStabilityRow(
                step=dump.step,
                layer=layer_number,
                split=split,
                split_seed=split_seed,
                num_parts=num_parts,
                part_tokens=int(len(indices[0])),
                num_pairs=len(correlations),
                mean_pairwise_correlation=float(np.mean(correlations)),
                min_pairwise_correlation=float(np.min(correlations)),
                stdev_pairwise_correlation=float(np.std(correlations)),
                mean_pairwise_linf=float(np.mean(linfs)),
                bias_correlation_full=bias_correlation_full,
                mean_bias_correlation_parts=float(np.mean(part_bias_correlations)),
            )
        )
    return rows


class HalfSplitRow(NamedTuple):
    """One (step, layer): both halves' prices against each other and against the stored bias.

    Carries intervals, not just point estimates, because every correlation here is over the 64
    experts and a point estimate at that width is not separable from a nearby one.
    """

    step: int
    layer: int
    rho: float  # corr(p*(A), p*(B)), the two halves' prices against each other
    corr_bias_a: float
    corr_bias_b: float
    kappa: float  # sqrt(corr_bias_a * corr_bias_b / rho) = corr(b_train, p_bar), NaN if <= 0
    # Fisher-z on rho, the cheap first-order bar. It assumes i.i.d. bivariate-normal pairs, which
    # 64 experts out of one LP solve are not, so read it as an order of magnitude and the bootstrap
    # below as the interval.
    rho_fisher_low: float
    rho_fisher_high: float
    # Percentile intervals from one joint resample of the experts, so kappa's interval carries the
    # shared randomness between its numerator and denominator rather than combining two separate
    # bars for quantities computed on the same 64 draws.
    rho_boot_low: float
    rho_boot_high: float
    kappa_boot_low: float
    kappa_boot_high: float
    boot_resamples: int
    kappa_boot_undefined: int  # resamples where corr_bias_a * corr_bias_b / rho was <= 0


def _centred_corr(u: np.ndarray, v: np.ndarray) -> float:
    """Correlation after mean-centering both vectors, matching :func:`dual_agreement`'s gauge."""
    u = u - u.mean()
    v = v - v.mean()
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.dot(u, v) / denom) if denom > 0 else float("nan")


def half_split_row(
    bias: np.ndarray,
    duals_a: np.ndarray,
    duals_b: np.ndarray,
    *,
    step: int,
    layer: int,
    resamples: int = 10000,
    seed: int = 0,
) -> HalfSplitRow:
    """`rho`, both bias correlations and `kappa`, each with a joint-bootstrap interval.

    The three correlations are recomputed together on every resample, because `kappa` is a ratio of
    quantities measured on the same 64 experts and two independent intervals would describe a
    quantity nobody computed. Experts are the resampling unit and the duals are held fixed, so this
    covers uncertainty over which experts one is averaging and **not** over which tokens the batch
    holds, because that is what the two halves are for.

    Re-centering happens inside the resample rather than once outside it, because the gauge is a
    property of the expert set being correlated and a resample is a different set.
    """
    rng = np.random.default_rng(seed)
    n = bias.shape[0]
    rho = _centred_corr(duals_a, duals_b)
    c_a = _centred_corr(bias, duals_a)
    c_b = _centred_corr(bias, duals_b)

    draws = rng.integers(0, n, size=(resamples, n))
    rhos = np.empty(resamples)
    kappas = []
    for i, idx in enumerate(draws):
        r = _centred_corr(duals_a[idx], duals_b[idx])
        rhos[i] = r
        ratio = _centred_corr(bias[idx], duals_a[idx]) * _centred_corr(bias[idx], duals_b[idx])
        kappas.append(np.sqrt(ratio / r) if r > 0 and ratio > 0 else np.nan)
    kappas = np.asarray(kappas)
    finite = kappas[np.isfinite(kappas)]

    # Fisher's n - 3 counts the two fitted means and the correlation itself. It does not count the
    # capacity constraint coupling the experts inside one LP solve, which is why it is the bar and
    # the bootstrap is the interval.
    z, se = np.arctanh(np.clip(rho, -0.999999, 0.999999)), 1.0 / np.sqrt(max(n - 3, 1))
    ratio_point = c_a * c_b / rho if rho > 0 and c_a * c_b > 0 else float("nan")

    return HalfSplitRow(
        step=step,
        layer=layer,
        rho=rho,
        corr_bias_a=c_a,
        corr_bias_b=c_b,
        kappa=float(np.sqrt(ratio_point)) if ratio_point == ratio_point else float("nan"),
        rho_fisher_low=float(np.tanh(z - 1.96 * se)),
        rho_fisher_high=float(np.tanh(z + 1.96 * se)),
        rho_boot_low=float(np.percentile(rhos, 2.5)),
        rho_boot_high=float(np.percentile(rhos, 97.5)),
        kappa_boot_low=float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
        kappa_boot_high=float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
        boot_resamples=resamples,
        kappa_boot_undefined=int(resamples - finite.size),
    )


class BatchScreen(NamedTuple):
    """Whether one token set's routing is concentrated enough to invalidate its prices."""

    admissible: bool
    max_load_over_balanced: float
    dead_experts: int
    reason: str


def screen_batch(routing_map: np.ndarray, topk: int) -> BatchScreen:
    """Refuse a token set whose deployed routing is too concentrated to price.

    ``routing_map`` is ``[tokens, experts]`` boolean for **one layer** and exactly the token set
    whose price is about to be computed. Screening the dump when a half is what gets scored misses
    the concentration, because averaging over a sound half dilutes it by roughly half.

    Two refusals, and both are about the oracle rather than the router. An expert the router left at
    zero tokens is still assigned ``L`` tokens by the LP, because capacity is tight, so its price is
    an artifact outright. And an expert the router loaded far above ``L`` forces a price that
    describes that concentration rather than the population.
    """
    load = routing_map.sum(axis=0)
    balanced = routing_map.shape[0] * topk / routing_map.shape[1]
    ratio = float(load.max() / balanced)
    dead = int((load == 0).sum())
    if dead:
        return BatchScreen(False, ratio, dead, f"{dead} experts received zero tokens")
    if ratio > CONCENTRATION_LIMIT:
        return BatchScreen(False, ratio, dead, f"busiest expert at {ratio:.2f}x balanced load")
    return BatchScreen(True, ratio, dead, "")

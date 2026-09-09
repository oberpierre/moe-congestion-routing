"""Router-saturation statistics computed straight from a probe dump's logits and routing map.

**Saturation** here means the sigmoid gate has stopped responding: winners sit at a logit where
``sigma'(z)`` is numerically zero, so no score-space loss (Rosenthal, ALF-LB, aux) has a gradient
that reaches them. That is a different quantity from ``probe_series.py``'s ``SaturationRow``,
which measures whether a token's *selected set* has stopped changing relative to a reference dump
-- one is a statement about the gate's local slope, the other about the realized assignment.

Every field here is computed from the pre-bias logits and the realized routing map alone: no
training, no GPU, and no ``combine``/``expert_bias`` array is read, because the question is what
the gate's own logit scale looks like, not what it decided to do with a bias term.

Every statistic here applies an elementwise sigmoid and its derivative to the logits, so a dump
whose router used ``softmax`` cannot be reduced: the numbers would be arithmetically valid and
mean nothing about that router. ``reduce_dump`` refuses one outright rather than emitting it.

``numpy`` and stdlib only, matching every other reader in ``metrics/`` (``probe_dump_format.py``'s
module docstring explains why: this has to load on a login node with no torch installed).

``frac_tok_frozen`` is the fraction of tokens whose K *selected* experts are all saturated. It used
to max ``sigp`` over all experts, so one responsive unselected expert kept a token counted as
unfrozen no matter how stuck its own winners were. Any CSV produced before this fix predates the
corrected definition, so its ``frac_tok_frozen`` column cannot be compared against rows from this
version.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy

from moe_congestion_routing.metrics.probe_dump_format import ROUTING_MAP_BITORDER
from moe_congestion_routing.metrics.probe_series import IncomparableProbes

# The largest logit magnitude a real training run produces is well under this, so it is not a
# clamp but the point past which fp32 sigmoid has no more bits left and returns exactly 1.0.
SATURATED_SIGMOID_LOGIT = 16.64


class GateRow(NamedTuple):
    run: str
    role: str
    asset: str
    token_sha256: str
    iteration: int
    layer: int
    n_tokens: int
    lse_mean: float
    logit_p999: float
    logit_max: float
    logit_mean: float
    logit_std: float
    sel_sig_mean: float
    sel_sigp_mean: float
    unsel_sigp_mean: float
    frac_sel_tied: float
    frac_tok_tie_ambiguous: float
    frac_sel_sat: float
    frac_tok_frozen: float
    margin_mean: float
    n_zero_token: int
    n_eff_2: float
    d0_size: int
    d0_logit_mean: float
    d0_lift: float
    l0_logit_mean: float
    d0_sigp_mean: float


class LayerBaseline(NamedTuple):
    """One layer's reference point, fixed at the first dump reduced for a run.

    ``dead_experts`` pins which experts count toward ``d0_*`` for every later dump in the series,
    and ``d0_logit_mean`` is that first dump's own ``d0_logit_mean``, which ``d0_lift`` is measured
    against so later dumps report drift from the start of training rather than from each other.
    """

    dead_experts: numpy.ndarray
    d0_logit_mean: float


def _sigmoid(x: numpy.ndarray) -> numpy.ndarray:
    """Elementwise logistic, evaluated in the input's own dtype so fp32 saturation is real."""
    one = numpy.asarray(1.0, dtype=x.dtype)
    return one / (one + numpy.exp(-x))


def _metadata_from_npz(data: Any) -> dict[str, Any]:
    """Parse the ``metadata`` array every dump in this tree carries.

    Its absence is a malformed dump rather than something to guess a default for, so this raises
    instead of falling back to an empty record.
    """
    if "metadata" not in data:
        raise ValueError("dump has no 'metadata' key, cannot read role/iteration/layer_numbers")
    return json.loads(str(data["metadata"]))


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Read just a dump's ``metadata`` dict, without touching ``logits``/``routing_map``.

    A ``.npz`` member is decompressed on access, so this is the cheap way to ask "what role is
    this asset" when picking which of several probed assets to reduce.
    """
    with numpy.load(path) as data:
        return _metadata_from_npz(data)


def _run_name_from_path(path: Path) -> str:
    """The run directory's name, read off the path rather than passed in.

    Every dump lives at ``<run_dir>/probes/[<asset>/]iter_%07d.npz``, so the parent
    of the ancestor literally named ``probes`` is the run directory regardless of whether
    this tree uses the flat layout or the per-asset one.
    """
    for parent in path.parents:
        if parent.name == "probes":
            return parent.parent.name
    raise ValueError(f"{path}: no ancestor directory named 'probes', cannot identify the run")


def select_asset_dir(probes_dir: Path, asset: str | None) -> Path:
    """The one asset directory under ``probes_dir`` to reduce.

    Per-asset layout: resolves ``asset`` by directory stem, or the lone subdirectory when there is
    exactly one and ``asset`` is ``None``. More than one subdirectory with ``asset`` unset raises,
    listing the stems, because every asset here shares ``role == "standing"`` so picking one by
    sort order would be silent rather than a real choice. Flat layout (``probes_dir`` holds
    ``*.npz`` directly): always ``probes_dir`` itself, checked against a named ``asset`` using the
    first dump's own ``moe_probe_batch`` metadata, because a flat directory has no subdirectory to
    name it with.
    """
    if not probes_dir.is_dir():
        raise FileNotFoundError(f"no probes/ directory under {probes_dir.parent}")
    subdirs = sorted(p for p in probes_dir.iterdir() if p.is_dir())
    if not subdirs:
        if asset is not None:
            dumps = sorted(probes_dir.glob("*.npz"))
            if dumps:
                found = Path(read_metadata(dumps[0])["moe_probe_batch"]).stem
                if found != asset:
                    raise IncomparableProbes(
                        f"{probes_dir}: holds asset {found!r}, not the requested {asset!r}"
                    )
        return probes_dir
    stems = sorted(d.name for d in subdirs)
    if asset is None:
        if len(subdirs) != 1:
            raise IncomparableProbes(
                f"{probes_dir}: {len(subdirs)} assets present {stems!r}. Pass --asset to pick one."
            )
        return subdirs[0]
    chosen = probes_dir / asset
    if chosen not in subdirs:
        raise IncomparableProbes(f"{probes_dir}: asset {asset!r} not found. Available: {stems!r}")
    return chosen


def reduce_dump(
    path: Path,
    *,
    sat: float,
    resp: float,
    baseline: dict[int, LayerBaseline] | None,
) -> tuple[int, dict[str, Any], list[GateRow], dict[int, LayerBaseline]]:
    """Reduce one probe dump to one ``GateRow`` per MoE layer.

    ``baseline`` is the layer -> ``LayerBaseline`` map built from the earliest dump this run has
    been reduced through so far, ``None`` (or missing a layer) meaning this call's own dump is
    that reference. Threading it across ascending-iteration calls for one run is what lets later
    dumps report ``d0_*`` fields, including ``d0_lift``, against a fixed baseline rather than
    against themselves or the previous dump.

    Refuses any dump whose router did not use ``sigmoid`` scoring, because every statistic below
    applies a sigmoid and its derivative to logits and a softmax router's logits are not in that
    space. Only ``logits`` and ``routing_map`` are otherwise read from the archive, and the bit
    order comes from the dump's own ``routing_map_bitorder``, because a dump written under a
    different convention would otherwise unpack into a silently wrong routing map rather than
    failing.
    """
    run = _run_name_from_path(path)
    baseline = dict(baseline) if baseline else {}

    with numpy.load(path) as data:
        meta = _metadata_from_npz(data)
        score_function = meta.get("moe_router_score_function")
        if score_function != "sigmoid":
            raise IncomparableProbes(
                f"{path}: score function is {score_function!r}, and only 'sigmoid' is supported "
                "here. Sigmoid saturation statistics do not describe a softmax router."
            )
        num_experts = int(meta["E"])
        bitorder = meta.get("routing_map_bitorder", ROUTING_MAP_BITORDER)
        logits = data["logits"].astype(numpy.float32, copy=False)
        routing_map = numpy.unpackbits(
            data["routing_map"], axis=-1, count=num_experts, bitorder=bitorder
        ).astype(bool)

    iteration = int(meta["iteration"])
    role = meta["role"]
    asset = Path(meta["moe_probe_batch"]).stem
    token_sha256 = meta["token_sha256"]
    topk = int(meta["K"])
    layer_numbers = meta["layer_numbers"]
    all_experts = numpy.arange(num_experts)

    rows = []
    for layer_index, layer in enumerate(layer_numbers):
        layer_logits = logits[layer_index]
        sel_mask = routing_map[layer_index]
        n_tokens = layer_logits.shape[0]

        logit_flat = layer_logits.reshape(-1)
        row_max = layer_logits.max(axis=1, keepdims=True)
        lse = row_max[:, 0] + numpy.log(numpy.exp(layer_logits - row_max).sum(axis=1))

        sig = _sigmoid(layer_logits)
        sigp = sig * (1.0 - sig)

        sel_logits = layer_logits[sel_mask]
        sel_only = numpy.where(sel_mask, layer_logits, numpy.inf)
        unsel_only = numpy.where(sel_mask, -numpy.inf, layer_logits)
        margin_mean = float((sel_only.min(axis=1) - unsel_only.max(axis=1)).mean())
        # A token's own K selected experts, not the full 64, is what its gate can actually move
        # load away from. Maxing sigp over every expert let one responsive unselected expert keep
        # a token counted as unfrozen even while all K of its winners were saturated.
        sel_sigp = numpy.where(sel_mask, sigp, -numpy.inf)

        # Ambiguity in compute_topk's tie-break needs more than K experts saturated: with exactly
        # K, every winner is pinned and there is no K+1'th saturated loser to be tied against.
        saturated_counts = (layer_logits >= SATURATED_SIGMOID_LOGIT).sum(axis=1)
        frac_tok_tie_ambiguous = float(numpy.mean(saturated_counts > topk))

        expert_counts = sel_mask.sum(axis=0)
        n_zero_token = int(numpy.sum(expert_counts == 0))
        sum_sq = float(numpy.sum(expert_counts.astype(numpy.float64) ** 2))
        # Participation ratio: E when load is uniform, falling toward the number of experts
        # actually receiving tokens as load concentrates on fewer of them.
        n_eff_2 = float(expert_counts.sum()) ** 2 / sum_sq if sum_sq > 0 else 0.0

        layer_baseline = baseline.get(layer)
        dead_experts = (
            numpy.flatnonzero(expert_counts == 0)
            if layer_baseline is None
            else layer_baseline.dead_experts
        )
        d0_size = int(dead_experts.size)
        if d0_size > 0:
            d0_logit_mean = float(layer_logits[:, dead_experts].mean())
            d0_sigp_mean = float(sigp[:, dead_experts].mean())
        else:
            d0_logit_mean = float("nan")
            d0_sigp_mean = float("nan")
        if layer_baseline is None:
            layer_baseline = LayerBaseline(dead_experts=dead_experts, d0_logit_mean=d0_logit_mean)
            baseline[layer] = layer_baseline
        d0_lift = d0_logit_mean - layer_baseline.d0_logit_mean

        live0_experts = numpy.setdiff1d(all_experts, dead_experts, assume_unique=True)
        l0_logit_mean = float(layer_logits[:, live0_experts].mean())

        rows.append(
            GateRow(
                run=run,
                role=role,
                asset=asset,
                token_sha256=token_sha256,
                iteration=iteration,
                layer=int(layer),
                n_tokens=int(n_tokens),
                lse_mean=float(lse.mean()),
                logit_p999=float(numpy.percentile(logit_flat, 99.9)),
                logit_max=float(logit_flat.max()),
                logit_mean=float(logit_flat.mean()),
                logit_std=float(logit_flat.std()),
                sel_sig_mean=float(sig[sel_mask].mean()),
                sel_sigp_mean=float(sigp[sel_mask].mean()),
                unsel_sigp_mean=float(sigp[~sel_mask].mean()),
                frac_sel_tied=float(numpy.mean(sel_logits >= SATURATED_SIGMOID_LOGIT)),
                frac_tok_tie_ambiguous=frac_tok_tie_ambiguous,
                frac_sel_sat=float(numpy.mean(sig[sel_mask] >= sat)),
                frac_tok_frozen=float(numpy.mean(sel_sigp.max(axis=1) < resp)),
                margin_mean=margin_mean,
                n_zero_token=n_zero_token,
                n_eff_2=n_eff_2,
                d0_size=d0_size,
                d0_logit_mean=d0_logit_mean,
                d0_lift=d0_lift,
                l0_logit_mean=l0_logit_mean,
                d0_sigp_mean=d0_sigp_mean,
            )
        )

    return iteration, meta, rows, baseline

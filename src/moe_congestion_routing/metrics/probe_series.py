"""Turns ``<run_dir>/probes/*.npz`` dumps into the per-layer router-saturation table.

**Router saturation** is the fraction of probe tokens whose selected expert set is identical to
the set the same token got in a reference dump, reported per MoE layer and lying in ``[0, 1]``.
A value of 1 means routing on this batch is fully frozen, so saturation rising early is a
precursor to expert collapse rather than a measure of it: OLMoE reports roughly 60% saturation
by 1% of training and 80% by 40%.

The reference is a *single* dump rather than each dump's predecessor, because the probe schedule
mixes a dense window with a coarse tail, so a step-to-step flip rate would have a denominator
that changes along the series and could not be compared across arms.

``numpy`` and stdlib only, so this loads on a login node or a laptop with no CUDA, no process
group and no torch installed, unlike ``router_probe.py``, which writes the dumps this module reads
and does need ``torch``. What the two share is in ``probe_dump_format``, which imports neither.

This module refuses three things outright rather than dropping rows or warning, because a
silently smaller table is the failure it exists to rule out: pooling two instruments, scoring a
non-``standing`` role by default, and comparing runs whose coarse grid differs.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy

from moe_congestion_routing.game.alflb import tie_margins, top_k_map
from moe_congestion_routing.metrics.probe_dump_format import ROUTING_MAP_BITORDER


class IncomparableProbes(ValueError):
    """Two dumps or two runs that must never be placed on one axis, or a dump we cannot read."""


def _sigmoid(x: numpy.ndarray) -> numpy.ndarray:
    """Elementwise logistic, evaluated entirely in the input's own dtype.

    The constant is materialized at that dtype rather than left a Python float, so the whole
    expression stays float32 for a float32 input regardless of how numpy is promoting scalars.
    """
    one = numpy.asarray(1.0, dtype=x.dtype)
    return one / (one + numpy.exp(-x))


@dataclass(frozen=True)
class ProbeDump:
    """One probe forward's dump: metadata always in memory, arrays opened on access.

    Holding the metadata is cheap at a few hundred bytes, whereas holding ``routing_map`` for
    every dump in a series is not, so ``routing_map()`` reopens the ``.npz`` each call rather than
    caching an array on the dataclass.
    """

    path: Path
    step: int
    meta: dict[str, Any]

    @property
    def token_sha256(self) -> str:
        return self.meta["token_sha256"]

    @property
    def role(self) -> str:
        return self.meta["role"]

    @property
    def coarse_interval(self) -> int:
        return self.meta["moe_probe_coarse_interval"]

    @property
    def layer_numbers(self) -> tuple[int, ...]:
        """Megatron layer numbers, in the order they index ``routing_map``'s first axis."""
        return tuple(self.meta["layer_numbers"])

    @property
    def num_experts(self) -> int:
        return self.meta["E"]

    @property
    def topk(self) -> int:
        return self.meta["K"]

    @property
    def num_sequences(self) -> int:
        """How many probe sequences were forwarded, so a token row range can be cut on a
        sequence boundary. Rows are sequence-major (``row = sequence * seq_length + position``),
        which ``probe_dump_format.TOKEN_AXIS_CONVENTION`` pins and every dump is written in."""
        return int(self.meta["moe_probe_seqs"])

    @property
    def score_function(self) -> str:
        """Megatron's ``moe_router_score_function`` for the run that wrote this dump."""
        return self.meta["moe_router_score_function"]

    @property
    def has_expert_bias(self) -> bool:
        return bool(self.meta["has_expert_bias"])

    def routing_map(self) -> numpy.ndarray:
        """``[L, N, E]`` bool: which experts each token selected, unpacked from the stored bits."""
        with numpy.load(self.path) as data:
            packed = data["routing_map"]
        return numpy.unpackbits(
            packed, axis=-1, count=self.num_experts, bitorder=ROUTING_MAP_BITORDER
        ).astype(bool)

    def logits(self) -> numpy.ndarray:
        """``[L, N, E]`` float32: the router's pre-bias output, as the model produced it."""
        with numpy.load(self.path) as data:
            return data["logits"]

    def expert_bias(self) -> numpy.ndarray:
        """``[L, E]`` float32: the ALF-LB bias this dump was routed with.

        Raises rather than returning zeros when the run carried no bias, because a zero bias and
        an absent one give the same arithmetic and completely different conclusions about a
        balancing method that is supposed to be doing something.
        """
        if not self.has_expert_bias:
            raise IncomparableProbes(
                f"{self.path}: this run has no expert_bias, so there is no ALF-LB price vector "
                "to read. Megatron maintains one only for a sigmoid or sqrtsoftplus arm"
            )
        with numpy.load(self.path) as data:
            return data["expert_bias"]

    def affinities(self) -> numpy.ndarray:
        """``[L, N, E]`` float64: the quantity the router adds its bias to.

        Evaluated in float32, the width the model used, then widened losslessly, so a
        disagreement with the model cannot come from us having used wider arithmetic than it did.

        This is not bit-identical to the model's own sigmoid and does not need to be. Numpy and
        torch differ on up to 4 ULP on about a third of the elements in a real dump, and the
        model ran on a GPU besides, so the property that matters is that top-K is insensitive to
        that: `selection_conformance` measures it directly rather than assuming it.
        """
        if self.score_function != "sigmoid":
            raise IncomparableProbes(
                f"{self.path}: score function is {self.score_function!r}, and only 'sigmoid' is "
                "supported. Top-K of sigmoid(z) + b is not top-K of z + b, so reading a "
                "softmax dump here would answer a different question without failing"
            )
        logits = self.logits().astype(numpy.float32, copy=False)
        return _sigmoid(logits).astype(numpy.float64)


def read_dump(path: Path | str) -> ProbeDump:
    """Parse one dump's metadata into a ``ProbeDump``, without loading its arrays."""
    path = Path(path)
    with numpy.load(path) as data:
        meta = json.loads(str(data["metadata"]))
    return ProbeDump(path=path, step=meta["iteration"], meta=meta)


@dataclass(frozen=True)
class ProbeSeries:
    """One run's dumps, ascending by step, already validated to share one instrument and role."""

    run_dir: Path
    arm: str | None
    dumps: tuple[ProbeDump, ...]
    asset: str | None = None
    """The dump directory's stem this series was read from, or ``None`` on the legacy flat
    layout, which holds exactly one asset and has no per-asset directory to name it with."""

    @property
    def token_sha256(self) -> str:
        return self.dumps[0].token_sha256

    @property
    def role(self) -> str:
        return self.dumps[0].role

    @property
    def coarse_interval(self) -> int:
        return self.dumps[0].coarse_interval


def _resolve_probes_dir(probes_dir: Path, asset: str | None) -> Path:
    """The directory to glob one asset's dumps from, and every refusal that layout can raise.

    Shared by every reader that turns ``(run_dir, asset)`` into dump paths, so the flat/per-asset
    layout decision lives in exactly one place. Per-asset layout (``probes_dir`` holds
    subdirectories): resolves ``asset`` by name, or the lone subdirectory when there is exactly
    one and ``asset`` is ``None``. Flat layout (``probes_dir`` holds ``*.npz`` directly, the only
    layout the writer produced before per-asset dumps): always ``probes_dir`` itself, since a flat
    directory holds exactly one asset and has no subdirectory to name it with. Matching a requested
    ``asset`` against what a flat directory actually holds needs a dump's own metadata, so that
    check is the caller's job.
    """
    subdirs = sorted(p for p in probes_dir.iterdir() if p.is_dir()) if probes_dir.exists() else []
    flat = probes_dir.exists() and any(probes_dir.glob("*.npz"))
    if flat and subdirs:
        raise IncomparableProbes(
            f"{probes_dir}: holds both loose dumps and asset subdirectories, which no writer "
            "produces, so this is a hand-edited directory and which layout is authoritative "
            "cannot be guessed"
        )
    if subdirs:
        stems = sorted(d.name for d in subdirs)
        if asset is None:
            if len(subdirs) != 1:
                raise IncomparableProbes(
                    f"{probes_dir}: {len(subdirs)} assets present {stems!r}. Pass an asset to "
                    "pick one"
                )
            return subdirs[0]
        chosen = probes_dir / asset
        if chosen not in subdirs:
            raise IncomparableProbes(
                f"{probes_dir}: asset {asset!r} not found. Available: {stems!r}"
            )
        return chosen
    return probes_dir


def probe_dump_path(run_dir: Path | str, step: int, *, asset: str | None = None) -> Path:
    """The single dump path for one ``(run, step, asset)``, on whichever layout this run used.

    Every caller that only needs one step's file, rather than a whole :class:`ProbeSeries`, goes
    through this rather than re-deriving ``probes/iter_%07d.npz`` or
    ``probes/<asset>/iter_%07d.npz`` at its own call site, so the layout decision is made once.
    """
    asset_dir = _resolve_probes_dir(Path(run_dir) / "probes", asset)
    path = asset_dir / f"iter_{step:07d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"no dump at {path}")
    return path


def read_series(
    run_dir: Path | str,
    *,
    asset: str | None = None,
    arm: str | None = None,
    allow_roles: Sequence[str] = ("standing",),
) -> ProbeSeries:
    """Read every dump under ``<run_dir>/probes/`` into one ascending-step ``ProbeSeries``.

    ``asset`` picks a dump directory's stem under the per-asset layout, or is left ``None`` on
    the legacy flat layout where there is only ever one. See :func:`_resolve_probes_dir` for the
    layout refusals this raises before any dump is even opened.

    Also refuses (``IncomparableProbes``) a dump whose ``role`` is outside ``allow_roles``, a
    non-``None`` ``asset`` that a flat directory's own dumps do not match, and two dumps in this
    run that disagree on ``token_sha256``, because any of those would silently pool an instrument
    this table was never meant to include. A run with no ``probes/`` directory, or one with no
    dumps in it, is a missing input rather than an incomparable one and raises
    ``FileNotFoundError`` naming the path.
    """
    run_dir = Path(run_dir)
    probes_dir = run_dir / "probes"
    asset_dir = _resolve_probes_dir(probes_dir, asset)
    is_flat = asset_dir == probes_dir
    paths = sorted(asset_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no probe dumps found under {probes_dir}")

    dumps = [read_dump(path) for path in paths]

    if is_flat and asset is not None:
        # A flat directory has no subdirectory to name it with, so the asset it holds is read
        # from the dumps' own record of which asset file wrote them.
        found = Path(dumps[0].meta["moe_probe_batch"]).stem
        if found != asset:
            raise IncomparableProbes(
                f"{probes_dir}: holds asset {found!r}, not the requested {asset!r}"
            )

    for dump in dumps:
        if dump.role not in allow_roles:
            raise IncomparableProbes(
                f"{dump.path}: role {dump.role!r} is not among the allowed roles "
                f"{tuple(allow_roles)!r}. Pass --allow-role {dump.role} to include it"
            )

    shas = {dump.token_sha256 for dump in dumps}
    if len(shas) > 1:
        raise IncomparableProbes(
            f"{probes_dir}: dumps disagree on token_sha256 {sorted(shas)!r}, so they are not "
            "the same instrument and cannot be pooled into one series"
        )

    resolved_asset = None if is_flat else asset_dir.name
    return ProbeSeries(
        run_dir=run_dir,
        arm=arm,
        dumps=tuple(sorted(dumps, key=lambda d: d.step)),
        asset=resolved_asset,
    )


@dataclass(frozen=True)
class SaturationRow:
    """One (run, step, layer) saturation observation: agreement with ``reference_step``'s map."""

    run_dir: str
    arm: str | None
    step: int
    layer: int
    reference_step: int
    agreement: float
    num_tokens: int
    role: str
    token_sha256: str
    coarse_interval: int


def _emitted_dumps(series: ProbeSeries, *, multi: bool) -> list[ProbeDump]:
    """Dumps that get a row: every dump alone, only the coarse grid across more than one series.

    The coarse-grid restriction is unconditional in the cross-run case: dense window steps line
    up across runs only when the windows themselves are identical, so emitting them here would
    silently compare different steps under the same column.
    """
    if not multi:
        return list(series.dumps)
    return [dump for dump in series.dumps if dump.step % series.coarse_interval == 0]


def _series_rows(
    series: ProbeSeries, *, multi: bool, reference_step: int | None
) -> list[SaturationRow]:
    emitted = _emitted_dumps(series, multi=multi)
    ref_step = emitted[-1].step if reference_step is None else reference_step
    ref_dump = next((dump for dump in series.dumps if dump.step == ref_step), None)
    if ref_dump is None:
        available = [dump.step for dump in series.dumps]
        raise ValueError(
            f"{series.run_dir}: reference_step {ref_step} is not among its available steps "
            f"{available}"
        )
    ref_map = ref_dump.routing_map()

    rows = []
    for dump in emitted:
        current_map = ref_map if dump.step == ref_step else dump.routing_map()
        # Set equality per token per layer, because saturation asks whether the selected set is the
        # same, not whether the top-k order or the combine weights are.
        agreement_by_layer = numpy.all(current_map == ref_map, axis=-1).mean(axis=-1)
        num_tokens = current_map.shape[1]
        for axis_index, layer_number in enumerate(dump.layer_numbers):
            rows.append(
                SaturationRow(
                    run_dir=str(series.run_dir),
                    arm=series.arm,
                    step=dump.step,
                    layer=layer_number,
                    reference_step=ref_step,
                    agreement=float(agreement_by_layer[axis_index]),
                    num_tokens=num_tokens,
                    role=dump.role,
                    token_sha256=dump.token_sha256,
                    coarse_interval=dump.coarse_interval,
                )
            )
    return rows


def saturation_rows(
    series: Sequence[ProbeSeries], *, reference_step: int | None = None
) -> list[SaturationRow]:
    """The router-saturation table over one or more series, in the order ``series`` was given.

    Refuses (``IncomparableProbes``) more than one series that disagree on ``token_sha256`` or on
    ``coarse_interval``, so two runs land in one table only when they are the same instrument on
    the same grid. With more than one series, only steps on the shared coarse grid are emitted;
    with exactly one, every dump is.
    """
    series = list(series)
    multi = len(series) > 1
    if multi:
        shas = {one.token_sha256 for one in series}
        if len(shas) > 1:
            raise IncomparableProbes(
                f"series disagree on token_sha256 {sorted(shas)!r}, so they are not the same "
                "instrument and cannot be compared"
            )
        intervals = {one.coarse_interval for one in series}
        if len(intervals) > 1:
            raise IncomparableProbes(
                f"series disagree on moe_probe_coarse_interval {sorted(intervals)!r}, so they "
                "have no shared grid to compare on"
            )

    rows = []
    for one in series:
        rows.extend(_series_rows(one, multi=multi, reference_step=reference_step))
    return rows


@dataclass(frozen=True)
class ConformanceRow:
    """One layer's answer to whether the offline replica selects what the model selected."""

    step: int
    layer: int
    num_tokens: int
    disagreeing_tokens: int
    untied_disagreements: int
    exact_ties: int
    """Tokens whose K-th and (K+1)-th float32 scores are bit-identical. Zero on every real dump
    measured so far, which is why the tie test is ULP-relative rather than exact."""

    tied_tokens: int
    """Tokens within ``CONFORMANCE_TIE_ULP`` float steps, so the replica's own arithmetic cannot
    resolve the ordering. Their disagreements are excluded from ``untied_disagreements``."""


# How many float steps of the model's own arithmetic count as a tie when checking that the offline
# replica routes the way the router did. Four, because that is the largest disagreement measured
# between numpy's sigmoid and torch's on a real dump, so a margin under it can be inverted by the
# replica rather than by any defect. Measured on this tree, a threshold of four exempts between two
# and eight tokens out of 262144 per dump, so the check keeps essentially all of its power.
CONFORMANCE_TIE_ULP = 4


def selection_conformance(dump: ProbeDump) -> list[ConformanceRow]:
    """Per layer, check that top-K of ``sigmoid(logits) + expert_bias`` is the dump's own map.

    This is what makes every later offline number about this dump worth reading, because it is
    the only check that the replica routes the way the model routed rather than merely
    plausibly. ``untied_disagreements`` is the one that must be zero: a token whose K-th and
    (K+1)-th scores are exactly equal was decided by a tie rule rather than by the affinities,
    and two implementations are free to disagree there.

    Scores are compared in float32, the arithmetic the model used, and a token counts as tied
    within :data:`CONFORMANCE_TIE_ULP` float steps rather than only at exact equality. An earlier
    version required exact equality, arguing that any nonzero float32 margin is an ordering both
    implementations see identically. That is false here, because our sigmoid is not the model's:
    numpy and torch differ by up to 4 ULP on about a third of a real dump, so a margin of one step
    can inverted by the replica alone. Exact equality also never occurred, so the exemption was
    dead and every near-tie counted as a defect.
    """
    affinities = dump.affinities()
    bias = dump.expert_bias()
    stored = dump.routing_map()
    k = dump.topk

    rows = []
    for axis_index, layer_number in enumerate(dump.layer_numbers):
        scores = affinities[axis_index].astype(numpy.float32) + bias[axis_index]
        replica = numpy.zeros_like(stored[axis_index])
        numpy.put_along_axis(replica, top_k_map(scores, k), True, axis=1)
        disagrees = ~numpy.all(replica == stored[axis_index], axis=1)
        margins = tie_margins(scores, k)
        kth = numpy.sort(scores, axis=1)[:, ::-1][:, k - 1]
        step_at_kth = numpy.spacing(kth).astype(numpy.float64)
        tied = margins.astype(numpy.float64) <= CONFORMANCE_TIE_ULP * step_at_kth
        rows.append(
            ConformanceRow(
                step=dump.step,
                layer=layer_number,
                num_tokens=int(stored.shape[1]),
                disagreeing_tokens=int(disagrees.sum()),
                untied_disagreements=int((disagrees & ~tied).sum()),
                exact_ties=int((margins == 0.0).sum()),
                tied_tokens=int(tied.sum()),
            )
        )
    return rows

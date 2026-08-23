"""Turns ``<run_dir>/probes/*.npz`` dumps into the per-layer router-saturation table.

**Router saturation** is the fraction of probe tokens whose selected expert set is identical to
the set the same token got in a reference dump, reported per MoE layer and lying in ``[0, 1]``.
A value of 1 means routing on this batch is fully frozen, so saturation rising early is a
precursor to expert collapse rather than a measure of it: OLMoE reports roughly 60% saturation
by 1% of training and 80% by 40%.

The reference is a *single* dump rather than each dump's predecessor, because the probe schedule
mixes a dense window with a coarse tail, so a step-to-step flip rate would have a denominator
that changes along the series and could not be compared across arms.

``numpy`` and stdlib only, so this loads on a login node with no CUDA and no process group,
unlike ``router_probe.py``, which writes the dumps this module reads and needs ``torch``. Only
``ROUTING_MAP_BITORDER`` is shared between the two, because the pack and the unpack must move
together.

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

from moe_congestion_routing.metrics.router_probe import ROUTING_MAP_BITORDER


class IncomparableProbes(ValueError):
    """Two dumps or two runs that must never be placed on one axis."""


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

    def routing_map(self) -> numpy.ndarray:
        """``[L, N, E]`` bool: which experts each token selected, unpacked from the stored bits."""
        with numpy.load(self.path) as data:
            packed = data["routing_map"]
        return numpy.unpackbits(
            packed, axis=-1, count=self.num_experts, bitorder=ROUTING_MAP_BITORDER
        ).astype(bool)


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

    @property
    def token_sha256(self) -> str:
        return self.dumps[0].token_sha256

    @property
    def role(self) -> str:
        return self.dumps[0].role

    @property
    def coarse_interval(self) -> int:
        return self.dumps[0].coarse_interval


def read_series(
    run_dir: Path | str,
    *,
    arm: str | None = None,
    allow_roles: Sequence[str] = ("standing",),
) -> ProbeSeries:
    """Read every dump under ``<run_dir>/probes/`` into one ascending-step ``ProbeSeries``.

    Refuses (``IncomparableProbes``) a dump whose ``role`` is outside ``allow_roles``, and refuses
    two dumps in this run that disagree on ``token_sha256``, because either would silently pool
    an instrument this table was never meant to include. A run with no ``probes/`` directory, or
    one with no dumps in it, is a missing input rather than an incomparable one and raises
    ``FileNotFoundError`` naming the path.
    """
    run_dir = Path(run_dir)
    probes_dir = run_dir / "probes"
    paths = sorted(probes_dir.glob("*.npz")) if probes_dir.exists() else []
    if not paths:
        raise FileNotFoundError(f"no probe dumps found under {probes_dir}")

    dumps = [read_dump(path) for path in paths]
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

    return ProbeSeries(run_dir=run_dir, arm=arm, dumps=tuple(sorted(dumps, key=lambda d: d.step)))


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

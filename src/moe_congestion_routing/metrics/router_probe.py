"""Per-layer router-state capture and the ``.npz`` dump writer for the probe tier."""

import json
import logging
import os
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy
import torch

logger = logging.getLogger(__name__)

# Every array is written in this row order: row n = sequence * seq_length + position, counting
# every probed sequence across every microbatch in probe-batch order. The router's own flatten
# is row n = position * micro_batch_size + sequence within a microbatch. ProbeCapture.record
# changes the ordering, so they are comparable across microbatche sizes.
TOKEN_AXIS_CONVENTION = "sequence-major: row n = sequence * seq_length + position"

# numpys default made explicit: if num_experts is not divisible by 8 the routing_map array is padded
# with zeros. Bitorder `big` controls that these bits appear on the right for big-endian.
ROUTING_MAP_BITORDER = "big"


class ProbeCapture:
    """Accumulates one probe forward's router state, one MoE layer and microbatch at a time.

    A probe forward runs several microbatches through the same model, so ``record`` is called
    once per (MoE layer, microbatch) pair. ``arrays()`` concatenates the microbatches back into
    the full probe batch along the token axis before stacking layers.
    """

    def __init__(self, iteration: int, micro_batch_size: int, topk: int) -> None:
        self.iteration = iteration
        self.micro_batch_size = micro_batch_size
        self.topk = topk
        self._logits: dict[int, list[torch.Tensor]] = {}
        self._routing_map: dict[int, list[torch.Tensor]] = {}
        self._combine: dict[int, list[torch.Tensor]] = {}
        self._expert_bias: dict[int, torch.Tensor | None] = {}

    def record(
        self,
        layer_number: int,
        logits: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        expert_bias: torch.Tensor | None,
    ) -> None:
        """Record one (layer, microbatch)'s router state: pre-bias logits, combine, assignment.

        ``combine`` is gathered from ``probs`` at ``routing_map``'s set bits, in ascending expert
        index order, the order ``arrays()`` matches against the packed bits. Validates that a
        token's selected count matches ``moe_router_topk``.
        """
        selected = routing_map.sum(dim=-1)
        if selected.numel() == 0:
            raise ValueError(f"layer {layer_number}: probe recorded zero tokens")
        if not torch.all(selected == self.topk):
            counts = sorted(int(c) for c in torch.unique(selected).tolist())
            raise ValueError(
                f"layer {layer_number}: expected exactly K={self.topk} experts per token "
                f"(moe_router_topk), got counts {counts}"
            )

        combine = probs[routing_map].reshape(logits.shape[0], self.topk)
        logits = _to_sequence_major(logits, self.micro_batch_size)
        routing_map = _to_sequence_major(routing_map, self.micro_batch_size)
        combine = _to_sequence_major(combine, self.micro_batch_size)
        self._logits.setdefault(layer_number, []).append(logits.detach().to(torch.float32).cpu())
        self._routing_map.setdefault(layer_number, []).append(
            routing_map.detach().to(torch.bool).cpu()
        )
        self._combine.setdefault(layer_number, []).append(combine.detach().to(torch.float32).cpu())
        if layer_number not in self._expert_bias:
            self._expert_bias[layer_number] = (
                None
                if expert_bias is None
                else expert_bias.detach().to(torch.float32).cpu().clone()
            )

    def arrays(self) -> dict[str, numpy.ndarray]:
        """Stack recorded state into per-layer ``[L, N, ...]`` arrays, one MoE layer axis."""
        if not self._logits:
            raise ValueError("no MoE layer recorded any router state for this probe")
        layer_numbers = sorted(self._logits)
        logits = numpy.stack(
            [torch.cat(self._logits[ln], dim=0).numpy() for ln in layer_numbers], axis=0
        )
        routing_map_bool = numpy.stack(
            [torch.cat(self._routing_map[ln], dim=0).numpy() for ln in layer_numbers], axis=0
        )
        combine = numpy.stack(
            [torch.cat(self._combine[ln], dim=0).numpy() for ln in layer_numbers], axis=0
        )
        out: dict[str, numpy.ndarray] = {
            "logits": logits.astype(numpy.float32),
            "routing_map": numpy.packbits(routing_map_bool, axis=-1, bitorder=ROUTING_MAP_BITORDER),
            "combine": combine.astype(numpy.float32),
            "layer_numbers": numpy.asarray(layer_numbers, dtype=numpy.int64),
        }
        has_bias = [self._expert_bias[ln] is not None for ln in layer_numbers]
        if any(has_bias):
            if not all(has_bias):
                missing = [
                    ln for ln, present in zip(layer_numbers, has_bias, strict=True) if not present
                ]
                raise ValueError(f"expert_bias present on some layers but missing on {missing}")
            out["expert_bias"] = numpy.stack(
                [self._expert_bias[ln].numpy() for ln in layer_numbers], axis=0
            ).astype(numpy.float32)
        return out


def _is_complete_dump(path: Path) -> bool:
    """A ``.npz`` is a zip archive, so a write truncated by a crash or preemption is (almost
    always) not a well-formed one: it is missing the central directory the writer flushes last.
    Used to tell a genuinely finished dump apart from wreckage left at the same path."""
    try:
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def write_probe_dump(path: str | Path, capture: ProbeCapture, meta: dict[str, Any]) -> bool:
    """Write one probe dump, returning whether it wrote (``False`` means it skipped).

    Skips only a *complete* existing dump, because a resume must never overwrite a genuine result
    but an interrupted write (not a well-formed ``.npz``) must be replaced.
    """
    path = Path(path)
    if path.exists():
        if _is_complete_dump(path):
            logger.info("probe dump %s already exists, skipping (not overwriting)", path)
            return False
        logger.warning(
            "probe dump %s exists but is not a complete .npz (likely a truncated write from a "
            "crash or preemption). Replacing it",
            path,
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = capture.arrays()
    layer_numbers = arrays.pop("layer_numbers").tolist()
    full_meta = dict(meta)
    full_meta.update(
        L=arrays["logits"].shape[0],
        N=arrays["logits"].shape[1],
        E=arrays["logits"].shape[2],
        K=capture.topk,
        micro_batch_size=capture.micro_batch_size,
        token_axis_convention=TOKEN_AXIS_CONVENTION,
        routing_map_bitorder=ROUTING_MAP_BITORDER,
        layer_numbers=layer_numbers,
        has_expert_bias="expert_bias" in arrays,
    )
    payload = dict(arrays)
    payload["metadata"] = numpy.array(json.dumps(full_meta))

    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            numpy.savez(tmp_file, **payload)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return True


_active: ProbeCapture | None = None


def active_capture() -> ProbeCapture | None:
    """Return the capture the current forward should record into, or ``None`` when inactive.

    The router hunk calls this on every forward; returning ``None`` outside a probe forward is
    what keeps ``record`` a no-op cost on the training path.
    """
    return _active


@contextmanager
def capturing(iteration: int, micro_batch_size: int, topk: int) -> Iterator[ProbeCapture]:
    """Activate a fresh ``ProbeCapture`` for the duration of one probe forward.

    Restores the previous active capture (``None`` in every real use) on exit even if the forward
    raises, so a failed probe cannot leave a stale capture recording into the next training step.
    """
    global _active
    capture = ProbeCapture(iteration=iteration, micro_batch_size=micro_batch_size, topk=topk)
    previous = _active
    _active = capture
    try:
        yield capture
    finally:
        _active = previous


def _to_sequence_major(tensor: torch.Tensor, micro_batch_size: int) -> torch.Tensor:
    """Permute one microbatch's rows from the router's own order into ``(sequence, position)``.

    The router flattens ``[seq_length, micro_batch_size, ...]``, giving
    ``row = position * micro_batch_size + sequence`` within this microbatch. Un-flattening and
    swapping those two axes makes each microbatch contribute a contiguous block of whole
    sequences, so concatenating microbatches in order yields sequence-major rows overall.
    """
    total_tokens = tensor.shape[0]
    if total_tokens % micro_batch_size != 0:
        raise ValueError(
            f"microbatch token count {total_tokens} is not a multiple of "
            f"micro_batch_size {micro_batch_size}, so the token axis cannot be canonicalised"
        )
    seq_length = total_tokens // micro_batch_size
    rest = tensor.shape[1:]
    return (
        tensor.reshape(seq_length, micro_batch_size, *rest)
        .transpose(0, 1)
        .contiguous()
        .reshape(total_tokens, *rest)
    )

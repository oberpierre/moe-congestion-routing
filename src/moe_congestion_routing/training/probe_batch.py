"""Loader for a frozen probe-batch asset (``assets/probe/*.npz``).

A probe asset holds raw token ids frozen at extraction time, not a dataset reference to
re-derive them from. This module only reads that frozen array, it never touches an
``IndexedDataset`` or a split.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy

if TYPE_CHECKING:
    import torch  # only for the probe_micro_batches type hint, not imported at runtime

PROBE_ASSET_VERSION = 1

PROBE_ROLES = ("standing", "skewed", "dev")


def tail_window(
    sequence_lengths: numpy.ndarray, target_tokens: int, max_tail_fraction: float
) -> tuple[int, int, float]:
    """Return ``(start, end, tail_fraction)``, the half-open document range of the minimal
    tail whose combined token count reaches ``target_tokens``.

    The tail commonly overshoots ``target_tokens`` because the last document included is
    kept whole here, so the caller truncates the concatenated tokens afterward rather than
    this function ending mid-document.
    """
    num_documents = sequence_lengths.shape[0]
    cumulative = 0
    n_tail = 0
    for n_tail in range(1, num_documents + 1):
        cumulative += int(sequence_lengths[num_documents - n_tail])
        if cumulative >= target_tokens:
            break
    else:
        raise ValueError(
            f"blob has only {cumulative} tokens across all {num_documents} documents, "
            f"need {target_tokens}"
        )

    tail_fraction = n_tail / num_documents
    if tail_fraction > max_tail_fraction:
        raise ValueError(
            f"reaching target_tokens={target_tokens} needs the last {n_tail}/{num_documents} "
            f"documents (tail_fraction={tail_fraction:.6g}), which exceeds "
            f"max_tail_fraction={max_tail_fraction}"
        )
    return num_documents - n_tail, num_documents, tail_fraction


@dataclass(frozen=True)
class ProbeBatch:
    """A frozen probe instrument, loaded from an ``assets/probe/*.npz`` file."""

    tokens: numpy.ndarray
    """``[S, seq_length + 1]`` int32; the raw ids, one extra trailing token per sequence."""

    seq_labels: numpy.ndarray
    """``[S]`` int32 cluster/domain id per sequence, ``-1`` where the blob has none."""

    token_sha256: str
    """sha256 of ``tokens.tobytes()``, recomputed and checked against this at load time."""

    role: str
    """One of ``"standing"``, ``"skewed"``, ``"dev"``."""

    provenance: dict[str, Any]
    """The full provenance dict written at extraction time."""

    @property
    def num_sequences(self) -> int:
        """``S``, the number of sequences in the batch."""
        return self.tokens.shape[0]

    @property
    def seq_length(self) -> int:
        """The model sequence length (``tokens.shape[1] - 1``, since ``tokens`` carries a
        trailing label token)."""
        return self.tokens.shape[1] - 1


def load_probe_batch(path: str | Path) -> ProbeBatch:
    """Load and validate a probe asset, raising ``ValueError`` naming ``path`` on any defect.

    The sha256 check is the point of this function: a hand-edited or corrupted instrument must
    fail the first time it is loaded rather than silently measuring the router on the wrong
    tokens.
    """
    path = Path(path)
    with numpy.load(path, allow_pickle=False) as data:
        tokens = data["tokens"]
        seq_labels = data["seq_labels"]
        provenance = json.loads(str(data["provenance"]))

    if provenance.get("asset_version") != PROBE_ASSET_VERSION:
        raise ValueError(
            f"{path}: asset_version {provenance.get('asset_version')!r} != "
            f"{PROBE_ASSET_VERSION} (this loader's expected version)"
        )
    if tokens.dtype != numpy.int32 or tokens.ndim != 2:
        raise ValueError(
            f"{path}: tokens must be int32 2-D, got dtype={tokens.dtype} ndim={tokens.ndim}"
        )
    role = provenance.get("role")
    if role not in PROBE_ROLES:
        raise ValueError(f"{path}: role {role!r} is not one of {PROBE_ROLES}")

    expected_shape = (provenance.get("S"), provenance.get("seq_length", -1) + 1)
    if tokens.shape != expected_shape:
        raise ValueError(
            f"{path}: provenance S={provenance.get('S')!r} seq_length="
            f"{provenance.get('seq_length')!r} disagrees with tokens.shape {tokens.shape}"
        )
    if seq_labels.shape != (tokens.shape[0],):
        raise ValueError(
            f"{path}: seq_labels shape {seq_labels.shape} != ({tokens.shape[0]},), "
            f"the number of sequences in tokens"
        )
    if seq_labels.dtype != numpy.int32:
        raise ValueError(f"{path}: seq_labels must be int32, got dtype={seq_labels.dtype}")

    recomputed_sha256 = hashlib.sha256(tokens.tobytes()).hexdigest()
    recorded_sha256 = provenance.get("token_sha256")
    if recomputed_sha256 != recorded_sha256:
        raise ValueError(
            f"{path}: recomputed token sha256 {recomputed_sha256} != recorded "
            f"{recorded_sha256} (asset is corrupted or was hand-edited)"
        )
    recomputed_seq_labels_sha256 = hashlib.sha256(seq_labels.tobytes()).hexdigest()
    recorded_seq_labels_sha256 = provenance.get("seq_labels_sha256")
    if recomputed_seq_labels_sha256 != recorded_seq_labels_sha256:
        raise ValueError(
            f"{path}: recomputed seq_labels sha256 {recomputed_seq_labels_sha256} != recorded "
            f"{recorded_seq_labels_sha256} (asset is corrupted or was hand-edited)"
        )

    tokens.setflags(write=False)  # make a probe asset immutable once loaded

    return ProbeBatch(
        tokens=tokens,
        seq_labels=seq_labels,
        token_sha256=recorded_sha256,
        role=role,
        provenance=provenance,
    )


def probe_micro_batches(
    batch: ProbeBatch,
    *,
    micro_batch_size: int,
    num_sequences: int | None = None,
    seq_length: int,
    eod_token: int,
    reset_position_ids: bool = False,
    reset_attention_mask: bool = False,
    eod_mask_loss: bool = False,
    create_attention_mask: bool = False,
    device: str | None = None,
) -> "list[dict[str, torch.Tensor | None]]":
    """Split ``batch`` into Megatron-shaped micro-batch dicts.

    Reuses Megatron's own ``_get_ltor_masks_and_position_ids`` (imported function-locally, so
    this module stays importable without torch or megatron) instead of reimplementing it.
    """
    import torch
    from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids

    if num_sequences is None:
        num_sequences = batch.num_sequences
    if num_sequences <= 0:
        raise ValueError(f"num_sequences must be positive, got {num_sequences}")
    if num_sequences > batch.num_sequences:
        raise ValueError(
            f"num_sequences {num_sequences} exceeds batch.num_sequences {batch.num_sequences}"
        )
    if num_sequences % micro_batch_size != 0:
        raise ValueError(
            f"num_sequences {num_sequences} is not divisible by micro_batch_size {micro_batch_size}"
        )
    if seq_length != batch.seq_length:
        raise ValueError(f"seq_length {seq_length} != asset seq_length {batch.seq_length}")

    tokens_all = torch.from_numpy(batch.tokens[:num_sequences].astype("int64"))
    if device is not None:
        tokens_all = tokens_all.to(device)
    text = tokens_all
    tokens = text[:, :-1].contiguous()
    labels = text[:, 1:].contiguous()

    micro_batches = []
    for start in range(0, num_sequences, micro_batch_size):
        stop = start + micro_batch_size
        mb_tokens = tokens[start:stop]
        mb_labels = labels[start:stop]

        loss_masks = []
        position_ids_list = []
        attention_masks = []
        for row in mb_tokens:
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                row,
                eod_token,
                reset_position_ids,
                reset_attention_mask,
                eod_mask_loss,
                create_attention_mask,
            )
            loss_masks.append(loss_mask)
            position_ids_list.append(position_ids)
            attention_masks.append(attention_mask)

        micro_batches.append(
            {
                "tokens": mb_tokens,
                "labels": mb_labels,
                "loss_mask": torch.stack(loss_masks),
                "position_ids": torch.stack(position_ids_list),
                "attention_mask": (torch.stack(attention_masks) if create_attention_mask else None),
            }
        )
    return micro_batches

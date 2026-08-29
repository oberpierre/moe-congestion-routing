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
    import torch  # This import supplies only the probe_micro_batches type hint and never runs.

PROBE_ASSET_VERSION = 1

PROBE_ROLES = ("standing", "skewed", "dev")


def compute_provenance_sha256(provenance: dict[str, Any]) -> str:
    """Return the sha256 of every provenance field except ``provenance_sha256`` itself."""
    fields = {key: value for key, value in provenance.items() if key != "provenance_sha256"}
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


# How many slots the strided sampler lays across the held-out split, as a multiple of the
# documents it estimates it needs. At 1 a split whose sampled documents run shorter than the mean
# would walk off the end and fail, whereas at 2 it can take twice the estimate and stay inside.
# The cost is spanning about half the split rather than all of it when lengths are typical. That
# is still about 3700 times the span of the contiguous rule on the primary blob, whose 48965906
# documents and 65-document window the committed standing asset records.
STRIDE_SPREAD = 2


def strided_window(
    sequence_lengths: numpy.ndarray,
    target_tokens: int,
    max_tail_fraction: float,
    stride_offset: int = 0,
) -> tuple[numpy.ndarray, int, float]:
    """Return ``(document_indices, stride, span_fraction)`` for a probe spread across the
    held-out split, rather than taken from one contiguous run of documents at its end.

    :func:`tail_window` takes the *minimal* tail, which on a real blob is a few dozen adjacent
    documents. Adjacent documents in a web corpus can share a crawl, a domain or a topic, so
    that instrument measures the router on one neighbourhood while the bias it is compared
    against tracks batches Megatron draws from the whole corpus. Striding removes that mismatch
    while keeping everything the contiguous rule bought: no seed, no shuffle index, no index
    cache, a deterministic result, and every sampled document inside the held-out split.

    ``max_tail_fraction`` still names the held-out split, but bounds a different thing here: not
    how far back a window reaches, but that every sampled document lies within it.

    ``stride_offset`` shifts the starting phase by that many documents, so a second call at a
    different offset in ``[0, stride)`` samples a document set disjoint from an offset-0 call on
    the same blob. ``stride_offset=0`` reproduces the original, offset-less behaviour exactly.
    """
    num_documents = int(sequence_lengths.shape[0])
    split_size = int(max_tail_fraction * num_documents)
    if split_size < 1:
        raise ValueError(
            f"max_tail_fraction={max_tail_fraction} over {num_documents} documents leaves no "
            "held-out split to sample from"
        )
    split_start = num_documents - split_size

    mean_length = float(sequence_lengths[split_start:].mean())
    if mean_length <= 0:
        raise ValueError(f"the last {split_size} documents hold no tokens")
    estimate = max(1, -(-target_tokens // int(mean_length)))
    stride = max(1, split_size // (STRIDE_SPREAD * estimate))

    # The stride is only known once computed above, so the offset can only be validated here
    # rather than by the caller before this call.
    if not 0 <= stride_offset < stride:
        raise ValueError(
            f"stride_offset={stride_offset} must satisfy 0 <= stride_offset < stride, where "
            f"stride={stride} for this blob and these arguments"
        )

    indices = []
    cumulative = 0
    for index in range(split_start + stride_offset, num_documents, stride):
        indices.append(index)
        cumulative += int(sequence_lengths[index])
        if cumulative >= target_tokens:
            break
    else:
        raise ValueError(
            f"striding the last {split_size} documents at stride {stride} and offset "
            f"{stride_offset} reaches only {cumulative} tokens across {len(indices)} "
            f"documents, need {target_tokens}"
        )

    span_fraction = (indices[-1] + 1 - indices[0]) / num_documents
    return numpy.array(indices, dtype=numpy.int64), stride, span_fraction


def spread_window(
    sequence_lengths: numpy.ndarray,
    target_tokens: int,
    max_tail_fraction: float,
    stride_offset: int = 0,
) -> tuple[numpy.ndarray, int, float]:
    """Return ``(document_indices, stride, span_fraction)`` for a grid laid across the
    **entire** held-out split, rather than across the first ``1/STRIDE_SPREAD`` of it the way
    :func:`strided_window` does.

    ``span_fraction`` here is a fraction of the held-out split, not of the blob. The caller records
    it as ``span_fraction_of_split`` for that reason, because a provenance reader comparing a bare
    ``span_fraction`` across two assets would be comparing two different denominators. Unlike the
    other two windows: every decision behind this sampler is stated in split units, so a
    blob-fraction number here would read as a different, much smaller quantity next to them.

    The caller must assemble the asset with :func:`water_fill`'s per-document allocation rather
    than whole-document concatenation truncated to ``target_tokens``, or the span this grid
    buys collapses back to a contiguous-looking prefix of it.
    """
    num_documents = int(sequence_lengths.shape[0])
    split_size = int(max_tail_fraction * num_documents)
    if split_size < 1:
        raise ValueError(
            f"max_tail_fraction={max_tail_fraction} over {num_documents} documents leaves no "
            "held-out split to sample from"
        )
    split_start = num_documents - split_size

    split_lengths = sequence_lengths[split_start:]
    mean_length = float(split_lengths.mean())
    if mean_length <= 0:
        raise ValueError(f"the last {split_size} documents hold no tokens")
    if int(split_lengths.sum()) < target_tokens:
        raise ValueError(
            f"the held-out split holds only {int(split_lengths.sum())} tokens across "
            f"{split_size} documents, need {target_tokens}"
        )

    # Scanning from n_docs=1 lets the search settle on whatever tiny grid happens to land on a
    # long document: on the primary blob that is a 2-document grid at stride 9940, feasible only
    # because one of the two holds 79,798 tokens. Starting from the split-mean-based estimate
    # forbids that degenerate case, so a future "simplification" that drops this floor would
    # silently reintroduce it.
    n_docs = max(1, -(-target_tokens // int(mean_length)))

    # Exhaustive over every n_docs it visits and returns the first feasible one. Grids at n_docs
    # and n_docs + 1 are not nested, so there is no monotonicity property in feasibility to lean
    # on, only the arithmetic fact that split_size // n_docs shrinks as n_docs grows.
    while True:
        stride = split_size // n_docs
        if stride < 1:
            raise ValueError(
                f"no stride fits {n_docs} documents into a held-out split of {split_size} documents"
            )
        # The stride is only known once computed above, so the offset can only be validated
        # here. Because stride shrinks (or holds) as n_docs grows, a failure here recurs for
        # every larger n_docs too, so there is no point continuing the scan past it.
        if not 0 <= stride_offset < stride:
            raise ValueError(
                f"stride_offset={stride_offset} must satisfy 0 <= stride_offset < stride, "
                f"where stride={stride} for this blob and these arguments"
            )
        indices = split_start + stride_offset + numpy.arange(n_docs, dtype=numpy.int64) * stride
        if int(sequence_lengths[indices].sum()) >= target_tokens:
            break
        n_docs += 1

    # n_docs slots at stride split_size // n_docs cover the split by construction, so the
    # smallest feasible n_docs is also the largest stride among grids that span the whole split.
    # That is why no STRIDE_SPREAD-like margin is needed here: that constant exists in
    # strided_window to stop its walk running off the end, and this grid cannot run off the end
    # because it is built to fit.
    span_fraction = (int(indices[-1]) + 1 - int(indices[0])) / split_size
    return indices, stride, span_fraction


def water_fill(lengths: numpy.ndarray, target_tokens: int) -> tuple[numpy.ndarray, int]:
    """Return ``(allocation, cap)``: take ``min(length, cap)`` tokens from each document in
    ``lengths``, with the smallest integer ``cap`` such that the allocations sum to at least
    ``target_tokens``, then trim the overshoot from the largest allocations (ties by lowest
    index) so the sum is exactly ``target_tokens`` and no document drops to zero.

    ``sum(min(length, q))`` is non-decreasing in ``q``, because raising the cap can only add
    allocation, never remove it. That is a provable property of this function rather than the
    empirical, hole-riddled one the stride scan in :func:`spread_window` cannot claim, so
    bisecting on ``q`` here is exact where bisecting on a stride was not.
    """
    lengths = numpy.asarray(lengths, dtype=numpy.int64)
    total = int(lengths.sum())
    if total < target_tokens:
        raise ValueError(
            f"lengths sum to {total} tokens across {lengths.size} documents, need {target_tokens}"
        )

    lo, hi = 0, int(lengths.max())
    while lo < hi:
        mid = (lo + hi) // 2
        if int(numpy.minimum(lengths, mid).sum()) >= target_tokens:
            hi = mid
        else:
            lo = mid + 1
    cap = lo

    allocation = numpy.minimum(lengths, cap)
    excess = int(allocation.sum()) - target_tokens
    if excess > 0:
        # The largest allocations are exactly the capped ones (every length >= cap ties at cap),
        # so trimming there first is what keeps q a real bound on any single document's share.
        # lexsort's last key is primary, so this orders by allocation descending, index ascending.
        order = numpy.lexsort((numpy.arange(lengths.size), -allocation))
        allocation = allocation.copy()
        remaining = excess
        for i in order:
            if remaining <= 0:
                break
            take = min(int(allocation[i]) - 1, remaining)
            if take <= 0:
                continue
            allocation[i] -= take
            remaining -= take
        if remaining > 0:
            raise ValueError(
                f"cannot trim {excess} excess tokens down to target_tokens={target_tokens} "
                "without reducing some document to zero"
            )

    return allocation.astype(numpy.int64), cap


@dataclass(frozen=True)
class ProbeBatch:
    """A frozen probe batch, loaded from an ``assets/probe/*.npz`` file."""

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

    The sha256 checks are the point of this function, because every comparison across training
    steps and across arms assumes these token ids never change. A hand-edited or corrupted file
    must therefore fail the first time it is loaded, rather than silently measuring the router on
    different tokens than the runs it will be compared against.
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
    if not path.name.startswith(f"{role}_"):
        raise ValueError(
            f"{path}: role {role!r} disagrees with filename {path.name!r}, which must "
            f"start with '{role}_' (the writer's own naming rule, enforced here too)"
        )

    recomputed_provenance_sha256 = compute_provenance_sha256(provenance)
    recorded_provenance_sha256 = provenance.get("provenance_sha256")
    if recomputed_provenance_sha256 != recorded_provenance_sha256:
        raise ValueError(
            f"{path}: recomputed provenance_sha256 {recomputed_provenance_sha256} != recorded "
            f"{recorded_provenance_sha256} (a provenance field was edited after extraction)"
        )

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

    # Freeze both arrays as changes would mutate what the probe measures
    tokens.setflags(write=False)
    seq_labels.setflags(write=False)

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
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
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

import hashlib
import json
from pathlib import Path

import numpy
import pytest

from moe_congestion_routing.training.megatron_path import MegatronLMNotVendoredError, ensure_on_path
from moe_congestion_routing.training.probe_batch import (
    STRIDE_SPREAD,
    load_probe_batch,
    probe_micro_batches,
    strided_window,
    tail_window,
)

_ASSET = (
    Path(__file__).resolve().parents[3] / "assets" / "probe" / "dev_climblab_c1valid_16x1024.npz"
)


def _write_asset(path, tokens, seq_labels, provenance):
    numpy.savez(
        path, tokens=tokens, seq_labels=seq_labels, provenance=numpy.array(json.dumps(provenance))
    )


def _provenance(**overrides):
    base = {
        "asset_version": 1,
        "role": "dev",
        "data_prefix": "x",
        "num_documents": 10,
        "tail_doc_range": [8, 10],
        "tail_fraction": 0.2,
        "max_tail_fraction": 0.2,
        "S": 4,
        "seq_length": 3,
        "token_sha256": "",
        "seq_labels_sha256": "",
        "blob_dtype": "uint16",
    }
    base.update(overrides)
    return base


def test_load_probe_batch_on_committed_dev_asset():
    batch = load_probe_batch(_ASSET)
    assert batch.num_sequences == 16
    assert batch.seq_length == 1024
    assert batch.role == "dev"
    assert (batch.seq_labels == 1).all()


def test_load_probe_batch_rejects_unknown_role(tmp_path):
    tokens = numpy.arange(4 * 4, dtype=numpy.int32).reshape(4, 4)
    seq_labels = numpy.full(4, -1, dtype=numpy.int32)
    sha = hashlib.sha256(tokens.tobytes()).hexdigest()
    path = tmp_path / "bogus_x_4x3.npz"
    _write_asset(path, tokens, seq_labels, _provenance(role="bogus", token_sha256=sha))

    with pytest.raises(ValueError, match="role"):
        load_probe_batch(path)


def test_load_probe_batch_rejects_a_tampered_copy(tmp_path):
    """Flipping one token after extraction must be caught by the sha256 check."""
    copy_path = tmp_path / "dev_copy.npz"
    with numpy.load(_ASSET) as data:
        tokens = data["tokens"].copy()
        seq_labels = data["seq_labels"].copy()
        provenance = json.loads(str(data["provenance"]))

    tokens[0, 0] += 1  # This tampers with tokens, so the recorded sha256 no longer matches.
    _write_asset(copy_path, tokens, seq_labels, provenance)

    with pytest.raises(ValueError, match="sha256"):
        load_probe_batch(copy_path)


def test_load_probe_batch_rejects_tampered_seq_labels_values(tmp_path):
    """``seq_labels`` drives per-cluster slicing, so an edited label must be caught the same
    way an edited token is: the sha256 check, not silently accepted."""
    copy_path = tmp_path / "dev_copy.npz"
    with numpy.load(_ASSET) as data:
        tokens = data["tokens"].copy()
        seq_labels = data["seq_labels"].copy()
        provenance = json.loads(str(data["provenance"]))

    # This tampers with seq_labels, so the recorded seq_labels_sha256 no longer matches.
    seq_labels[0] = seq_labels[0] + 1
    _write_asset(copy_path, tokens, seq_labels, provenance)

    with pytest.raises(ValueError, match="sha256"):
        load_probe_batch(copy_path)


def test_load_probe_batch_rejects_seq_labels_wrong_length(tmp_path):
    """A ``seq_labels`` array shorter than ``tokens`` would silently misalign labels with
    sequences if the shapes were never compared against each other."""
    copy_path = tmp_path / "dev_copy.npz"
    with numpy.load(_ASSET) as data:
        tokens = data["tokens"].copy()
        provenance = json.loads(str(data["provenance"]))

    # This is tampered to 3 rows, whereas tokens has 16, so the shapes disagree.
    short_seq_labels = numpy.full(3, 1, dtype=numpy.int32)
    _write_asset(copy_path, tokens, short_seq_labels, provenance)

    with pytest.raises(ValueError, match="shape"):
        load_probe_batch(copy_path)


def test_load_probe_batch_rejects_a_tampered_role_field(tmp_path):
    """Editing only ``provenance["role"]`` without renaming the file must not survive: the
    filename no longer starts with the edited role's prefix, so the writer's own naming
    rule (enforced here at load time too) catches it before the hash check even runs."""
    copy_path = tmp_path / "dev_climblab_c1valid_16x1024.npz"
    with numpy.load(_ASSET) as data:
        tokens = data["tokens"].copy()
        seq_labels = data["seq_labels"].copy()
        provenance = json.loads(str(data["provenance"]))

    provenance["role"] = "standing"  # This tampers with role, leaving the filename as "dev_...".
    _write_asset(copy_path, tokens, seq_labels, provenance)

    with pytest.raises(ValueError, match="start with"):
        load_probe_batch(copy_path)


def test_load_probe_batch_rejects_a_tampered_role_field_with_matching_rename(tmp_path):
    """Renaming the copy to match the edited role satisfies the filename check, so only
    ``provenance_sha256``, which covers the role field itself, catches this. The two
    checks fail differently: this is the one the filename check alone would miss."""
    copy_path = tmp_path / "standing_climblab_c1valid_16x1024.npz"
    with numpy.load(_ASSET) as data:
        tokens = data["tokens"].copy()
        seq_labels = data["seq_labels"].copy()
        provenance = json.loads(str(data["provenance"]))

    # This tampers with role to match the new filename's "standing_" prefix.
    provenance["role"] = "standing"
    _write_asset(copy_path, tokens, seq_labels, provenance)

    with pytest.raises(ValueError, match="provenance_sha256"):
        load_probe_batch(copy_path)


def test_committed_asset_records_exactly_the_provenance_fields():
    batch = load_probe_batch(_ASSET)
    assert set(batch.provenance) == {
        "asset_version",
        "role",
        "data_prefix",
        "num_documents",
        "tail_doc_range",
        "tail_fraction",
        "max_tail_fraction",
        "S",
        "seq_length",
        "token_sha256",
        "seq_labels_sha256",
        "provenance_sha256",
        "blob_dtype",
    }


def test_load_probe_batch_freezes_both_tokens_and_seq_labels():
    """A ``skewed`` asset's per-cluster slicing reads ``seq_labels``, so those labels are as much
    a part of what the asset measures as ``tokens`` is, and both must come back read-only."""
    batch = load_probe_batch(_ASSET)
    assert batch.tokens.flags.writeable is False
    assert batch.seq_labels.flags.writeable is False


def test_tail_window_returns_the_minimal_tail_covering_the_target():
    lengths = numpy.array([3, 3, 3, 3])
    start, end, tail_fraction = tail_window(lengths, target_tokens=5, max_tail_fraction=1.0)
    assert (start, end) == (2, 4)
    assert tail_fraction == pytest.approx(0.5)


def test_tail_window_stops_as_soon_as_the_target_is_exactly_reached():
    """Pins the ``cumulative >= target_tokens`` boundary: a tail that lands on the target
    exactly must take that document and stop, not continue as ``>`` alone would."""
    lengths = numpy.array([3, 3, 3, 3])
    start, end, tail_fraction = tail_window(lengths, target_tokens=6, max_tail_fraction=1.0)
    assert (start, end) == (2, 4)
    assert tail_fraction == pytest.approx(0.5)


def test_tail_window_allows_the_tail_to_overshoot_the_target():
    """A document is never split, so the tail's total token count can exceed ``target_tokens``
    once the last needed document is included whole. The caller needs to truncate afterwards."""
    lengths = numpy.array([5, 5, 5, 5])
    start, end, tail_fraction = tail_window(lengths, target_tokens=12, max_tail_fraction=1.0)
    assert (start, end) == (1, 4)
    assert tail_fraction == pytest.approx(0.75)


def test_tail_window_rejects_a_tail_exceeding_max_tail_fraction():
    lengths = numpy.array([3, 3, 3, 3])
    with pytest.raises(ValueError, match="tail_fraction"):
        tail_window(lengths, target_tokens=5, max_tail_fraction=0.25)


def test_tail_window_rejects_insufficient_total_tokens():
    lengths = numpy.array([3, 3, 3, 3])
    with pytest.raises(ValueError, match="need"):
        tail_window(lengths, target_tokens=100, max_tail_fraction=1.0)


def _skip_unless_megatron_available():
    pytest.importorskip("triton", reason="megatron.core requires triton, unavailable on macOS")
    try:
        ensure_on_path()
    except MegatronLMNotVendoredError as e:
        pytest.skip(str(e))
    pytest.importorskip("megatron.core.datasets.gpt_dataset")


def test_probe_micro_batches_splits_and_matches_asset_tokens():
    _skip_unless_megatron_available()
    torch = pytest.importorskip("torch")

    batch = load_probe_batch(_ASSET)
    micro_batches = probe_micro_batches(
        batch, micro_batch_size=4, seq_length=batch.seq_length, eod_token=50256
    )

    assert len(micro_batches) == 4
    for mb in micro_batches:
        assert mb.keys() == {"tokens", "labels", "loss_mask", "position_ids", "attention_mask"}
        # This is None because create_attention_mask defaults to False.
        assert mb["attention_mask"] is None

    tokens = torch.cat([mb["tokens"] for mb in micro_batches], dim=0)
    labels = torch.cat([mb["labels"] for mb in micro_batches], dim=0)
    assert torch.equal(tokens, torch.from_numpy(batch.tokens[:, :-1].astype("int64")))
    assert torch.equal(labels, torch.from_numpy(batch.tokens[:, 1:].astype("int64")))


def test_probe_micro_batches_rejects_non_dividing_micro_batch_size():
    _skip_unless_megatron_available()
    pytest.importorskip("torch")

    batch = load_probe_batch(_ASSET)
    with pytest.raises(ValueError, match=r"16.*5|5.*16"):
        probe_micro_batches(batch, micro_batch_size=5, seq_length=batch.seq_length, eod_token=0)


def test_probe_micro_batches_rejects_seq_length_mismatch():
    _skip_unless_megatron_available()
    pytest.importorskip("torch")

    batch = load_probe_batch(_ASSET)
    with pytest.raises(ValueError, match=r"512.*1024|1024.*512"):
        probe_micro_batches(batch, micro_batch_size=4, seq_length=512, eod_token=0)


def test_probe_micro_batches_rejects_num_sequences_above_the_asset():
    _skip_unless_megatron_available()
    pytest.importorskip("torch")

    batch = load_probe_batch(_ASSET)
    with pytest.raises(ValueError, match=r"18.*16|16.*18"):
        probe_micro_batches(
            batch, micro_batch_size=6, num_sequences=18, seq_length=batch.seq_length, eod_token=0
        )


def test_probe_micro_batches_rejects_non_positive_num_sequences():
    _skip_unless_megatron_available()
    pytest.importorskip("torch")

    batch = load_probe_batch(_ASSET)
    with pytest.raises(ValueError, match="positive"):
        probe_micro_batches(
            batch, micro_batch_size=4, num_sequences=0, seq_length=batch.seq_length, eod_token=0
        )


# --- strided_window: a probe spread across the held-out split -------------------------------


def _uniform_lengths(num_documents, length=500):
    return numpy.full(num_documents, length, dtype=numpy.int64)


def test_strided_window_spreads_over_the_split_while_reading_the_same_documents():
    """The reason the mode exists, pinned as the property that does not depend on blob size.

    Both rules read about the same number of documents. The contiguous one reads them from one
    neighbourhood at the very end of the blob, and this one spreads them over the held-out split,
    covering a 1/STRIDE_SPREAD share of it. How many times wider that is grows with the blob, so
    it is not the thing to assert: it is 74x here and about 3700x on the primary blob, whose
    48965906 documents and 65-document window are both recorded in the committed asset.
    """
    lengths = _uniform_lengths(1_000_000)
    target = 32_784

    start, end, tail_fraction = tail_window(lengths, target, 0.01)
    indices, stride, span_fraction = strided_window(lengths, target, 0.01)

    assert stride > 1
    assert span_fraction == pytest.approx(0.01 / STRIDE_SPREAD, rel=0.05)
    assert span_fraction > tail_fraction * 50
    # Not at the cost of reading a different amount of the corpus.
    assert abs(indices.size - (end - start)) <= 2


def test_strided_window_stays_inside_the_held_out_split():
    # The guard that matters: a sampled document outside the split is a training document, and
    # the probe would then measure the router on data it was fit to.
    num_documents = 1_000_000
    lengths = _uniform_lengths(num_documents)

    indices, _, _ = strided_window(lengths, 32_784, 0.01)

    split_start = num_documents - int(0.01 * num_documents)
    assert int(indices.min()) >= split_start
    assert int(indices.max()) < num_documents


def test_strided_window_is_evenly_spaced_and_deterministic():
    lengths = _uniform_lengths(1_000_000)

    first, stride, _ = strided_window(lengths, 32_784, 0.01)
    again, stride_again, _ = strided_window(lengths, 32_784, 0.01)

    numpy.testing.assert_array_equal(first, again)
    assert stride == stride_again
    # Even spacing, so the sample carries no structure of its own beyond the stride.
    numpy.testing.assert_array_equal(numpy.diff(first), numpy.full(first.size - 1, stride))


def test_strided_window_reaches_the_token_target_when_documents_run_short():
    """Documents shorter than the split mean make the walk take more of them than estimated.

    The stride is laid out with room for that, so this must return a longer sample rather than
    running off the end of the split, which is the failure STRIDE_SPREAD exists to prevent.
    """
    num_documents = 1_000_000
    lengths = _uniform_lengths(num_documents)
    # Every document the stride will actually land on holds a fifth of the mean.
    split_start = num_documents - int(0.01 * num_documents)
    lengths[split_start::2] = 100

    indices, _, _ = strided_window(lengths, 32_784, 0.01)

    assert int(lengths[indices].sum()) >= 32_784
    assert int(indices.max()) < num_documents


def test_strided_window_refuses_a_split_that_cannot_supply_the_tokens():
    lengths = _uniform_lengths(10_000, length=1)
    with pytest.raises(ValueError, match="reaches only"):
        strided_window(lengths, 32_784, 0.01)


def test_strided_window_refuses_a_fraction_that_leaves_no_split():
    lengths = _uniform_lengths(50)
    with pytest.raises(ValueError, match="no held-out split"):
        strided_window(lengths, 100, 0.001)


# --- strided_window: stride_offset, the knob the third probe asset is drawn with -------------


def test_strided_window_default_offset_reproduces_the_original_indices():
    """``stride_offset=0`` must be indistinguishable from calling without the parameter at all,
    because every asset extracted before this parameter existed depends on that."""
    lengths = _uniform_lengths(1_000_000)

    without_param, stride_a, span_a = strided_window(lengths, 32_784, 0.01)
    with_zero, stride_b, span_b = strided_window(lengths, 32_784, 0.01, stride_offset=0)

    numpy.testing.assert_array_equal(without_param, with_zero)
    assert stride_a == stride_b
    assert span_a == pytest.approx(span_b)


def test_strided_window_offset_shifts_every_index_by_the_offset():
    """This is the test that bites: it fails if ``stride_offset`` is dropped from the index
    arithmetic (the loop would then always start at ``split_start``, ignoring the parameter),
    and it passes once the offset is folded into the starting index as the contract requires.
    """
    lengths = _uniform_lengths(1_000_000)

    base, stride, _ = strided_window(lengths, 32_784, 0.01, stride_offset=0)
    offset = stride // 2
    shifted, shifted_stride, _ = strided_window(lengths, 32_784, 0.01, stride_offset=offset)

    assert shifted_stride == stride
    # Same count and same stride, but every index moved by exactly `offset`.
    assert shifted.size == base.size
    numpy.testing.assert_array_equal(shifted, base + offset)


def test_strided_window_offsets_are_disjoint_from_the_zero_offset_draw():
    lengths = _uniform_lengths(1_000_000)

    base, stride, _ = strided_window(lengths, 32_784, 0.01, stride_offset=0)
    offset_draw, _, _ = strided_window(lengths, 32_784, 0.01, stride_offset=stride // 2)

    assert set(base.tolist()).isdisjoint(offset_draw.tolist())


def test_strided_window_rejects_an_offset_outside_the_stride():
    lengths = _uniform_lengths(1_000_000)
    _, stride, _ = strided_window(lengths, 32_784, 0.01)

    with pytest.raises(ValueError, match=f"stride={stride}"):
        strided_window(lengths, 32_784, 0.01, stride_offset=stride)


def test_strided_window_rejects_a_negative_offset():
    lengths = _uniform_lengths(1_000_000)

    with pytest.raises(ValueError, match="stride_offset"):
        strided_window(lengths, 32_784, 0.01, stride_offset=-1)

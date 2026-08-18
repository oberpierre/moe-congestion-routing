import json
import os
import stat

import numpy
import pytest
import torch

from moe_congestion_routing.metrics.router_probe import (
    ProbeCapture,
    active_capture,
    capturing,
    write_probe_dump,
)


def _map(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.bool)


def test_active_capture_is_none_outside_a_capturing_block():
    assert active_capture() is None


def test_capturing_activates_and_restores_on_exit():
    assert active_capture() is None
    with capturing(iteration=10, micro_batch_size=1, topk=2) as capture:
        assert active_capture() is capture
    assert active_capture() is None


def test_capturing_restores_even_if_the_body_raises():
    with pytest.raises(RuntimeError), capturing(iteration=0, micro_batch_size=1, topk=2):
        raise RuntimeError("boom")
    assert active_capture() is None


def test_record_gathers_combine_in_ascending_expert_order():
    capture = ProbeCapture(iteration=0, micro_batch_size=1, topk=2)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    probs = torch.zeros(2, 4)
    probs[0, 0], probs[0, 2] = 0.3, 0.7
    probs[1, 1], probs[1, 3] = 0.4, 0.6
    routing_map = _map([[1, 0, 1, 0], [0, 1, 0, 1]])
    capture.record(2, logits, probs, routing_map, None)
    arrays = capture.arrays()
    numpy.testing.assert_allclose(arrays["combine"][0], [[0.3, 0.7], [0.4, 0.6]])
    unpacked = numpy.unpackbits(arrays["routing_map"][0], axis=-1)[:, :4]
    numpy.testing.assert_array_equal(unpacked, routing_map.numpy().astype(numpy.uint8))


def test_record_concatenates_microbatches_along_the_token_axis():
    # micro_batch_size=1 makes each microbatch a single sequence of one token, so canonicalising
    # the token axis is a no-op here and concatenation order is exactly call order.
    capture = ProbeCapture(iteration=0, micro_batch_size=1, topk=2)
    logits_a = torch.zeros(2, 4)
    logits_b = torch.ones(3, 4)
    routing_map = _map([[1, 1, 0, 0]])
    probs = torch.zeros(1, 4)
    probs[0, 0], probs[0, 1] = 0.5, 0.5
    for row in range(2):
        capture.record(2, logits_a[row : row + 1], probs, routing_map, None)
    for row in range(3):
        capture.record(2, logits_b[row : row + 1], probs, routing_map, None)
    arrays = capture.arrays()
    assert arrays["logits"].shape == (1, 5, 4)


def test_record_stacks_layers_in_ascending_layer_number_order():
    capture = ProbeCapture(iteration=0, micro_batch_size=1, topk=2)
    routing_map = _map([[1, 1, 0, 0]])
    probs = torch.zeros(1, 4)
    probs[0, 0], probs[0, 1] = 0.5, 0.5
    capture.record(5, torch.full((1, 4), 5.0), probs, routing_map, None)
    capture.record(2, torch.full((1, 4), 2.0), probs, routing_map, None)
    arrays = capture.arrays()
    assert arrays["layer_numbers"].tolist() == [2, 5]
    assert arrays["logits"][0, 0, 0] == 2.0
    assert arrays["logits"][1, 0, 0] == 5.0


def test_record_rejects_a_token_with_the_wrong_number_of_selected_experts():
    capture = ProbeCapture(iteration=0, micro_batch_size=1, topk=2)
    routing_map = _map([[1, 1, 0, 0], [1, 0, 0, 0]])
    probs = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="expected exactly K=2"):
        capture.record(2, torch.zeros(2, 4), probs, routing_map, None)


def test_record_rejects_a_later_call_disagreeing_with_the_configured_topk():
    # K is fixed at construction from moe_router_topk (not inferred from the data), so a later
    # call whose data disagrees is caught on that call, not silently averaged in.
    capture = ProbeCapture(iteration=0, micro_batch_size=1, topk=2)
    probs2 = torch.zeros(1, 4)
    probs2[0, :2] = 0.5
    capture.record(2, torch.zeros(1, 4), probs2, _map([[1, 1, 0, 0]]), None)
    probs1 = torch.zeros(1, 4)
    probs1[0, 0] = 1.0
    with pytest.raises(ValueError, match="expected exactly K=2"):
        capture.record(3, torch.zeros(1, 4), probs1, _map([[1, 0, 0, 0]]), None)


def test_arrays_raises_when_nothing_was_recorded():
    with pytest.raises(ValueError, match="no MoE layer recorded"):
        ProbeCapture(iteration=0, micro_batch_size=1, topk=2).arrays()


def test_expert_bias_omitted_when_every_layer_has_none():
    capture = ProbeCapture(iteration=0, micro_batch_size=1, topk=2)
    probs = torch.zeros(1, 4)
    probs[0, :2] = 0.5
    capture.record(2, torch.zeros(1, 4), probs, _map([[1, 1, 0, 0]]), None)
    arrays = capture.arrays()
    assert "expert_bias" not in arrays


def test_expert_bias_present_when_every_layer_has_one():
    capture = ProbeCapture(iteration=0, micro_batch_size=1, topk=2)
    probs = torch.zeros(1, 4)
    probs[0, :2] = 0.5
    capture.record(2, torch.zeros(1, 4), probs, _map([[1, 1, 0, 0]]), torch.zeros(4))
    capture.record(5, torch.zeros(1, 4), probs, _map([[1, 1, 0, 0]]), torch.ones(4))
    arrays = capture.arrays()
    assert arrays["expert_bias"].shape == (2, 4)
    numpy.testing.assert_allclose(arrays["expert_bias"][1], numpy.ones(4))


def test_expert_bias_partial_across_layers_raises():
    capture = ProbeCapture(iteration=0, micro_batch_size=1, topk=2)
    probs = torch.zeros(1, 4)
    probs[0, :2] = 0.5
    capture.record(2, torch.zeros(1, 4), probs, _map([[1, 1, 0, 0]]), torch.zeros(4))
    capture.record(5, torch.zeros(1, 4), probs, _map([[1, 1, 0, 0]]), None)
    with pytest.raises(ValueError, match="expert_bias present on some layers"):
        capture.arrays()


def test_record_canonicalises_all_three_arrays_under_the_same_permutation():
    # A prior version of this test gave every row the same routing_map and combine, so a bug
    # permuting logits alone while leaving the other two in raw router order would still pass
    # here (and every other test in this file uses micro_batch_size=1, where the permutation is
    # the identity). Each row below selects its own expert and carries its own combine value,
    # both keyed to the row's canonical index n, so all three arrays must share one permutation
    # for every assertion below to hold.
    micro_batch_size, seq_length = 2, 3
    num_sequences = 2 * micro_batch_size
    num_experts = num_sequences * seq_length  # one expert slot per row, so selection is 1:1 with n
    capture = ProbeCapture(iteration=0, micro_batch_size=micro_batch_size, topk=1)
    for mb in range(2):
        logits_rows, probs_rows, map_rows = [], [], []
        for s in range(seq_length):
            for b in range(micro_batch_size):
                sequence = mb * micro_batch_size + b
                n = sequence * seq_length + s
                logits_rows.append([100 * sequence + s] * num_experts)
                probs_row = [0.0] * num_experts
                probs_row[n] = 0.5 + 0.01 * n
                probs_rows.append(probs_row)
                map_row = [False] * num_experts
                map_row[n] = True
                map_rows.append(map_row)
        capture.record(
            2,
            torch.tensor(logits_rows, dtype=torch.float32),
            torch.tensor(probs_rows, dtype=torch.float32),
            torch.tensor(map_rows, dtype=torch.bool),
            None,
        )

    arrays = capture.arrays()
    logits = arrays["logits"][0]
    unpacked = numpy.unpackbits(arrays["routing_map"][0], axis=-1, bitorder="big")[:, :num_experts]
    combine = arrays["combine"][0]
    for n in range(num_sequences * seq_length):
        sequence, position = divmod(n, seq_length)
        assert logits[n, 0] == 100 * sequence + position
        assert numpy.nonzero(unpacked[n])[0].tolist() == [n]
        numpy.testing.assert_allclose(combine[n, 0], 0.5 + 0.01 * n)


def _one_layer_capture() -> ProbeCapture:
    capture = ProbeCapture(iteration=6, micro_batch_size=1, topk=2)
    probs = torch.zeros(1, 4)
    probs[0, :2] = 0.5
    capture.record(2, torch.zeros(1, 4), probs, _map([[1, 1, 0, 0]]), None)
    return capture


def test_write_probe_dump_writes_arrays_and_metadata(tmp_path):
    capture = _one_layer_capture()
    path = tmp_path / "iter_0000006.npz"
    wrote = write_probe_dump(path, capture, {"iteration": 6, "role": "dev"})
    assert wrote is True
    with numpy.load(path) as data:
        assert data["logits"].dtype == numpy.float32
        assert data["combine"].dtype == numpy.float32
        meta = json.loads(str(data["metadata"]))
    assert meta["L"] == 1
    assert meta["N"] == 1
    assert meta["E"] == 4
    assert meta["K"] == 2
    assert meta["micro_batch_size"] == 1
    assert meta["token_axis_convention"]
    assert meta["routing_map_bitorder"] == "big"
    assert meta["role"] == "dev"
    assert meta["has_expert_bias"] is False


@pytest.mark.parametrize("umask, expected", [(0o022, 0o644), (0o002, 0o664)])
def test_write_probe_dump_uses_the_mode_open_would_have_given(
    tmp_path, monkeypatch, umask, expected
):
    calls = []

    def fake_umask(new):
        calls.append(new)
        return umask

    monkeypatch.setattr(os, "umask", fake_umask)
    path = tmp_path / "iter_0000006.npz"
    write_probe_dump(path, _one_layer_capture(), {"iteration": 6})

    assert stat.S_IMODE(path.stat().st_mode) == expected
    # POSIX has no getter, so reading the umask means setting it, whereas failing to restore it
    # would leave the whole training process at 0 and every later file world-writable.
    assert calls == [0, umask]


def test_write_probe_dump_skips_an_existing_complete_file(tmp_path, caplog):
    path = tmp_path / "iter_0000006.npz"
    write_probe_dump(path, _one_layer_capture(), {"iteration": 6})
    mtime_before = path.stat().st_mtime_ns
    with caplog.at_level("INFO"):
        wrote = write_probe_dump(path, _one_layer_capture(), {"iteration": 6})
    assert wrote is False
    assert path.stat().st_mtime_ns == mtime_before
    assert str(path) in caplog.text


def test_write_probe_dump_replaces_a_truncated_file_rather_than_skipping(tmp_path, caplog):
    # A truncated .npz at the final path is wreckage from an interrupted write (a crash or
    # preemption mid-savez), not a finished dump. Treating "exists" as "done" would poison this
    # step permanently, since every later resume would skip it forever.
    path = tmp_path / "iter_0000006.npz"
    path.write_bytes(b"not a real npz")
    with caplog.at_level("WARNING"):
        wrote = write_probe_dump(path, _one_layer_capture(), {"iteration": 6})
    assert wrote is True
    assert "not a complete" in caplog.text
    with numpy.load(path) as data:
        assert data["logits"].dtype == numpy.float32


def test_write_probe_dump_leaves_no_temp_file_behind_on_a_mid_write_failure(tmp_path, monkeypatch):
    import moe_congestion_routing.metrics.router_probe as router_probe_module

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(router_probe_module.numpy, "savez", _raise)
    path = tmp_path / "iter_0000006.npz"
    with pytest.raises(OSError, match="disk full"):
        write_probe_dump(path, _one_layer_capture(), {"iteration": 6})
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []

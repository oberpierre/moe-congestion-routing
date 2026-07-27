import os
import tempfile

import numpy
import pyarrow
import pyarrow.parquet
import pytest

from moe_congestion_routing.data.convert import convert_shards, download_shards, iter_token_rows
from moe_congestion_routing.training.megatron_path import MegatronLMNotVendoredError, ensure_on_path


def _write_parquet(path, rows, column="tokens"):
    table = pyarrow.table({column: pyarrow.array(rows, type=pyarrow.list_(pyarrow.int32()))})
    pyarrow.parquet.write_table(table, str(path))


def test_iter_token_rows_yields_each_rows_tokens(tmp_path):
    rows = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    path = tmp_path / "shard.parquet"
    _write_parquet(path, rows)
    assert list(iter_token_rows(path, "tokens")) == rows


def test_iter_token_rows_raises_on_missing_column(tmp_path):
    path = tmp_path / "shard.parquet"
    _write_parquet(path, [[1, 2]], column="tokens")
    with pytest.raises(KeyError, match="input_ids"):
        list(iter_token_rows(path, "input_ids"))


def test_download_shards_preserves_order_and_passes_repo_args(monkeypatch):
    """Concurrent fetch must return paths in input-shard order (convert_shards depends on it)."""
    import huggingface_hub

    seen = []

    def fake_download(*, repo_id, filename, repo_type, cache_dir):
        seen.append((repo_id, filename, repo_type, cache_dir))
        return f"/cache/{filename}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    shards = [f"cluster_1/cluster_1_{i:03d}.tokenized.parquet" for i in range(5)]
    paths = download_shards("nvidia/Nemotron-ClimbLab", shards, "/cache", max_workers=4)

    assert [p.name for p in paths] == [f"cluster_1_{i:03d}.tokenized.parquet" for i in range(5)]
    assert all(repo_id == "nvidia/Nemotron-ClimbLab" for repo_id, _, _, _ in seen)
    assert all(repo_type == "dataset" for _, _, repo_type, _ in seen)
    assert {filename for _, filename, _, _ in seen} == set(shards)


def test_download_shards_logs_thread_count(monkeypatch, caplog):
    """The effective thread count (capped by shard count) is logged at INFO."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: f"/c/{kw['filename']}")

    shards = ["a.parquet", "b.parquet", "c.parquet"]
    with caplog.at_level("INFO", logger="moe_congestion_routing.data.convert"):
        download_shards("repo", shards, "/cache", max_workers=8)  # capped to 3 by shard count

    assert "Downloading 3 shard(s) with 3 thread(s)" in caplog.text


def test_download_shards_empty_is_noop(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda **_: pytest.fail("should not download")
    )
    assert download_shards("repo", [], "/cache") == []


def test_convert_shards_yields_correct_results():
    """End-to-end test of convert_shards, including empty row skipping and EOD appending."""
    pytest.importorskip("triton", reason="megatron.core requires triton, unavailable on macOS")
    try:
        ensure_on_path()
    except MegatronLMNotVendoredError as e:
        pytest.skip(str(e))

    IndexedDataset = pytest.importorskip("megatron.core.datasets.indexed_dataset").IndexedDataset

    d = tempfile.mkdtemp()
    rows = [[1, 2, 3], [], [50256, 7], [10, 11, 12, 13]]  # includes an empty row (skipped)
    t = pyarrow.table({"tokens": pyarrow.array(rows, type=pyarrow.list_(pyarrow.int32()))})
    p1 = os.path.join(d, "a.parquet")
    pyarrow.parquet.write_table(t, p1)

    # (1) plain conversion, uint16, no EOD
    stats = convert_shards([p1], os.path.join(d, "plain"), dtype=numpy.uint16)
    ds = IndexedDataset(os.path.join(d, "plain"))
    nonempty = [r for r in rows if r]
    assert stats.num_documents == len(nonempty), stats
    assert stats.num_tokens == sum(len(r) for r in nonempty), stats
    assert len(ds) == len(nonempty)
    assert [ds[i].tolist() for i in range(len(ds))] == nonempty
    assert ds[0].dtype == numpy.uint16
    assert stats.bin_bytes == sum(len(r) for r in nonempty) * 2  # uint16 = 2 bytes/token

    # (2) append_eod appends the eod id and lengthens each document by one
    stats2 = convert_shards(
        [p1], os.path.join(d, "eod"), dtype=numpy.uint16, append_eod=True, eod_token_id=50256
    )
    ds2 = IndexedDataset(os.path.join(d, "eod"))
    assert [ds2[i].tolist() for i in range(len(ds2))] == [r + [50256] for r in nonempty]
    assert stats2.num_tokens == stats.num_tokens + len(nonempty)

    # (3) two shards concatenate in order
    stats3 = convert_shards([p1, p1], os.path.join(d, "double"), dtype=numpy.uint16)
    assert stats3.num_documents == 2 * len(nonempty)

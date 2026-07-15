"""Tests for the orchestration in ``run_preparation``.

These mock out ``download_shards`` and ``convert_shards`` (the network- and Megatron/GPU-bound
steps) so the wiring is exercised on CPU: that every distinct shard is downloaded exactly once
in a single concurrent batch, and that each job is converted from the right local paths in the
right order.
"""

import json
from pathlib import Path

from moe_congestion_routing.data import prepare_dataset
from moe_congestion_routing.data.climblab import ConversionJob
from moe_congestion_routing.data.config import DataPrepConfig
from moe_congestion_routing.data.convert import ConversionStats


def _config(tmp_path):
    return DataPrepConfig(
        output_dir=str(tmp_path / "out"),
        clusters=["cluster_1", "cluster_2"],
        held_out_clusters=["cluster_3"],
        shards_per_cluster=2,
        val_shards_per_cluster=1,
    )


def _cluster_to_shards():
    return {
        c: [f"{c}/{c}_{i:03d}.parquet" for i in range(2)]
        for c in ("cluster_1", "cluster_2", "cluster_3")
    }


def _patch_io(monkeypatch):
    """Replace the two heavy steps; return the recorded (download_calls, convert_calls)."""
    download_calls = []
    convert_calls = []

    def fake_download(dataset_repo, shards, cache_dir, **_):
        shards = list(shards)
        download_calls.append(shards)
        # local path is a stable, invertible function of the repo-relative shard path
        return [Path(f"/local/{s.replace('/', '__')}") for s in shards]

    def fake_convert(parquet_paths, output_prefix, **_):
        paths = list(parquet_paths)
        convert_calls.append((Path(output_prefix).name, paths))
        return ConversionStats(
            prefix=Path(output_prefix).name,
            num_documents=len(paths),
            num_tokens=10 * len(paths),
            bin_bytes=20 * len(paths),
            idx_bytes=5,
        )

    monkeypatch.setattr(prepare_dataset, "download_shards", fake_download)
    monkeypatch.setattr(prepare_dataset, "convert_shards", fake_convert)
    return download_calls, convert_calls


def test_downloads_every_shard_once_and_maps_them_per_job(tmp_path, monkeypatch):
    download_calls, convert_calls = _patch_io(monkeypatch)

    prepared = prepare_dataset.run_preparation(_config(tmp_path), _cluster_to_shards())

    # (1) exactly one concurrent download batch, holding every distinct shard once, in order.
    assert len(download_calls) == 1
    assert download_calls[0] == [
        "cluster_1/cluster_1_000.parquet",
        "cluster_1/cluster_1_001.parquet",
        "cluster_2/cluster_2_000.parquet",
        "cluster_2/cluster_2_001.parquet",
        "cluster_3/cluster_3_000.parquet",
        "cluster_3/cluster_3_001.parquet",
    ]

    # (2) each job converted from the local path(s) for its shards, preserving order.
    def local(shard):
        return Path(f"/local/{shard.replace('/', '__')}")

    assert dict(convert_calls) == {
        "cluster_1_train": [local("cluster_1/cluster_1_000.parquet")],
        "cluster_1_valid": [local("cluster_1/cluster_1_001.parquet")],
        "cluster_2_train": [local("cluster_2/cluster_2_000.parquet")],
        "cluster_2_valid": [local("cluster_2/cluster_2_001.parquet")],
        "cluster_3_holdout": [
            local("cluster_3/cluster_3_000.parquet"),
            local("cluster_3/cluster_3_001.parquet"),
        ],
    }

    # (3) manifest records one prefix per job with its source shards preserved.
    manifest = json.loads((Path(_config(tmp_path).output_dir) / "manifest.json").read_text())
    by_prefix = {p["prefix"]: p for p in manifest["prefixes"]}
    assert set(by_prefix) == {p.prefix for p in prepared}
    assert by_prefix["cluster_3_holdout"]["shards"] == [
        "cluster_3/cluster_3_000.parquet",
        "cluster_3/cluster_3_001.parquet",
    ]


def test_shared_shard_across_jobs_is_downloaded_once(tmp_path, monkeypatch):
    """If two jobs ever reference the same shard, it must be fetched once and mapped to both."""
    download_calls, convert_calls = _patch_io(monkeypatch)

    shared = "cluster_1/cluster_1_000.parquet"
    jobs = [
        ConversionJob("job_a", "train", "cluster_1", (shared,)),
        ConversionJob("job_b", "valid", "cluster_1", (shared,)),
    ]
    monkeypatch.setattr(prepare_dataset, "plan_conversions", lambda *a, **k: jobs)

    prepare_dataset.run_preparation(_config(tmp_path), _cluster_to_shards())

    assert download_calls == [[shared]]  # deduped: one shard, one batch
    local = Path(f"/local/{shared.replace('/', '__')}")
    assert dict(convert_calls) == {"job_a": [local], "job_b": [local]}

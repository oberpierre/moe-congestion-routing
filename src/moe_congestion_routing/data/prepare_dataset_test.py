"""Tests for the orchestration in ``run_preparation``.

These mock out ``download_shards`` and ``convert_shards`` (the network- and Megatron/GPU-bound
steps) so the wiring is exercised on CPU: that every distinct shard is downloaded exactly once
in a single concurrent batch, and that each job is converted from the right local paths in the
right order.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from moe_congestion_routing.data import prepare_dataset
from moe_congestion_routing.data.climblab import ConversionJob
from moe_congestion_routing.data.config import DataPrepConfig
from moe_congestion_routing.data.convert import ConversionStats


def _config(tmp_path, **overrides):
    base = {
        "output_dir": str(tmp_path / "out"),
        "clusters": ["cluster_1", "cluster_2"],
        "held_out_clusters": ["cluster_3"],
        "shards_per_cluster": 2,
        "val_shards_per_cluster": 1,
    }
    base.update(overrides)
    return DataPrepConfig(**base)


def _cluster_to_shards():
    return {
        c: [f"{c}/{c}_{i:03d}.parquet" for i in range(2)]
        for c in ("cluster_1", "cluster_2", "cluster_3")
    }


def _patch_io(monkeypatch):
    """Replace the two heavy steps; record their args on a namespace.

    ``rec.download_calls`` (per-call shard lists), ``rec.cache_dirs`` / ``rec.download_workers``
    (the cache dir + thread count passed to ``download_shards``), and ``rec.convert_calls``
    (``(prefix_name, local_paths)`` per job).
    """
    rec = SimpleNamespace(download_calls=[], convert_calls=[], cache_dirs=[], download_workers=[])

    def fake_download(dataset_repo, shards, cache_dir, *, max_workers=None, **_):
        shards = list(shards)
        rec.download_calls.append(shards)
        rec.cache_dirs.append(cache_dir)
        rec.download_workers.append(max_workers)
        # local path is a stable, invertible function of the repo-relative shard path
        return [Path(f"/local/{s.replace('/', '__')}") for s in shards]

    def fake_convert(parquet_paths, output_prefix, **_):
        paths = list(parquet_paths)
        rec.convert_calls.append((Path(output_prefix).name, paths))
        return ConversionStats(
            prefix=Path(output_prefix).name,
            num_documents=len(paths),
            num_tokens=10 * len(paths),
            bin_bytes=20 * len(paths),
            idx_bytes=5,
        )

    monkeypatch.setattr(prepare_dataset, "download_shards", fake_download)
    monkeypatch.setattr(prepare_dataset, "convert_shards", fake_convert)
    return rec


def test_downloads_every_shard_once_and_maps_them_per_job(tmp_path, monkeypatch):
    # convert_workers=1 keeps conversion inline so the monkeypatched convert_shards is reached
    # (a ProcessPoolExecutor would run the real one in a subprocess, out of the patch's scope).
    rec = _patch_io(monkeypatch)

    config = _config(tmp_path, convert_workers=1)
    prepared = prepare_dataset.run_preparation(config, _cluster_to_shards())

    # (0) shards are cached in the config-resolved cache dir (<output_dir>/_hf_cache by default),
    #     and downloaded with the config's thread count.
    assert rec.cache_dirs == [config.cache_path]
    assert config.cache_path == Path(config.output_dir) / "_hf_cache"
    assert rec.download_workers == [config.download_workers]

    # (1) exactly one concurrent download batch, holding every distinct shard once, in order.
    assert len(rec.download_calls) == 1
    assert rec.download_calls[0] == [
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

    assert dict(rec.convert_calls) == {
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
    manifest = json.loads((Path(config.output_dir) / "manifest.json").read_text())
    by_prefix = {p["prefix"]: p for p in manifest["prefixes"]}
    assert set(by_prefix) == {p.prefix for p in prepared}
    assert by_prefix["cluster_3_holdout"]["shards"] == [
        "cluster_3/cluster_3_000.parquet",
        "cluster_3/cluster_3_001.parquet",
    ]


def test_shared_shard_across_jobs_is_downloaded_once(tmp_path, monkeypatch):
    """If two jobs ever reference the same shard, it must be fetched once and mapped to both."""
    rec = _patch_io(monkeypatch)

    shared = "cluster_1/cluster_1_000.parquet"
    jobs = [
        ConversionJob("job_a", "train", "cluster_1", (shared,)),
        ConversionJob("job_b", "valid", "cluster_1", (shared,)),
    ]
    monkeypatch.setattr(prepare_dataset, "plan_conversions", lambda *a, **k: jobs)

    prepare_dataset.run_preparation(_config(tmp_path, convert_workers=1), _cluster_to_shards())

    assert rec.download_calls == [[shared]]  # deduped: one shard, one batch
    local = Path(f"/local/{shared.replace('/', '__')}")
    assert dict(rec.convert_calls) == {"job_a": [local], "job_b": [local]}


class _RecordingPool:
    """Stand-in for ProcessPoolExecutor: records how it was built and runs ``map`` inline.

    Running inline (in-process) lets us assert the requested worker count without spawning
    real subprocesses, and keeps the monkeypatched ``convert_shards`` in scope.
    """

    def __init__(self, max_workers=None, mp_context=None):
        self.max_workers = max_workers
        self.mp_context = mp_context

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, *iterables):
        # Non-strict like the real Executor.map: run_preparation passes an infinite repeat(config).
        return [fn(*args) for args in zip(*iterables)]  # noqa: B905


def _record_pools(monkeypatch):
    """Patch ProcessPoolExecutor with the recording stand-in; return the list of pools built."""
    created = []

    def factory(max_workers=None, mp_context=None):
        pool = _RecordingPool(max_workers=max_workers, mp_context=mp_context)
        created.append(pool)
        return pool

    monkeypatch.setattr(prepare_dataset, "ProcessPoolExecutor", factory)
    return created


def test_convert_worker_count_is_capped_by_job_count(tmp_path, monkeypatch):
    """Asking for more workers than jobs caps at the job count, and the pool uses spawn."""
    _patch_io(monkeypatch)
    created = _record_pools(monkeypatch)

    # config plans 5 jobs; request 100 workers -> min(100, 5) = 5.
    prepare_dataset.run_preparation(_config(tmp_path, convert_workers=100), _cluster_to_shards())

    assert [pool.max_workers for pool in created] == [5]
    assert created[0].mp_context.get_start_method() == "spawn"


def test_convert_worker_count_defaults_to_cpu_count(tmp_path, monkeypatch):
    """convert_workers=None uses os.cpu_count(), still bounded by the job count."""
    _patch_io(monkeypatch)
    created = _record_pools(monkeypatch)
    monkeypatch.setattr(prepare_dataset.os, "cpu_count", lambda: 3)

    prepare_dataset.run_preparation(_config(tmp_path, convert_workers=None), _cluster_to_shards())

    assert [pool.max_workers for pool in created] == [3]  # min(cpu_count=3, jobs=5)


def test_convert_workers_one_runs_inline_without_a_pool(tmp_path, monkeypatch):
    """convert_workers=1 takes the inline path: no process pool is ever constructed."""
    rec = _patch_io(monkeypatch)
    created = _record_pools(monkeypatch)

    prepare_dataset.run_preparation(_config(tmp_path, convert_workers=1), _cluster_to_shards())

    assert created == []  # pool never built
    assert len(rec.convert_calls) == 5  # yet all 5 jobs still converted


def test_download_thread_count_comes_from_config(tmp_path, monkeypatch):
    """download_shards receives config.download_workers as its thread count."""
    rec = _patch_io(monkeypatch)

    prepare_dataset.run_preparation(
        _config(tmp_path, convert_workers=1, download_workers=4), _cluster_to_shards()
    )

    assert rec.download_workers == [4]


def test_logs_convert_process_count_for_pool(tmp_path, monkeypatch, caplog):
    """The parallel path logs how many worker processes are used (here capped to 5 jobs)."""
    _patch_io(monkeypatch)
    _record_pools(monkeypatch)

    with caplog.at_level("INFO", logger="moe_congestion_routing.data.prepare_dataset"):
        prepare_dataset.run_preparation(
            _config(tmp_path, convert_workers=100), _cluster_to_shards()
        )

    assert "across 5 worker process(es)" in caplog.text


def test_logs_convert_single_process_for_inline(tmp_path, monkeypatch, caplog):
    """The inline path logs that a single process is used."""
    _patch_io(monkeypatch)

    with caplog.at_level("INFO", logger="moe_congestion_routing.data.prepare_dataset"):
        prepare_dataset.run_preparation(_config(tmp_path, convert_workers=1), _cluster_to_shards())

    assert "inline in a single process" in caplog.text


def test_parallel_convert_path_builds_every_prefix(tmp_path, monkeypatch):
    """The real ProcessPoolExecutor path (convert_workers>1): convert_shards runs for real in
    worker processes, so this catches pickling / ordering / independence bugs the mocked tests
    (inline path only) cannot. Gated on megatron like the convert_shards test."""
    import pyarrow
    import pyarrow.parquet

    pytest.importorskip("triton", reason="megatron.core requires triton, unavailable on macOS")
    from moe_congestion_routing.training.megatron_path import (
        MegatronLMNotVendoredError,
        ensure_on_path,
    )

    try:
        ensure_on_path()
    except MegatronLMNotVendoredError as e:
        pytest.skip(str(e))
    IndexedDataset = pytest.importorskip("megatron.core.datasets.indexed_dataset").IndexedDataset

    # A distinct tiny parquet per planned shard; unique token values let us verify concat order.
    c2s = _cluster_to_shards()
    rows_by_shard, local_by_shard = {}, {}
    for i, cluster in enumerate(sorted(c2s)):
        for j, shard in enumerate(c2s[cluster]):
            rows = [[i * 10 + j, i * 10 + j + 1]]  # one document, two tokens, unique per shard
            path = tmp_path / f"{cluster}__{j}.parquet"
            table = pyarrow.table({"tokens": pyarrow.array(rows, pyarrow.list_(pyarrow.int32()))})
            pyarrow.parquet.write_table(table, str(path))
            rows_by_shard[shard], local_by_shard[shard] = rows, path

    monkeypatch.setattr(
        prepare_dataset,
        "download_shards",
        lambda repo, shards, cache, **_: [local_by_shard[s] for s in shards],
    )

    prepared = prepare_dataset.run_preparation(_config(tmp_path, convert_workers=2), c2s)

    out = Path(_config(tmp_path).output_dir)
    for prefix in prepared:
        ds = IndexedDataset(str(out / prefix.prefix))
        assert len(ds) == prefix.num_documents
    # cluster_3's two holdout shards must be concatenated in shard order, across the pool.
    holdout = IndexedDataset(str(out / "cluster_3_holdout"))
    assert [holdout[k].tolist() for k in range(len(holdout))] == (
        rows_by_shard["cluster_3/cluster_3_000.parquet"]
        + rows_by_shard["cluster_3/cluster_3_001.parquet"]
    )

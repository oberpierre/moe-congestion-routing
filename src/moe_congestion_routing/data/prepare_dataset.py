"""Orchestrate ClimbLab preparation end to end: list shards -> plan splits -> download ->
convert -> write a manifest.

The manifest (``<output_dir>/manifest.json``) records the exact config, and for every built
prefix its role, source cluster, source shards, and token/byte counts. It is the provenance
record the verification/analysis and determinism scripts read back, so they never have to
re-list the remote dataset.
"""

import dataclasses
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path

from moe_congestion_routing.data.climblab import plan_conversions
from moe_congestion_routing.data.config import DataPrepConfig
from moe_congestion_routing.data.convert import ConversionStats, convert_shards, download_shards


@dataclass(frozen=True)
class PreparedPrefix:
    """One built ``.bin``/``.idx`` prefix plus its source and counts."""

    prefix: str
    role: str
    cluster: str
    shards: list[str]
    num_documents: int
    num_tokens: int
    bin_bytes: int
    idx_bytes: int


def _convert_job(
    local_shards: list[Path], output_prefix: str, config: DataPrepConfig
) -> ConversionStats:
    """Convert one job's shards into a ``.bin``/``.idx`` prefix.

    Module-level (not a closure) so a ``ProcessPoolExecutor`` can pickle and dispatch it. Reads
    ``convert_shards`` through the module global on purpose, so the inline (``workers == 1``)
    path stays monkeypatch-able in tests.
    """
    return convert_shards(
        local_shards,
        output_prefix,
        dtype=config.numpy_dtype,
        token_column=config.token_column,
        append_eod=config.append_eod,
        eod_token_id=config.eod_token_id,
    )


def run_preparation(
    config: DataPrepConfig,
    cluster_to_shards: dict[str, list[str]] | None = None,
    *,
    convert_workers: int | None = None,
) -> list[PreparedPrefix]:
    """Build all planned prefixes and write the manifest.

    Args:
        config: the preparation config. Shards are cached in ``config.cache_path``.
        cluster_to_shards: pre-listed ``{cluster: [shard paths]}``; if ``None`` it is fetched
            from the HF Hub. Injecting it lets callers (and tests) skip the network listing.
        convert_workers: number of *processes* (not threads) used to convert jobs in parallel,
            as conversion is GIL-bound (pyarrow decode -> Python lists -> numpy). As each worker
            holds a whole shard's tokens in memory and shards reach up to 4 GB, take care of OOM.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config.cache_path
    jobs = plan_conversions(config, cluster_to_shards)

    # Download shards across all jobs in one concurrent batch, so shards from different jobs
    # overlap instead of downloading sequentially.
    all_shards = list(dict.fromkeys(shard for job in jobs for shard in job.shards))
    local_paths = download_shards(config.dataset_repo, all_shards, cache_dir)
    shard_to_local = dict(zip(all_shards, local_paths, strict=True))

    # Convert jobs in parallel across processes. Each job reads distinct shards and writes a
    # distinct output prefix, so they are independent; results are collected in job order.
    local_per_job = [[shard_to_local[shard] for shard in job.shards] for job in jobs]
    prefixes = [str(output_dir / job.prefix) for job in jobs]
    workers = min(convert_workers or os.cpu_count() or 1, len(jobs)) if jobs else 0

    if workers <= 1:
        stats = [
            _convert_job(shards, prefix, config)
            for shards, prefix in zip(local_per_job, prefixes, strict=True)
        ]
    else:
        # spawn (not fork): the parent is multi-threaded here (pyarrow/torch keep worker
        # threads), and forking a multi-threaded process risks deadlocking the child.
        pool_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=pool_ctx) as pool:
            stats = list(pool.map(_convert_job, local_per_job, prefixes, repeat(config)))

    prepared = [
        PreparedPrefix(
            prefix=job.prefix,
            role=job.role,
            cluster=job.cluster,
            shards=list(job.shards),
            num_documents=stat.num_documents,
            num_tokens=stat.num_tokens,
            bin_bytes=stat.bin_bytes,
            idx_bytes=stat.idx_bytes,
        )
        for job, stat in zip(jobs, stats, strict=True)
    ]

    write_manifest(config, prepared, output_dir / "manifest.json")
    return prepared


def write_manifest(
    config: DataPrepConfig, prepared: list[PreparedPrefix], path: str | Path
) -> None:
    """Write the provenance manifest as json."""
    payload = {
        "config": dataclasses.asdict(config),
        "prefixes": [dataclasses.asdict(p) for p in prepared],
    }
    Path(path).write_text(json.dumps(payload, indent=2))

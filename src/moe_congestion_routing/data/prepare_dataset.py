"""Orchestrate ClimbLab preparation end to end: list shards -> plan splits -> download ->
convert -> write a manifest.

The manifest (``<output_dir>/manifest.json``) records the exact config, and for every built
prefix its role, source cluster, source shards, and token/byte counts. It is the provenance
record the verification/analysis and determinism scripts read back, so they never have to
re-list the remote dataset.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from moe_congestion_routing.data.climblab import plan_conversions
from moe_congestion_routing.data.config import DataPrepConfig
from moe_congestion_routing.data.convert import convert_shards, download_shards


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


def run_preparation(
    config: DataPrepConfig,
    cluster_to_shards: dict[str, list[str]] | None = None,
    download_dir: str | Path | None = None,
) -> list[PreparedPrefix]:
    """Build all planned prefixes and write the manifest.

    Args:
        config: the preparation config.
        cluster_to_shards: pre-listed ``{cluster: [shard paths]}``; if ``None`` it is fetched
            from the HF Hub. Injecting it lets callers (and tests) skip the network listing.
        download_dir: where shards are cached; defaults to ``<output_dir>/_hf_cache``.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(download_dir) if download_dir is not None else output_dir / "_hf_cache"
    jobs = plan_conversions(config, cluster_to_shards)

    # Download shards across all jobs in one concurrent batch, so shards from different jobs
    # overlap instead of downloading sequentially.
    all_shards = list(dict.fromkeys(shard for job in jobs for shard in job.shards))
    local_paths = download_shards(config.dataset_repo, all_shards, cache_dir)
    shard_to_local = dict(zip(all_shards, local_paths, strict=True))

    prepared: list[PreparedPrefix] = []
    for job in jobs:
        local_shards = [shard_to_local[shard] for shard in job.shards]
        stats = convert_shards(
            local_shards,
            output_dir / job.prefix,
            dtype=config.numpy_dtype,
            token_column=config.token_column,
            append_eod=config.append_eod,
            eod_token_id=config.eod_token_id,
        )
        prepared.append(
            PreparedPrefix(
                prefix=job.prefix,
                role=job.role,
                cluster=job.cluster,
                shards=list(job.shards),
                num_documents=stats.num_documents,
                num_tokens=stats.num_tokens,
                bin_bytes=stats.bin_bytes,
                idx_bytes=stats.idx_bytes,
            )
        )

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

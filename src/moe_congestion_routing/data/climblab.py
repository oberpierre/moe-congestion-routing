"""Cluster/shard discovery and train/validation split planning for ClimbLab.

Two concerns, deliberately separated so the split logic is unit-testable without a network:

* `plan_conversions` is pure — given ``{cluster: [shard paths]}`` it returns the list of
  conversion jobs (one output ``.bin``/``.idx`` prefix per ``(cluster, split)``), applying
  the per-cluster shard budget and carving disjoint train / held-out-validation shard sets.
* `list_cluster_shards` / `available_clusters` are thin Hugging Face Hub wrappers that
  enumerate the parquet shards actually present in the dataset repo.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath

from moe_congestion_routing.data.config import DataPrepConfig

# Split roles. "train" and the two validation flavours the experiment needs: per-cluster
# in-distribution "valid" shards, and whole "holdout" clusters for domain shift (E3).
TRAIN = "train"
VALID = "valid"
HOLDOUT = "holdout"


@dataclass(frozen=True)
class ConversionJob:
    """One ``.bin``/``.idx`` prefix to build from a disjoint set of parquet shards."""

    prefix: str
    """Output prefix name (no directory, no extension), e.g. ``cluster_00_train``."""

    role: str
    """One of ``TRAIN`` / ``VALID`` / ``HOLDOUT``."""

    cluster: str
    """Source cluster folder name (kept for analysis, e.g. per-cluster BPB)."""

    shards: tuple[str, ...]
    """Repo-relative parquet paths feeding this prefix."""


def _prefix_token(cluster: str) -> str:
    """Turn a (possibly nested) cluster folder name into a filesystem-prefix-safe token."""
    return cluster.strip("/").replace("/", "_")


def _budgeted(shards: list[str], cap: int | None) -> list[str]:
    """Deterministically take up to ``cap`` shards (sorted, so selection is reproducible)."""
    ordered = sorted(shards)
    return ordered if cap is None else ordered[:cap]


def plan_conversions(
    config: DataPrepConfig,
    cluster_to_shards: dict[str, list[str]] | None = None,
) -> list[ConversionJob]:
    """Plan the ``.bin``/``.idx`` prefixes to build (pure; no I/O).

    For each training cluster: take up to ``shards_per_cluster`` shards (deterministically),
    hold out the last ``val_shards_per_cluster`` of them as a ``_valid`` prefix, and emit the
    rest as a ``_train`` prefix. Each held-out cluster becomes a single ``_holdout`` prefix.
    Train and validation shard sets are disjoint by construction (no leakage).

    Raises:
        KeyError: if a requested cluster is absent from ``cluster_to_shards``.
        ValueError: if a cluster yields no shards, or no train shard remains after holdout.
    """
    jobs: list[ConversionJob] = []

    if cluster_to_shards is None:
        cluster_to_shards = list_cluster_shards(config)

    for cluster in config.clusters:
        if cluster not in cluster_to_shards:
            raise KeyError(f"cluster {cluster!r} not found among available shards")
        shards = _budgeted(cluster_to_shards[cluster], config.shards_per_cluster)
        if not shards:
            raise ValueError(f"cluster {cluster!r} has no parquet shards")

        n_val = config.val_shards_per_cluster
        if n_val >= len(shards):
            raise ValueError(
                f"cluster {cluster!r}: val_shards_per_cluster={n_val} leaves no train shards "
                f"(only {len(shards)} shard(s) available after the budget)"
            )
        val_shards = shards[len(shards) - n_val :] if n_val else []
        train_shards = shards[: len(shards) - n_val] if n_val else shards

        token = _prefix_token(cluster)
        jobs.append(ConversionJob(f"{token}_{TRAIN}", TRAIN, cluster, tuple(train_shards)))
        if val_shards:
            jobs.append(ConversionJob(f"{token}_{VALID}", VALID, cluster, tuple(val_shards)))

    for cluster in config.held_out_clusters:
        if cluster not in cluster_to_shards:
            raise KeyError(f"held-out cluster {cluster!r} not found among available shards")
        shards = _budgeted(cluster_to_shards[cluster], config.shards_per_cluster)
        if not shards:
            raise ValueError(f"held-out cluster {cluster!r} has no parquet shards")
        token = _prefix_token(cluster)
        jobs.append(ConversionJob(f"{token}_{HOLDOUT}", HOLDOUT, cluster, tuple(shards)))

    return jobs


def _group_parquet_by_cluster(files: list[str], clusters: list[str]) -> dict[str, list[str]]:
    """Group repo-relative parquet paths by which requested cluster folder they live under.

    A file belongs to ``cluster`` if ``cluster`` appears as one of its parent path segments.
    We accept all files ending in ``.parquet`` ignoring the rest and return sorted lists/cluster.
    """
    wanted = set(clusters)
    grouped: dict[str, list[str]] = {c: [] for c in clusters}
    for f in files:
        if not f.endswith(".parquet"):
            continue
        parents = set(PurePosixPath(f).parts[:-1])
        for cluster in wanted & parents:
            grouped[cluster].append(f)
    return grouped


def list_cluster_shards(
    config: DataPrepConfig, clusters: list[str] | None = None
) -> dict[str, list[str]]:
    """List parquet shards per requested cluster from the HF dataset repo (network)."""
    from huggingface_hub import HfApi

    requested = clusters if clusters is not None else [*config.clusters, *config.held_out_clusters]
    files = HfApi().list_repo_files(config.dataset_repo, repo_type="dataset")
    grouped = _group_parquet_by_cluster(files, requested)

    missing = sorted(c for c, s in grouped.items() if not s)
    if missing:
        raise ValueError(
            f"no parquet shards found for cluster(s) {missing} in {config.dataset_repo}; "
            f"check names against available_clusters()"
        )
    return {c: sorted(s) for c, s in grouped.items()}


def available_clusters(dataset_repo: str) -> list[str]:
    """Discover cluster folder names (parents of parquet files) in the HF dataset repo."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(dataset_repo, repo_type="dataset")
    clusters = {
        PurePosixPath(f).parts[-2]
        for f in files
        if f.endswith(".parquet") and len(PurePosixPath(f).parts) >= 2
    }
    return sorted(clusters)

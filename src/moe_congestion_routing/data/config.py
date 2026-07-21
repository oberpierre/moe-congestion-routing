"""Configuration for Nemotron-Climb data preparation (ClimbLab + ClimbMix variants)."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy
import yaml

# The Climb datasets ship pre-tokenized with the GPT-2 tokenizer (vocab 50257).
# Fits into uint16 but we support int32 as well.
_DTYPES: dict[str, type[numpy.number]] = {"uint16": numpy.uint16, "int32": numpy.int32}


@dataclass(frozen=True)
class VariantSpec:
    """Immutable description of one Nemotron-Climb dataset variant."""

    repo: str
    """Hugging Face dataset repo id the shards live in."""

    layout: str
    """``clustered`` (shards grouped by cluster folder, e.g. ClimbLab) or ``flat`` (a single set
    of shards, e.g. ClimbMix). Drives discovery and split planning in ``climb.py``."""

    shard_prefix: str = ""
    """Flat variants only: the repo sub-path the shards live under (e.g. ``climbmix_small/``).
    Empty means the repo root."""


# Built-in variants. Each is a concrete (repo, layout, discovery) bundle; the config just names
# one. Add ClimbMix-full (jsonl) here when needed — it only adds a format axis to the reader.
VARIANTS: dict[str, VariantSpec] = {
    "climblab": VariantSpec("nvidia/Nemotron-ClimbLab", "clustered"),
    "climbmix_small": VariantSpec("nvidia/Nemotron-ClimbMix", "flat", "climbmix_small/"),
}


@dataclass(frozen=True)
class DataPrepConfig:
    """Everything the preparation + loading pipeline needs, loadable from a yaml file."""

    variant: str
    """Dataset variant: a key into :data:`VARIANTS` (``climblab`` | ``climbmix_small``). Drives the
    repo, shard discovery, and split layout (clustered vs flat)."""

    output_dir: str
    """Directory the ``.bin``/``.idx`` prefixes are written to."""

    # --- clustered variants (ClimbLab) only ---
    clusters: list[str] = field(default_factory=list)
    """Cluster folders used for training + per-cluster validation. Clustered variants only."""

    held_out_clusters: list[str] = field(default_factory=list)
    """Clusters reserved entirely as held-out (domain-shift) validation. Clustered variants only."""

    shards_per_cluster: int | None = None
    """Cap on parquet shards taken per cluster (``None`` = all). Clustered variants only."""

    val_shards_per_cluster: int = 0
    """Shards per training cluster held out as in-distribution validation (train gets the rest).
    Clustered variants only."""

    # --- flat variants (ClimbMix) only ---
    max_shards: int | None = None
    """Cap on the number of (sorted) shards taken for a flat variant (``None`` = all). The train /
    valid split is left to Megatron's ``--split`` at train time, so no split is planned here."""

    # --- shared ---
    cache_dir: str | None = None
    """Directory the downloaded shards are cached in. ``None`` resolves to
    ``<output_dir>/_hf_cache`` via :pyattr:`cache_path`."""

    download_workers: int = 8
    """Concurrent *threads* used to download shards (I/O-bound HTTPS; they share link bandwidth)."""

    convert_workers: int | None = None
    """*Processes* used to convert shards in parallel (GIL-bound work). ``None`` uses
    ``os.cpu_count()``, ``1`` runs inline; either way bounded by the job count."""

    dataset_repo: str | None = None
    """Override for the repo id; ``None`` uses the variant's (see :pyattr:`repo`)."""

    token_column: str = "tokens"
    """Column/field holding the pre-tokenized integer token-id sequence per row."""

    dtype: str = "uint16"
    """On-disk token dtype; one of ``_DTYPES``."""

    append_eod: bool = False
    """Append ``eod_token_id`` after each document (row) when building the ``.bin``."""

    eod_token_id: int = 50256
    """GPT-2 end-of-text id, used when ``append_eod`` and by the GPTDataset tokenizer shim."""

    vocab_size: int = 50257
    """GPT-2 vocabulary size; used for dtype sanity and the tokenizer shim."""

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {self.variant!r}")
        if self.layout == "clustered":
            self._validate_clustered()
        else:
            self._validate_flat()
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {self.dtype!r}")
        if self.append_eod and not 0 <= self.eod_token_id < self.vocab_size:
            raise ValueError("eod_token_id must be within [0, vocab_size)")
        if self.download_workers < 1:
            raise ValueError("download_workers must be >= 1")
        if self.convert_workers is not None and self.convert_workers < 1:
            raise ValueError("convert_workers must be >= 1 (or null for os.cpu_count())")

    def _validate_clustered(self) -> None:
        if not self.clusters:
            raise ValueError("clusters must be a non-empty list for a clustered variant")
        overlap = sorted(set(self.clusters) & set(self.held_out_clusters))
        if overlap:
            raise ValueError(f"clusters and held_out_clusters must be disjoint; overlap: {overlap}")
        if self.val_shards_per_cluster < 0:
            raise ValueError("val_shards_per_cluster must be >= 0")
        if self.shards_per_cluster is not None:
            if self.shards_per_cluster < 1:
                raise ValueError("shards_per_cluster must be >= 1 (or null for all shards)")
            if self.val_shards_per_cluster >= self.shards_per_cluster:
                raise ValueError(
                    "val_shards_per_cluster must leave at least one train shard per cluster"
                )

    def _validate_flat(self) -> None:
        if self.clusters or self.held_out_clusters:
            raise ValueError(
                f"clusters/held_out_clusters are not used for the flat variant {self.variant!r}"
            )
        if self.max_shards is not None and self.max_shards < 1:
            raise ValueError("max_shards must be >= 1 (or null for all shards)")

    @property
    def spec(self) -> VariantSpec:
        """The :class:`VariantSpec` for this config's ``variant``."""
        return VARIANTS[self.variant]

    @property
    def repo(self) -> str:
        """Effective HF dataset repo: ``dataset_repo`` if set, else the variant's."""
        return self.dataset_repo or self.spec.repo

    @property
    def layout(self) -> str:
        """The variant's layout: ``clustered`` or ``flat``."""
        return self.spec.layout

    @property
    def numpy_dtype(self) -> type[numpy.number]:
        """The numpy dtype the ``.bin`` tokens are stored as."""
        return _DTYPES[self.dtype]

    @property
    def cache_path(self) -> Path:
        """Resolved shard cache directory: ``cache_dir`` if set, else ``<output_dir>/_hf_cache``."""
        return Path(self.cache_dir) if self.cache_dir else Path(self.output_dir) / "_hf_cache"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataPrepConfig":
        """Build a config from a yaml file. Unknown keys raise ``TypeError`` (fail loud)."""
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a valid yaml mapping, got {type(data).__name__}")
        return cls(**data)

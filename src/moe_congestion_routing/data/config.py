"""Configuration for ClimbLab data preparation."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy
import yaml

# ClimbLab ships pre-tokenized with the GPT-2 tokenizer (vocab 50257).
# Fits into uint16 but we support int32 as well.
_DTYPES: dict[str, type[numpy.number]] = {"uint16": numpy.uint16, "int32": numpy.int32}


@dataclass(frozen=True)
class DataPrepConfig:
    """Everything the preparation + loading pipeline needs, loadable from a yaml file."""

    output_dir: str
    """Directory the ``.bin``/``.idx`` prefixes are written to."""

    clusters: list[str]
    """ClimbLab cluster folder names used for training (and for per-cluster validation)."""

    cache_dir: str | None = None
    """Directory the downloaded parquet shards are cached in. ``None`` (the default) resolves to
    ``<output_dir>/_hf_cache`` via :pyattr:`cache_path`."""

    dataset_repo: str = "nvidia/Nemotron-ClimbLab"
    """Hugging Face dataset repo id to pull shards from."""

    held_out_clusters: list[str] = field(default_factory=list)
    """Clusters reserved entirely as held-out validation."""

    shards_per_cluster: int | None = None
    """Cap on parquet shards taken per cluster (``None`` = all shards)."""

    val_shards_per_cluster: int = 0
    """
    Shards per training cluster held out as (in-distribution) validation.
    For train = shards_per_cluster - val_shards_per_cluster.
    """

    token_column: str = "tokens"
    """Parquet column holding the pre-tokenized integer token-id sequence per row."""

    dtype: str = "uint16"
    """On-disk token dtype; one of ``_DTYPES``."""

    append_eod: bool = False
    """Append ``eod_token_id`` after each document (row) when building the ``.bin``."""

    eod_token_id: int = 50256
    """GPT-2 end-of-text id, used when ``append_eod`` and by the GPTDataset tokenizer shim."""

    vocab_size: int = 50257
    """GPT-2 vocabulary size; used for dtype sanity and the tokenizer shim."""

    seed: int = 1234
    """Random seed threaded into the downstream ``GPTDataset`` global shuffle order."""

    seq_length: int = 2048
    """Sequence length of the samples the downstream ``GPTDataset`` packs tokens into."""

    def __post_init__(self) -> None:
        if not self.clusters:
            raise ValueError("clusters must be a non-empty list")
        overlap = sorted(set(self.clusters) & set(self.held_out_clusters))
        if overlap:
            raise ValueError(f"clusters and held_out_clusters must be disjoint; overlap: {overlap}")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {self.dtype!r}")
        if self.val_shards_per_cluster < 0:
            raise ValueError("val_shards_per_cluster must be >= 0")
        if self.shards_per_cluster is not None:
            if self.shards_per_cluster < 1:
                raise ValueError("shards_per_cluster must be >= 1 (or null for all shards)")
            if self.val_shards_per_cluster >= self.shards_per_cluster:
                raise ValueError(
                    "val_shards_per_cluster must leave at least one train shard per cluster"
                )
        if self.append_eod and not 0 <= self.eod_token_id < self.vocab_size:
            raise ValueError("eod_token_id must be within [0, vocab_size)")

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

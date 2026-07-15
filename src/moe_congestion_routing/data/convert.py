"""Convert pre-tokenized ClimbLab parquet shards into Megatron ``.bin``/``.idx`` prefixes.

ClimbLab ships already tokenized (one GPT-2 token-id sequence per parquet row), so there is
no tokenizer step: we read the token-id column straight into Megatron's
``IndexedDatasetBuilder``. Each parquet row becomes one *document* (``add_document`` records
a boundary in ``document_indices``, which is exactly what ``GPTDataset`` shuffles over).
"""

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy
import pyarrow.parquet as parquet

from moe_congestion_routing.training.megatron_path import ensure_on_path


@dataclass(frozen=True)
class ConversionStats:
    """Result of building one ``.bin``/``.idx`` prefix."""

    prefix: str
    num_documents: int
    num_tokens: int
    bin_bytes: int
    idx_bytes: int


def iter_token_rows(parquet_path: str | Path, token_column: str) -> Iterator[list[int]]:
    """Yield each row's token-id list from ``token_column``, streaming by row group."""
    reader = parquet.ParquetFile(str(parquet_path))
    if token_column not in reader.schema_arrow.names:
        raise KeyError(
            f"column {token_column!r} not in {parquet_path} (columns: {reader.schema_arrow.names})"
        )
    for batch in reader.iter_batches(columns=[token_column]):
        for value in batch.column(0):
            yield value.as_py()


def convert_shards(
    parquet_paths: Iterable[str | Path],
    output_prefix: str | Path,
    *,
    dtype: type[numpy.number],
    token_column: str = "tokens",
    append_eod: bool = False,
    eod_token_id: int = 50256,
) -> ConversionStats:
    """Build a single ``.bin``/``.idx`` prefix from one or more parquet shards.

    Rows are appended in the order the shards are given, one document per row. Empty rows are
    skipped (they would create zero-length documents Megatron cannot sample from).
    """
    ensure_on_path()
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    output_prefix = str(output_prefix)
    builder = IndexedDatasetBuilder(output_prefix + ".bin", dtype=dtype)

    num_documents = 0
    num_tokens = 0
    for path in parquet_paths:
        for tokens in iter_token_rows(path, token_column):
            if not tokens:  # skip empty/null rows *before* EOD, never emit a lone-EOD document
                continue
            if append_eod:
                tokens = [*tokens, eod_token_id]
            builder.add_document(numpy.asarray(tokens, dtype=dtype), [len(tokens)])
            num_documents += 1
            num_tokens += len(tokens)
    builder.finalize(output_prefix + ".idx")

    return ConversionStats(
        prefix=os.path.basename(output_prefix),
        num_documents=num_documents,
        num_tokens=num_tokens,
        bin_bytes=os.path.getsize(output_prefix + ".bin"),
        idx_bytes=os.path.getsize(output_prefix + ".idx"),
    )


def download_shards(
    dataset_repo: str,
    shards: Iterable[str],
    cache_dir: str | Path,
    *,
    max_workers: int = 8,
) -> list[Path]:
    """Download parquet shards from the HF dataset repo, returning local file paths.

    Shards download concurrently up to ``max_workers`` at a time. Returned paths preserve the
    input shard order, which ``convert_shards`` relies on for deterministic document ordering.
    """
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    shards = list(shards)

    def fetch(shard: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=dataset_repo,
                filename=shard,
                repo_type="dataset",
                cache_dir=str(cache_dir),
            )
        )

    workers = max(1, min(max_workers, len(shards)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fetch, shards))  # map preserves input order

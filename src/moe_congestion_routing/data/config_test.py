import re
from pathlib import Path

import numpy
import pytest

from moe_congestion_routing.data.config import DataPrepConfig


def _valid(**overrides):
    base = {"output_dir": "out", "clusters": ["c0", "c1"]}
    base.update(overrides)
    return DataPrepConfig(**base)


def test_defaults_and_numpy_dtype():
    cfg = _valid()
    assert cfg.dataset_repo == "nvidia/Nemotron-ClimbLab"
    assert cfg.token_column == "tokens"
    assert cfg.numpy_dtype is numpy.uint16
    assert _valid(dtype="int32").numpy_dtype is numpy.int32


def test_cache_path_defaults_under_output_dir():
    assert _valid(output_dir="out").cache_path == Path("out") / "_hf_cache"


def test_cache_path_uses_explicit_cache_dir():
    cfg = _valid(output_dir="out", cache_dir="/scratch/shards")
    assert cfg.cache_path == Path("/scratch/shards")


def test_from_yaml_roundtrip(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "output_dir: out\n"
        "clusters: [a, b]\n"
        "held_out_clusters: [c]\n"
        "shards_per_cluster: 3\n"
        "val_shards_per_cluster: 1\n"
        "dtype: int32\n"
    )
    cfg = DataPrepConfig.from_yaml(path)
    assert cfg.clusters == ["a", "b"]
    assert cfg.held_out_clusters == ["c"]
    assert cfg.shards_per_cluster == 3
    assert cfg.numpy_dtype is numpy.int32


def test_from_yaml_rejects_unknown_key(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("output_dir: out\nclusters: [a]\nbogus_key: 1\n")
    with pytest.raises(TypeError, match=re.escape("unexpected keyword argument 'bogus_key'")):
        DataPrepConfig.from_yaml(path)


def test_from_yaml_rejects_non_mapping(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must contain a valid yaml mapping"):
        DataPrepConfig.from_yaml(path)


def test_rejects_empty_clusters():
    with pytest.raises(ValueError, match="clusters must be a non-empty list"):
        DataPrepConfig(output_dir="out", clusters=[])


def test_rejects_overlapping_train_and_holdout():
    with pytest.raises(ValueError, match="disjoint"):
        _valid(clusters=["a", "b"], held_out_clusters=["b"])


def test_rejects_unknown_dtype():
    with pytest.raises(
        ValueError, match=re.escape("dtype must be one of ['int32', 'uint16'], got 'float64'")
    ):
        _valid(dtype="float64")


def test_rejects_val_not_leaving_a_train_shard():
    with pytest.raises(ValueError, match="must leave at least one train shard"):
        _valid(shards_per_cluster=2, val_shards_per_cluster=2)


def test_rejects_negative_val_shards():
    with pytest.raises(ValueError, match="val_shards_per_cluster must be >= 0"):
        _valid(val_shards_per_cluster=-1)


def test_rejects_eod_out_of_range_when_appending():
    with pytest.raises(ValueError, match=re.escape("eod_token_id must be within [0, vocab_size)")):
        _valid(append_eod=True, eod_token_id=99999, vocab_size=50257)


def test_worker_defaults():
    cfg = _valid()
    assert cfg.download_workers == 8
    assert cfg.convert_workers is None  # None => os.cpu_count() at run time


def test_rejects_non_positive_download_workers():
    with pytest.raises(ValueError, match="download_workers must be >= 1"):
        _valid(download_workers=0)


def test_rejects_non_positive_convert_workers():
    with pytest.raises(ValueError, match="convert_workers must be >= 1"):
        _valid(convert_workers=0)


def test_convert_workers_allows_none():
    assert _valid(convert_workers=None).convert_workers is None

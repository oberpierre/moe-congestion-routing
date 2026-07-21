import re
from pathlib import Path

import numpy
import pytest

from moe_congestion_routing.data.config import DataPrepConfig


def _valid(**overrides):
    base = {"variant": "climblab", "output_dir": "out", "clusters": ["c0", "c1"]}
    base.update(overrides)
    return DataPrepConfig(**base)


def _flat(**overrides):
    base = {"variant": "climbmix_small", "output_dir": "out"}
    base.update(overrides)
    return DataPrepConfig(**base)


def test_defaults_and_numpy_dtype():
    cfg = _valid()
    assert cfg.repo == "nvidia/Nemotron-ClimbLab"  # derived from the variant
    assert cfg.dataset_repo is None  # no explicit override
    assert cfg.token_column == "tokens"
    assert cfg.numpy_dtype is numpy.uint16
    assert _valid(dtype="int32").numpy_dtype is numpy.int32


def test_variant_drives_repo_and_layout():
    assert _valid().layout == "clustered"
    flat = _flat()
    assert flat.layout == "flat"
    assert flat.repo == "nvidia/Nemotron-ClimbMix"
    assert _flat(dataset_repo="me/fork").repo == "me/fork"  # explicit override wins


def test_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant must be one of"):
        DataPrepConfig(variant="bogus", output_dir="out")


def test_flat_rejects_cluster_fields():
    with pytest.raises(ValueError, match="not used for the flat variant"):
        _flat(clusters=["c0"])


def test_flat_rejects_non_positive_max_shards():
    with pytest.raises(ValueError, match="max_shards must be >= 1"):
        _flat(max_shards=0)


def test_cache_path_defaults_under_output_dir():
    assert _valid(output_dir="out").cache_path == Path("out") / "_hf_cache"


def test_cache_path_uses_explicit_cache_dir():
    cfg = _valid(output_dir="out", cache_dir="/scratch/shards")
    assert cfg.cache_path == Path("/scratch/shards")


def test_from_yaml_roundtrip(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "variant: climblab\n"
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


def test_from_yaml_expands_env_var_in_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_STORE", "/store/me")
    monkeypatch.setenv("SCRATCH", "/scratch/me")
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "variant: climbmix_small\n"
        "output_dir: ${DATA_STORE}/datasets/climbmix_small\n"
        "cache_dir: ${SCRATCH}/hf_cache\n"
    )
    cfg = DataPrepConfig.from_yaml(path)
    assert cfg.output_dir == "/store/me/datasets/climbmix_small"
    assert cfg.cache_path == Path("/scratch/me/hf_cache")


def test_from_yaml_fails_on_unresolved_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_STORE", raising=False)
    path = tmp_path / "cfg.yaml"
    path.write_text("variant: climbmix_small\noutput_dir: ${DATA_STORE}/datasets/climbmix_small\n")
    with pytest.raises(ValueError, match="unresolved environment variable"):
        DataPrepConfig.from_yaml(path)


def test_from_yaml_rejects_unknown_key(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("variant: climblab\noutput_dir: out\nclusters: [a]\nbogus_key: 1\n")
    with pytest.raises(TypeError, match=re.escape("unexpected keyword argument 'bogus_key'")):
        DataPrepConfig.from_yaml(path)


def test_from_yaml_rejects_non_mapping(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must contain a valid yaml mapping"):
        DataPrepConfig.from_yaml(path)


def test_rejects_empty_clusters():
    with pytest.raises(ValueError, match="clusters must be a non-empty list"):
        DataPrepConfig(variant="climblab", output_dir="out", clusters=[])


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

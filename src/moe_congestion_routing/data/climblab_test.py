import pytest

from moe_congestion_routing.data.climblab import (
    ConversionJob,
    _group_parquet_by_cluster,
    plan_conversions,
)
from moe_congestion_routing.data.config import DataPrepConfig


def _shards(cluster, n):
    # Intentionally reverse-ordered so tests confirm plan_conversions sorts deterministically.
    return [f"{cluster}/shard-{i:04d}.parquet" for i in reversed(range(n))]


def test_plan_splits_train_val_and_holdout_disjointly():
    cfg = DataPrepConfig(
        output_dir="out",
        clusters=["c0", "c1"],
        held_out_clusters=["c2"],
        shards_per_cluster=3,
        val_shards_per_cluster=1,
    )
    avail = {"c0": _shards("c0", 5), "c1": _shards("c1", 5), "c2": _shards("c2", 4)}

    jobs = {j.prefix: j for j in plan_conversions(cfg, avail)}

    assert set(jobs) == {"c0_train", "c0_valid", "c1_train", "c1_valid", "c2_holdout"}
    # Budget of 3 applied to sorted shards; last (0002) held out for val, 0000/0001 train.
    assert jobs["c0_train"].shards == ("c0/shard-0000.parquet", "c0/shard-0001.parquet")
    assert jobs["c0_valid"].shards == ("c0/shard-0002.parquet",)
    assert jobs["c0_train"].role == "train" and jobs["c0_valid"].role == "valid"
    # No leakage: train and val shard sets are disjoint per cluster.
    assert not set(jobs["c0_train"].shards) & set(jobs["c0_valid"].shards)
    # Whole held-out cluster, capped to the budget, all shards -> one holdout prefix.
    assert jobs["c2_holdout"].role == "holdout"
    assert jobs["c2_holdout"].shards == (
        "c2/shard-0000.parquet",
        "c2/shard-0001.parquet",
        "c2/shard-0002.parquet",
    )


def test_no_val_shards_emits_only_train_prefixes():
    cfg = DataPrepConfig(output_dir="out", clusters=["c0"], val_shards_per_cluster=0)
    jobs = plan_conversions(cfg, {"c0": _shards("c0", 3)})
    assert [j.prefix for j in jobs] == ["c0_train"]
    assert len(jobs[0].shards) == 3


def test_no_budget_takes_all_shards_sorted():
    cfg = DataPrepConfig(output_dir="out", clusters=["c0"], shards_per_cluster=None)
    (job,) = plan_conversions(cfg, {"c0": _shards("c0", 4)})
    assert job.shards == tuple(f"c0/shard-{i:04d}.parquet" for i in range(4))


def test_missing_cluster_raises():
    cfg = DataPrepConfig(output_dir="out", clusters=["c0", "foo"])
    with pytest.raises(KeyError, match="cluster 'foo' not found among available shards"):
        plan_conversions(cfg, {"c0": _shards("c0", 2)})


def test_validation_consuming_all_shards_raises():
    cfg = DataPrepConfig(output_dir="out", clusters=["c0"], val_shards_per_cluster=2)
    with pytest.raises(ValueError, match="leaves no train shards"):
        plan_conversions(cfg, {"c0": _shards("c0", 2)})


def test_group_parquet_by_cluster_handles_nesting_and_ignores_non_parquet():
    files = [
        "cluster_0/part-0.parquet",
        "data/cluster_1/part-0.parquet",  # nested one level down
        "cluster_1/part-1.parquet",
        "cluster_0/README.md",  # ignored (not parquet)
        "cluster_9/part-0.parquet",  # not requested
    ]
    grouped = _group_parquet_by_cluster(files, ["cluster_0", "cluster_1"])
    assert grouped == {
        "cluster_0": ["cluster_0/part-0.parquet"],
        # NOTE: Limitation will be picked up as part of cluster_1 even though folder is different,
        # we do not handle this as we do not run into this in practice.
        "cluster_1": ["data/cluster_1/part-0.parquet", "cluster_1/part-1.parquet"],
    }


def test_conversion_job_is_hashable_frozen():
    job = ConversionJob("c0_train", "train", "c0", ("c0/a.parquet",))
    assert job in {job}  # frozen dataclass -> hashable

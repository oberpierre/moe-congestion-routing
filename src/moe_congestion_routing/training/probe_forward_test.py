import os
import subprocess
import types

import pytest

from moe_congestion_routing.probe_windows import parse_windows
from moe_congestion_routing.training.probe_forward import (
    probe_fires,
    validate_probe_setup,
)

_ASSET = "assets/probe/dev_climblab_c1valid_16x1024.npz"


def test_probe_fires_on_the_coarse_grid():
    assert probe_fires(6, coarse=6, dense=2, windows=[])
    assert not probe_fires(5, coarse=6, dense=2, windows=[])


def test_probe_fires_disabled_when_coarse_is_zero_and_no_window_matches():
    assert not probe_fires(5, coarse=0, dense=2, windows=[(0, 4)])


def test_probe_fires_inside_a_dense_window_anchored_at_its_start():
    windows = [(3200, 3450)]
    assert probe_fires(3200, coarse=250, dense=25, windows=windows)
    assert probe_fires(3225, coarse=250, dense=25, windows=windows)
    assert not probe_fires(3210, coarse=250, dense=25, windows=windows)
    assert not probe_fires(3460, coarse=250, dense=25, windows=windows)


def test_probe_fires_is_a_union_not_a_double_count():
    # 250 lands on both the coarse grid and the dense window's anchor-aligned grid; the caller
    # must still see a single True, since probe_fires reports "fires", not "how many rules fire".
    windows = [(0, 500)]
    assert probe_fires(250, coarse=250, dense=25, windows=windows)


def test_probe_fires_outside_every_window_and_off_grid_is_false():
    assert not probe_fires(7, coarse=6, dense=2, windows=[(0, 4)])


def test_fleet_cadence_windows_parse_and_the_anneal_start_fires_once():
    # base_cluster.yaml's cadence: a dense window over warmup plus one over the WSD anneal
    # (train_iters - lr_wsd_decay_iters = 5000 to train_iters = 5500). Step 5000 lands on both the
    # coarse grid (a multiple of 250) and the second window's own anchor, and probe_fires reports a
    # union, so this must be a single True rather than a double-fire.
    windows = parse_windows(["0:500", "5000:5500"])
    assert windows == [(0, 500), (5000, 5500)]
    assert probe_fires(5000, coarse=250, dense=25, windows=windows) is True


def test_validate_probe_setup_checks_seq_length_agreement():
    # args.moe_probe_batch is always a list by the time it reaches here: argparse's nargs='+'
    # never hands back a bare string, even for one asset.
    args = types.SimpleNamespace(moe_probe_batch=[_ASSET], seq_length=999)
    with pytest.raises(ValueError, match="seq_length"):
        validate_probe_setup(args)


def test_validate_probe_setup_accepts_a_tracked_dev_asset(tmp_path):
    args = types.SimpleNamespace(
        moe_probe_batch=[_ASSET], seq_length=1024, moe_probe_dir=str(tmp_path / "probes")
    )
    validate_probe_setup(args)
    assert (tmp_path / "probes").is_dir()


def test_validate_probe_setup_checks_every_asset_in_the_list():
    # A second, bad-seq_length asset later in the list must still be caught, not just the first.
    args = types.SimpleNamespace(moe_probe_batch=[_ASSET, _ASSET], seq_length=999)
    with pytest.raises(ValueError, match="seq_length"):
        validate_probe_setup(args)


def test_validate_probe_setup_rejects_an_untracked_standing_asset(tmp_path, monkeypatch):
    # A standing-role copy that git does not track: load_probe_batch itself enforces that the
    # filename's leading role prefix agrees with the recorded role, so this asset is renamed to
    # match before being dropped somewhere git has never seen it.
    import numpy

    with numpy.load(_ASSET, allow_pickle=False) as data:
        tokens, seq_labels, provenance = data["tokens"], data["seq_labels"], str(data["provenance"])
    import json

    prov = json.loads(provenance)
    prov["role"] = "standing"
    from moe_congestion_routing.training.probe_batch import compute_provenance_sha256

    prov["provenance_sha256"] = compute_provenance_sha256(prov)
    untracked_dir = tmp_path / "untracked_repo"
    untracked_dir.mkdir()
    monkeypatch.chdir(untracked_dir)
    asset_path = untracked_dir / "standing_climblab_c1valid_16x1024.npz"
    numpy.savez(asset_path, tokens=tokens, seq_labels=seq_labels, provenance=json.dumps(prov))
    subprocess.run(["git", "init", "-q"], cwd=untracked_dir, check=True)

    args = types.SimpleNamespace(
        moe_probe_batch=[str(asset_path)],
        seq_length=1024,
        moe_probe_dir=str(untracked_dir / "probes"),
    )
    with pytest.raises(ValueError, match="does not.*track it"):
        validate_probe_setup(args)


def test_validate_probe_setup_rejects_an_unwritable_probe_dir(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory write permissions")
    probe_dir = tmp_path / "readonly"
    probe_dir.mkdir(mode=0o500)
    try:
        args = types.SimpleNamespace(
            moe_probe_batch=[_ASSET], seq_length=1024, moe_probe_dir=str(probe_dir)
        )
        with pytest.raises(ValueError, match="not writable"):
            validate_probe_setup(args)
    finally:
        probe_dir.chmod(0o700)

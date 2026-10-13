import subprocess
import sys

import numpy as np
import pytest

from moe_congestion_routing.game.ensemble import Instance, affinities


def test_affinities_shape_and_range():
    inst = Instance(n=64, e=8, k=2, separation=2.0, seed=0)
    a = affinities(inst)
    assert a.shape == (64, 8)
    assert a.dtype == np.float64
    assert np.all(a > 0.0) and np.all(a < 1.0)


def test_affinities_deterministic_in_seed():
    inst1 = Instance(n=64, e=8, k=2, separation=2.0, seed=3)
    inst2 = Instance(n=64, e=8, k=2, separation=2.0, seed=3)
    np.testing.assert_array_equal(affinities(inst1), affinities(inst2))


def test_affinities_differ_by_seed():
    a = affinities(Instance(n=64, e=8, k=2, separation=2.0, seed=0))
    b = affinities(Instance(n=64, e=8, k=2, separation=2.0, seed=1))
    assert not np.allclose(a, b)


def test_small_separation_crowds_affinities_near_one_half():
    # separation is the pre-sigmoid scale, so a small value keeps every pre-sigmoid draw close
    # to 0 and every affinity close to sigmoid(0) = 0.5.
    inst = Instance(n=2000, e=8, k=2, separation=0.02, seed=0)
    a = affinities(inst)
    assert np.max(np.abs(a - 0.5)) < 0.05


def test_large_separation_spreads_affinities_toward_the_extremes():
    inst = Instance(n=2000, e=8, k=2, separation=2.0, seed=0)
    a = affinities(inst)
    assert np.std(a) > np.std(affinities(Instance(n=2000, e=8, k=2, separation=0.02, seed=0)))


@pytest.mark.parametrize("field", ["n", "e", "k", "separation", "seed"])
def test_instance_is_frozen(field):
    inst = Instance(n=4, e=3, k=1, separation=1.0, seed=0)
    with pytest.raises(AttributeError):
        setattr(inst, field, getattr(inst, field))


def test_importing_ensemble_does_not_pull_in_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import moe_congestion_routing.game.ensemble, sys; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

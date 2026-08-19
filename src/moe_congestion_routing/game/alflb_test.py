import math
import subprocess
import sys

import numpy as np
import pytest

from moe_congestion_routing.game.alflb import iterate, tie_margins, top_k_map


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------------------------
# One step reproduces the hand-computed update, including that a balanced expert does not move.
# ---------------------------------------------------------------------------------------------


def test_one_step_matches_hand_computed_update():
    # 6 tokens, 3 experts, top-1: rows 0-2 prefer expert 0, rows 3-4 prefer expert 1, row 5
    # prefers expert 2, so loads are [3, 2, 1] against a balanced load of 6*1/3 = 2. Expert 1
    # is exactly balanced and its bias must stay 0.
    a = np.array(
        [
            [5.0, 1.0, 1.0],
            [5.0, 1.0, 1.0],
            [5.0, 1.0, 1.0],
            [1.0, 5.0, 1.0],
            [1.0, 5.0, 1.0],
            [1.0, 1.0, 5.0],
        ]
    )
    eta = 0.1
    result = iterate(a, k=1, eta=eta, steps=1, mode="deployed")
    np.testing.assert_allclose(result.bias, [-eta, 0.0, eta])
    assert result.bias[1] == 0.0  # np.sign(0) == 0, not an approximation
    assert result.steps == 1


# ---------------------------------------------------------------------------------------------
# top_k_map ties to the lowest expert index. torch.topk does not: on this exact input it
# returns [1, 2], not [0, 1], so this is our own convention rather than a replication of it.
# ---------------------------------------------------------------------------------------------


def test_top_k_map_breaks_ties_to_lowest_index():
    y = np.array([[0.5, 0.5, 0.5, 0.1]])
    assert top_k_map(y, k=2).tolist() == [[0, 1]]


# ---------------------------------------------------------------------------------------------
# Deployed mode finds an exact cycle, and replaying from the returned bias reproduces it.
# ---------------------------------------------------------------------------------------------


def test_deployed_mode_cycle_is_reproducible_from_the_returned_bias():
    rng = np.random.default_rng(0)
    a = _sigmoid(2 * rng.standard_normal((64, 8)))
    n, e = a.shape
    k = 2
    eta = 1e-2
    result = iterate(a, k, eta=eta, steps=2000, mode="deployed")

    assert isinstance(result.cycle_length, int)
    assert result.cycle_length > 0

    # Manually replay the update rule from the returned bias, using the same public
    # top_k_map the module itself uses, and check it returns to that bias after exactly
    # one cycle_length worth of steps.
    balanced_load = n * k / e
    bias = result.bias.copy()
    for _ in range(result.cycle_length):
        idx = top_k_map(a + bias, k)
        x = np.zeros((n, e), dtype=bool)
        x[np.repeat(np.arange(n), k), idx.ravel()] = True
        load = x.sum(axis=0)
        bias = bias + eta * np.sign(balanced_load - load)
    np.testing.assert_allclose(bias, result.bias)


# ---------------------------------------------------------------------------------------------
# band_width and cycle_max_load on a fixed instance, at two step sizes.
# ---------------------------------------------------------------------------------------------


def test_band_width_equals_eta_exactly_for_a_period_two_cycle():
    # A period-2 cycle forces this: each expert's two sign steps over one period must
    # cancel to close the loop, so a moving expert takes one +eta step and one -eta step,
    # and the peak-to-peak of those two points is exactly eta.
    rng = np.random.default_rng(1)
    a = _sigmoid(2 * rng.standard_normal((512, 8)))
    for eta in (1e-3, 1e-2):
        result = iterate(a, k=2, eta=eta, steps=5000, mode="deployed")
        assert result.cycle_length == 2
        assert result.band_width == pytest.approx(eta)


def test_cycle_max_load_grows_with_eta():
    # Unlike band_width, the realized load overflow is not an identity of the cycle
    # length: it depends on how far the affinities let the bias push loads before the
    # sign flips, so this is where the fixed-step oscillation actually shows up.
    rng = np.random.default_rng(1)
    a = _sigmoid(2 * rng.standard_normal((512, 8)))
    small = iterate(a, k=2, eta=1e-3, steps=5000, mode="deployed")
    large = iterate(a, k=2, eta=1e-2, steps=5000, mode="deployed")
    assert large.cycle_max_load > small.cycle_max_load


# ---------------------------------------------------------------------------------------------
# eta_schedule requires mode="annealed".
# ---------------------------------------------------------------------------------------------


def test_eta_schedule_with_deployed_mode_raises():
    with pytest.raises(ValueError, match="annealed"):
        iterate(
            np.ones((4, 3)),
            k=1,
            eta=1e-2,
            steps=1,
            mode="deployed",
            eta_schedule=lambda t: 1e-2,
        )


def test_annealed_bias_reaches_a_hand_computed_endpoint():
    # Both tokens prefer expert 0, so the bias walks to [-S, +S] with S the running sum of
    # 0.1/sqrt(t+1). Token 1 flips once 2S > 0.6, which first holds at t=5 (S = 0.323167).
    # Load is then [1, 1], so every sign is zero and the bias is converged for good.
    a = np.array([[0.9, 0.1], [0.8, 0.2]])
    expected = sum(0.1 / math.sqrt(t + 1) for t in range(5))
    result = iterate(a, k=1, eta=0.1, steps=50, mode="annealed")

    np.testing.assert_allclose(result.bias, [-expected, expected])
    assert result.settled_at == 5
    assert result.steps < 50  # stopped at the fixed point instead of spending the budget
    np.testing.assert_array_equal(result.x, [[True, False], [False, True]])

    # The same instance under a fixed step settles somewhere else, so the endpoint is a
    # property of the schedule and not merely of the affinities.
    assert not np.allclose(iterate(a, k=1, eta=0.1, steps=50, mode="deployed").bias, result.bias)


def test_annealing_converges_where_a_fixed_step_orbits_forever():
    # The substantive difference between the modes, on one instance. A fixed step keeps the load
    # off balance forever inside a band of eta, whereas a decaying step reaches exact balance.
    rng = np.random.default_rng(1)
    a = _sigmoid(2 * rng.standard_normal((512, 8)))
    steps = 3000
    deployed = iterate(a, k=2, eta=1e-2, steps=steps, mode="deployed")
    annealed = iterate(a, k=2, eta=1e-2, steps=steps, mode="annealed")

    assert deployed.settled_at is None  # orbits: never exactly balanced
    assert deployed.cycle_length is not None
    assert deployed.band_width == pytest.approx(1e-2)

    assert annealed.settled_at is not None  # converged, and stopped there
    assert annealed.steps == annealed.settled_at + 1
    assert annealed.steps < steps
    assert annealed.band_width < deployed.band_width


def test_settled_at_is_none_when_exact_balance_is_unreachable():
    # Loads are integers, so n*k/E must be one for every sign to vanish. At 5*2/3 it cannot,
    # and the honest report is that the budget ran out rather than that anything converged.
    a = _sigmoid(2 * np.random.default_rng(0).standard_normal((5, 3)))
    result = iterate(a, k=2, eta=1e-2, steps=200, mode="annealed")
    assert result.settled_at is None
    assert result.steps == 200


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"k": 9, "eta": 1e-2, "steps": 5}, "1 <= k <= num_experts"),
        ({"k": 0, "eta": 1e-2, "steps": 5}, "1 <= k <= num_experts"),
        ({"k": 2, "eta": 0.0, "steps": 5}, "eta must be positive"),
        ({"k": 2, "eta": -1e-2, "steps": 5}, "eta must be positive"),
        ({"k": 2, "eta": 1e-2, "steps": 5, "mode": "deploye"}, "mode must be one of"),
    ],
)
def test_iterate_refuses_inputs_it_cannot_honour(kwargs, match):
    # Each of these used to return a plausible result instead of raising: k > E silently gave
    # every token all E experts, a typo'd mode ran fixed-step with cycle detection off, and
    # eta=0 reported cycle_length=1 off a nan lattice coordinate.
    a = np.random.default_rng(0).random((16, 8))
    with pytest.raises(ValueError, match=match):
        iterate(a, **kwargs)


# ---------------------------------------------------------------------------------------------
# tie_margins.
# ---------------------------------------------------------------------------------------------


def test_tie_margins_shape_and_nonnegativity():
    rng = np.random.default_rng(3)
    y = rng.standard_normal((20, 6))
    margins = tie_margins(y, k=2)
    assert margins.shape == (20,)
    assert np.all(margins >= 0.0)


def test_tie_margins_zero_on_a_row_of_equal_values():
    y = np.array([[1.0, 1.0, 1.0, 1.0]])
    assert tie_margins(y, k=2)[0] == 0.0


# ---------------------------------------------------------------------------------------------
# Update-rule conformance against Megatron's own get_updated_expert_bias.
# ---------------------------------------------------------------------------------------------
# This is the only test requiring a GPU and the vendored Megatron-LM submodule, so it uses
# pytest.importorskip and does not import torch at module load.


@pytest.mark.parametrize(
    "counts",
    [
        [10, 20, 30, 40],
        [0, 0, 0, 0],
        [5, 5, 5, 5],
        [100, 1, 1, 1, 1, 1, 1, 1],
    ],
)
def test_update_matches_megatron_get_updated_expert_bias(counts, tmp_path):
    pytest.importorskip("triton", reason="megatron.core requires triton, unavailable on macOS")
    from moe_congestion_routing.training.megatron_path import (
        MegatronLMNotVendoredError,
        ensure_on_path,
    )

    try:
        ensure_on_path()
    except MegatronLMNotVendoredError as e:
        pytest.skip(str(e))
    torch = pytest.importorskip("torch")
    moe_utils = pytest.importorskip("megatron.core.transformer.moe.moe_utils")

    # get_updated_expert_bias calls torch.distributed.all_reduce unconditionally. So we
    # create a single-rank (no-op) gloo group so the function is callable on CPU here.
    started_here = not torch.distributed.is_initialized()
    if started_here:
        torch.distributed.init_process_group(
            backend="gloo", world_size=1, rank=0, init_method=f"file://{tmp_path / 'store'}"
        )
    try:
        eta = 1e-3
        counts_arr = np.array(counts, dtype=np.float64)
        balanced_load = counts_arr.sum() / len(counts)
        expected_delta = eta * np.sign(balanced_load - counts_arr)

        tokens = torch.tensor(counts, dtype=torch.float64)
        bias = torch.zeros(len(counts), dtype=torch.float64)
        updated = moe_utils.get_updated_expert_bias(
            tokens, bias, eta, tp_dp_cp_group=torch.distributed.group.WORLD
        )
        actual_delta = (updated - bias).numpy()
        np.testing.assert_allclose(actual_delta, expected_delta)
    finally:
        if started_here:
            torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------------------------
# alflb.py imports no torch at module load, so it stays usable with no GPU.
# ---------------------------------------------------------------------------------------------


def test_importing_alflb_does_not_pull_in_torch():
    # A subprocess rather than an in-process sys.modules check, because other test
    # modules in this suite import torch and would otherwise pollute the check.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import moe_congestion_routing.game.alflb, sys; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

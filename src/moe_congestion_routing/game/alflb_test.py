import math
import subprocess
import sys

import numpy as np
import pytest

from moe_congestion_routing.game.alflb import (
    bias_update,
    iterate,
    tie_margins,
    top_k_map,
)
from moe_congestion_routing.game.ensemble import N512_E8_K2_SEP2_SEED1, affinities


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


def test_tie_margins_refuses_k_equal_to_num_experts():
    # iterate accepts k == num_experts, so tie_margins must not be the one entry point that
    # raises IndexError from an out-of-bounds column instead of the package's ValueError.
    a = np.random.default_rng(0).random((4, 3))
    with pytest.raises(ValueError, match="1 <= k < num_experts"):
        tie_margins(a, 3)


def test_the_returned_assignment_is_the_top_k_of_the_returned_bias():
    # The fields must describe one state, not two. x, bias and load are computed at different
    # points in the loop, so this is the invariant that ties them together: re-deriving the
    # assignment from the bias the caller was handed has to reproduce the assignment it was
    # handed, otherwise a caller comparing them is comparing different iterations.
    a = _sigmoid(2 * np.random.default_rng(4).standard_normal((64, 8)))
    result = iterate(a, k=2, eta=1e-2, steps=500, mode="deployed")
    rederived = np.zeros_like(result.x)
    np.put_along_axis(rederived, top_k_map(a + result.bias, 2), True, axis=1)
    np.testing.assert_array_equal(result.x, rederived)
    assert result.max_load == int(result.x.sum(axis=0).max())


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
    a = affinities(N512_E8_K2_SEP2_SEED1)
    for eta in (1e-3, 1e-2):
        result = iterate(a, k=2, eta=eta, steps=5000, mode="deployed")
        assert result.cycle_length == 2
        assert result.band_width == pytest.approx(eta)


def test_cycle_max_load_grows_with_eta():
    # Unlike band_width, the realized load overflow is not an identity of the cycle
    # length: it depends on how far the affinities let the bias push loads before the
    # sign flips, so this is where the fixed-step oscillation actually shows up.
    a = affinities(N512_E8_K2_SEP2_SEED1)
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
    a = affinities(N512_E8_K2_SEP2_SEED1)
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
        ({"k": 2, "eta": 1e-2, "steps": 0}, "steps must be >= 1"),
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
# closest_approach and worst_phase.
# ---------------------------------------------------------------------------------------------


def test_closest_approach_and_worst_phase_pull_opposite_ways_on_the_same_cycle():
    # Both cycle phases carry the same max load (130), so only the objective term in each key
    # decides which phase each field reports: closest_approach keeps the higher-objective phase,
    # worst_phase keeps the lower-objective one. Dropping either key's objective term would let
    # the two fields agree, which this pins against.
    a = affinities(N512_E8_K2_SEP2_SEED1)
    result = iterate(a, k=2, eta=1e-3, steps=2000, mode="deployed")

    assert result.worst_phase is not None
    assert result.closest_approach.objective > result.worst_phase.objective


def test_worst_phase_is_none_when_no_cycle_was_found():
    # Annealed mode never runs cycle detection, so it can never report a worst_phase.
    a = affinities(N512_E8_K2_SEP2_SEED1)
    result = iterate(a, k=2, eta=1e-2, steps=2000, mode="annealed")
    assert result.worst_phase is None


def test_closest_approach_step_matches_settled_at_when_the_run_settles():
    a = affinities(N512_E8_K2_SEP2_SEED1)
    result = iterate(a, k=2, eta=1e-2, steps=2000, mode="annealed")
    assert result.settled_at is not None
    assert result.closest_approach.step == result.settled_at


def test_closest_approach_breaks_ties_by_objective_not_earliest_step():
    # Steps 1, 2, 3 and 4 all tie at the minimum max load of 2, and only step 2's objective
    # term in the sort key is highest: token 3 stays on expert 1 at affinity 0.9 there, where
    # steps 1 and 3 send it to expert 2 at 0.8. Dropping the objective term from the key would
    # select step 1, the earliest tied step, instead.
    a = np.array(
        [
            [0.8, 0.9, 0.0],
            [0.0, 0.7, 0.3],
            [0.6, 0.1, 0.9],
            [0.4, 0.9, 0.8],
        ]
    )
    result = iterate(a, k=1, eta=0.1, steps=5, mode="deployed")
    assert result.closest_approach.step == 2
    assert result.closest_approach.objective == pytest.approx(3.3)


def test_a_run_that_starts_balanced_settles_at_step_zero_without_moving_the_bias():
    # The fixed-point check runs before the update, so a perfectly balanced start must settle
    # immediately and the bias must never move.
    a = np.array(
        [
            [0.9, 0.8, 0.1, 0.0],
            [0.9, 0.8, 0.1, 0.0],
            [0.0, 0.1, 0.8, 0.9],
            [0.0, 0.1, 0.8, 0.9],
        ]
    )
    result = iterate(a, k=2, eta=1e-1, steps=5, mode="deployed")
    assert result.settled_at == 0
    assert result.steps == 1
    np.testing.assert_array_equal(result.bias, [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.x.sum(axis=0), [2, 2, 2, 2])
    assert result.closest_approach.step == 0


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
        expected_delta = bias_update(counts_arr, balanced_load, eta)

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

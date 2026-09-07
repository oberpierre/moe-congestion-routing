import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from moe_congestion_routing.losses import rosenthal
from moe_congestion_routing.losses.cost_families import (
    COST_EXPONENTS,
    COST_FAMILIES,
    DEFAULT_LAMBDA,
    ORACLE_COST_FAMILIES,
    ROSENTHAL_TYPES,
    VARIANTS,
    _softplus,
    _softplus_inverse,
    check_variant,
    cost_exponent,
    discrete_potential,
    first_arc_above_price,
    marginal_cost,
    pressure_bound,
)
from moe_congestion_routing.losses.rosenthal import congestion_potential
from moe_congestion_routing.training.pretrain_config import MoEPretrainConfig, build_megatron_args


def test_no_torch_import():
    # The whole point of splitting this module out: a config's --dry-run must not
    # pay for importing torch just for verification.
    script = (
        "import sys; "
        "import moe_congestion_routing.losses.cost_families; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_registry_keys_agree():
    # COST_EXPONENTS is only a subset now that softplus_barrier trains without an exponent,
    # whereas DEFAULT_LAMBDA stays an equality since every trainable family needs a default lambda.
    assert set(COST_EXPONENTS) <= set(COST_FAMILIES)
    assert set(DEFAULT_LAMBDA) == set(COST_FAMILIES)


def test_cost_exponent_known_families():
    assert cost_exponent("linear") == 1
    assert cost_exponent("quadratic") == 2


def test_cost_exponent_unknown_raises_with_offending_value():
    with pytest.raises(ValueError, match="bogus"):
        cost_exponent("bogus")


def test_check_variant_known_variants_do_not_raise():
    for variant in VARIANTS:
        check_variant(variant)  # must not raise


def test_check_variant_unknown_raises_with_offending_value():
    with pytest.raises(ValueError, match="bogus"):
        check_variant("bogus")


def test_rosenthal_types_are_the_two_congestion_balancing_types():
    assert ROSENTHAL_TYPES == ("rosenthal", "global_rosenthal")


def test_pressure_bound_is_coeff_times_c_of_e_over_k():
    # hard's bound is coeff * lam * (E/K)**p, the marginal cost evaluated at the highest relative
    # load a single expert can reach under top-k selection, using cost_exponent's p for the family.
    assert pressure_bound(
        2.0, 3.0, num_experts=8, topk=2, cost_family="linear", variant="hard"
    ).value == pytest.approx(2.0 * 3.0 * (8 / 2) ** 1)
    assert pressure_bound(
        2.0, 3.0, num_experts=8, topk=2, cost_family="quadratic", variant="hard"
    ).value == pytest.approx(2.0 * 3.0 * (8 / 2) ** 2)


def test_pressure_bound_soft_variant_is_coeff_times_c_of_e():
    # soft's bound is coeff * lam * E**p. Softmax mass has no selection cap, so it can concentrate
    # up to E rather than stopping at E/K the way a hard selection count does.
    assert pressure_bound(
        2.0, 3.0, num_experts=8, topk=2, cost_family="linear", variant="soft"
    ).value == pytest.approx(2.0 * 3.0 * 8**1)
    assert pressure_bound(
        2.0, 3.0, num_experts=8, topk=2, cost_family="quadratic", variant="soft"
    ).value == pytest.approx(2.0 * 3.0 * 8**2)


def test_pressure_bound_soft_is_topk_to_the_p_times_hard():
    # The soft bound E**p is topk**p times the hard bound (E/K)**p at identical coeff, lam,
    # num_experts, topk and cost_family. That factor is the whole reason variant is required.
    for cost_family, p in COST_EXPONENTS.items():
        hard = pressure_bound(
            2.0, 3.0, num_experts=8, topk=2, cost_family=cost_family, variant="hard"
        )
        soft = pressure_bound(
            2.0, 3.0, num_experts=8, topk=2, cost_family=cost_family, variant="soft"
        )
        assert soft.value == pytest.approx(hard.value * 2**p)


def test_pressure_bound_unknown_cost_family_raises():
    with pytest.raises(ValueError, match="bogus"):
        pressure_bound(1.0, 1.0, num_experts=4, topk=1, cost_family="bogus", variant="hard")


def test_pressure_bound_unknown_variant_raises():
    with pytest.raises(ValueError, match="bogus"):
        pressure_bound(1.0, 1.0, num_experts=4, topk=1, cost_family="linear", variant="bogus")


def test_pressure_bound_hard_expr_is_the_capped_ratio_with_default_names():
    # Default names match MoEPretrainConfig's own field names, so pretrain_config.py's caller
    # needs no override. expr carries the exponent baked in, since it is the complete price
    # expression rather than a base a caller then exponentiates itself.
    bound = pressure_bound(1.0, 1.0, num_experts=4, topk=2, cost_family="linear", variant="hard")
    assert bound.expr == "(num_experts/moe_router_topk)**1"


def test_pressure_bound_soft_expr_is_the_uncapped_count_with_default_names():
    bound = pressure_bound(1.0, 1.0, num_experts=4, topk=2, cost_family="linear", variant="soft")
    assert bound.expr == "num_experts**1"


def test_pressure_bound_expr_uses_caller_supplied_names():
    # Megatron's TransformerConfig field is num_moe_experts rather than num_experts, so the caller
    # passes its own name and the printed warning names a flag the reader can actually set.
    bound = pressure_bound(
        1.0,
        1.0,
        num_experts=4,
        topk=2,
        cost_family="linear",
        variant="hard",
        num_experts_name="num_moe_experts",
    )
    assert bound.expr == "(num_moe_experts/moe_router_topk)**1"


def test_pressure_bound_barrier_expr_is_the_softplus_form():
    bound = pressure_bound(
        1.0, 1.0, num_experts=4, topk=2, cost_family="softplus_barrier", variant="hard"
    )
    assert bound.expr == "softplus(((num_experts/moe_router_topk) - 1)/0.1)"
    assert bound.value == pytest.approx(math.log1p(math.exp(10.0)))


def test_default_lambda_cost_exponents_key_mismatch_raises_at_import():
    # Proves the guard constrains the module rather than restating an invariant that already
    # holds. A mutated copy of the module's own source, with DEFAULT_LAMBDA missing a key
    # COST_EXPONENTS has, must make the module-level check raise exactly as a real edit adding a
    # family to one dict and forgetting the other would.
    import moe_congestion_routing.losses.cost_families as cost_families_module

    source = pathlib.Path(cost_families_module.__file__).read_text()
    mutated = source.replace(
        'DEFAULT_LAMBDA: dict[str, float] = {"linear": 1.0, "quadratic": 0.5, '
        '"softplus_barrier": 0.2}',
        'DEFAULT_LAMBDA: dict[str, float] = {"linear": 1.0}',
    )
    assert mutated != source  # guard against the replacement silently matching nothing
    with pytest.raises(ValueError, match="disagree"):
        exec(compile(mutated, "<mutated cost_families>", "exec"), {"__name__": "mutated"})


def test_cost_families_key_mismatch_raises_at_import():
    import moe_congestion_routing.losses.cost_families as cost_families_module

    source = pathlib.Path(cost_families_module.__file__).read_text()
    mutated = source.replace(
        'COST_FAMILIES: tuple[str, ...] = ("linear", "quadratic", "softplus_barrier")',
        'COST_FAMILIES: tuple[str, ...] = ("linear", "quadratic", "softplus_barrier", "power")',
    )
    assert mutated != source
    with pytest.raises(ValueError, match="disagree"):
        exec(compile(mutated, "<mutated cost_families>", "exec"), {"__name__": "mutated"})


# N=64, K=4, E=8 is the shape the raw-to-normalized factor was measured on,
# so balanced_load L = N*K/E = 32 and every load vector below sums to N*K = 256.
_N, _K, _E = 64, 4, 8
_BALANCED_LOAD = _N * _K / _E

_LOAD_VECTORS = {
    "balanced": np.full(_E, _BALANCED_LOAD, dtype=np.int64),
    "concentrated": np.array([256 - 7, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64),
    "uneven": np.array([80, 60, 40, 30, 20, 15, 8, 3], dtype=np.int64),
}


@pytest.mark.parametrize("cost_family", COST_FAMILIES)
@pytest.mark.parametrize("lam", [1.0, 0.5, 2.5])
@pytest.mark.parametrize("loads_name", sorted(_LOAD_VECTORS))
def test_discrete_potential_pins_against_torch_congestion_potential(cost_family, lam, loads_name):
    # rosenthal.congestion_potential returns Phi_cong/(N*K) as float32, whereas discrete_potential
    # is the raw sum, so the two are compared with that factor restored rather than
    # directly, and the float32 side sets the tolerance.
    loads = _LOAD_VECTORS[loads_name]
    assert loads.sum() == _N * _K

    numpy_value = discrete_potential(loads, _BALANCED_LOAD, lam=lam, cost_family=cost_family)
    torch_value = congestion_potential(
        torch.tensor(loads),
        total_num_tokens=_N,
        topk=_K,
        num_experts=_E,
        lam=lam,
        cost_family=cost_family,
    )
    assert numpy_value == pytest.approx(float(torch_value) * (_N * _K), rel=1e-5, abs=1e-4)


@pytest.mark.parametrize("cost_family", COST_FAMILIES)
@pytest.mark.parametrize("lam", [1.0, 0.5, 2.5])
def test_marginal_cost_is_pinned_to_the_torch_cost(cost_family, lam):
    # marginal_cost is a second implementation of rosenthal.cost with the j/L division folded in,
    # so it needs the same pin discrete_potential has. The fixed point below cannot supply it: at
    # j == L every family agrees, so an exponent dropped entirely would still pass there.
    j = np.arange(1, 4 * int(_BALANCED_LOAD) + 1, dtype=np.float64)
    numpy_value = marginal_cost(j, _BALANCED_LOAD, lam=lam, cost_family=cost_family)
    torch_value = rosenthal.cost(torch.tensor(j / _BALANCED_LOAD), cost_family, lam=lam)
    assert numpy_value == pytest.approx(torch_value.numpy(), rel=1e-5, abs=1e-4)


@pytest.mark.parametrize("cost_family", COST_EXPONENTS)
@pytest.mark.parametrize("lam", [1.0, 0.5, 2.5])
def test_marginal_cost_at_balanced_load_is_lam(cost_family, lam):
    # The definition's own fixed point: at j == L the relative load j/L is exactly 1, so the price
    # collapses to lam regardless of the exponent p. Catches an off-by-one in the 1-based index.
    # Power families only: the barrier's price at x=1 is lam*softplus(0)=lam*ln(2), not lam.
    assert marginal_cost(
        _BALANCED_LOAD, _BALANCED_LOAD, lam=lam, cost_family=cost_family
    ) == pytest.approx(lam)


@pytest.mark.parametrize("cost_family", COST_FAMILIES)
def test_discrete_potential_on_zero_loads_is_zero(cost_family):
    zero_loads = np.zeros(_E, dtype=np.int64)
    assert discrete_potential(zero_loads, _BALANCED_LOAD, cost_family=cost_family) == 0.0


@pytest.mark.parametrize("cost_family", COST_FAMILIES)
def test_discrete_potential_increases_moving_a_token_to_the_heavier_expert(cost_family):
    # Convexity of the marginal cost is what makes the LP oracle fill arcs in increasing-price
    # order, so moving one token from the lighter expert to the heavier one must strictly raise
    # the potential.
    loads = np.array([3, 7, 2, 4], dtype=np.int64)
    before = discrete_potential(loads, _BALANCED_LOAD, cost_family=cost_family)

    moved = loads.copy()
    moved[np.argmin(moved)] -= 1
    moved[np.argmax(moved)] += 1
    after = discrete_potential(moved, _BALANCED_LOAD, cost_family=cost_family)

    assert after > before


def test_marginal_cost_unknown_cost_family_raises():
    with pytest.raises(ValueError, match="bogus"):
        marginal_cost(1, _BALANCED_LOAD, cost_family="bogus")


def test_discrete_potential_unknown_cost_family_raises():
    with pytest.raises(ValueError, match="bogus"):
        discrete_potential(np.array([1, 2]), _BALANCED_LOAD, cost_family="bogus")


# Several (threshold, balanced_load) pairs, including the real shape's measured span and its
# balanced load, plus a threshold at and below zero to exercise the early-return branch.
_THRESHOLD_CASES = [
    (0.202505, 2048.0),
    (5.0, 32.0),
    (0.0, 10.0),
    (-3.0, 10.0),
]


@pytest.mark.parametrize("cost_family", COST_FAMILIES)
@pytest.mark.parametrize("lam", [0.125, 1.0, 3.0])
@pytest.mark.parametrize("threshold,balanced_load", _THRESHOLD_CASES)
def test_first_arc_above_price_agrees_with_brute_force(cost_family, lam, threshold, balanced_load):
    j = first_arc_above_price(threshold, balanced_load, lam=lam, cost_family=cost_family)
    assert float(marginal_cost(j, balanced_load, lam=lam, cost_family=cost_family)) > threshold
    if j > 1:
        below = float(marginal_cost(j - 1, balanced_load, lam=lam, cost_family=cost_family))
        assert below <= threshold


def test_first_arc_above_price_nonpositive_lam_raises():
    with pytest.raises(ValueError, match="lam"):
        first_arc_above_price(1.0, 10.0, lam=0.0, cost_family="linear")
    with pytest.raises(ValueError, match="lam"):
        first_arc_above_price(1.0, 10.0, lam=-1.0, cost_family="linear")


def test_first_arc_above_price_unknown_cost_family_raises():
    with pytest.raises(ValueError, match="bogus"):
        first_arc_above_price(1.0, 10.0, cost_family="bogus")


# A fixed grid of j and balanced_load unrelated to any other test in this file, so a defect that
# happens to cancel out on the tables above would still show up here. The reference is the power
# law spelled out again directly, `lam*(j/L)**p`, rather than a value snapshotted before the
# registry became record-based, so a change to how the record is stored cannot silently change
# what "bit-identical" means.
_GRID_J = np.arange(1, 4001, dtype=np.float64)
_GRID_L = 733.0


@pytest.mark.parametrize("cost_family,p", [("linear", 1), ("quadratic", 2)])
@pytest.mark.parametrize("lam", [1.0, 0.4, 3.1])
def test_power_families_are_bit_identical_to_the_direct_formula(cost_family, p, lam):
    expected = lam * (_GRID_J / _GRID_L) ** p
    assert marginal_cost(_GRID_J, _GRID_L, lam=lam, cost_family=cost_family) == pytest.approx(
        expected, rel=0, abs=0
    )

    loads = np.array([50, 0, 733, 4000], dtype=np.int64)
    expected_potential = sum(
        float(np.sum(lam * (np.arange(1, int(n) + 1, dtype=np.float64) / _GRID_L) ** p))
        for n in loads
    )
    assert discrete_potential(loads, _GRID_L, lam=lam, cost_family=cost_family) == pytest.approx(
        expected_potential, rel=0, abs=0
    )

    for threshold in (0.01, 0.5, 2.0, 10.0):
        expected_j = math.floor(_GRID_L * (threshold / lam) ** (1.0 / p)) + 1
        assert (
            first_arc_above_price(threshold, _GRID_L, lam=lam, cost_family=cost_family)
            == expected_j
        )


def test_barrier_price_is_near_zero_below_balance_and_rises_sharply_above_it():
    # x = j/L. Below 1 (under-loaded) the barrier should sit near 0, whereas above 1 it should
    # rise steeply, which is the whole point of a capacity barrier over a power-law cost.
    below = marginal_cost(0.1 * _GRID_L, _GRID_L, cost_family="softplus_barrier", tau=0.1)
    at_balance = marginal_cost(1.0 * _GRID_L, _GRID_L, cost_family="softplus_barrier", tau=0.1)
    above = marginal_cost(1.5 * _GRID_L, _GRID_L, cost_family="softplus_barrier", tau=0.1)
    assert float(below) < 0.01
    assert float(at_balance) < float(above)
    assert float(above) > 4.0


def test_barrier_price_does_not_overflow_at_x_eight():
    # (x-1)/tau = 70 at x=8, tau=0.1, the largest relative load this fleet reaches, so a direct
    # np.exp of that argument would already be inf.
    value = marginal_cost(8.0 * _GRID_L, _GRID_L, lam=1.0, cost_family="softplus_barrier", tau=0.1)
    assert math.isfinite(float(value))
    assert float(value) == pytest.approx(70.0, abs=1e-6)


_BARRIER_THRESHOLD_CASES = [0.0, -3.0, 1e-4, 0.01, 0.1, 0.5, 2.0, 10.0]


@pytest.mark.parametrize("tau", [0.5, 0.1, 0.02])
@pytest.mark.parametrize("threshold", _BARRIER_THRESHOLD_CASES)
def test_first_arc_above_price_round_trips_for_the_barrier(threshold, tau):
    j = first_arc_above_price(threshold, _GRID_L, cost_family="softplus_barrier", tau=tau)
    assert j >= 1
    assert float(marginal_cost(j, _GRID_L, cost_family="softplus_barrier", tau=tau)) > threshold
    if j > 1:
        below = float(marginal_cost(j - 1, _GRID_L, cost_family="softplus_barrier", tau=tau))
        assert below <= threshold


def test_first_arc_above_price_nonpositive_threshold_returns_the_first_arc():
    # The stated branch: threshold <= 0 returns 1 directly for every family, power or barrier,
    # because marginal_cost(1, ...) is already > 0 and the general formula is not even evaluated.
    assert first_arc_above_price(0.0, _GRID_L, cost_family="linear") == 1
    assert first_arc_above_price(-5.0, _GRID_L, cost_family="quadratic") == 1
    assert first_arc_above_price(0.0, _GRID_L, cost_family="softplus_barrier", tau=0.1) == 1
    assert first_arc_above_price(-5.0, _GRID_L, cost_family="softplus_barrier", tau=0.1) == 1


@pytest.mark.parametrize("cost_family", COST_EXPONENTS)
def test_tau_on_a_power_family_raises(cost_family):
    with pytest.raises(ValueError, match="tau"):
        marginal_cost(1, _GRID_L, cost_family=cost_family, tau=0.1)
    with pytest.raises(ValueError, match="tau"):
        first_arc_above_price(1.0, _GRID_L, cost_family=cost_family, tau=0.1)
    with pytest.raises(ValueError, match="tau"):
        discrete_potential(np.array([1, 2]), _GRID_L, cost_family=cost_family, tau=0.1)


def test_barrier_with_no_tau_uses_the_registry_default():
    default = marginal_cost(1.5 * _GRID_L, _GRID_L, cost_family="softplus_barrier")
    explicit = marginal_cost(1.5 * _GRID_L, _GRID_L, cost_family="softplus_barrier", tau=0.1)
    assert float(default) == pytest.approx(float(explicit))
    other_tau = marginal_cost(1.5 * _GRID_L, _GRID_L, cost_family="softplus_barrier", tau=0.5)
    assert float(default) != pytest.approx(float(other_tau))


def test_trainable_set_now_equals_the_oracle_set():
    # softplus_barrier was oracle-only (a proper subset) until this round promoted it into
    # COST_FAMILIES at the 'hard' variant, so the two registries coincide until a future
    # oracle-only family reopens the gap.
    assert set(COST_FAMILIES) == set(ORACLE_COST_FAMILIES)


def test_cost_exponent_softplus_barrier_raises_without_offering_it_as_a_choice():
    # Matching on the family name alone passed against a message that rejected the barrier and
    # then listed it among the permitted values, because that message spelled COST_FAMILIES
    # while the check reads COST_EXPONENTS. The assertion is on the offered set for that reason.
    with pytest.raises(ValueError) as excinfo:
        cost_exponent("softplus_barrier")
    message = str(excinfo.value)
    assert "softplus_barrier" in message
    offered = message.split("expected one of", 1)[1]
    assert "softplus_barrier" not in offered
    for family in COST_EXPONENTS:
        assert family in offered


def test_marginal_cost_and_discrete_potential_accept_softplus_barrier():
    # The registry change's whole point: the oracle-side functions must price a family
    # cost_exponent itself still refuses, rather than the two disagreeing on what exists.
    assert math.isfinite(float(marginal_cost(1, _GRID_L, cost_family="softplus_barrier", tau=0.1)))
    assert math.isfinite(
        discrete_potential(np.array([10, 20]), _GRID_L, cost_family="softplus_barrier", tau=0.1)
    )


def test_config_naming_softplus_barrier_hard_is_accepted():
    # Asserted through the public config path, training/pretrain_config.py's own validation,
    # rather than by reading COST_FAMILIES out of this module, because that validation is what a
    # yaml file actually goes through.
    cfg = MoEPretrainConfig(
        train_data_path="/data/train",
        lr_wsd_decay_iters=10,
        moe_router_load_balancing_type="rosenthal",
        moe_rosenthal_cost="softplus_barrier",
        moe_rosenthal_variant="hard",
    )
    build_megatron_args(cfg)  # must not raise


def test_config_naming_softplus_barrier_soft_is_rejected_naming_the_dilogarithm():
    cfg = MoEPretrainConfig(
        train_data_path="/data/train",
        lr_wsd_decay_iters=10,
        moe_router_load_balancing_type="rosenthal",
        moe_rosenthal_cost="softplus_barrier",
        moe_rosenthal_variant="soft",
    )
    with pytest.raises(ValueError, match="dilogarithm"):
        build_megatron_args(cfg)


@pytest.mark.parametrize("bad_tau", [0.0, -0.1])
def test_a_non_positive_tau_raises_on_every_entry_point(bad_tau):
    """Both failures are silent without the guard, and `tau=0` is the natural thing to type
    when probing the hard-capacity limit the calibration sweep converges toward.

    At `tau == 0` the price is `0/0` at balanced load and `+inf` above it, and at `tau < 0` the
    barrier decreases in load, so `first_arc_above_price` returns an arc whose price sits *below*
    the threshold it was asked to exceed.
    """
    for call in (
        lambda: marginal_cost(733, 733.0, cost_family="softplus_barrier", tau=bad_tau),
        lambda: first_arc_above_price(0.5, 733.0, cost_family="softplus_barrier", tau=bad_tau),
        lambda: discrete_potential(
            np.array([1.0]), 733.0, cost_family="softplus_barrier", tau=bad_tau
        ),
    ):
        with pytest.raises(ValueError, match="tau must be positive"):
            call()


@pytest.mark.parametrize("y", [50.0, 300.0, 700.0])
def test_softplus_inverse_round_trips_in_the_large_y_branch(y):
    """The stable rewrite for large `y` is the whole point of `_softplus_inverse`, and nothing
    else in the suite reaches it: the barrier thresholds this repo actually prices sit near 0.2
    to 3.5, so the branch would ship unexercised. A naive `log(exp(y) - 1)` overflows near 709.
    """
    z = _softplus_inverse(np.float64(y))
    assert np.isfinite(z)
    assert _softplus(z) == pytest.approx(y, rel=1e-12)


@pytest.mark.parametrize("bad_tau", [0.0, -0.5])
def test_pressure_bound_rejects_a_non_positive_tau(bad_tau, monkeypatch):
    # pressure_bound was the one barrier price path reading record.tau directly instead of
    # through _resolve_tau, so a backwards barrier produced a plausible bound and no warning.
    import moe_congestion_routing.losses.cost_families as cf

    monkeypatch.setitem(cf._ORACLE_RECORDS, "softplus_barrier", cf._BarrierCost(tau=bad_tau))
    with pytest.raises(ValueError, match="tau must be positive"):
        pressure_bound(0.01, 0.2, 4, 2, "softplus_barrier", "hard")


def test_pressure_bound_barrier_value_agrees_with_marginal_cost():
    # pressure_bound states the barrier price a second time inside this module. Pinned to
    # marginal_cost at x = base rather than to a hand-computed constant, so the two cannot drift.
    coeff, lam, num_experts, topk = 0.01, 0.2, 4, 2
    base = num_experts / topk
    bound = pressure_bound(coeff, lam, num_experts, topk, "softplus_barrier", "hard")
    want = coeff * float(marginal_cost(base, 1.0, lam=lam, cost_family="softplus_barrier"))
    assert bound.value == pytest.approx(want, rel=1e-12)

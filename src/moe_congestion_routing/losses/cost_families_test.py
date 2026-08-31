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
    ROSENTHAL_TYPES,
    VARIANTS,
    check_variant,
    cost_exponent,
    discrete_potential,
    marginal_cost,
    pressure_bound,
)
from moe_congestion_routing.losses.rosenthal import congestion_potential


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
    assert set(COST_EXPONENTS) == set(COST_FAMILIES)
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
    # needs no override.
    bound = pressure_bound(1.0, 1.0, num_experts=4, topk=2, cost_family="linear", variant="hard")
    assert bound.expr == "(num_experts/moe_router_topk)"


def test_pressure_bound_soft_expr_is_the_uncapped_count_with_default_names():
    bound = pressure_bound(1.0, 1.0, num_experts=4, topk=2, cost_family="linear", variant="soft")
    assert bound.expr == "num_experts"


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
    assert bound.expr == "(num_moe_experts/moe_router_topk)"


def test_default_lambda_cost_exponents_key_mismatch_raises_at_import():
    # Proves the guard constrains the module rather than restating an invariant that already
    # holds. A mutated copy of the module's own source, with DEFAULT_LAMBDA missing a key
    # COST_EXPONENTS has, must make the module-level check raise exactly as a real edit adding a
    # family to one dict and forgetting the other would.
    import moe_congestion_routing.losses.cost_families as cost_families_module

    source = pathlib.Path(cost_families_module.__file__).read_text()
    mutated = source.replace(
        'DEFAULT_LAMBDA: dict[str, float] = {"linear": 1.0, "quadratic": 0.5}',
        'DEFAULT_LAMBDA: dict[str, float] = {"linear": 1.0}',
    )
    assert mutated != source  # guard against the replacement silently matching nothing
    with pytest.raises(ValueError, match="disagree"):
        exec(compile(mutated, "<mutated cost_families>", "exec"), {"__name__": "mutated"})


def test_cost_families_key_mismatch_raises_at_import():
    import moe_congestion_routing.losses.cost_families as cost_families_module

    source = pathlib.Path(cost_families_module.__file__).read_text()
    mutated = source.replace(
        'COST_FAMILIES: tuple[str, ...] = ("linear", "quadratic")',
        'COST_FAMILIES: tuple[str, ...] = ("linear", "quadratic", "power")',
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


@pytest.mark.parametrize("cost_family", COST_FAMILIES)
@pytest.mark.parametrize("lam", [1.0, 0.5, 2.5])
def test_marginal_cost_at_balanced_load_is_lam(cost_family, lam):
    # The definition's own fixed point: at j == L the relative load j/L is exactly 1, so the price
    # collapses to lam regardless of the exponent p. Catches an off-by-one in the 1-based index.
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

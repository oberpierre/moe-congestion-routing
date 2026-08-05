import pathlib
import subprocess
import sys

import pytest

from moe_congestion_routing.losses.cost_families import (
    COST_EXPONENTS,
    COST_FAMILIES,
    DEFAULT_LAMBDA,
    ROSENTHAL_TYPES,
    VARIANTS,
    check_variant,
    cost_exponent,
    pressure_bound,
)


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
    # coeff * lam * (E/K)**p -- the marginal cost c(.) evaluated at the highest relative load any
    # single expert can reach under top-k selection, matching cost_exponent's own p per family.
    assert pressure_bound(2.0, 3.0, num_experts=8, topk=2, cost_family="linear") == pytest.approx(
        2.0 * 3.0 * (8 / 2) ** 1
    )
    assert pressure_bound(
        2.0, 3.0, num_experts=8, topk=2, cost_family="quadratic"
    ) == pytest.approx(2.0 * 3.0 * (8 / 2) ** 2)


def test_pressure_bound_unknown_cost_family_raises():
    with pytest.raises(ValueError, match="bogus"):
        pressure_bound(1.0, 1.0, num_experts=4, topk=1, cost_family="bogus")


def test_default_lambda_cost_exponents_key_mismatch_raises_at_import():
    # Proves the guard actually constrains the module rather than restating an invariant that
    # already holds: execute a mutated copy of the module's own source (same shape, DEFAULT_LAMBDA
    # missing a key COST_EXPONENTS has) and watch the module-level check raise, exactly as it would
    # for a real edit that added a cost family to one dict and forgot the other.
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

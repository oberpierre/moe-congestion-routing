"""Registry of congestion cost family names, exponents and lambda defaults.

Deliberately ``torch``-free: ``training/pretrain_config.py`` imports the names
from here to validate a config, and ``--dry-run`` should not require ``torch``.
"""

COST_FAMILIES: tuple[str, ...] = ("linear", "quadratic")
VARIANTS: tuple[str, ...] = ("hard", "soft")
COST_EXPONENTS: dict[str, int] = {"linear": 1, "quadratic": 2}
DEFAULT_LAMBDA: dict[str, float] = {"linear": 1.0, "quadratic": 0.5}

# The two Megatron ``moe_router_load_balancing_type`` values that select the congestion loss
# (micro-batch vs global-batch reduction).
ROSENTHAL_TYPES: tuple[str, ...] = ("rosenthal", "global_rosenthal")

# Guard to keep declarations in sync and prevent runtime KeyError, promoting it to import time
# failure instead.
if not (set(DEFAULT_LAMBDA) == set(COST_EXPONENTS) == set(COST_FAMILIES)):
    raise ValueError(
        "COST_FAMILIES, COST_EXPONENTS and DEFAULT_LAMBDA disagree on cost families: "
        f"{sorted(COST_FAMILIES)} != {sorted(COST_EXPONENTS)} != {sorted(DEFAULT_LAMBDA)}"
    )


def cost_exponent(cost_family: str) -> int:
    """The exponent ``p`` of ``c(x) = lam * x**p`` for ``cost_family``, or raise."""
    if cost_family not in COST_EXPONENTS:
        raise ValueError(f"unknown cost family {cost_family!r}; expected one of {COST_FAMILIES}")
    return COST_EXPONENTS[cost_family]


def check_variant(variant: str) -> None:
    """Raise unless ``variant`` is a known loss variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


def pressure_bound(
    coeff: float, lam: float, num_experts: int, topk: int, cost_family: str
) -> float:
    """Sanity bound on the cong pressure at full imbalance under rel. load: ``coeff * c(E/K)``."""
    return coeff * lam * (num_experts / topk) ** cost_exponent(cost_family)

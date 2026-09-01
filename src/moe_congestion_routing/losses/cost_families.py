"""Registry of congestion cost family names, exponents and lambda defaults.

Deliberately ``torch``-free: ``training/pretrain_config.py`` imports the names
from here to validate a config, and ``--dry-run`` should not require ``torch``.
"""

import math
from typing import NamedTuple

import numpy as np

COST_FAMILIES: tuple[str, ...] = ("linear", "quadratic")
VARIANTS: tuple[str, ...] = ("hard", "soft")
COST_EXPONENTS: dict[str, int] = {"linear": 1, "quadratic": 2}
DEFAULT_LAMBDA: dict[str, float] = {"linear": 1.0, "quadratic": 0.5}

# The two Megatron ``moe_router_load_balancing_type`` values that select the congestion loss
# (micro-batch vs global-batch reduction).
ROSENTHAL_TYPES: tuple[str, ...] = ("rosenthal", "global_rosenthal")

# Keeps the three declarations in sync. A family added to one and forgotten in another becomes an
# import-time failure here rather than a KeyError at some later call site.
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


def marginal_cost(
    j: np.ndarray | int,
    balanced_load: float,
    *,
    lam: float = 1.0,
    cost_family: str = "linear",
) -> np.ndarray:
    """Marginal price of the ``j``-th (1-based) token routed to an expert, ``lam*(j/L)**p``.

    ``j`` is the arc index in the oracle's per-expert cost-flow graph, so this is the price the LP
    oracle assigns to that arc. Float64, matching the LP's own precision rather than the torch
    loss's float32.
    """
    p = cost_exponent(cost_family)
    j_arr = np.asarray(j, dtype=np.float64)
    return lam * (j_arr / balanced_load) ** p


def first_arc_above_price(
    threshold: float,
    balanced_load: float,
    *,
    lam: float = 1.0,
    cost_family: str = "linear",
) -> int:
    """The smallest 1-based ``j`` with ``marginal_cost(j, ...) > threshold``, in closed form.

    ``marginal_cost`` is strictly increasing in ``j`` for ``lam > 0``, so the crossing point
    ``x* = balanced_load * (threshold/lam)**(1/p)`` is unique and ``j = floor(x*) + 1``.
    ``threshold <= 0`` returns 1 directly, because ``marginal_cost(1, ...) > 0`` already, and the
    real branch's fractional power would otherwise be taken of a negative base under the quadratic
    family. This is the inverse of ``marginal_cost`` and is what lets a caller size an arc schedule
    from a price bound without building the schedule first.
    """
    p = cost_exponent(cost_family)
    if lam <= 0:
        raise ValueError(
            f"lam must be positive, got {lam}: at lam <= 0 every marginal cost is 0, so no j "
            "ever exceeds a positive threshold and the inverse is undefined"
        )
    if threshold <= 0:
        return 1
    x_star = balanced_load * (threshold / lam) ** (1.0 / p)
    return math.floor(x_star) + 1


def discrete_potential(
    loads: np.ndarray,
    balanced_load: float,
    *,
    lam: float = 1.0,
    cost_family: str = "linear",
) -> float:
    """Discrete Rosenthal potential ``sum_e sum_{j=1..n_e} lam*(j/L)**p`` over realized loads.

    Raw sum, unnormalized by ``N*K``, unlike ``rosenthal.congestion_potential``, because this is
    the scored quantity the LP oracle's objective is checked against.
    """
    # Validate eagerly, because an all-zero load vector never enters the loop and would otherwise
    # accept an unknown family silently.
    cost_exponent(cost_family)
    total = 0.0
    for n in np.asarray(loads):
        n_int = int(n)
        if n_int <= 0:
            continue
        j = np.arange(1, n_int + 1, dtype=np.float64)
        # Through marginal_cost rather than inline, so the exponent is applied in exactly one place
        # and a defect in the price cannot cancel itself out of the potential.
        total += float(np.sum(marginal_cost(j, balanced_load, lam=lam, cost_family=cost_family)))
    return total


def check_variant(variant: str) -> None:
    """Raise unless ``variant`` is a known loss variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


class PressureBound(NamedTuple):
    """A sanity-bound value paired with the expression string that produced it.

    Every caller that warns about ``value`` also has to name which expression it came from, so
    returning both together keeps that branch in one place. A caller re-deriving the expression by
    testing ``variant`` again could pick the wrong branch while ``value`` stayed correct, and
    nothing would catch it, because the value-pinning tests do not assert on warning strings.
    """

    value: float
    expr: str


def pressure_bound(
    coeff: float,
    lam: float,
    num_experts: int,
    topk: int,
    cost_family: str,
    variant: str,
    *,
    num_experts_name: str = "num_experts",
    topk_name: str = "moe_router_topk",
) -> PressureBound:
    """Sanity bound on the congestion pressure at full imbalance, for ``variant``'s relative load.

    ``hard``'s relative load is a top-k selection count, so it is capped at ``E/K``, the value it
    reaches when every token routes to one expert. ``soft``'s relative load is softmax mass, which
    has no selection cap and can concentrate up to ``E``. The two variants therefore reach
    different worst cases even though both loads average to ``E``. ``variant`` is required rather
    than defaulted, because a default that returned the ``hard`` bound for a soft arm would
    under-warn by exactly ``K**p``, which is the quiet failure this check exists to prevent.

    ``num_experts_name`` and ``topk_name`` let each caller spell the bound with its own config
    field's name, ``num_experts`` for ``MoEPretrainConfig`` and ``num_moe_experts`` for Megatron's
    ``TransformerConfig``, while the branch deciding which expression applies stays here rather
    than being repeated at every call site.
    """
    check_variant(variant)
    if variant == "hard":
        base = num_experts / topk
        expr = f"({num_experts_name}/{topk_name})"
    else:
        base = num_experts
        expr = num_experts_name
    value = coeff * lam * base ** cost_exponent(cost_family)
    return PressureBound(value, expr)

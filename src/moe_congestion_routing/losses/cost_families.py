"""Registry of congestion cost family names, exponents, lambda defaults and shape parameters.

``COST_FAMILIES`` is what the trainable Rosenthal loss supports and what a config may name:
``linear``, ``quadratic`` and ``softplus_barrier``, each at either variant. A family with no
exponent (the barrier) is a record rather than an entry in ``COST_EXPONENTS``, which is why
``marginal_cost``, ``first_arc_above_price`` and ``discrete_potential`` route through
``_oracle_family`` instead of ``cost_exponent`` for every family, power or barrier alike.
``ORACLE_COST_FAMILIES`` names the same set today, but it stays a separate registry because it
answers a different question (priceable by the offline LP/incremental-arc oracle, vs. trainable
by the router) and may diverge again if a future family is ever oracle-only.

Deliberately ``torch``-free: ``training/pretrain_config.py`` imports the names
from here to validate a config, and ``--dry-run`` should not require ``torch``. The one function
that needs a torch price for the barrier (``cost()`` in ``losses/rosenthal.py``) reads its shape
parameter, ``tau``, from ``barrier_tau()`` below rather than restating it.
"""

import math
from typing import NamedTuple

import numpy as np

COST_FAMILIES: tuple[str, ...] = ("linear", "quadratic", "softplus_barrier")
VARIANTS: tuple[str, ...] = ("hard", "soft")
COST_EXPONENTS: dict[str, int] = {"linear": 1, "quadratic": 2}
DEFAULT_LAMBDA: dict[str, float] = {"linear": 1.0, "quadratic": 0.5, "softplus_barrier": 0.2}

# The two Megatron ``moe_router_load_balancing_type`` values that select the congestion loss
# (micro-batch vs global-batch reduction).
ROSENTHAL_TYPES: tuple[str, ...] = ("rosenthal", "global_rosenthal")

# The barrier has no exponent, so only DEFAULT_LAMBDA has to cover every family. A power family
# left out of COST_EXPONENTS is not caught here, and fails at its first cost() call instead.
if not (set(COST_EXPONENTS) <= set(COST_FAMILIES)):
    raise ValueError(
        "COST_EXPONENTS names a family COST_FAMILIES does not: "
        f"{sorted(set(COST_EXPONENTS) - set(COST_FAMILIES))}"
    )
if set(DEFAULT_LAMBDA) != set(COST_FAMILIES):
    raise ValueError(
        "COST_FAMILIES and DEFAULT_LAMBDA disagree on cost families: "
        f"{sorted(COST_FAMILIES)} != {sorted(DEFAULT_LAMBDA)}"
    )


class _PowerCost(NamedTuple):
    """``c(x) = lam * x**exponent``, the shape every trainable family currently has."""

    exponent: int


class _BarrierCost(NamedTuple):
    """``c(x) = lam * softplus((x - 1) / tau)``, with a registry default for ``tau``."""

    tau: float


# Numbers, not callables, so this file needs no torch and a config can be checked without it.
# The two price functions (numpy here, torch in rosenthal.py) read the shape from this one place.
_ORACLE_RECORDS: dict[str, _PowerCost | _BarrierCost] = {
    family: _PowerCost(exponent) for family, exponent in COST_EXPONENTS.items()
}
_ORACLE_RECORDS["softplus_barrier"] = _BarrierCost(tau=0.1)
ORACLE_COST_FAMILIES: tuple[str, ...] = tuple(_ORACLE_RECORDS)


def _oracle_family(cost_family: str) -> _PowerCost | _BarrierCost:
    """The record backing ``cost_family`` for the oracle-side price functions, or raise."""
    if cost_family not in _ORACLE_RECORDS:
        raise ValueError(
            f"unknown cost family {cost_family!r}, expected one of {ORACLE_COST_FAMILIES}"
        )
    return _ORACLE_RECORDS[cost_family]


def barrier_tau(cost_family: str) -> float:
    """``tau`` for a barrier family, read from the one registry rather than restated at each
    caller. Raises for a power family, mirroring ``_resolve_tau``'s guard, since a power family
    has no ``tau`` to read. The only caller outside this module is ``losses/rosenthal.py``'s
    ``cost()``, which needs ``tau`` but must stay the only place that knows about ``torch``.
    """
    record = _oracle_family(cost_family)
    if isinstance(record, _PowerCost):
        raise ValueError(
            f"tau is not a parameter of cost_family {cost_family!r} because only "
            "'softplus_barrier' has a tau"
        )
    return record.tau


def _resolve_tau(record: _PowerCost | _BarrierCost, cost_family: str, tau: float | None) -> float:
    """``tau`` for a barrier record, defaulted from the registry, or raise if it was passed to a
    power family, which has no shape to override and would leave a caller measuring nothing."""
    if isinstance(record, _PowerCost):
        if tau is not None:
            raise ValueError(
                f"tau is not a parameter of cost_family {cost_family!r}; only "
                "'softplus_barrier' takes tau"
            )
        return 0.0
    resolved = record.tau if tau is None else tau
    # Both failures below are silent without this guard, which is why it is a raise and not a
    # clamp. At tau == 0 the price is 0/0 = NaN exactly at balanced load and +inf above it, and
    # at tau < 0 the barrier runs backwards, so marginal_cost stops increasing in j and
    # first_arc_above_price returns an arc whose price is below the threshold it was given.
    if resolved <= 0:
        raise ValueError(
            f"tau must be positive, got {resolved}: at tau == 0 the price is NaN at balanced "
            f"load and infinite above it, and at tau < 0 the barrier decreases in load, which "
            f"breaks the monotonicity first_arc_above_price inverts"
        )
    return resolved


def _softplus(z: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(1 + exp(z))``, avoiding overflow at the ``z`` up to ~70 this
    fleet's loads reach at ``tau = 0.1``, where a direct ``np.exp(z)`` would already be inf."""
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def _softplus_inverse(y: float) -> float:
    """``log(exp(y) - 1)``, the inverse of softplus, for ``y > 0``.

    ``math.expm1(y)`` is exact for small ``y`` but its argument overflows once ``y`` is a few
    hundred, so past that point this rewrites to ``y + log1p(-exp(-y))``, which is stable because
    ``exp(-y)`` alone underflows to 0 rather than the whole expression overflowing to inf.
    """
    if y > 30.0:
        return y + math.log1p(-math.exp(-y))
    return math.log(math.expm1(y))


def cost_exponent(cost_family: str) -> int:
    """The exponent ``p`` of ``c(x) = lam * x**p`` for ``cost_family``, or raise.

    Only the power families have a ``p``, so this raises for ``'softplus_barrier'`` exactly as it
    raises for any other name COST_EXPONENTS does not carry, rather than inventing one.
    """
    if cost_family not in COST_EXPONENTS:
        raise ValueError(
            f"{cost_family!r} has no power-law exponent, expected one of "
            f"{tuple(COST_EXPONENTS)}. COST_EXPONENTS is a strict subset of COST_FAMILIES, so a "
            "trainable family may be absent from it without being unknown."
        )
    return COST_EXPONENTS[cost_family]


def marginal_cost(
    j: np.ndarray | int,
    balanced_load: float,
    *,
    lam: float = 1.0,
    cost_family: str = "linear",
    tau: float | None = None,
) -> np.ndarray:
    """Marginal price of the ``j``-th (1-based) token routed to an expert.

    ``lam*(j/L)**p`` for a power family, or ``lam*softplus((j/L - 1)/tau)`` for the barrier,
    where ``tau`` defaults to the family's own registry value and raises if given to a power
    family. ``j`` is the arc index in the oracle's per-expert cost-flow graph, so this is the
    price the LP oracle assigns to that arc. Float64, matching the LP's own precision rather than
    the torch loss's float32.
    """
    record = _oracle_family(cost_family)
    tau_value = _resolve_tau(record, cost_family, tau)
    j_arr = np.asarray(j, dtype=np.float64)
    x = j_arr / balanced_load
    if isinstance(record, _PowerCost):
        return lam * x**record.exponent
    return lam * _softplus((x - 1.0) / tau_value)


def first_arc_above_price(
    threshold: float,
    balanced_load: float,
    *,
    lam: float = 1.0,
    cost_family: str = "linear",
    tau: float | None = None,
) -> int:
    """The smallest 1-based ``j`` with ``marginal_cost(j, ...) > threshold``, in closed form.

    ``marginal_cost`` is strictly increasing in ``j`` for ``lam > 0``, so the crossing point is
    unique. For a power family it is ``x* = balanced_load * (threshold/lam)**(1/p)``, unchanged.
    For the barrier it inverts softplus, ``x* = balanced_load * (1 + tau*log(exp(threshold/lam)
    - 1))``, which can fall below the ``j=1`` arc when even the very first arc's price already
    exceeds ``threshold``, so the result is clamped to ``max(1, floor(x*) + 1)`` rather than
    returned negative or zero. ``threshold <= 0`` returns 1 directly, because
    ``marginal_cost(1, ...) > 0`` already, and the real branch's fractional power would otherwise
    be taken of a negative base under the quadratic family. This is the inverse of
    ``marginal_cost`` and is what lets a caller size an arc schedule from a price bound without
    building the schedule first.
    """
    record = _oracle_family(cost_family)
    tau_value = _resolve_tau(record, cost_family, tau)
    if lam <= 0:
        raise ValueError(
            f"lam must be positive, got {lam}: at lam <= 0 every marginal cost is 0, so no j "
            "ever exceeds a positive threshold and the inverse is undefined"
        )
    if threshold <= 0:
        return 1
    if isinstance(record, _PowerCost):
        x_star = balanced_load * (threshold / lam) ** (1.0 / record.exponent)
    else:
        x_star = balanced_load * (1.0 + tau_value * _softplus_inverse(threshold / lam))
    return max(1, math.floor(x_star) + 1)


def discrete_potential(
    loads: np.ndarray,
    balanced_load: float,
    *,
    lam: float = 1.0,
    cost_family: str = "linear",
    tau: float | None = None,
) -> float:
    """Discrete Rosenthal potential ``sum_e sum_{j=1..n_e} marginal_cost(j, ...)`` over realized
    loads.

    Raw sum, unnormalized by ``N*K``, unlike ``rosenthal.congestion_potential``, because this is
    the scored quantity the LP oracle's objective is checked against.
    """
    # Validated eagerly against the oracle's wider family set, because an all-zero load vector
    # never enters the loop and would otherwise accept an unknown family (or a misused tau)
    # silently.
    record = _oracle_family(cost_family)
    _resolve_tau(record, cost_family, tau)
    total = 0.0
    for n in np.asarray(loads):
        n_int = int(n)
        if n_int <= 0:
            continue
        j = np.arange(1, n_int + 1, dtype=np.float64)
        # Through marginal_cost rather than inline, so the price is computed in exactly one place
        # and a defect there cannot cancel itself out of the potential.
        total += float(
            np.sum(marginal_cost(j, balanced_load, lam=lam, cost_family=cost_family, tau=tau))
        )
    return total


def check_variant(variant: str) -> None:
    """Raise unless ``variant`` is a known loss variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")


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

    ``expr`` is the COMPLETE price expression, exponent or softplus-form included, rather than a
    base a caller then exponentiates itself: a caller re-deriving which price applies by testing
    ``cost_family`` a second time could print an expression that does not match ``value``.
    """
    check_variant(variant)
    if variant == "hard":
        base = num_experts / topk
        base_expr = f"({num_experts_name}/{topk_name})"
    else:
        base = num_experts
        base_expr = num_experts_name
    record = _oracle_family(cost_family)
    if isinstance(record, _PowerCost):
        value = coeff * lam * base**record.exponent
        expr = f"{base_expr}**{record.exponent}"
    else:
        # _resolve_tau, not record.tau, because it is the only place tau is checked positive.
        # A negative tau prints a bound from a backwards barrier, and no test reads warning text.
        tau = _resolve_tau(record, cost_family, None)
        value = coeff * lam * float(_softplus(np.asarray((base - 1.0) / tau)))
        expr = f"softplus(({base_expr} - 1)/{tau})"
    return PressureBound(value, expr)

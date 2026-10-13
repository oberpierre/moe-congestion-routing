"""Rosenthal congestion loss: cost families, loss variants, pressure, and the discrete potential.

Per MoE layer, over the token set fixed by the balancing type (micro-batch or global-batch,
changing what ``tokens_per_expert`` and ``total_num_tokens`` are reduced over, never the
math below):

    N          tokens the counts are reduced over (``total_num_tokens``)
    E, K       experts, top-k (``num_experts``, ``topk``)
    L = N*K/E  balanced load: the load each expert carries if assignment were perfectly uniform
    prob_sum   [E], the local rank's sum over tokens of the pre-top-k routing scores;
               differentiable
    n          [E], hard per-expert counts; always treated as detached (a count carries no gradient
               regardless of what the caller passes)
    P = prob_sum / N        mean gate mass per expert
    u = E*P                 soft relative load: differentiable, tracks the router's own scores
    u_hat = E*n/(N*K)       hard relative load: detached, tracks the realized assignment

    This loss is defined on per-token normalized routing scores: whatever score function produced
    them, ``prob_sum`` is built from scores that sum to 1 per token, so ``sum_e prob_sum_e = N``.
    That is what keeps the congestion game unweighted, not a coincidence of softmax in particular:
    Megatron normalizes sigmoid and sqrtsoftplus the same way, so the identity holds under any of
    the three. Top-k selection independently makes K assignments per token, so ``sum_e n_e = N*K``.
    ``u`` and ``u_hat`` both have the same mass (``sum_e u = sum_e u_hat = E``) and both equal 1 at
    a perfectly balanced batch, for every K.

``hard`` is linear in ``prob_sum``, so per-rank contributions sum to the whole batch, same as
Megatron's own decomposition. ``soft``'s potential is degree ``p+1`` and does not decompose,
so ``rosenthal_loss`` never differentiates it: it prices the local ``prob_sum`` against ``c(·)``
evaluated at the detached, already-reduced ``global_prob_sum`` (defaulting to ``prob_sum``,
exact at group size 1). Required because sum of C(·) != C(sum of ·) for every cost family.

Two power-law cost families, `c(x)` the marginal cost of relative load `x` and
`C(x) = integral_0^x c`:

    linear:     p=1  c(x) = lam*x     C(x) = lam*x**2/2  default lam = 1.0
    quadratic:  p=2  c(x) = lam*x**2  C(x) = lam*x**3/3  default lam = 0.5
                                       (slope-matched to linear at x=1: lam_p = lam_1/p)

A third family, `softplus_barrier` (`c(x) = lam*softplus((x-1)/tau)`), trains at both variants.
Its `C(x)` has no elementary form but is an exact dilogarithm (see `cost_antiderivative` and
`_dilog_neg_exp` below), needed only by `soft`'s logged value. See `losses/cost_families.py` for
its registry entry and `tau`.

Both loss variants share the prefactor ``alpha/E`` (``alpha = moe_aux_loss_coeff``), which equals
``alpha*L/(N*K)``. So the loss is the Rosenthal potential per assignment of which there are N*K:

    soft:  L_CG = alpha*(1/E)*(lam/(p+1)) * sum_e u[e]**(p+1)        - continuous potential
    hard:  L_CG = alpha*(1/E) * sum_e detach(lam*u_hat[e]**p) * u[e] - linearized at the realized
                                                                       load

    dL_CG/dP[e] = alpha*lam*u[e]**p       (soft)
                = alpha*lam*u_hat[e]**p   (hard)

Both are ``alpha * c(relative load)``, the congestion price, evaluated at the expected load
(soft) or the realized load (hard). They coincide iff u = u_hat, i.e. prob_sum = n/K elementwise.
The gap between them is what frac_gate_l1 measures.

Exact-Switch identity, for every K: ``hard`` + ``linear`` + ``lam=1`` reduces algebraically to
``alpha*E/(K*N**2) * sum_e n[e]*prob_sum[e]``, which is Megatron's own
``switch_load_balancing_loss_func`` expression verbatim (its ``E * sum(f_i * P_i)`` written out).
This is not a K=1 special case: substituting u = E*prob_sum/N and u_hat = E*n/(N*K) into the
hard-variant sum makes every factor of K and E cancel except the ones in the identity above.

The discrete Rosenthal congestion potential itself, computed from the realized assignment alone,
not of any loss the model was trained against. Using the closed form from the identity:
``sum_{j=1}^{n} j = n(n+1)/2`` (linear) and ``sum_{j=1}^{n} j**2 = n(n+1)(2n+1)/6`` (quadratic):

    Phi_cong = sum_e sum_{j=1..n_e} c(j/L)
             = sum_e lam*n_e*(n_e+1) / (2*L)                      (linear)
             = sum_e lam*n_e*(n_e+1)*(2*n_e+1) / (6*L**2)         (quadratic)

``congestion_potential`` at perfect balance is the right-endpoint Riemann sum of ``C(1)`` over L
subintervals, so it exceeds it by ``lam/(2L)``. Exact for linear, to leading order for quadratic.
That is the same ``lam/(2L)`` the discretization-gap identity pins at ``u == u_hat``; the balanced
case is its special instance.
"""

import math
from collections.abc import Callable

import torch

from moe_congestion_routing.losses.cost_families import (
    COST_EXPONENTS,
    COST_FAMILIES,
    DEFAULT_LAMBDA,
    VARIANTS,
    barrier_tau,
    check_variant,
    cost_exponent,
)

__all__ = [
    "COST_EXPONENTS",
    "COST_FAMILIES",
    "DEFAULT_LAMBDA",
    "VARIANTS",
    "balanced_load",
    "congestion_potential",
    "cost",
    "cost_antiderivative",
    "hard_relative_load",
    "pressure",
    "price_bias_step",
    "relative_loads",
    "rosenthal_loss",
]


def _as_float_tensor(value: float | torch.Tensor) -> torch.Tensor:
    return value.float() if isinstance(value, torch.Tensor) else torch.tensor(float(value))


def cost(x: torch.Tensor, cost_family: str, lam: float = 1.0) -> torch.Tensor:
    """Marginal congestion cost: ``lam * x**p`` for a power family, or
    ``lam * softplus((x - 1) / tau)`` for ``'softplus_barrier'``, with ``tau`` read from the one
    shared registry (``barrier_tau``) so this and ``cost_families.marginal_cost`` cannot silently
    price the same family differently. ``torch.nn.functional.softplus`` is already
    overflow-safe, unlike a direct ``exp``."""
    if cost_family == "softplus_barrier":
        tau = barrier_tau(cost_family)
        return lam * torch.nn.functional.softplus((x.float() - 1.0) / tau)
    p = cost_exponent(cost_family)
    return lam * x.float() ** p


# 24 terms because the series argument never exceeds 0.5, so the tail is under 0.5**24/576,
# about 1e-10, which is far below what float32 carries.
_DILOG_SERIES_TERMS = 24


def _dilog_neg_exp(z: torch.Tensor) -> torch.Tensor:
    """``Li2(-e^z)`` for any real ``z``, staying finite where ``exp(z)`` alone would overflow.

    Two folds, each needed. ``Li2(-e^z) = -pi^2/6 - z^2/2 - Li2(-e^-z)`` moves ``z > 0`` onto a
    negative exponent, which is what avoids the overflow. The Landen transform
    ``Li2(w) = -0.5*log(1-w)**2 - Li2(w/(w-1))`` then shrinks the series argument to at most 0.5,
    without which the series at ``w = -1`` would need thousands of terms.
    """
    z = z.float()
    w = -torch.exp(-z.abs())  # always in [-1, 0], never overflows since the exponent is <= 0
    v = w / (w - 1.0)  # in [0, 0.5]
    series = torch.zeros_like(v)
    v_pow = torch.ones_like(v)
    for k in range(1, _DILOG_SERIES_TERMS + 1):
        v_pow = v_pow * v
        series = series + v_pow / (k * k)
    li2_w = -0.5 * torch.log1p(-w) ** 2 - series
    return torch.where(z > 0, -(math.pi**2) / 6 - z**2 / 2 - li2_w, li2_w)


def cost_antiderivative(x: torch.Tensor, cost_family: str, lam: float = 1.0) -> torch.Tensor:
    """``C(x) = integral_0^x c``: ``lam * x**(p+1) / (p+1)`` for a power family, or the exact
    dilogarithm form for ``softplus_barrier``. Only the ``soft`` variant calls this, for its
    logged value, because ``hard``'s loss never needs an antiderivative.
    """
    if cost_family == "softplus_barrier":
        tau = barrier_tau(cost_family)
        x = x.float()
        z = (x - 1.0) / tau
        const = _dilog_neg_exp(torch.tensor(-1.0 / tau, dtype=x.dtype, device=x.device))
        return lam * tau * (const - _dilog_neg_exp(z))
    p = cost_exponent(cost_family)
    return lam * x.float() ** (p + 1) / (p + 1)


def balanced_load(
    total_num_tokens: float | torch.Tensor, topk: int, num_experts: int
) -> torch.Tensor:
    """``L = N*K/E``, the load each expert carries if assignment were perfectly uniform."""
    total = _as_float_tensor(total_num_tokens)
    return total * topk / num_experts


def hard_relative_load(
    tokens_per_expert: torch.Tensor,
    total_num_tokens: float | torch.Tensor,
    topk: int,
    num_experts: int,
) -> torch.Tensor:
    """``u_hat = E*n/(N*K)``, the detached hard relative load per expert.

    ``n`` is a count, not a score, so it carries no gradient regardless of what the caller passes
    in, therefore ``.detach()`` is required.
    """
    total = _as_float_tensor(total_num_tokens)
    n = tokens_per_expert.float().detach()
    return (num_experts * n / (total * topk)).detach()


def relative_loads(
    prob_sum: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: float | torch.Tensor,
    topk: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(u, u_hat)``: soft (differentiable) and hard (detached) relative load per expert.

    Both equal 1 at a perfectly balanced batch, for every K. ``u_hat`` is detached because
    ``tokens_per_expert`` is a count, not a score, and carries no gradient.
    """
    total = _as_float_tensor(total_num_tokens)
    prob_sum = prob_sum.float()

    p_e = prob_sum / total  # [E], mean gate mass
    u = num_experts * p_e
    u_hat = hard_relative_load(tokens_per_expert, total, topk, num_experts)
    return u, u_hat


def _assert_conserves_global_mass(u_glob: torch.Tensor, num_experts: int) -> None:
    """Cheap invariant check on an explicitly supplied ``global_prob_sum``.

    Per-token routing-score mass sums to 1 for every score function this loss supports, so
    ``sum_e u_glob`` must equal E however the reduce group was formed. A mismatch means
    ``global_prob_sum`` was reduced over the wrong group.
    """
    total = u_glob.sum()
    expected = torch.tensor(float(num_experts), dtype=total.dtype)
    assert torch.allclose(total, expected, rtol=1e-3, atol=1e-3), (
        f"sum_e u_glob = {total.item():.4f} != E = {num_experts} (conservation invariant); "
        "check global_prob_sum was reduced over the same group as the counts, on the same "
        "pre-drop token set"
    )


def rosenthal_loss(
    prob_sum: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: float | torch.Tensor,
    topk: int,
    num_experts: int,
    coeff: float,
    lam: float,
    variant: str,
    cost_family: str,
    *,
    global_prob_sum: torch.Tensor | None = None,
) -> torch.Tensor:
    """The trainable Rosenthal congestion loss, 0-d float32, differentiable through ``prob_sum``.

    ``global_prob_sum`` is the ``[E]`` detached, already-reduced sum of routing scores over the
    whole reduce group, in fp32. When ``None`` it defaults to ``prob_sum``.
    """
    check_variant(variant)
    total = _as_float_tensor(total_num_tokens)
    u, u_hat = relative_loads(prob_sum, tokens_per_expert, total, topk, num_experts)
    global_prob_sum_supplied = global_prob_sum is not None
    glob = prob_sum if global_prob_sum is None else global_prob_sum
    u_glob, _ = relative_loads(glob, tokens_per_expert, total, topk, num_experts)
    if global_prob_sum_supplied:
        _assert_conserves_global_mass(u_glob, num_experts)
    prefactor = coeff / num_experts

    if variant == "soft":
        # Straight-through: forward value is the potential C(u_glob) (what gets logged),
        # backward flows only through carrier, whose gradient is alpha*c(u_glob). Which is the
        # exact gradient of that potential, since u is linear in prob_sum. c(u_glob) is a
        # detached, globally-synced coefficient, so no gradient crosses a rank boundary.
        # Do not collapse this into one expression: the potential and its gradient are
        # different-degree polynomials in u_glob (p+1 vs p); only the split gives both.
        value = prefactor * cost_antiderivative(u_glob, cost_family, lam).sum()
        weight = cost(u_glob, cost_family, lam).detach()
        carrier = coeff * (weight * prob_sum.float() / total).sum()
        return (value.detach() + (carrier - carrier.detach())).float()

    # hard: linearize the marginal cost c(.) at the REALIZED (detached) load, then price the
    # router's own soft mass u against that fixed weight. This is exactly what makes the hard
    # variant linear in prob_sum. Each rank can contribute its own local prob_sum against
    # globally-reduced counts and the per-rank losses sum to the whole-batch loss, which is what
    # switch_load_balancing_loss_func already relies on and this variant is built to match at
    # lam=1, linear (see the module docstring's exact-Switch identity).
    weight = cost(u_hat, cost_family, lam).detach()
    return (prefactor * (weight * u).sum()).float()


def pressure(
    prob_sum: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: float | torch.Tensor,
    topk: int,
    num_experts: int,
    coeff: float,
    lam: float,
    variant: str,
    cost_family: str,
    *,
    global_prob_sum: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-expert congestion price ``alpha * c(load)``, ``[E]`` float32, always detached."""
    check_variant(variant)
    total = _as_float_tensor(total_num_tokens)
    u, u_hat = relative_loads(prob_sum, tokens_per_expert, total, topk, num_experts)
    global_prob_sum_supplied = global_prob_sum is not None
    glob = prob_sum if global_prob_sum is None else global_prob_sum
    u_glob, _ = relative_loads(glob, tokens_per_expert, total, topk, num_experts)
    if global_prob_sum_supplied:
        _assert_conserves_global_mass(u_glob, num_experts)
    load = u_glob if variant == "soft" else u_hat
    return (coeff * cost(load.detach(), cost_family, lam)).float()


def price_bias_step(
    tokens_per_expert: torch.Tensor, *, cost_family: str, lam: float
) -> torch.Tensor:
    """Rosenthal-price step for the ``rosenthal_price`` expert-bias update rule.

    ``clip(c(u/L) - mean(c(u/L)), -1, 1)`` along the last axis, ``L = mean(u)``. Runs under
    ``no_grad`` in float32 and always returns a detached tensor. The caller applies
    ``bias <- bias - rate * price_bias_step(...)``, which reduces to ALF-LB's own
    ``sign(mean(u) - u)`` rule elementwise-in-sign for the linear cost family, because a linear
    price is monotone in ``u`` and centering never changes that monotonicity.

    Accepts ``(experts,)`` or stacked ``(layers, experts)`` counts, matching what
    ``get_updated_expert_bias`` holds after its all-reduce: both the mean and the centering run
    along the last axis so layers never mix. A layer whose counts are all zero has an undefined
    ``u/L`` and steps by zero there, matching what ``sign(0 - 0)`` already gives on that input.
    """
    with torch.no_grad():
        u = tokens_per_expert.float()
        load = u.mean(dim=-1, keepdim=True)
        safe_load = torch.where(load == 0, torch.ones_like(load), load)
        price = cost(u / safe_load, cost_family, lam)
        centered = price - price.mean(dim=-1, keepdim=True)
        step = torch.clamp(centered, -1.0, 1.0)
        return torch.where(load == 0, torch.zeros_like(step), step).detach()


def _potential_closed_form_linear(
    n: torch.Tensor, balanced_load: torch.Tensor, lam: float
) -> torch.Tensor:
    return (lam * n * (n + 1) / (2 * balanced_load)).sum()


def _potential_closed_form_quadratic(
    n: torch.Tensor, balanced_load: torch.Tensor, lam: float
) -> torch.Tensor:
    return (lam * n * (n + 1) * (2 * n + 1) / (6 * balanced_load**2)).sum()


def _potential_closed_form_barrier(
    n: torch.Tensor, balanced_load: torch.Tensor, lam: float
) -> torch.Tensor:
    """``sum_{j=1..n_e} c(j/L)`` for the barrier, by prefix sum.

    This is the DISCRETE sum over token ranks, not the integral ``cost_antiderivative`` computes.
    No closed form is known for it, whereas the integral has one.

    One arange sized at the largest realized count answers every expert at once, because expert
    ``e``'s partial sum is entry ``n_e`` of it. A per-expert loop instead costs a device sync and
    ``E`` launches on a path ``phi_cong`` runs every step, every layer, for every arm.
    """
    counts = n.long()
    max_n = int(counts.max().item()) if counts.numel() else 0
    if max_n <= 0:
        return torch.zeros((), dtype=torch.float32, device=n.device)
    j = torch.arange(1, max_n + 1, dtype=torch.float32, device=n.device)
    prices = cost(j / balanced_load, "softplus_barrier", lam)
    # The leading zero is what makes an expert with no tokens cost nothing. Without it every
    # idle expert would be charged for one arc, quietly.
    prefix = torch.cat([torch.zeros(1, dtype=prices.dtype, device=prices.device), prices.cumsum(0)])
    return prefix[counts.clamp(min=0, max=max_n)].sum()


# A closed form for the discrete rank sum avoids the loop. Only the power families have one.
_POTENTIAL_CLOSED_FORMS: dict[str, Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = {
    "linear": _potential_closed_form_linear,
    "quadratic": _potential_closed_form_quadratic,
    "softplus_barrier": _potential_closed_form_barrier,
}

if set(_POTENTIAL_CLOSED_FORMS) != set(COST_FAMILIES):
    raise ValueError(
        "congestion_potential's closed forms disagree with COST_FAMILIES: "
        f"{sorted(_POTENTIAL_CLOSED_FORMS)} != {sorted(COST_FAMILIES)}"
    )


def congestion_potential(
    tokens_per_expert: torch.Tensor,
    total_num_tokens: float | torch.Tensor,
    topk: int,
    num_experts: int,
    lam: float = 1.0,
    cost_family: str = "linear",
) -> torch.Tensor:
    """Discrete Rosenthal congestion potential ``Phi_cong / (N*K)``, 0-d float32, always detached.

    The congestion part of the potential function (omitting affinities). Requires the realized
    assignments alone. Normalized per assignment (N*K terms in the sum, one per token-expert
    pairing), same denominator the loss prefactor uses, so the two are directly comparable.
    """
    # Checked here, not via cost_exponent, which rejects the barrier even though this can price it.
    if cost_family not in _POTENTIAL_CLOSED_FORMS:
        raise ValueError(f"unknown cost family {cost_family!r}, expected one of {COST_FAMILIES}")
    n = tokens_per_expert.float().detach()
    total = _as_float_tensor(total_num_tokens)
    load = balanced_load(total, topk, num_experts)  # L = N*K/E

    phi = _POTENTIAL_CLOSED_FORMS[cost_family](n, load, lam)

    return (phi / (total * topk)).float().detach()

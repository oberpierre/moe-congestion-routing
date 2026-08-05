"""Rosenthal congestion loss: cost families, loss variants, pressure, and the discrete potential.

Per MoE layer, over the token set fixed by the balancing type (micro-batch or global-batch,
changing what ``tokens_per_expert`` and ``total_num_tokens`` are reduced over, never the
math below):

    N          tokens the counts are reduced over (``total_num_tokens``)
    E, K       experts, top-k (``num_experts``, ``topk``)
    L = N*K/E  balanced load: the load each expert carries if assignment were perfectly uniform
    prob_sum   [E], the local rank's sum over tokens of the pre-top-k softmax scores; differentiable
    n          [E], hard per-expert counts; always treated as detached (a count carries no gradient
               regardless of what the caller passes)
    P = prob_sum / N        mean gate mass per expert
    u = E*P                 soft relative load: differentiable, tracks the router's own scores
    u_hat = E*n/(N*K)       hard relative load: detached, tracks the realized assignment

    ``u`` and ``u_hat`` both have the same mass (``sum_e u = sum_e u_hat = E``): softmax sums to 1
    per token, so ``sum_e prob_sum_e = N``, while top-k selection makes K assignments per token,
    so ``sum_e n_e = N*K``. Both equal 1 at a perfectly balanced batch, for every K.

``hard`` is linear in ``prob_sum``, so per-rank contributions sum to the whole batch, same as
Megatron's own decomposition. ``soft``'s potential is degree ``p+1`` and does not decompose,
so ``rosenthal_loss`` never differentiates it: it prices the local ``prob_sum`` against ``c(·)``
evaluated at the detached, already-reduced ``global_prob_sum`` (defaulting to ``prob_sum``,
exact at group size 1). Required because sum of C(·) != C(sum of ·) for every cost family.

Two cost families, `c(x)` the marginal cost of relative load `x` and `C(x) = integral_0^x c`:

    linear:     p=1  c(x) = lam*x     C(x) = lam*x**2/2  default lam = 1.0
    quadratic:  p=2  c(x) = lam*x**2  C(x) = lam*x**3/3  default lam = 0.5
                                       (slope-matched to linear at x=1: lam_p = lam_1/p)

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

from collections.abc import Callable

import torch

from moe_congestion_routing.losses.cost_families import (
    COST_EXPONENTS,
    COST_FAMILIES,
    DEFAULT_LAMBDA,
    VARIANTS,
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
    "relative_loads",
    "rosenthal_loss",
]


def _as_float_tensor(value: float | torch.Tensor) -> torch.Tensor:
    return value.float() if isinstance(value, torch.Tensor) else torch.tensor(float(value))


def cost(x: torch.Tensor, cost_family: str, lam: float = 1.0) -> torch.Tensor:
    """Marginal congestion cost ``c(x) = lam * x**p`` for the given cost family."""
    p = cost_exponent(cost_family)
    return lam * x.float() ** p


def cost_antiderivative(x: torch.Tensor, cost_family: str, lam: float = 1.0) -> torch.Tensor:
    """``C(x) = integral_0^x c``, i.e. ``lam * x**(p+1) / (p+1)``."""
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

    Per-token softmax mass sums to 1, so ``sum_e u_glob`` must equal E regardless of how the
    reduce group was formed. A mismatch means ``global_prob_sum`` was reduced over the wrong group.
    """
    total = u_glob.sum()
    expected = torch.tensor(float(num_experts), dtype=total.dtype)
    assert torch.allclose(total, expected, rtol=1e-3, atol=1e-3), (
        f"sum_e u_glob = {total.item():.4f} != E = {num_experts} (0015 conservation invariant); "
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


def _potential_closed_form_linear(
    n: torch.Tensor, balanced_load: torch.Tensor, lam: float
) -> torch.Tensor:
    return (lam * n * (n + 1) / (2 * balanced_load)).sum()


def _potential_closed_form_quadratic(
    n: torch.Tensor, balanced_load: torch.Tensor, lam: float
) -> torch.Tensor:
    return (lam * n * (n + 1) * (2 * n + 1) / (6 * balanced_load**2)).sum()


# congestion_potential cannot route through cost() since it evaluates sum_{j=1..n} c(j/L) in closed
# form precisely to avoid the loop over per-expert token rank, so each family's closed form is
# held explicitly here instead. Enforces closed forms are available for the cost family.
_POTENTIAL_CLOSED_FORMS: dict[str, Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = {
    "linear": _potential_closed_form_linear,
    "quadratic": _potential_closed_form_quadratic,
}

if not (set(_POTENTIAL_CLOSED_FORMS) == set(COST_EXPONENTS) == set(COST_FAMILIES)):
    raise ValueError(
        "congestion_potential's closed forms disagree with COST_EXPONENTS/COST_FAMILIES: "
        f"{sorted(_POTENTIAL_CLOSED_FORMS)} != {sorted(COST_EXPONENTS)} != {sorted(COST_FAMILIES)}"
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
    # Validates cost_family and, since _POTENTIAL_CLOSED_FORMS' keys are checked above to match
    # COST_EXPONENTS' keys exactly, validates membership in _POTENTIAL_CLOSED_FORMS too.
    cost_exponent(cost_family)
    n = tokens_per_expert.float().detach()
    total = _as_float_tensor(total_num_tokens)
    load = balanced_load(total, topk, num_experts)  # L = N*K/E

    phi = _POTENTIAL_CLOSED_FORMS[cost_family](n, load, lam)

    return (phi / (total * topk)).float().detach()

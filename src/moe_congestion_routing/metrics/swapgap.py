"""SwapGap -- an epsilon-equilibrium gap for congestion-priced MoE routing.

Total *profitable single-swap deviation mass* of a routing: for each token, how much it would gain
by moving from its worst currently-selected expert to its best unselected one, under a congestion
cost ``c_e`` on expert load. Zero = no token has a profitable single swap = the routing is
one-swap-stable (a PNE). It needs no equilibrium solver.

Per token ``i`` with selected set ``s_i`` and per-expert hard load ``n_e = |{j : e in s_j}|``:

    g_i = [ max_{e' notin s_i}( a_ie' - c_e'(n_e' + 1))     # best expert to switch TO (load +1)
          - min_{e    in  s_i}( a_ie  - c_e(n_e)      )]_+  # worst expert we currently sit ON

where ``a_ie`` is the token's affinity for expert ``e`` -- the raw (unbiased) router logit
``sg(z_ie)``. For bias arms pass the *pre-bias* logits as affinities and the *realized* (biased)
assignment as the routing map: SwapGap then measures residual profitable swaps given how the model
actually routed.

The congestion cost is on the **normalized load** ``n_e / L``, where ``L = T*k/E`` is the
balanced (uniform) load, so the cost is scale-free in batch size and expert count. A balanced
expert sits at ``n_e / L = 1``. A collapsed top-k expert reaches ``E/k``. ``lambda`` is thus a
dimensionless congestion-to-affinity price, and the cost ranges are bounded: linear in
``[0, lambda*E/k]``, quadratic in ``[0, lambda*(E/k)^2]``.
The **per-token mean** ``(1/T) sum_i g_i`` is reported (batch-size invariant on the output side).
Range is ``[0, inf)`` but is effectively bounded by the affinity spread plus ``lambda*(E/k)^2``,
0 is the equilibrium.

Raw hard counts were tried and rejected: at ``lambda = 1`` they put the metric in the millions.
"""

from collections.abc import Callable

import torch


def _linear(load: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * load


def _quadratic(load: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * load**2


def _zero(load: torch.Tensor, lam: float) -> torch.Tensor:
    # Sanity baseline: with no congestion cost, SwapGap of a top-k-on-affinity routing is 0 by
    # construction; for a bias arm it measures the assignment's distance from affinity-greedy.
    return torch.zeros_like(load)


SWAPGAP_COSTS: dict[str, Callable[[torch.Tensor, float], torch.Tensor]] = {
    "linear": _linear,
    "quadratic": _quadratic,
    "zero": _zero,
}


def swapgap(
    logits: torch.Tensor,
    routing_map: torch.Tensor,
    cost: str,
    *,
    lam: float = 1.0,
) -> torch.Tensor:
    """Per-token-mean SwapGap for one MoE layer under congestion cost ``cost`` on hard counts.

    Args:
        logits: ``[T, E]`` raw router logits = affinities ``a = sg(z)`` (unbiased). For bias arms,
            the pre-bias logits.
        routing_map: ``[T, E]`` bool, the realized top-k assignment (biased for bias arms).
        cost: a key in :data:`SWAPGAP_COSTS`.
        lam: congestion strength ``lambda`` (dimensionless price on the normalized load).

    Returns:
        Scalar tensor in ``[0, inf)``; 0 iff no token has a profitable single swap.
    """
    if cost not in SWAPGAP_COSTS:
        raise ValueError(f"unknown SwapGap cost {cost!r}; expected one of {sorted(SWAPGAP_COSTS)}")

    num_tokens = logits.shape[0]
    if num_tokens == 0:
        return logits.new_zeros(())

    a = logits.float()  # [T, E] affinities
    selected = routing_map.bool()  # [T, E]
    load = selected.sum(dim=0).float()  # [E] hard load n_e
    lbar = load.mean()  # balanced load L = T*k/E; normalize so the cost is scale-free

    cost_fn = SWAPGAP_COSTS[cost]
    cost_stay = cost_fn(load / lbar, lam)  # [E] c_e(n_e / L): sit on a selected expert
    cost_join = cost_fn((load + 1.0) / lbar, lam)  # [E] c_e((n_e + 1) / L): join a new expert

    net_stay = a - cost_stay  # [T, E] payoff of staying on e
    net_join = a - cost_join  # [T, E] payoff of switching to e'

    neg_inf = torch.finfo(a.dtype).min
    pos_inf = torch.finfo(a.dtype).max
    # Best expert to switch TO: max of net_join over UNSELECTED experts (mask selected out).
    best_alt = net_join.masked_fill(selected, neg_inf).max(dim=1).values  # [T]
    # Worst expert we currently sit ON: min of net_stay over SELECTED experts (mask unselected out).
    worst_current = net_stay.masked_fill(~selected, pos_inf).min(dim=1).values  # [T]

    gap = (best_alt - worst_current).clamp_min(0.0)  # [T]
    return gap.mean()

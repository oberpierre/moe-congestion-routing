import pytest
import torch

from moe_congestion_routing.metrics.swapgap import SWAPGAP_COSTS, _is_admissible_price, swapgap


def _map(rows: list[list[int]]) -> torch.Tensor:
    """Build a [T, E] bool routing map from 0/1 rows."""
    return torch.tensor(rows, dtype=torch.bool)


def test_topk_on_affinity_with_zero_cost_is_equilibrium():
    # Assignment = the actual top-1 by logit with no congestion cost no token can profitably swap.
    logits = torch.tensor([[1.0, 3.0, 2.0], [0.5, 0.1, 4.0]])
    routing_map = _map([[0, 1, 0], [0, 0, 1]])  # each token on its highest-logit expert
    assert swapgap(logits, routing_map, "zero") == pytest.approx(0.0)


def test_assignment_off_the_affinity_greedy_choice():
    # Token prefers expert 1 (logit 3) but is assigned expert 0 (logit 1): under zero cost the gap
    # is the affinity difference max(3,2) - 1 = 2. This is the bias-arm shape (realized != greedy).
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    routing_map = _map([[1, 0, 0]])
    assert swapgap(logits, routing_map, "zero") == pytest.approx(2.0)


def test_lambda_is_not_multipliable():
    # Both tokens crowd expert 0 (load 2); expert 1 is empty. Affinity gap (2 vs 0) is what a swap
    # must overcome. At lambda=1 the congestion isn't enough -> gap 0; at lambda=3 it is -> gap 1.
    # 3 * 0 != 1: SwapGap is convex-piecewise-linear, NOT linear, in lambda (argmax/argmin/clamp).
    logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    routing_map = _map([[1, 0], [1, 0]])
    assert swapgap(logits, routing_map, "linear", lam=1.0) == pytest.approx(0.0)
    assert swapgap(logits, routing_map, "linear", lam=3.0) == pytest.approx(1.0)


def test_quadratic_penalizes_overload_more_than_linear():
    # 3 tokens crowd expert 0 (load 3), 1 on expert 1, all affinities equal -> only congestion
    # drives swaps. L = 4/2 = 2, so normalized loads are [1.5, 0.5]. Each crowded token
    # (linear): stay c(1.5)=1.5, join expert1 at c((1+1)/2)=1 -> gap 1 - 1.5 ... = 0.5. Quadratic:
    # stay c(1.5)=2.25, join c(1)=1 -> gap 1.25. The lone token on expert 1 has no profitable swap.
    logits = torch.zeros(4, 2)
    routing_map = _map([[1, 0], [1, 0], [1, 0], [0, 1]])
    sg_lin = swapgap(logits, routing_map, "linear")
    sg_quad = swapgap(logits, routing_map, "quadratic")
    assert sg_lin == pytest.approx(0.375)  # (0.5 + 0.5 + 0.5 + 0) / 4
    assert sg_quad == pytest.approx(0.9375)  # (1.25 + 1.25 + 1.25 + 0) / 4
    assert sg_quad > sg_lin


def test_join_cost_uses_load_plus_one():
    # Two tokens on expert 0 (load 2), expert 1 empty. Joining expert 1 must be priced at load
    # n+1 = 1 (cost 1), not at n = 0 (cost 0). With the +1 convention the gap is 0; pricing the
    # join at n would wrongly yield 1. Guards the own-load convention.
    logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    routing_map = _map([[1, 0], [1, 0]])
    assert swapgap(logits, routing_map, "linear", lam=1.0) == pytest.approx(0.0)


def test_quadratic_flags_high_affinity_collapse_linear_misses():
    # 32 tokens, E=8, top-1, affinity (15, 0, ...) -> collapse onto expert 0 (load 32). L = 32/8
    # = 4, so the collapsed normalized load is E/k = 8. Linear stay-cost is 8 < affinity 15, so the
    # linear cost cannot overcome the affinity advantage -> gap 0 (the known limitation of the
    # normalized form). Quadratic stay-cost is 8^2 = 64 >> 15, so it flags the collapse:
    # gap = c(8) - c(~0) - 15 = 64 - ~0 - 15 = ~48.94 per token.
    logits = torch.zeros(32, 8)
    logits[:, 0] = 15.0
    routing_map = _map([[1, 0, 0, 0, 0, 0, 0, 0]] * 32)
    assert swapgap(logits, routing_map, "linear", lam=1.0) == pytest.approx(0.0)
    assert swapgap(logits, routing_map, "quadratic", lam=1.0) == pytest.approx(48.9375)


def test_top2_routing():
    # k=2 of E=4: each token sits on two experts; the reduction handles multi-select selected sets.
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    routing_map = _map([[1, 1, 0, 0], [0, 0, 1, 1]])  # top-2 by logit each
    assert swapgap(logits, routing_map, "zero") == pytest.approx(0.0)  # greedy + no cost


def test_empty_batch_returns_zero():
    logits = torch.zeros(0, 8)
    routing_map = torch.zeros(0, 8, dtype=torch.bool)
    assert swapgap(logits, routing_map, "linear") == pytest.approx(0.0)


def test_unknown_cost_raises():
    with pytest.raises(ValueError, match="unknown SwapGap cost"):
        swapgap(torch.zeros(2, 4), _map([[1, 0, 0, 0], [1, 0, 0, 0]]), "foobar")


def test_cost_function_values():
    load = torch.tensor([0.0, 2.0, 4.0])
    assert torch.allclose(SWAPGAP_COSTS["linear"](load, 1.0), torch.tensor([0.0, 2.0, 4.0]))
    assert torch.allclose(SWAPGAP_COSTS["quadratic"](load, 1.0), torch.tensor([0.0, 4.0, 16.0]))
    assert torch.allclose(SWAPGAP_COSTS["zero"](load, 1.0), torch.zeros(3))


def test_admissibility_rejects_a_hard_barrier_price():
    # 0 below a capacity threshold, inf above it
    def hard_barrier(load: torch.Tensor, lam: float) -> torch.Tensor:
        del lam
        return torch.where(load <= 1.25, torch.zeros_like(load), torch.full_like(load, torch.inf))

    assert not _is_admissible_price(hard_barrier)


def test_admissibility_rejects_a_finite_non_monotone_price():
    # Finite everywhere but strictly decreasing over roughly half of [0, 8]
    def oscillating(load: torch.Tensor, lam: float) -> torch.Tensor:
        del lam
        return load + 2.0 * torch.sin(2.0 * torch.pi * load / 5.0)

    assert not _is_admissible_price(oscillating)

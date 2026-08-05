import pathlib

import pytest
import torch

from moe_congestion_routing.losses.cost_families import pressure_bound
from moe_congestion_routing.losses.rosenthal import (
    congestion_potential,
    cost_antiderivative,
    pressure,
    relative_loads,
    rosenthal_loss,
)


def _topk_counts(logits: torch.Tensor, topk: int) -> torch.Tensor:
    """Hard per-expert counts from ``logits``' own top-k routing, mirroring the router itself."""
    idx = logits.topk(topk, dim=-1).indices
    routing_map = torch.zeros_like(logits, dtype=torch.bool)
    routing_map.scatter_(1, idx, True)
    return routing_map.sum(dim=0).float()


def _conserving_fixture(
    num_experts: int, total_num_tokens: float, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A (prob_sum, tokens_per_expert) pair conserving total mass like a real router: prob_sum
    sums to N (per-token softmax mass) and tokens_per_expert sums to N*topk (K assignments per
    token).
    """
    raw = torch.rand(num_experts)
    prob_sum = raw / raw.sum() * total_num_tokens
    assignments = torch.randint(0, num_experts, (int(total_num_tokens) * topk,))
    tokens_per_expert = torch.bincount(assignments, minlength=num_experts).float()
    return prob_sum, tokens_per_expert


def _switch_loss(
    prob_sum: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: float,
    topk: int,
    coeff: float,
) -> torch.Tensor:
    """Megatron's switch_load_balancing_loss_func, written out locally: E * sum(f_i * P_i)."""
    num_experts = prob_sum.shape[0]
    scale = coeff * num_experts / (topk * total_num_tokens * total_num_tokens)
    return scale * (prob_sum * tokens_per_expert).sum()


def test_switch_equality_for_every_topk():
    # hard + linear + lam=1 must reproduce switch_load_balancing_loss_func's own expression
    # exactly, for every K. The topk factors cancel algebraically, hence holds for K>1.
    torch.manual_seed(0)
    num_experts = 6
    total_num_tokens = 40.0
    coeff = 0.5
    for topk in (1, 4, 8):
        prob_sum = torch.rand(num_experts) * total_num_tokens / num_experts
        tokens_per_expert = torch.randint(0, 20, (num_experts,)).float()
        got = rosenthal_loss(
            prob_sum,
            tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            coeff=coeff,
            lam=1.0,
            variant="hard",
            cost_family="linear",
        )
        want = _switch_loss(prob_sum, tokens_per_expert, total_num_tokens, topk, coeff)
        assert torch.allclose(got, want)


@pytest.mark.parametrize("variant", ["soft", "hard"])
@pytest.mark.parametrize("cost_family", ["linear", "quadratic"])
def test_gradient_identity(variant, cost_family):
    # d(loss)/d(prob_sum) = coeff*lam*u**p / N (soft) or coeff*lam*u_hat**p / N (hard); the /N is
    # the chain rule through P = prob_sum / N.
    torch.manual_seed(1)
    num_experts = 5
    topk = 2
    total_num_tokens = 30.0
    coeff = 0.7
    lam = 1.3
    prob_sum = (torch.rand(num_experts) * 3 + 0.5).requires_grad_(True)
    tokens_per_expert = torch.randint(1, 10, (num_experts,)).float()

    loss = rosenthal_loss(
        prob_sum,
        tokens_per_expert,
        total_num_tokens,
        topk,
        num_experts,
        coeff=coeff,
        lam=lam,
        variant=variant,
        cost_family=cost_family,
    )
    (grad,) = torch.autograd.grad(loss, prob_sum)

    u, u_hat = relative_loads(
        prob_sum.detach(), tokens_per_expert, total_num_tokens, topk, num_experts
    )
    p = {"linear": 1, "quadratic": 2}[cost_family]
    load = u if variant == "soft" else u_hat
    expected = coeff * lam * load**p / total_num_tokens
    assert torch.allclose(grad, expected, atol=1e-6)


@pytest.mark.parametrize("variant", ["soft", "hard"])
@pytest.mark.parametrize("cost_family", ["linear", "quadratic"])
def test_stable_at_balance(variant, cost_family):
    # The gradient should be zero at a perfectly balanced batch.
    num_experts = 4
    topk = 2
    num_tokens = 12  # multiple of num_experts, so a whole number of cycles balances exactly
    base = torch.tensor([2.0, 1.0, 0.5, 0.0])
    rows = torch.stack([torch.roll(base, shifts=i % num_experts) for i in range(num_tokens)])
    logits = rows.clone().requires_grad_(True)

    tokens_per_expert = _topk_counts(logits.detach(), topk)
    scores = torch.softmax(logits, dim=-1)
    prob_sum = scores.sum(dim=0)

    loss = rosenthal_loss(
        prob_sum,
        tokens_per_expert,
        float(num_tokens),
        topk,
        num_experts,
        coeff=1.0,
        lam=1.0,
        variant=variant,
        cost_family=cost_family,
    )
    (grad,) = torch.autograd.grad(loss, logits)
    assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-6)


@pytest.mark.parametrize("variant", ["soft", "hard"])
@pytest.mark.parametrize("cost_family", ["linear", "quadratic"])
def test_pressure_scale_bound(variant, cost_family):
    # u_hat is bounded above by E/K (all N tokens collapsed onto one expert, the most any single
    # expert's relative load can reach under top-k selection), so hard pressure inherits the bound
    # c(E/K), which is exactly what pressure_bound(..., variant="hard") expresses. u has no such
    # selection cap, it is softmax MASS, not a count, and can concentrate on one expert up to
    # u = E (bounded by the simplex total, not by K) making the soft bound c(E), which is what
    # pressure_bound(..., variant="soft") expresses.
    torch.manual_seed(3)
    num_experts = 8
    topk = 3
    total_num_tokens = 50.0
    coeff = 0.9
    lam = 1.1

    prob_sum, tokens_per_expert = _conserving_fixture(num_experts, total_num_tokens, topk)

    pr = pressure(
        prob_sum,
        tokens_per_expert,
        total_num_tokens,
        topk,
        num_experts,
        coeff=coeff,
        lam=lam,
        variant=variant,
        cost_family=cost_family,
    )
    bound = pressure_bound(coeff, lam, num_experts, topk, cost_family, variant=variant)
    assert pr.max() <= bound.value + 1e-4


@pytest.mark.parametrize("cost_family", ["linear", "quadratic"])
@pytest.mark.parametrize("topk", [1, 2, 8])
def test_soft_hard_gradient_parity_at_balance(topk, cost_family):
    # soft and hard must exert IDENTICAL pressure at a perfectly balanced batch, for every K.
    num_experts, total_num_tokens = 64, 512.0
    tokens_per_expert = torch.full((num_experts,), total_num_tokens * topk / num_experts)

    grad = {}
    for variant in ("hard", "soft"):
        prob_sum = torch.full((num_experts,), total_num_tokens / num_experts).requires_grad_(True)
        loss = rosenthal_loss(
            prob_sum,
            tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            coeff=1.0,
            lam=1.0,
            variant=variant,
            cost_family=cost_family,
        )
        (g,) = torch.autograd.grad(loss, prob_sum)
        grad[variant] = g * total_num_tokens  # dL/dP = dL/d(prob_sum) * N

    assert torch.allclose(grad["hard"], grad["soft"], atol=1e-5)
    assert torch.allclose(grad["hard"], torch.ones(num_experts), atol=1e-5)


def test_relative_loads_conserve_total_mass():
    # Both u and u_hat sum to E over experts, for ANY prob_sum/tokens_per_expert that came from a
    # real router, not just at balance.
    torch.manual_seed(8)
    num_experts = 6
    topk = 2
    num_tokens = 11
    logits = torch.randn(num_tokens, num_experts)
    scores = torch.softmax(logits, dim=-1)
    prob_sum = scores.sum(dim=0)
    tokens_per_expert = _topk_counts(logits, topk)

    u, u_hat = relative_loads(prob_sum, tokens_per_expert, float(num_tokens), topk, num_experts)
    assert torch.allclose(u.sum(), torch.tensor(float(num_experts)), atol=1e-5)
    assert torch.allclose(u_hat.sum(), torch.tensor(float(num_experts)), atol=1e-5)


@pytest.mark.parametrize("topk", [1, 2, 8])
def test_congestion_potential_exact_value_at_balance(topk):
    # Exact finite-L closed forms at perfect balance, not the C(1) limit this converges to as
    # L grows, so this pins the (N*K) denominator precisely, without a tolerance loose enough to
    # hide a stray factor of K.
    num_experts = 5
    balanced_load = 20.0  # L
    total_num_tokens = balanced_load * num_experts / topk  # N such that L = N*topk/E
    tokens_per_expert = torch.full((num_experts,), balanced_load)
    lam = 1.3

    expected = {
        "linear": lam * (balanced_load + 1) / (2 * balanced_load),
        "quadratic": lam * (balanced_load + 1) * (2 * balanced_load + 1) / (6 * balanced_load**2),
    }
    for cost_family, want in expected.items():
        phi = congestion_potential(
            tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            lam=lam,
            cost_family=cost_family,
        )
        assert torch.allclose(phi, torch.tensor(want), atol=1e-6)


def test_hard_variant_is_rank_additive():
    # Splitting prob_sum into two parts (same counts, same N) and summing the two hard losses must
    # reproduce the whole-batch hard loss. Verifies the distributed path which relies on
    # combining per-rank prob_sum against globally-reduced counts.
    torch.manual_seed(4)
    num_experts = 5
    topk = 2
    total_num_tokens = 24.0
    coeff = 1.0
    lam = 1.0
    tokens_per_expert = torch.randint(1, 10, (num_experts,)).float()
    prob_sum_a = torch.rand(num_experts) * 3
    prob_sum_b = torch.rand(num_experts) * 3
    prob_sum_whole = prob_sum_a + prob_sum_b

    for cost_family in ("linear", "quadratic"):
        loss_a = rosenthal_loss(
            prob_sum_a,
            tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            coeff=coeff,
            lam=lam,
            variant="hard",
            cost_family=cost_family,
        )
        loss_b = rosenthal_loss(
            prob_sum_b,
            tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            coeff=coeff,
            lam=lam,
            variant="hard",
            cost_family=cost_family,
        )
        loss_whole = rosenthal_loss(
            prob_sum_whole,
            tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            coeff=coeff,
            lam=lam,
            variant="hard",
            cost_family=cost_family,
        )
        assert torch.allclose(loss_a + loss_b, loss_whole)


def test_soft_variant_is_not_rank_additive():
    # The soft variant's potential form C(u) is degree p+1 in u, hence degree p+1 in prob_sum
    # (u is linear in prob_sum), so the same split-and-sum trick must NOT reproduce the
    # whole-batch loss: C(a) + C(b) != C(a + b) for a nonlinear C.
    torch.manual_seed(5)
    num_experts = 5
    topk = 2
    total_num_tokens = 24.0
    tokens_per_expert = torch.randint(1, 10, (num_experts,)).float()
    prob_sum_a = torch.rand(num_experts) * 3 + 0.1
    prob_sum_b = torch.rand(num_experts) * 3 + 0.1
    prob_sum_whole = prob_sum_a + prob_sum_b

    kwargs = {
        "tokens_per_expert": tokens_per_expert,
        "total_num_tokens": total_num_tokens,
        "topk": topk,
        "num_experts": num_experts,
        "coeff": 1.0,
        "lam": 1.0,
        "variant": "soft",
        "cost_family": "quadratic",
    }
    loss_a = rosenthal_loss(prob_sum_a, **kwargs)
    loss_b = rosenthal_loss(prob_sum_b, **kwargs)
    loss_whole = rosenthal_loss(prob_sum_whole, **kwargs)
    assert not torch.allclose(loss_a + loss_b, loss_whole)


@pytest.mark.parametrize("variant", ["soft", "hard"])
@pytest.mark.parametrize("cost_family", ["linear", "quadratic"])
def test_gradient_synced_coefficient_reproduces_reference(variant, cost_family):
    torch.manual_seed(12)
    num_experts = 6
    topk = 2
    total_num_tokens = 50.0
    coeff = 0.6
    lam = 1.0
    tokens_per_expert = torch.randint(1, 10, (num_experts,)).float()

    raw = torch.rand(num_experts)
    prob_sum_whole = raw / raw.sum() * total_num_tokens  # conserves N, like a real router
    split = torch.rand(num_experts)
    prob_sum_a = (prob_sum_whole * split).detach().requires_grad_(True)
    prob_sum_b = (prob_sum_whole * (1 - split)).detach().requires_grad_(True)

    kwargs = {
        "tokens_per_expert": tokens_per_expert,
        "total_num_tokens": total_num_tokens,
        "topk": topk,
        "num_experts": num_experts,
        "coeff": coeff,
        "lam": lam,
        "variant": variant,
        "cost_family": cost_family,
    }

    prob_sum_whole_ref = prob_sum_whole.clone().requires_grad_(True)
    loss_whole = rosenthal_loss(prob_sum_whole_ref, **kwargs)
    (grad_whole,) = torch.autograd.grad(loss_whole, prob_sum_whole_ref)

    global_prob_sum = prob_sum_a.detach() + prob_sum_b.detach()
    loss_a = rosenthal_loss(prob_sum_a, global_prob_sum=global_prob_sum, **kwargs)
    loss_b = rosenthal_loss(prob_sum_b, global_prob_sum=global_prob_sum, **kwargs)
    (grad_a,) = torch.autograd.grad(loss_a, prob_sum_a)
    (grad_b,) = torch.autograd.grad(loss_b, prob_sum_b)

    assert torch.allclose(grad_a, grad_whole, atol=1e-6)
    assert torch.allclose(grad_b, grad_whole, atol=1e-6)
    if variant == "soft":
        # The straight-through VALUE is the same whole-batch potential on every rank that shares
        # the same global_prob_sum, so summing it double-counts rather than reconstructing the
        # whole (test_soft_variant_is_not_rank_additive pins this same nonlinearity directly).
        assert not torch.allclose(loss_a + loss_b, loss_whole)
    else:
        # hard IS rank-additive (test_hard_variant_is_rank_additive pins this directly): its
        # value is linear in prob_sum, so summing the two shards' losses reconstructs the whole.
        assert torch.allclose(loss_a + loss_b, loss_whole)


@pytest.mark.parametrize("cost_family", ["linear", "quadratic"])
def test_soft_value_matches_potential_of_global_not_local_prob_sum(cost_family):
    torch.manual_seed(15)
    num_experts = 5
    topk = 2
    total_num_tokens = 24.0
    coeff = 0.6
    lam = 1.0
    tokens_per_expert = torch.randint(1, 10, (num_experts,)).float()
    prob_sum_local = torch.rand(num_experts) * 2 + 0.1
    global_prob_sum, _ = _conserving_fixture(num_experts, total_num_tokens, topk)
    assert not torch.allclose(prob_sum_local, global_prob_sum)  # else the mutation goes unnoticed

    got = rosenthal_loss(
        prob_sum_local,
        tokens_per_expert,
        total_num_tokens,
        topk,
        num_experts,
        coeff=coeff,
        lam=lam,
        variant="soft",
        cost_family=cost_family,
        global_prob_sum=global_prob_sum,
    )

    u_glob, _ = relative_loads(
        global_prob_sum, tokens_per_expert, total_num_tokens, topk, num_experts
    )
    expected = (coeff / num_experts) * cost_antiderivative(u_glob, cost_family, lam).sum()
    assert torch.allclose(got, expected)

    u_local, _ = relative_loads(
        prob_sum_local, tokens_per_expert, total_num_tokens, topk, num_experts
    )
    wrong = (coeff / num_experts) * cost_antiderivative(u_local, cost_family, lam).sum()
    assert not torch.allclose(got, wrong)


@pytest.mark.parametrize("cost_family", ["linear", "quadratic"])
def test_soft_loss_value_is_potential(cost_family):
    # Regression guard for the straight-through construction: the carrier, evaluated as a plain
    # expression coeff*sum(c(u_glob)*prob_sum/total), equals (p+1) times the continuized potential
    torch.manual_seed(13)
    num_experts = 5
    topk = 2
    total_num_tokens = 30.0
    coeff = 0.5
    lam = 1.0
    tokens_per_expert = torch.randint(1, 10, (num_experts,)).float()
    prob_sum = torch.rand(num_experts) * 3 + 0.1

    u, _ = relative_loads(prob_sum, tokens_per_expert, total_num_tokens, topk, num_experts)
    potential = (coeff / num_experts) * cost_antiderivative(u, cost_family, lam).sum()

    got = rosenthal_loss(
        prob_sum,
        tokens_per_expert,
        total_num_tokens,
        topk,
        num_experts,
        coeff=coeff,
        lam=lam,
        variant="soft",
        cost_family=cost_family,
    )
    p = {"linear": 1, "quadratic": 2}[cost_family]
    assert torch.allclose(got, potential)
    assert not torch.allclose(got, potential * (p + 1))


def test_global_prob_sum_conservation_assert_fires_on_mismatch():
    num_experts = 4
    topk = 2
    total_num_tokens = 20.0
    tokens_per_expert = torch.full((num_experts,), 10.0)
    prob_sum = torch.full((num_experts,), total_num_tokens / num_experts)  # conserves N

    with pytest.raises(AssertionError, match="conservation invariant"):
        rosenthal_loss(
            prob_sum,
            tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            coeff=1.0,
            lam=1.0,
            variant="soft",
            cost_family="linear",
            global_prob_sum=prob_sum * 5,  # does not conserve sum_e u_glob = E
        )


def test_pressure_global_prob_sum_matches_loss_gradient_coefficient():
    torch.manual_seed(14)
    num_experts = 5
    topk = 2
    total_num_tokens = 24.0
    coeff = 0.8
    lam = 1.0
    tokens_per_expert = torch.randint(1, 10, (num_experts,)).float()
    prob_sum_local = torch.rand(num_experts) * 2 + 0.1
    global_prob_sum, _ = _conserving_fixture(num_experts, total_num_tokens, topk)

    pr = pressure(
        prob_sum_local,
        tokens_per_expert,
        total_num_tokens,
        topk,
        num_experts,
        coeff=coeff,
        lam=lam,
        variant="soft",
        cost_family="linear",
        global_prob_sum=global_prob_sum,
    )
    u_glob, _ = relative_loads(
        global_prob_sum, tokens_per_expert, total_num_tokens, topk, num_experts
    )
    expected = coeff * lam * u_glob
    assert torch.allclose(pr, expected)
    u_local, _ = relative_loads(
        prob_sum_local, tokens_per_expert, total_num_tokens, topk, num_experts
    )
    assert not torch.allclose(pr, coeff * lam * u_local)


@pytest.mark.parametrize("topk", [1, 2, 8])
def test_discretization_gap_at_u_equals_u_hat(topk):
    # discretization gap Lemma: at u = u_hat, congestion_potential - loss = lam*E/(2*N*topk) =
    # lam/(2*L) exactly
    torch.manual_seed(6)
    num_experts = 7
    lam = 1.0

    tokens_per_expert = torch.randint(0, 15, (num_experts,)).float()
    total_num_tokens = tokens_per_expert.sum().item() / topk  # sum_e n_e = N*topk
    prob_sum = tokens_per_expert / topk  # makes u == u_hat for every topk, conserving sum == N

    loss = rosenthal_loss(
        prob_sum,
        tokens_per_expert,
        total_num_tokens,
        topk,
        num_experts,
        coeff=1.0,
        lam=lam,
        variant="soft",
        cost_family="linear",
    )
    phi = congestion_potential(
        tokens_per_expert, total_num_tokens, topk, num_experts, lam=lam, cost_family="linear"
    )
    expected_gap = lam * num_experts / (2 * total_num_tokens * topk)
    assert torch.allclose(phi - loss, torch.tensor(expected_gap))


def test_unknown_variant_raises_with_offending_value():
    prob_sum = torch.ones(3)
    tokens_per_expert = torch.ones(3)
    with pytest.raises(ValueError, match="bogus"):
        rosenthal_loss(
            prob_sum,
            tokens_per_expert,
            10.0,
            2,
            3,
            coeff=1.0,
            lam=1.0,
            variant="bogus",
            cost_family="linear",
        )


def test_unknown_cost_family_raises_with_offending_value():
    prob_sum = torch.ones(3)
    tokens_per_expert = torch.ones(3)
    with pytest.raises(ValueError, match="bogus"):
        rosenthal_loss(
            prob_sum,
            tokens_per_expert,
            10.0,
            2,
            3,
            coeff=1.0,
            lam=1.0,
            variant="hard",
            cost_family="bogus",
        )
    with pytest.raises(ValueError, match="bogus"):
        congestion_potential(tokens_per_expert, 10.0, 2, 3, cost_family="bogus")


def test_potential_closed_forms_key_mismatch_raises_at_import():
    # Same shape as cost_families.py's own guard: _POTENTIAL_CLOSED_FORMS must be checked against
    # COST_FAMILIES as well as COST_EXPONENTS, since COST_FAMILIES is the literal every consumer
    # actually validates membership against. Dropping a family's closed form here, with
    # COST_EXPONENTS/COST_FAMILIES left untouched, must fail at import rather than falling through
    # to whichever branch happened to be checked last in congestion_potential.
    import moe_congestion_routing.losses.rosenthal as rosenthal_module

    source = pathlib.Path(rosenthal_module.__file__).read_text()
    mutated = source.replace(
        "_POTENTIAL_CLOSED_FORMS: dict[str, Callable[[torch.Tensor, torch.Tensor, float], "
        'torch.Tensor]] = {\n    "linear": _potential_closed_form_linear,\n    '
        '"quadratic": _potential_closed_form_quadratic,\n}',
        "_POTENTIAL_CLOSED_FORMS: dict[str, Callable[[torch.Tensor, torch.Tensor, float], "
        'torch.Tensor]] = {\n    "linear": _potential_closed_form_linear,\n}',
    )
    assert mutated != source  # guard against the replacement silently matching nothing
    with pytest.raises(ValueError, match="disagree"):
        exec(compile(mutated, "<mutated rosenthal>", "exec"), {"__name__": "mutated"})

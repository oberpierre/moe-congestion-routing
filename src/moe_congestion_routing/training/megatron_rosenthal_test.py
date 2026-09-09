"""Tests for patch `0003` (Rosenthal router load-balancing type) that need no GPU.

Everything here calls loss functions and ``TransformerConfig`` directly with plain tensors, and
never constructs a ``TopKRouter`` or ``MoELayer`` through ``__init__``. ``Router.gating()`` moves
the gate weights to ``torch.cuda.current_device()`` on first forward, and ``TopKRouter.__init__``
allocates CUDA buffers whenever expert bias or a global-batch balancing type is active, so even
construction needs a GPU for those configs. ``is_aux_loss_enabled`` and ``get_aux_loss_coeff`` are
reached through ``TopKRouter.__new__(TopKRouter)`` with hand-set attributes, which skips
``__init__`` and therefore CUDA while still calling the real patched method bodies.

Skips cleanly, at module level, on a machine with no ``triton`` (macOS) or where ``Megatron-LM``
is not vendored and patched (``git submodule update --init`` and ``./scripts/apply-patches.sh``).
"""

import warnings
from types import SimpleNamespace

import pytest
import torch

from moe_congestion_routing.losses.rosenthal import rosenthal_loss
from moe_congestion_routing.training.megatron_path import MegatronLMNotVendoredError, ensure_on_path

# Module-level skip rather than per-test, because every test below needs megatron.core. The names
# below are assigned from importorskip's return value rather than imported, since a plain import
# statement after this point would trip ruff's E402.
pytest.importorskip("triton", reason="megatron.core requires triton, unavailable on macOS")
try:
    ensure_on_path()
except MegatronLMNotVendoredError as e:
    pytest.skip(str(e), allow_module_level=True)

TransformerConfig = pytest.importorskip(
    "megatron.core.transformer.transformer_config"
).TransformerConfig
_moe_utils = pytest.importorskip("megatron.core.transformer.moe.moe_utils")
_router_module = pytest.importorskip("megatron.core.transformer.moe.router")
switch_load_balancing_loss_func = _moe_utils.switch_load_balancing_loss_func
TopKRouter = _router_module.TopKRouter


def _base_kwargs(**overrides) -> dict:
    """A minimal MoE TransformerConfig, rosenthal-selected, that constructs without a GPU."""
    kwargs = {
        "num_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_moe_experts": 8,
        "moe_router_topk": 2,
        "moe_router_load_balancing_type": "rosenthal",
        "moe_aux_loss_coeff": 0.01,
    }
    kwargs.update(overrides)
    return kwargs


def _quiet_transformer_config(**overrides):
    """Construct a TransformerConfig, suppressing the unrelated moe_ffn_hidden_size / cuda_graph
    UserWarnings TransformerConfig.__post_init__ always emits for this minimal config, so a test
    asserting on OUR warning (or on none at all) is not confused by them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TransformerConfig(**_base_kwargs(**overrides))


# ---------------------------------------------------------------------------------------------
# rosenthal_loss(hard, linear, lambda=1) == switch_load_balancing_loss_func, value and grad
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("topk", [1, 8])
def test_rosenthal_hard_linear_equals_switch_value(topk):
    torch.manual_seed(0)
    num_tokens, num_experts, coeff = 37, 8, 0.02
    scores = torch.rand(num_tokens, num_experts, requires_grad=True)
    counts = torch.randint(0, 10, (num_experts,)).float()

    switch = switch_load_balancing_loss_func(
        probs=scores,
        tokens_per_expert=counts,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        moe_aux_loss_coeff=coeff,
        fused=False,
    )
    rosenthal = rosenthal_loss(
        prob_sum=scores.sum(dim=0),
        tokens_per_expert=counts,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        coeff=coeff,
        lam=1.0,
        variant="hard",
        cost_family="linear",
    )
    assert torch.allclose(switch, rosenthal, atol=1e-5)


@pytest.mark.parametrize("topk", [1, 8])
def test_rosenthal_hard_linear_equals_switch_grad_wrt_scores(topk):
    torch.manual_seed(0)
    num_tokens, num_experts, coeff = 37, 8, 0.02
    counts = torch.randint(0, 10, (num_experts,)).float()

    scores_switch = torch.rand(num_tokens, num_experts, requires_grad=True)
    switch = switch_load_balancing_loss_func(
        probs=scores_switch,
        tokens_per_expert=counts,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        moe_aux_loss_coeff=coeff,
        fused=False,
    )
    (switch_grad,) = torch.autograd.grad(switch, scores_switch)

    scores_rosenthal = scores_switch.detach().clone().requires_grad_(True)
    rosenthal = rosenthal_loss(
        prob_sum=scores_rosenthal.sum(dim=0),
        tokens_per_expert=counts,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        coeff=coeff,
        lam=1.0,
        variant="hard",
        cost_family="linear",
    )
    (rosenthal_grad,) = torch.autograd.grad(rosenthal, scores_rosenthal)

    assert torch.allclose(switch_grad, rosenthal_grad, atol=1e-5)


# ---------------------------------------------------------------------------------------------
# The equivalence above uses an arbitrary probability tensor, so it holds regardless of which
# score function produced it. Patch 0008 lets the aux-loss carrier differ from selection, and the
# risk it introduces is plumbing, not math: every aux-loss consumer (switch, rosenthal) reads the
# same `scores_for_aux_loss` tensor from one `compute_routing_scores_for_aux_loss` call, so a
# change that routed the carrier override to one loss and not the other would desync them. These
# two tests run that real function under both score functions to pin against that.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("carrier", ["sigmoid", "softmax"])
def test_rosenthal_hard_linear_equals_switch_under_both_carriers(carrier):
    torch.manual_seed(0)
    num_tokens, num_experts, topk, coeff = 37, 8, 2, 0.02
    logits = torch.randn(num_tokens, num_experts)
    _, scores = _moe_utils.compute_routing_scores_for_aux_loss(logits, topk, carrier)
    counts = torch.randint(0, 10, (num_experts,)).float()

    switch = switch_load_balancing_loss_func(
        probs=scores,
        tokens_per_expert=counts,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        moe_aux_loss_coeff=coeff,
        fused=False,
    )
    rosenthal = rosenthal_loss(
        prob_sum=scores.sum(dim=0),
        tokens_per_expert=counts,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        coeff=coeff,
        lam=1.0,
        variant="hard",
        cost_family="linear",
    )
    assert torch.allclose(switch, rosenthal, atol=1e-5)


@pytest.mark.parametrize("carrier", ["sigmoid", "softmax"])
def test_rosenthal_hard_linear_equals_switch_grad_under_both_carriers(carrier):
    torch.manual_seed(0)
    num_tokens, num_experts, topk, coeff = 37, 8, 2, 0.02
    counts = torch.randint(0, 10, (num_experts,)).float()

    logits_switch = torch.randn(num_tokens, num_experts, requires_grad=True)
    _, scores_switch = _moe_utils.compute_routing_scores_for_aux_loss(logits_switch, topk, carrier)
    switch = switch_load_balancing_loss_func(
        probs=scores_switch,
        tokens_per_expert=counts,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        moe_aux_loss_coeff=coeff,
        fused=False,
    )
    (switch_grad,) = torch.autograd.grad(switch, logits_switch)

    logits_rosenthal = logits_switch.detach().clone().requires_grad_(True)
    _, scores_rosenthal = _moe_utils.compute_routing_scores_for_aux_loss(
        logits_rosenthal, topk, carrier
    )
    rosenthal = rosenthal_loss(
        prob_sum=scores_rosenthal.sum(dim=0),
        tokens_per_expert=counts,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        coeff=coeff,
        lam=1.0,
        variant="hard",
        cost_family="linear",
    )
    (rosenthal_grad,) = torch.autograd.grad(rosenthal, logits_rosenthal)

    assert torch.allclose(switch_grad, rosenthal_grad, atol=1e-5)


# ---------------------------------------------------------------------------------------------
# TransformerConfig validation rules
# ---------------------------------------------------------------------------------------------


def test_rule1_unknown_variant_raises():
    with pytest.raises(ValueError, match="moe_rosenthal_variant"):
        _quiet_transformer_config(moe_rosenthal_variant="bogus")


def test_rule2_unknown_cost_family_raises():
    with pytest.raises(ValueError, match="moe_rosenthal_cost"):
        _quiet_transformer_config(moe_rosenthal_cost="bogus")


def test_rule3_nonpositive_lambda_raises():
    with pytest.raises(ValueError, match="moe_rosenthal_lambda"):
        _quiet_transformer_config(moe_rosenthal_lambda=0.0)


def test_rule4_retired_sigmoid_score_function_no_longer_rejected():
    # Rule 4 is retired. compute_routing_scores_for_aux_loss already per-token normalizes sigmoid
    # scores the same way it normalizes softmax, so the loss's conservation invariant holds under
    # sigmoid too and a rosenthal type must construct without raising.
    _quiet_transformer_config(moe_router_score_function="sigmoid")


def test_global_rosenthal_soft_no_longer_rejected():
    _quiet_transformer_config(
        moe_router_load_balancing_type="global_rosenthal", moe_rosenthal_variant="soft"
    )


def test_softplus_barrier_soft_no_longer_rejected():
    # softplus_barrier's antiderivative is now an exact dilogarithm, so the patched
    # TransformerConfig must construct without raising at the soft variant, mirroring the
    # global_rosenthal retirement above.
    _quiet_transformer_config(moe_rosenthal_cost="softplus_barrier", moe_rosenthal_variant="soft")


def test_rosenthal_soft_no_longer_requires_tensor_model_parallel_size_one():
    _quiet_transformer_config(
        moe_rosenthal_variant="soft",
        tensor_model_parallel_size=2,
        num_attention_heads=8,  # must stay divisible by tensor_model_parallel_size
        # An unrelated Megatron constraint ("Bias in Moe is only supported when ETP==1"), reached
        # only because the retired rule 6 no longer rejects this config first. Not something this
        # patch enforces or tests.
        add_bias_linear=False,
    )


def test_rule7_log_grad_ratio_without_rosenthal_type_raises():
    with pytest.raises(ValueError, match="moe_rosenthal_log_grad_ratio"):
        _quiet_transformer_config(
            moe_router_load_balancing_type="aux_loss",
            moe_rosenthal_log_grad_ratio=True,
        )


def test_rule7_log_grad_ratio_with_rosenthal_type_does_not_raise():
    _quiet_transformer_config(moe_rosenthal_log_grad_ratio=True)


def test_rule7_log_grad_ratio_rejects_rosenthal_combined_with_another_type():
    # An `any(t in ROSENTHAL_TYPES ...)` check would accept this list, because it asks only that a
    # rosenthal type be included, not that the selection be one. That matters because patch 0004's
    # "cg" probe wraps the single `logits` tensor feeding compute_routing_scores_for_aux_loss, and
    # every active aux loss type reads that same tensor. With this list, rosenthal_grad_norm_cg
    # would record the combined gradient of seq_aux_loss and rosenthal rather than rosenthal's own.
    # It is reachable only through a direct Megatron launch, which is what this copy of the
    # validation exists for.
    with pytest.raises(ValueError, match="moe_rosenthal_log_grad_ratio"):
        _quiet_transformer_config(
            moe_router_load_balancing_type=["seq_aux_loss", "rosenthal"],
            moe_aux_loss_coeff=[0.01, 0.01],
            moe_rosenthal_log_grad_ratio=True,
        )


def test_rule8_pressure_above_sanity_bound_warns():
    # Deliberately not routed through _quiet_transformer_config, whose blanket
    # simplefilter("ignore") would swallow our own warning along with the unrelated ones.
    # pytest.warns needs only one matching warning, so the unrelated ones are harmless here.
    with pytest.warns(UserWarning, match="exceeds the sanity bound of 1 at full imbalance"):
        TransformerConfig(**_base_kwargs(moe_rosenthal_lambda=100.0))


def test_rule8_pressure_within_sanity_bound_does_not_warn_about_pressure():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _quiet_transformer_config(moe_rosenthal_lambda=1.0)
    assert not any("sanity bound" in str(w.message) for w in caught)


def test_rule8_soft_variant_warns_with_the_soft_bound_expression():
    # pretrain_config_test.py covers our own copy's soft branch, so this test covers the patched
    # Megatron file's. soft's bound is num_moe_experts**p, since softmax mass is uncapped by top-k
    # selection, not (num_moe_experts/moe_router_topk)**p as hard's is, and the message must say so.
    with pytest.warns(UserWarning, match=r"num_moe_experts\*\*"):
        TransformerConfig(**_base_kwargs(moe_rosenthal_variant="soft", moe_rosenthal_lambda=100.0))


# ---------------------------------------------------------------------------------------------
# moe_router_bias_update_rule validation. moe_router_load_balancing_type stays "none",
# this arm's own setting, so these checks are provably reached outside the rosenthal-loss block
# above, which _rosenthal_types_selected leaves empty for a "none" arm.
# ---------------------------------------------------------------------------------------------


def _price_rule_kwargs(**overrides) -> dict:
    kwargs = {
        "num_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_moe_experts": 8,
        "moe_router_topk": 2,
        "add_bias_linear": False,
        "moe_router_load_balancing_type": "none",
        "moe_router_enable_expert_bias": True,
        "moe_router_score_function": "sigmoid",
        "moe_router_bias_update_rule": "rosenthal_price",
    }
    kwargs.update(overrides)
    return kwargs


def _quiet_price_rule_config(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TransformerConfig(**_price_rule_kwargs(**overrides))


def test_bias_rule_unknown_rule_raises():
    with pytest.raises(ValueError, match="moe_router_bias_update_rule"):
        _quiet_price_rule_config(moe_router_bias_update_rule="bogus")


def test_bias_rule_rosenthal_price_without_expert_bias_raises():
    with pytest.raises(ValueError, match="moe_router_enable_expert_bias"):
        _quiet_price_rule_config(moe_router_enable_expert_bias=False)


def test_bias_rule_rosenthal_price_unknown_cost_family_raises():
    with pytest.raises(ValueError, match="moe_rosenthal_cost"):
        _quiet_price_rule_config(moe_rosenthal_cost="not_a_real_family")


def test_bias_rule_rosenthal_price_nonpositive_lambda_raises():
    with pytest.raises(ValueError, match="moe_rosenthal_lambda"):
        _quiet_price_rule_config(moe_rosenthal_lambda=-3.0)


def test_bias_rule_rosenthal_price_with_valid_settings_does_not_raise():
    _quiet_price_rule_config()


# ---------------------------------------------------------------------------------------------
# List-form moe_router_load_balancing_type. The field holds either a string or a list, since
# argparse's nargs='+' always produces a list and validate_args collapses only a single-element
# one. So a list combining rosenthal with another type, or selecting both rosenthal types at once,
# must still see every rule below.
# ---------------------------------------------------------------------------------------------


def test_rule4_retired_sigmoid_score_function_no_longer_rejected_when_combined_with_another_type():
    # Same retirement as the scalar case above, checked through the list-form path so a future
    # change to that path's own normalization is not missed just because it always tests scalar.
    _quiet_transformer_config(
        moe_router_load_balancing_type=["seq_aux_loss", "rosenthal"],
        # One coefficient per entry, as Megatron's own list-form validation requires.
        moe_aux_loss_coeff=[0.01, 0.01],
        moe_router_score_function="sigmoid",
    )


def test_rosenthal_and_global_rosenthal_together_raises_exclusivity_error():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _quiet_transformer_config(
            moe_router_load_balancing_type=["rosenthal", "global_rosenthal"],
            # One coefficient per entry, as Megatron's own list-form validation requires.
            moe_aux_loss_coeff=[0.01, 0.01],
        )


def test_scalar_rosenthal_with_softmax_is_accepted():
    _quiet_transformer_config(
        moe_router_load_balancing_type="rosenthal", moe_router_score_function="softmax"
    )


# ---------------------------------------------------------------------------------------------
# The string-matching guards Megatron's own dispatch (is_aux_loss_enabled aside) depends on.
# ---------------------------------------------------------------------------------------------


def test_aux_loss_is_not_a_substring_of_rosenthal():
    assert "aux_loss" not in "rosenthal"


def test_aux_loss_is_not_a_substring_of_global_rosenthal():
    assert "aux_loss" not in "global_rosenthal"


def test_global_aux_loss_is_not_a_substring_of_global_rosenthal():
    assert "global_aux_loss" not in "global_rosenthal"


def test_rosenthal_is_a_substring_of_global_rosenthal():
    assert "rosenthal" in "global_rosenthal"


# ---------------------------------------------------------------------------------------------
# is_aux_loss_enabled(). Without it the router never computes scores_for_aux_loss and the run
# trains with no balancing at all while the config says otherwise, with no error.
# ---------------------------------------------------------------------------------------------


def _bare_router(routing_type: str, aux_loss_coeff: float) -> "TopKRouter":
    """A TopKRouter carrying only the attributes is_aux_loss_enabled and get_aux_loss_coeff read.

    Built via __new__ so that __init__, and its CUDA buffer allocation, never runs.
    """
    router = TopKRouter.__new__(TopKRouter)
    router.routing_type = routing_type
    router.config = SimpleNamespace(moe_aux_loss_coeff=aux_loss_coeff)
    return router


def test_is_aux_loss_enabled_true_for_rosenthal():
    assert _bare_router("rosenthal", 0.01).is_aux_loss_enabled() is True


def test_is_aux_loss_enabled_true_for_global_rosenthal():
    assert _bare_router("global_rosenthal", 0.01).is_aux_loss_enabled() is True


def test_is_aux_loss_enabled_false_when_coeff_is_zero():
    assert _bare_router("rosenthal", 0.0).is_aux_loss_enabled() is False


def test_is_aux_loss_enabled_false_for_none_balancing():
    assert _bare_router("none", 0.0).is_aux_loss_enabled() is False


# ---------------------------------------------------------------------------------------------
# rosenthal_price sign convention, checked through the real get_updated_expert_bias.
# torch.distributed.all_reduce inside it is unconditional, so a single-rank no-op gloo group
# (the same pattern game/alflb_test.py uses) makes it callable on CPU.
# ---------------------------------------------------------------------------------------------


def test_price_rule_moves_bias_against_load_and_matches_alflb_at_collapse(tmp_path):
    started_here = not torch.distributed.is_initialized()
    if started_here:
        torch.distributed.init_process_group(
            backend="gloo", world_size=1, rank=0, init_method=f"file://{tmp_path / 'store'}"
        )
    try:
        rate = 1e-3
        counts = torch.tensor([1.0, 3.0, 5.0, 2.0, 4.0, 0.0, 6.0, 3.0])  # mean 3.0
        bias = torch.zeros_like(counts)
        updated = _moe_utils.get_updated_expert_bias(
            counts.clone(),
            bias,
            rate,
            tp_dp_cp_group=torch.distributed.group.WORLD,
            bias_update_rule="rosenthal_price",
            cost_family="linear",
            lam=1.0,
        )
        delta = updated - bias
        above_mean = counts > counts.mean()
        below_mean = counts < counts.mean()
        assert (delta[above_mean] < 0).all()
        assert (delta[below_mean] > 0).all()
        assert delta.abs().max().item() <= rate + 1e-6  # fp32 rounding at the clip boundary

        # Total collapse (one winner at 8x the mean, the rest dead) saturates the clip on both
        # sides, so the step must equal the rate exactly rather than merely stay bounded by it.
        collapse = torch.zeros(8)
        collapse[0] = 8.0
        collapse_bias = torch.zeros(8)
        collapse_updated = _moe_utils.get_updated_expert_bias(
            collapse.clone(),
            collapse_bias,
            rate,
            tp_dp_cp_group=torch.distributed.group.WORLD,
            bias_update_rule="rosenthal_price",
            cost_family="linear",
            lam=1.0,
        )
        collapse_delta = collapse_updated - collapse_bias
        assert collapse_delta[0].item() == pytest.approx(-rate)
        assert collapse_delta[1:].tolist() == pytest.approx([rate] * 7)
    finally:
        if started_here:
            torch.distributed.destroy_process_group()

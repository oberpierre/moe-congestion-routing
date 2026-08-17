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
# TransformerConfig validation rules 1-6 raise; rule 8 warns.
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


def test_rule4_sigmoid_score_function_raises():
    with pytest.raises(ValueError, match="moe_router_score_function"):
        _quiet_transformer_config(moe_router_score_function="sigmoid")


def test_global_rosenthal_soft_no_longer_rejected():
    # Rules 5 and 6 are retired. The synced-coefficient construction makes 'soft' correct at any
    # reduce-group size, so global_rosenthal with soft must construct without raising.
    _quiet_transformer_config(
        moe_router_load_balancing_type="global_rosenthal", moe_rosenthal_variant="soft"
    )


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
    with pytest.warns(UserWarning, match="congestion pressure at full imbalance"):
        TransformerConfig(**_base_kwargs(moe_rosenthal_lambda=100.0))


def test_rule8_pressure_within_sanity_bound_does_not_warn_about_pressure():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _quiet_transformer_config(moe_rosenthal_lambda=1.0)
    assert not any("congestion pressure" in str(w.message) for w in caught)


def test_rule8_soft_variant_warns_with_the_soft_bound_expression():
    # pretrain_config_test.py covers our own copy's soft branch, so this test covers the patched
    # Megatron file's. soft's bound is num_moe_experts**p, since softmax mass is uncapped by top-k
    # selection, not (num_moe_experts/moe_router_topk)**p as hard's is, and the message must say so.
    with pytest.warns(UserWarning, match=r"num_moe_experts\*\*"):
        TransformerConfig(**_base_kwargs(moe_rosenthal_variant="soft", moe_rosenthal_lambda=100.0))


# ---------------------------------------------------------------------------------------------
# List-form moe_router_load_balancing_type. The field holds either a string or a list, since
# argparse's nargs='+' always produces a list and validate_args collapses only a single-element
# one. So a list combining rosenthal with another type, or selecting both rosenthal types at once,
# must still see every rule below.
# ---------------------------------------------------------------------------------------------


def test_rule4_sigmoid_score_function_raises_for_rosenthal_combined_with_another_type():
    with pytest.raises(ValueError, match="moe_router_score_function"):
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

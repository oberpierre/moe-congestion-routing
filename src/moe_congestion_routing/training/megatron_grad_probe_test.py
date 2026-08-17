"""Tests for patch `0004` (logit-space gradient probes) that need no GPU.

``RouterGradProbe`` is a plain ``torch.autograd.Function``, identity in forward and a
norm-and-pass-through in backward, so it is exercised directly with CPU tensors rather than through
a constructed ``TopKRouter`` or ``MoELayer``. ``Router.gating()`` moves the gate weights to
``torch.cuda.current_device()`` on first forward, so a real router forward would need a GPU.

The property this file exists to prove is that two probes placed on two different consumers of the
same upstream tensor each record only their own branch's gradient, never a mixture of the two and
never the other branch's alone. ``test_two_probes_on_sibling_branches_record_only_their_own_branch_
gradient`` checks it by building a toy two-branch graph with two different known linear maps and
asserting each recorded norm matches its own branch's closed form. Were the probes wired to the
same branch, or were ``backward`` accumulating across calls, the two recorded values would collide
on one branch's number instead of each matching its own.

Skips cleanly, at module level, on a machine with no ``triton`` (macOS) or where ``Megatron-LM`` is
not vendored and patched (``git submodule update --init`` and ``./scripts/apply-patches.sh``).
"""

import pytest

from moe_congestion_routing.training.megatron_path import MegatronLMNotVendoredError, ensure_on_path

# Module-level skip rather than per-test, because every test below needs megatron.core. The names
# below are assigned from importorskip's return value rather than imported, since a plain import
# statement after this point would trip ruff's E402.
pytest.importorskip("triton", reason="megatron.core requires triton, unavailable on macOS")
try:
    ensure_on_path()
except MegatronLMNotVendoredError as e:
    pytest.skip(str(e), allow_module_level=True)

torch = pytest.importorskip("torch")
_moe_utils = pytest.importorskip("megatron.core.transformer.moe.moe_utils")
_moe_logging = pytest.importorskip("megatron.core.transformer.moe.moe_logging")
RouterGradProbe = _moe_utils.RouterGradProbe
MoEMetricsTracker = _moe_logging.MoEMetricsTracker
set_moe_metrics_tracker = _moe_logging.set_moe_metrics_tracker
destroy_moe_metrics_tracker = _moe_logging.destroy_moe_metrics_tracker


@pytest.fixture(autouse=True)
def _fresh_tracker():
    # The metrics tracker is a module-level global, which is megatron.core's own pattern rather
    # than ours. Each test therefore gets a fresh instance instead of accumulating into whatever a
    # previous test, or a previous module, left behind.
    set_moe_metrics_tracker(MoEMetricsTracker())
    yield
    destroy_moe_metrics_tracker()


def test_forward_is_exact_identity():
    x = torch.randn(5, 3)
    y = RouterGradProbe.apply(x, "probe_identity", 1, 1)
    assert torch.equal(y, x)


def test_backward_passes_gradient_through_unchanged():
    x = torch.randn(5, 3, requires_grad=True)
    y = RouterGradProbe.apply(x, "probe_passthrough", 1, 1)
    seed = torch.randn(5, 3)
    y.backward(seed)
    assert torch.equal(x.grad, seed)


def test_backward_records_the_gradient_norm_under_its_own_name():
    x = torch.randn(4, requires_grad=True)
    y = RouterGradProbe.apply(x, "probe_norm", layer_number=2, num_layers=3)
    seed = torch.tensor([3.0, 0.0, 0.0, 4.0])  # norm 5, deliberately not 1 so a stub value can't
    y.backward(seed)  # coincidentally pass.

    tracker = _moe_logging.get_moe_metrics_tracker()
    assert tracker.metrics["probe_norm"].values[1].item() == pytest.approx(5.0)  # layer 2 -> idx 1


def test_two_probes_on_sibling_branches_record_only_their_own_branch_gradient():
    # y = f(probe_a(x)) + g(probe_b(x)), where f and g are two different known linear maps on the
    # same upstream x. If both probes landed on one branch, or if backward mixed the two, the
    # recorded norms would collide on a single branch's value. With genuinely separate branches
    # each must equal its own closed-form gradient norm exactly.
    torch.manual_seed(0)
    x = torch.randn(4, requires_grad=True)
    w_f = torch.tensor([1.0, 2.0, 3.0, 4.0])  # ||w_f|| = sqrt(30)
    w_g = torch.tensor([5.0, 0.0, 0.0, 0.0])  # ||w_g|| = 5, deliberately different from ||w_f||

    probe_a = RouterGradProbe.apply(x, "branch_task", 1, 1)
    probe_b = RouterGradProbe.apply(x, "branch_cg", 1, 1)

    f = (w_f * probe_a).sum()  # df/d(probe_a) = w_f
    g = (w_g * probe_b).sum()  # dg/d(probe_b) = w_g
    y = f + g
    y.backward()

    tracker = _moe_logging.get_moe_metrics_tracker()
    task_norm = tracker.metrics["branch_task"].values[0].item()
    cg_norm = tracker.metrics["branch_cg"].values[0].item()

    assert task_norm == pytest.approx(w_f.norm().item())
    assert cg_norm == pytest.approx(w_g.norm().item())
    # And, since x's own gradient is the sum of both branches (autograd does that outside the
    # probe, not the probe's concern), x.grad is neither of the two recorded norms' vectors alone.
    assert torch.equal(x.grad, w_f + w_g)


def test_probe_on_a_branch_shared_by_two_losses_records_their_combined_norm():
    # Documents the quiet failure patch 0003's rule 7 exists to prevent. router.py's "cg" probe
    # wraps the single `logits` tensor feeding compute_routing_scores_for_aux_loss, and every
    # active aux loss type reads that same tensor, unlike the two sibling branches above which are
    # genuinely separate autograd nodes. If moe_rosenthal_log_grad_ratio were allowed alongside
    # another balancing type, this probe would record the sum of both losses' gradients under
    # "rosenthal_grad_norm_cg" rather than the congestion loss's own contribution, and it would do
    # so with no error, since forward and backward are exact identities that see only what reaches
    # them. This test therefore puts two losses on one probe on purpose.
    torch.manual_seed(1)
    x = torch.randn(4, requires_grad=True)
    w_rosenthal = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w_other_aux_loss = torch.tensor([0.0, 1.0, 0.0, 1.0])

    probe = RouterGradProbe.apply(x, "rosenthal_grad_norm_cg", 1, 1)
    rosenthal_loss = (w_rosenthal * probe).sum()
    other_aux_loss = (w_other_aux_loss * probe).sum()
    (rosenthal_loss + other_aux_loss).backward()

    tracker = _moe_logging.get_moe_metrics_tracker()
    recorded = tracker.metrics["rosenthal_grad_norm_cg"].values[0].item()

    combined_norm = (w_rosenthal + w_other_aux_loss).norm().item()
    assert recorded == pytest.approx(combined_norm)
    assert recorded != pytest.approx(w_rosenthal.norm().item())

"""Every committed ``configs/train/*.yaml`` emits only flags the pinned Megatron accepts.

This tests **our emitter**, not the configs. It asserts nothing about what any particular config
sets or inherits, only that ``build_megatron_args`` never hands Megatron's real
``megatron.training.arguments.parse_args`` a flag name, value or choice it rejects. A wrong flag
name otherwise builds a plausible command and fails only once the job reaches the front of the
cluster queue, which is the failure this test exists to move into ``uv run pytest``.

Requires ``./scripts/apply-patches.sh`` to have been run. The rosenthal balancing types and the
``--moe-rosenthal-*`` flags exist in Megatron's argparse only once patch ``0003`` is applied, so
without the patches this fails on every rosenthal config. That is a missing prerequisite rather
than a broken config or a bug in this test.

Skips cleanly (not silently, not as a no-op) when ``megatron.core`` is unavailable: on macOS,
because ``megatron.core`` imports ``triton`` several packages deep and there is no macOS wheel;
and wherever the ``Megatron-LM`` submodule hasn't been checked out at all.
"""

import contextlib
import io
import sys
from pathlib import Path

import pytest

from moe_congestion_routing.training.megatron_path import MegatronLMNotVendoredError, ensure_on_path
from moe_congestion_routing.training.pretrain_config import MoEPretrainConfig, build_megatron_args

try:
    ensure_on_path()
except MegatronLMNotVendoredError as e:
    pytest.skip(str(e), allow_module_level=True)

# Importing megatron.core pulls in torch.distributed, pipeline_parallel and paged_stash, which
# reaches triton, and triton has no macOS wheel. importorskip turns that ImportError into a skip.
pytest.importorskip("megatron.core")
parse_args = pytest.importorskip("megatron.training.arguments").parse_args

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAIN_CONFIGS = sorted((_REPO_ROOT / "configs" / "train").glob("*.yaml"))
assert _TRAIN_CONFIGS, f"expected at least one configs/train/*.yaml under {_REPO_ROOT}"

# These two are shared backbones rather than launchable arms. build_megatron_args is still tried
# for them like any other config, but a ValueError raised while building their own args, as opposed
# to a rejection from Megatron, is treated as "not independently launchable" and skipped.
_BASE_CONFIGS = {"base_cluster.yaml", "base_local.yaml"}


@pytest.mark.parametrize("config_path", _TRAIN_CONFIGS, ids=lambda p: p.name)
def test_config_emits_args_megatron_argparse_accepts(config_path, monkeypatch):
    # base_cluster.yaml and the configs extending it reference ${DATA_STORE}. Setting it here
    # rather than requiring it in the environment is safe, because its value is irrelevant to
    # argparse, which never touches the filesystem for a plain string flag like --data-path.
    monkeypatch.setenv("DATA_STORE", "/tmp/megatron-argparse-test-data-store")

    cfg = MoEPretrainConfig.from_yaml(config_path).resolved(_REPO_ROOT)
    try:
        argv = build_megatron_args(cfg)
    except ValueError:
        if config_path.name in _BASE_CONFIGS:
            pytest.skip(f"{config_path.name} is not independently launchable on its own")
        raise

    # parse_args is used rather than parse_and_validate_args, which would need a real distributed
    # runtime. It reads sys.argv directly and writes usage text to stdout on a bad flag, so that
    # output is captured and a failure reports the offending flag instead of dumping argparse usage.
    monkeypatch.setattr(sys, "argv", ["pretrain_gpt.py", *argv])
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            parse_args()
    except SystemExit as e:
        pytest.fail(
            f"{config_path.name}: parse_args() rejected the emitted command "
            f"(SystemExit {e.code}):\n{captured.getvalue()}"
        )

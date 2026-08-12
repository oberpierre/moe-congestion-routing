from pathlib import Path

from moe_congestion_routing.checkpoint_args import (
    CHECKPOINT_OVERRIDE_ARGS,
    checkpoint_override_argv,
)
from moe_congestion_routing.training.pretrain_config import MoEPretrainConfig

_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "train"


def test_mandatory_epsilon_matches_the_reference_architecture_it_was_trained_with():
    # CHECKPOINT_OVERRIDE_ARGS' --norm-epsilon is only correct because it equals what
    # base_cluster.yaml itself trains with, so the two must move together: derive the expected
    # value from the config rather than hardcoding it here, or the suite cannot catch the two
    # drifting apart when both are edited "consistently" by hand.
    cfg = MoEPretrainConfig.from_yaml(_CONFIGS / "base_cluster.yaml")
    epsilon_flag = next(a for a in CHECKPOINT_OVERRIDE_ARGS if a.startswith("--norm-epsilon "))
    # Compare as floats: str(1e-6) is "1e-06" but the flag carries "1e-6", so a string
    # comparison would fail on formatting rather than on an actual drift between the two.
    assert float(epsilon_flag.split()[1]) == cfg.norm_epsilon


def test_checkpoint_override_argv_is_checkpoint_override_args_split_into_tokens():
    assert checkpoint_override_argv() == [
        tok for arg in CHECKPOINT_OVERRIDE_ARGS for tok in arg.split()
    ]

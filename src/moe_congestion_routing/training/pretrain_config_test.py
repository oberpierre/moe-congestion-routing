import pytest

from moe_congestion_routing.training.pretrain_config import (
    MoEPretrainConfig,
    build_launch_command,
    build_megatron_args,
)


def _pairs(args: list[str]) -> dict[str, str]:
    """Flags that take a value -> value (ignores bare boolean flags like --bf16)."""
    out = {}
    for i, tok in enumerate(args):
        if tok.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[tok] = args[i + 1]
    return out


def test_from_yaml_roundtrip(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "num_experts: 16\nmoe_router_topk: 1\ntrain_iters: 5\nmoe_aux_loss_coeff: 0.02\n"
    )
    cfg = MoEPretrainConfig.from_yaml(path)
    assert cfg.num_experts == 16
    assert cfg.moe_router_topk == 1
    assert cfg.train_iters == 5
    assert cfg.moe_aux_loss_coeff == 0.02
    assert cfg.tokenizer_type == "NullTokenizer"  # default preserved


def test_from_yaml_rejects_unknown_key(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("nonexistent_key: 1\n")
    with pytest.raises(TypeError):
        MoEPretrainConfig.from_yaml(path)


def test_build_megatron_args_carries_moe_and_tokenizer():
    cfg = MoEPretrainConfig(num_experts=8, moe_router_topk=2, moe_aux_loss_coeff=0.01)
    pairs = _pairs(build_megatron_args(cfg))
    assert pairs["--num-experts"] == "8"
    assert pairs["--moe-router-topk"] == "2"
    assert pairs["--moe-router-load-balancing-type"] == "aux_loss"
    assert pairs["--moe-aux-loss-coeff"] == "0.01"
    assert pairs["--tokenizer-type"] == "NullTokenizer"
    assert pairs["--vocab-size"] == "50257"  # NullTokenizer eod = 50256 = <|endoftext|>
    assert pairs["--transformer-impl"] == "transformer_engine"
    assert pairs["--attention-backend"] == "auto"


def test_build_megatron_args_disables_apex_megatron_fusions():
    # Under TE these apex/Megatron fusion paths stay off (apex absent locally; TE fuses its own).
    args = build_megatron_args(MoEPretrainConfig())
    assert "--no-persist-layer-norm" in args
    assert "--no-gradient-accumulation-fusion" in args


def test_build_megatron_args_attention_backend_overridable():
    args = build_megatron_args(MoEPretrainConfig(attention_backend="unfused"))
    assert _pairs(args)["--attention-backend"] == "unfused"


def test_build_megatron_args_toggles_optional_flags():
    on = build_megatron_args(MoEPretrainConfig(bf16=True, save="ckpt", save_interval=10))
    assert "--bf16" in on
    assert _pairs(on)["--save"] == "ckpt"
    assert _pairs(on)["--save-interval"] == "10"

    off = build_megatron_args(MoEPretrainConfig(bf16=False, save=None, save_interval=None))
    assert "--bf16" not in off
    assert "--save" not in off
    assert "--save-interval" not in off


def test_build_megatron_args_emits_load_when_set():
    on = build_megatron_args(MoEPretrainConfig(load="/ckpt/dir"))
    assert _pairs(on)["--load"] == "/ckpt/dir"
    assert "--load" not in build_megatron_args(MoEPretrainConfig(load=None))


def test_build_megatron_args_emits_ckpt_step_to_pin_iteration():
    # load points at the checkpoints DIR; ckpt_step selects which iter_<N>/ inside it.
    on = build_megatron_args(MoEPretrainConfig(load="/ckpt/dir", ckpt_step=200))
    assert _pairs(on)["--ckpt-step"] == "200"
    assert "--ckpt-step" not in build_megatron_args(MoEPretrainConfig(load="/ckpt/dir"))


def test_resolved_absolutises_paths_and_derives_cache(tmp_path):
    cfg = MoEPretrainConfig(
        train_data_path="artifacts/x_train",
        valid_data_path="artifacts/x_valid",
        output_dir="artifacts/run",
        data_cache_path=None,
    )
    r = cfg.resolved(tmp_path)
    assert r.train_data_path == str(tmp_path / "artifacts/x_train")
    assert r.valid_data_path == str(tmp_path / "artifacts/x_valid")
    assert r.data_cache_path == str(tmp_path / "artifacts/run/cache")  # derived from output_dir


def test_resolved_keeps_absolute_paths(tmp_path):
    cfg = MoEPretrainConfig(train_data_path="/abs/train", data_cache_path="/abs/cache")
    r = cfg.resolved(tmp_path)
    assert r.train_data_path == "/abs/train"
    assert r.data_cache_path == "/abs/cache"


def test_resolved_absolutises_checkpoint_paths(tmp_path):
    r = MoEPretrainConfig(save="ckpt/out", load="ckpt/in").resolved(tmp_path)
    assert r.save == str(tmp_path / "ckpt/out")
    assert r.load == str(tmp_path / "ckpt/in")
    # unset stays None (launcher derives the per-run save dir when save_interval is on)
    assert MoEPretrainConfig().resolved(tmp_path).save is None


def test_build_launch_command_wraps_torchrun():
    cfg = MoEPretrainConfig()
    cmd = build_launch_command(cfg, "/repo/Megatron-LM/pretrain_gpt.py", nproc=1)
    assert cmd[0] == "torchrun"
    assert "--standalone" in cmd
    assert cmd[_pairs_index(cmd, "--nproc-per-node")] == "1"
    assert "/repo/Megatron-LM/pretrain_gpt.py" in cmd
    assert "--num-experts" in cmd  # megatron args appended after the script


def _pairs_index(args: list[str], flag: str) -> int:
    return args.index(flag) + 1

import pytest

from moe_congestion_routing.training.infer_config import (
    MoEInferConfig,
    build_infer_launch_command,
    build_infer_megatron_args,
)


def _pairs(args: list[str]) -> dict[str, str]:
    """Flags that take a single value -> value (ignores bare flags and the variadic --prompts)."""
    out = {}
    for i, tok in enumerate(args):
        if tok.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[tok] = args[i + 1]
    return out


def test_from_yaml_roundtrip(tmp_path):
    path = tmp_path / "infer.yaml"
    path.write_text("load: run/checkpoints\nnum_tokens_to_generate: 16\ntop_k: 5\n")
    cfg = MoEInferConfig.from_yaml(path)
    assert cfg.load == "run/checkpoints"
    assert cfg.num_tokens_to_generate == 16
    assert cfg.top_k == 5
    assert cfg.tokenizer_type == "HuggingFaceTokenizer"  # default preserved


def test_from_yaml_rejects_unknown_key(tmp_path):
    path = tmp_path / "infer.yaml"
    path.write_text("nonexistent_key: 1\n")
    with pytest.raises(TypeError):
        MoEInferConfig.from_yaml(path)


def test_build_args_requires_load():
    with pytest.raises(ValueError, match="load is required"):
        build_infer_megatron_args(MoEInferConfig(load=None))


def test_build_args_carries_tokenizer_checkpoint_and_generation():
    cfg = MoEInferConfig(load="/run/checkpoints", num_tokens_to_generate=24, top_k=1)
    pairs = _pairs(build_infer_megatron_args(cfg))
    assert pairs["--tokenizer-type"] == "HuggingFaceTokenizer"
    assert pairs["--tokenizer-model"] == "assets/tokenizer/gpt2"
    assert pairs["--vocab-size"] == "50257"
    assert pairs["--load"] == "/run/checkpoints"
    assert pairs["--num-tokens-to-generate"] == "24"
    assert pairs["--top_k"] == "1"  # greedy; note the underscore flag name Megatron expects


def test_build_args_emits_ckpt_step_when_set():
    on = build_infer_megatron_args(MoEInferConfig(load="/x", ckpt_step=200))
    assert _pairs(on)["--ckpt-step"] == "200"
    assert "--ckpt-step" not in build_infer_megatron_args(MoEInferConfig(load="/x"))


def test_build_args_prompts_are_variadic_and_last():
    cfg = MoEInferConfig(load="/x", prompts=["The capital of France is", "Hello there"])
    args = build_infer_megatron_args(cfg)
    # --prompts must be the final flag so its nargs='+' list can't eat a following flag's value.
    assert args[-3:] == ["--prompts", "The capital of France is", "Hello there"]


def test_build_args_uses_legacy_static_engine_by_default():
    # legacy static avoids the dynamic-batching context that requires flash-attn (uninstalled).
    assert "--use-legacy-static-engine" in build_infer_megatron_args(MoEInferConfig(load="/x"))
    off = build_infer_megatron_args(MoEInferConfig(load="/x", use_legacy_static_engine=False))
    assert "--use-legacy-static-engine" not in off


def test_build_args_disables_fused_kernels_for_local_impl():
    args = build_infer_megatron_args(MoEInferConfig(load="/x"))
    for flag in (
        "--no-persist-layer-norm",
        "--no-gradient-accumulation-fusion",
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--disable-bias-linear",
    ):
        assert flag in args


def test_resolved_absolutises_load_and_tokenizer(tmp_path):
    r = MoEInferConfig(load="artifacts/run/checkpoints").resolved(tmp_path)
    assert r.load == str(tmp_path / "artifacts/run/checkpoints")
    assert r.tokenizer_model == str(tmp_path / "assets/tokenizer/gpt2")
    # a None load stays None (the launcher fills it from --load, then validates)
    assert MoEInferConfig(load=None).resolved(tmp_path).load is None


def test_build_launch_command_wraps_torchrun():
    cmd = build_infer_launch_command(
        MoEInferConfig(load="/x"),
        "/repo/Megatron-LM/examples/inference/advanced/gpt_static_inference.py",
    )
    assert cmd[0] == "torchrun"
    assert "--standalone" in cmd
    assert cmd[-1] == "The capital of France is"  # variadic prompt tail survives wrapping

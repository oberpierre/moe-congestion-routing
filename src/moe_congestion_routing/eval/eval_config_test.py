import shlex
from pathlib import Path

import pytest

from moe_congestion_routing.eval.eval_config import (
    EvalConfig,
    build_lm_eval_args,
    eval_output_dir,
)


def _cfg(**kw) -> EvalConfig:
    kw.setdefault("ckpt_step", 20)
    # eval_output_dir (and therefore build_lm_eval_args) needs a way to name the output path;
    # give every test a load path unless it overrides load or output_dir itself.
    if kw.get("load") is None and kw.get("output_dir") is None:
        kw.setdefault("load", "/ckpt/dir")
    return EvalConfig(**kw)


def _model_args(args: list[str]) -> dict[str, str]:
    """Parse ``--model_args``'s comma-joined ``key=value`` string into a dict."""
    raw = args[args.index("--model_args") + 1]
    out = {}
    for part in raw.split(","):
        k, _, v = part.partition("=")
        out[k] = v
    return out


def _flag(args: list[str], name: str) -> str | None:
    """Value of a top-level ``--flag value`` pair, or None if the flag is absent."""
    if name not in args:
        return None
    return args[args.index(name) + 1]


# --- from_yaml / extends -----------------------------------------------------------------


def test_from_yaml_roundtrip(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("ckpt_step: 40\ntasks: [arc_easy]\nnum_fewshot: 0\n")
    cfg = EvalConfig.from_yaml(path)
    assert cfg.ckpt_step == 40
    assert cfg.tasks == ["arc_easy"]
    assert cfg.num_fewshot == 0
    assert cfg.tokenizer_type == "HuggingFaceTokenizer"  # default preserved
    assert cfg.tokenizer_model == "assets/tokenizer/gpt2"  # default preserved


def test_from_yaml_rejects_unknown_key(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("ckpt_step: 20\nnonexistent_key: 1\n")
    with pytest.raises(TypeError):
        EvalConfig.from_yaml(path)


def test_from_yaml_requires_ckpt_step(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("tasks: [arc_easy]\n")
    with pytest.raises(TypeError):
        EvalConfig.from_yaml(path)


def test_from_yaml_extends_merges_base_with_override(tmp_path):
    (tmp_path / "base.yaml").write_text("ckpt_step: 10\ntasks: [arc_easy]\nseed: 1\n")
    arm = tmp_path / "arm.yaml"
    arm.write_text("extends: base.yaml\nckpt_step: 20\n")
    cfg = EvalConfig.from_yaml(arm)
    assert cfg.tasks == ["arc_easy"]  # inherited from base
    assert cfg.seed == 1  # inherited from base
    assert cfg.ckpt_step == 20  # arm overrides base


def test_from_yaml_extends_rejects_cycles(tmp_path):
    (tmp_path / "x.yaml").write_text("extends: y.yaml\nckpt_step: 1\n")
    (tmp_path / "y.yaml").write_text("extends: x.yaml\nckpt_step: 1\n")
    with pytest.raises(ValueError, match="circular"):
        EvalConfig.from_yaml(tmp_path / "x.yaml")


# --- build_lm_eval_args: the emitter -----------------------------------------------------


def test_checkpoint_and_tokenizer_reach_model_args():
    cfg = _cfg(ckpt_step=20, load="/ckpt/dir")
    model_args = _model_args(build_lm_eval_args(cfg))
    assert model_args["load"] == "/ckpt/dir"
    assert model_args["ckpt_step"] == "20"
    assert model_args["tokenizer_type"] == "HuggingFaceTokenizer"
    assert model_args["tokenizer_model"] == "assets/tokenizer/gpt2"


def test_tokenizer_override_reaches_model_args():
    cfg = _cfg(tokenizer_type="NullTokenizer", tokenizer_model="gpt2-alt")
    model_args = _model_args(build_lm_eval_args(cfg))
    assert model_args["tokenizer_type"] == "NullTokenizer"
    assert model_args["tokenizer_model"] == "gpt2-alt"


def test_absent_optional_model_args_keys_are_omitted():
    cfg = _cfg(seq_length=None, micro_batch_size=None)
    model_args = _model_args(build_lm_eval_args(cfg))
    assert "seq_length" not in model_args
    assert "micro_batch_size" not in model_args


def test_optional_model_args_keys_emitted_when_set():
    cfg = _cfg(seq_length=1024, micro_batch_size=8)
    model_args = _model_args(build_lm_eval_args(cfg))
    assert model_args["seq_length"] == "1024"
    assert model_args["micro_batch_size"] == "8"


def test_load_omitted_from_model_args_when_unset():
    # output_dir stands in for the run-directory-from-checkpoint derivation, which needs `load`.
    cfg = _cfg(load=None, output_dir="/out")
    model_args = _model_args(build_lm_eval_args(cfg))
    assert "load" not in model_args


def test_absent_optional_top_level_flags_are_omitted_not_emitted_empty():
    cfg = _cfg(tasks=[], num_fewshot=None, batch_size=None, limit=None)
    args = build_lm_eval_args(cfg)
    assert "--tasks" not in args
    assert "--num_fewshot" not in args
    assert "--batch_size" not in args
    assert "--limit" not in args


def test_top_level_flags_emitted_when_set():
    cfg = _cfg(tasks=["arc_easy", "piqa"], num_fewshot=10, batch_size=16, limit=50)
    args = build_lm_eval_args(cfg)
    assert _flag(args, "--tasks") == "arc_easy,piqa"
    assert _flag(args, "--num_fewshot") == "10"
    assert _flag(args, "--batch_size") == "16"
    assert _flag(args, "--limit") == "50"


def test_both_seeds_reach_the_seed_flag():
    cfg = _cfg(seed=42, fewshot_seed=7)
    args = build_lm_eval_args(cfg)
    assert _flag(args, "--seed") == "42,42,42,7"


def test_argv_with_space_containing_value_survives_shlex_roundtrip():
    # --model_args' value carries extra_args' own space-separated flag list, so it is a single
    # argv element that itself contains a space. A launcher must print/record it with
    # shlex.join (or equivalent quoting), not " ".join, or pasting the printed command back
    # into a shell splits that element into two and drops or misplaces a flag.
    cfg = _cfg(extra_args="--no-rope-fusion")
    args = build_lm_eval_args(cfg)
    cmd = ["python", "-m", "lm_eval", *args]
    model_args_value = args[args.index("--model_args") + 1]
    assert " " in model_args_value  # otherwise this test would not exercise the bug
    assert shlex.split(shlex.join(cmd)) == cmd


def test_extra_args_passes_through():
    cfg = _cfg(extra_args="--no-rope-fusion")
    model_args = _model_args(build_lm_eval_args(cfg))
    assert "--no-rope-fusion" in model_args["extra_args"]


def test_mandatory_flags_present_with_no_extra_args_set():
    cfg = _cfg(extra_args="")
    model_args = _model_args(build_lm_eval_args(cfg))
    assert "--no-use-tokenizer-model-from-checkpoint-args" in model_args["extra_args"]
    assert "--no-gradient-accumulation-fusion" in model_args["extra_args"]


def test_mandatory_flags_present_even_when_config_sets_extra_args():
    # A config cannot drop the two mandatory flags by setting its own extra_args -- they are
    # appended by build_lm_eval_args itself, not left to the config to include.
    cfg = _cfg(extra_args="--no-use-tokenizer-model-from-checkpoint-args")
    model_args = _model_args(build_lm_eval_args(cfg))
    assert model_args["extra_args"].count("--no-use-tokenizer-model-from-checkpoint-args") >= 1
    assert "--no-gradient-accumulation-fusion" in model_args["extra_args"]


def test_output_path_derives_from_the_checkpoint_run_dir(tmp_path):
    checkpoints = tmp_path / "run" / "checkpoints"
    checkpoints.mkdir(parents=True)
    cfg = _cfg(ckpt_step=20, load=str(checkpoints))
    path = eval_output_dir(cfg)
    assert path == tmp_path / "run" / "evals" / "iter_0000020"
    assert _flag(build_lm_eval_args(cfg), "--output_path") == str(path)


def test_output_dir_override_skips_the_evals_subdir(tmp_path):
    cfg = _cfg(
        ckpt_step=5473,
        load=str(tmp_path / "some" / "checkpoints"),
        output_dir=str(tmp_path / "flame"),
    )
    path = eval_output_dir(cfg)
    assert path == tmp_path / "flame" / "iter_0005473"


# --- resolved() ----------------------------------------------------------------------------


def test_resolved_absolutises_load_and_tokenizer_model(tmp_path):
    cfg = _cfg(load="artifacts/run/checkpoints", tokenizer_model="assets/tokenizer/gpt2")
    resolved = cfg.resolved(tmp_path)
    assert resolved.load == str(tmp_path / "artifacts" / "run" / "checkpoints")
    assert resolved.tokenizer_model == str(tmp_path / "assets" / "tokenizer" / "gpt2")


def test_resolved_keeps_load_none_when_unset(tmp_path):
    cfg = _cfg(load=None)
    resolved = cfg.resolved(tmp_path)
    assert resolved.load is None


def test_resolved_keeps_absolute_paths(tmp_path):
    abs_load = str(tmp_path / "abs" / "checkpoints")
    cfg = _cfg(load=abs_load)
    resolved = cfg.resolved(tmp_path)
    assert resolved.load == abs_load


def test_resolved_expands_env_vars_in_load(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_STORE", str(tmp_path / "store"))
    cfg = _cfg(load="${DATA_STORE}/run/checkpoints")
    resolved = cfg.resolved(Path("/unused"))
    assert resolved.load == str(tmp_path / "store" / "run" / "checkpoints")


def test_resolved_absolutises_output_dir_override(tmp_path):
    cfg = _cfg(output_dir="artifacts/eval/flame_290m")
    resolved = cfg.resolved(tmp_path)
    assert resolved.output_dir == str(tmp_path / "artifacts" / "eval" / "flame_290m")

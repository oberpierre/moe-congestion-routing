import shlex
from pathlib import Path

import pytest

from moe_congestion_routing.eval.eval_config import (
    EvalConfig,
    build_launch_command,
    build_lm_eval_args,
    eval_arm,
    eval_output_dir,
    eval_run_dir,
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
    path.write_text("ckpt_step: 40\ntasks: [arc_easy]\n")
    cfg = EvalConfig.from_yaml(path)
    assert cfg.ckpt_step == 40
    assert cfg.tasks == ["arc_easy"]
    assert cfg.tokenizer_type == "HuggingFaceTokenizer"  # default preserved
    assert cfg.tokenizer_model == "assets/tokenizer/gpt2"  # default preserved


def test_from_yaml_rejects_unknown_key(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("ckpt_step: 20\nnonexistent_key: 1\n")
    with pytest.raises(TypeError):
        EvalConfig.from_yaml(path)


def test_from_yaml_rejects_num_fewshot_key(tmp_path):
    # num_fewshot moved to lm-eval's own task/group configs -- a config still setting it here
    # is stale rather than silently accepted, because --num_fewshot would overwrite every task's
    # own split.
    path = tmp_path / "run.yaml"
    path.write_text("ckpt_step: 20\nnum_fewshot: 0\n")
    with pytest.raises(TypeError):
        EvalConfig.from_yaml(path)


def test_from_yaml_allows_missing_ckpt_step_and_load(tmp_path):
    # A config need not name a checkpoint at all: scripts/run_lm_eval.py's --load/--ckpt-step
    # can supply both later. require_launch_ready, not from_yaml, is what enforces they arrive
    # from somewhere before a run launches.
    path = tmp_path / "run.yaml"
    path.write_text("tasks: [arc_easy]\n")
    cfg = EvalConfig.from_yaml(path)
    assert cfg.ckpt_step is None
    assert cfg.load is None


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
    cfg = _cfg(tasks=[], batch_size=None, limit=None)
    args = build_lm_eval_args(cfg)
    assert "--tasks" not in args
    assert "--batch_size" not in args
    assert "--limit" not in args


def test_top_level_flags_emitted_when_set():
    cfg = _cfg(tasks=["arc_easy", "piqa"], batch_size=16, limit=50)
    args = build_lm_eval_args(cfg)
    assert _flag(args, "--tasks") == "arc_easy,piqa"
    assert _flag(args, "--batch_size") == "16"
    assert _flag(args, "--limit") == "50"


def test_include_path_flag_always_emitted_with_its_default():
    # Without --include_path the harness never looks outside its own built-in task registry, so
    # tasks: [flame_suite] fails with "task not found". Should always be emitted.
    cfg = _cfg(tasks=["arc_easy"])
    args = build_lm_eval_args(cfg)
    assert _flag(args, "--include_path") == "configs/eval/tasks"


def test_include_path_override_reaches_the_flag():
    cfg = _cfg(include_path="/custom/tasks")
    args = build_lm_eval_args(cfg)
    assert _flag(args, "--include_path") == "/custom/tasks"


def test_ckpt_step_omitted_from_model_args_when_unset(monkeypatch):
    import moe_congestion_routing.eval.eval_config as eval_config_module

    monkeypatch.setattr(eval_config_module, "eval_output_dir", lambda cfg: Path("/unused"))
    cfg = EvalConfig(ckpt_step=None, load="/ckpt/dir")
    model_args = _model_args(build_lm_eval_args(cfg))
    assert "ckpt_step" not in model_args


def test_num_fewshot_flag_never_emitted():
    # num_fewshot is not an EvalConfig field at all: passing --num_fewshot would overwrite every
    # task's own split, so the launcher must never be able to emit it regardless of what a
    # config sets on other fields.
    cfg = _cfg(tasks=["arc_easy"])
    args = build_lm_eval_args(cfg)
    assert "--num_fewshot" not in args


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


def test_output_dir_override_still_gets_the_evals_subdir(tmp_path):
    # An override IS the run directory directly (one level shallower than the derived case,
    # which has a <run_tag> to nest under), but the tail below it is identical either way.
    # The glob in results.py has to find both.
    cfg = _cfg(
        ckpt_step=5473,
        load=str(tmp_path / "some" / "checkpoints"),
        output_dir=str(tmp_path / "flame"),
    )
    path = eval_output_dir(cfg)
    assert path == tmp_path / "flame" / "evals" / "iter_0005473"


def test_eval_run_dir_derives_from_the_checkpoint_when_no_override(tmp_path):
    checkpoints = tmp_path / "run" / "checkpoints"
    checkpoints.mkdir(parents=True)
    cfg = _cfg(ckpt_step=20, load=str(checkpoints))
    assert eval_run_dir(cfg) == tmp_path / "run"


def test_eval_run_dir_is_the_override_itself(tmp_path):
    cfg = _cfg(ckpt_step=5473, output_dir=str(tmp_path / "flame"))
    assert eval_run_dir(cfg) == tmp_path / "flame"


# --- build_launch_command: nproc and devices move together ------------------------------------


@pytest.mark.parametrize("nproc", [1, 2, 4])
def test_nproc_per_node_and_devices_move_together(nproc):
    # The spec's actual point: a test that checked only one of these would pass even if the two
    # went out of sync, since --nproc-per-node and devices= are set from two different places in
    # a command built by hand. Asserting both from one `nproc` catches that they cannot drift.
    cmd = build_launch_command(_cfg(), nproc=nproc)
    assert _flag(cmd, "--nproc-per-node") == str(nproc)
    assert _model_args(cmd)["devices"] == str(nproc)


def test_build_launch_command_default_nproc_is_one():
    cmd = build_launch_command(_cfg())
    assert _flag(cmd, "--nproc-per-node") == "1"
    assert _model_args(cmd)["devices"] == "1"


def test_build_launch_command_runs_lm_eval_as_a_module_via_torch_distributed_run():
    cmd = build_launch_command(_cfg(), nproc=4)
    assert cmd[1:4] == ["-m", "torch.distributed.run", "--standalone"]
    assert "lm_eval" in cmd
    assert cmd[cmd.index("lm_eval") - 1] == "-m"


def test_build_launch_command_carries_the_rest_of_build_lm_eval_args():
    # devices= is the only thing build_launch_command adds; everything else (checkpoint,
    # tokenizer, tasks, seeds, output path) must still reach the command the same way
    # build_lm_eval_args produces it on its own.
    cfg = _cfg(tasks=["arc_easy"], seed=42, fewshot_seed=7)
    cmd = build_launch_command(cfg, nproc=2)
    model_args = _model_args(cmd)
    assert model_args["load"] == cfg.load
    assert model_args["ckpt_step"] == str(cfg.ckpt_step)
    assert _flag(cmd, "--tasks") == "arc_easy"
    assert _flag(cmd, "--seed") == "42,42,42,7"


def test_build_lm_eval_args_never_emits_devices():
    # devices is not an EvalConfig field, so the plain (non-launch) arg builder must never emit
    # it even though it shares the same underlying key=value construction as
    # build_launch_command.
    model_args = _model_args(build_lm_eval_args(_cfg()))
    assert "devices" not in model_args


# --- eval_arm --------------------------------------------------------------------------------


def test_eval_arm_derived_from_the_run_directorys_parent_when_launch_command_marks_it_ours(
    tmp_path,
):
    run_dir = tmp_path / "exp1" / "switch" / "20260101-000000"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run_dir / "launch_command.txt").write_text("some command\n")
    cfg = _cfg(ckpt_step=20, load=str(checkpoints))
    assert eval_arm(cfg) == "switch"


def test_eval_arm_is_the_output_dir_overrides_own_basename(tmp_path):
    cfg = _cfg(ckpt_step=5473, output_dir=str(tmp_path / "flame_290m"))
    assert eval_arm(cfg) == "flame_290m"


def test_eval_arm_returns_none_when_the_run_directory_has_no_launch_command_marker(tmp_path):
    # Right shape (a "switch"-looking directory two levels above the checkpoints), but nothing
    # our own launcher wrote there -- exactly the case a path-shape check alone cannot see.
    checkpoints = tmp_path / "exp1" / "switch" / "20260101-000000" / "checkpoints"
    checkpoints.mkdir(parents=True)
    cfg = _cfg(ckpt_step=20, load=str(checkpoints))
    assert eval_arm(cfg) is None


@pytest.mark.parametrize(
    "load",
    [
        "/data/flame_moe_290m/checkpoints",
        "/home/user/scratch/flame_moe_290m/checkpoints",
        "/home/user/downloads/checkpoints",
    ],
)
def test_eval_arm_returns_none_for_external_checkpoints_with_no_output_dir_override(load):
    # None of these directories were produced by scripts/run_moe_pretrain.py, so none of them
    # have a launch_command.txt regardless of how plausible the path looks.
    cfg = _cfg(ckpt_step=5473, load=load)
    assert eval_arm(cfg) is None


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


# --- require_launch_ready --------------------------------------------------------------------


def test_require_launch_ready_passes_when_both_set():
    cfg = _cfg(ckpt_step=20, load="/ckpt/dir")
    cfg.require_launch_ready()  # must not raise


def test_require_launch_ready_raises_naming_load_when_only_ckpt_step_missing():
    cfg = EvalConfig(ckpt_step=None, load="/ckpt/dir")
    with pytest.raises(ValueError, match="ckpt_step"):
        cfg.require_launch_ready()


def test_require_launch_ready_raises_naming_ckpt_step_when_only_load_missing():
    cfg = EvalConfig(ckpt_step=20, load=None)
    with pytest.raises(ValueError, match="load"):
        cfg.require_launch_ready()


def test_require_launch_ready_names_both_config_key_and_flag_for_each_missing_field():
    # A config supplying neither, launched with neither, must name both ways to supply each.
    cfg = EvalConfig(ckpt_step=None, load=None)
    with pytest.raises(ValueError) as excinfo:
        cfg.require_launch_ready()
    message = str(excinfo.value)
    assert "load" in message and "--load" in message
    assert "ckpt_step" in message and "--ckpt-step" in message


def test_resolved_absolutises_output_dir_override(tmp_path):
    cfg = _cfg(output_dir="artifacts/eval/flame_290m")
    resolved = cfg.resolved(tmp_path)
    assert resolved.output_dir == str(tmp_path / "artifacts" / "eval" / "flame_290m")


def test_resolved_absolutises_include_path():
    cfg = _cfg()
    resolved = cfg.resolved(Path("/repo"))
    assert resolved.include_path == "/repo/configs/eval/tasks"

import sys
from pathlib import Path

from moe_congestion_routing.checkpoint_args import checkpoint_override_argv
from moe_congestion_routing.infer.launch import GenerateRequest, build_generate_command


def _req(**kw) -> GenerateRequest:
    kw.setdefault("load", "/ckpt/dir")
    return GenerateRequest(**kw)


# --- resolved() --------------------------------------------------------------------------


def test_resolved_absolutises_a_relative_load(tmp_path):
    req = _req(load="artifacts/run/checkpoints")
    resolved = req.resolved(tmp_path)
    assert resolved.load == str(tmp_path / "artifacts" / "run" / "checkpoints")


def test_resolved_keeps_an_already_absolute_load(tmp_path):
    req = _req(load="/abs/checkpoints")
    resolved = req.resolved(tmp_path)
    assert resolved.load == "/abs/checkpoints"


def test_resolved_passes_a_hub_id_tokenizer_model_through_unchanged(tmp_path):
    req = _req(tokenizer_model="EleutherAI/pythia-12b")
    resolved = req.resolved(tmp_path)
    assert resolved.tokenizer_model == "EleutherAI/pythia-12b"


def test_resolved_absolutises_a_dot_slash_marked_tokenizer_model(tmp_path):
    req = _req(tokenizer_model="./assets/tokenizer/gpt2")
    resolved = req.resolved(tmp_path)
    assert resolved.tokenizer_model == str(tmp_path / "assets" / "tokenizer" / "gpt2")


def test_resolved_expands_env_var_load(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_STORE", str(tmp_path / "store"))
    req = _req(load="${DATA_STORE}/checkpoints")
    resolved = req.resolved(tmp_path)
    assert resolved.load == str(tmp_path / "store" / "checkpoints")


# --- build_generate_command ---------------------------------------------------------------


def test_command_runs_torch_distributed_run_under_the_current_interpreter():
    # Not a bare `torchrun`: the cluster venv borrows torch from the container and so has no
    # torchrun of its own, which would send PATH to the container's copy and its system python,
    # where the venv's transformers is invisible.
    req = _req()
    cmd = build_generate_command(req, "/repo/scripts/moe_generate.py", nproc=2)
    assert cmd[:7] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "2",
        "/repo/scripts/moe_generate.py",
    ]


def test_prompts_come_last_because_nargs_plus_would_swallow_anything_after_it():
    req = _req(prompts=("hello", "world"), passthrough=("--num-tokens-to-oom", "5"))
    cmd = build_generate_command(req, "gen.py")
    assert cmd[-3:] == ["--prompts", "hello", "world"]


def test_interactive_mode_omits_prompts_entirely():
    req = _req(interactive=True, prompts=("unused",))
    cmd = build_generate_command(req, "gen.py")
    assert "--prompts" not in cmd
    assert "--interactive" in cmd


def test_ckpt_step_omitted_when_none():
    req = _req(ckpt_step=None)
    cmd = build_generate_command(req, "gen.py")
    assert "--ckpt-step" not in cmd


def test_ckpt_step_included_when_set():
    req = _req(ckpt_step=100)
    cmd = build_generate_command(req, "gen.py")
    idx = cmd.index("--ckpt-step")
    assert cmd[idx + 1] == "100"


def test_checkpoint_override_args_are_present_and_in_between_engine_and_passthrough():
    req = _req(passthrough=("--seed", "1"))
    cmd = build_generate_command(req, "gen.py")
    override = checkpoint_override_argv()
    override_idx = cmd.index(override[0])
    assert cmd[override_idx : override_idx + len(override)] == override
    engine_idx = cmd.index("--engine")
    passthrough_idx = cmd.index("--seed")
    assert engine_idx < override_idx < passthrough_idx


def test_top_k_and_top_p_use_megatrons_own_underscore_spelling():
    req = _req(top_k=1, top_p=0.0)
    cmd = build_generate_command(req, "gen.py")
    assert "--top_k" in cmd
    assert "--top_p" in cmd
    assert "--top-k" not in cmd
    assert "--top-p" not in cmd


def test_mandatory_flags_present_in_order():
    req = _req()
    cmd = build_generate_command(req, "gen.py")
    mandatory = ["--use-checkpoint-args", "--bf16", "--transformer-impl", "transformer_engine"]
    start = cmd.index("--use-checkpoint-args")
    assert cmd[start : start + len(mandatory)] == mandatory


def test_load_and_generation_flags_use_the_request_values():
    req = _req(
        load="/ckpt",
        max_new_tokens=16,
        temperature=0.5,
        attention_backend="unfused",
        engine="static",
    )
    cmd = build_generate_command(req, "gen.py")
    assert cmd[cmd.index("--load") + 1] == "/ckpt"
    assert cmd[cmd.index("--num-tokens-to-generate") + 1] == "16"
    assert cmd[cmd.index("--temperature") + 1] == "0.5"
    assert cmd[cmd.index("--attention-backend") + 1] == "unfused"
    assert cmd[cmd.index("--engine") + 1] == "static"


def test_default_engine_is_auto():
    assert GenerateRequest(load="/ckpt").engine == "auto"


def test_generate_script_accepts_a_path(tmp_path: Path):
    # Located relative to --nproc-per-node rather than by absolute index, so adding a launcher
    # flag ahead of the script does not fail this test for the wrong reason.
    cmd = build_generate_command(_req(), tmp_path / "moe_generate.py")
    assert cmd[cmd.index("--nproc-per-node") + 2] == str(tmp_path / "moe_generate.py")

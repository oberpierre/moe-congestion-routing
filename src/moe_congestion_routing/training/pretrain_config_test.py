import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from moe_congestion_routing.losses.cost_families import DEFAULT_LAMBDA
from moe_congestion_routing.training.pretrain_config import (
    MoEPretrainConfig,
    build_launch_command,
    build_megatron_args,
    resolve_run_dir,
)

_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "train"


def _pairs(args: list[str]) -> dict[str, str]:
    """Flags that take a value -> value (ignores bare boolean flags like --bf16)."""
    out = {}
    for i, tok in enumerate(args):
        if tok.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[tok] = args[i + 1]
    return out


def _cfg(**kw) -> MoEPretrainConfig:
    """A build-args-ready config: fills the now-required train_data_path (yaml must set it)."""
    return MoEPretrainConfig(train_data_path="/data/train", **kw)


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


def test_from_yaml_extends_merges_base_with_override(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "num_experts: 8\nmoe_router_topk: 2\nhidden_size: 512\ntrain_data_path: /data/train\n"
    )
    arm = tmp_path / "arm.yaml"
    arm.write_text("extends: base.yaml\nmoe_router_load_balancing_type: none\nhidden_size: 1024\n")
    cfg = MoEPretrainConfig.from_yaml(arm)
    assert cfg.num_experts == 8  # inherited from base
    assert cfg.moe_router_topk == 2  # inherited from base
    assert cfg.moe_router_load_balancing_type == "none"  # from arm
    assert cfg.hidden_size == 1024  # arm overrides base
    assert cfg.train_data_path == "/data/train"


def test_from_yaml_extends_is_recursive_and_ordered(tmp_path):
    (tmp_path / "a.yaml").write_text("num_layers: 2\nhidden_size: 128\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nhidden_size: 256\n")
    (tmp_path / "c.yaml").write_text("extends: b.yaml\nnum_experts: 4\ntrain_data_path: /d\n")
    cfg = MoEPretrainConfig.from_yaml(tmp_path / "c.yaml")
    assert cfg.num_layers == 2  # from a (grandparent)
    assert cfg.hidden_size == 256  # b overrides a
    assert cfg.num_experts == 4  # from c


def test_from_yaml_extends_rejects_cycles(tmp_path):
    (tmp_path / "x.yaml").write_text("extends: y.yaml\n")
    (tmp_path / "y.yaml").write_text("extends: x.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        MoEPretrainConfig.from_yaml(tmp_path / "x.yaml")


def test_control_arm_disables_balancing():
    pairs = _pairs(build_megatron_args(_cfg(moe_router_load_balancing_type="none")))
    assert pairs["--moe-router-load-balancing-type"] == "none"
    # control never carries a bias flag
    assert "--moe-router-enable-expert-bias" not in build_megatron_args(
        _cfg(moe_router_load_balancing_type="none")
    )


def test_alflb_arm_emits_sigmoid_expert_bias():
    cfg = _cfg(
        moe_router_load_balancing_type="none",
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
        moe_router_bias_update_rate=1e-2,
    )
    args = build_megatron_args(cfg)
    pairs = _pairs(args)
    assert "--moe-router-enable-expert-bias" in args
    assert pairs["--moe-router-bias-update-rate"] == "0.01"
    assert pairs["--moe-router-score-function"] == "sigmoid"


def test_score_function_defaults_softmax_and_bias_off():
    args = build_megatron_args(_cfg())
    assert _pairs(args)["--moe-router-score-function"] == "softmax"
    assert "--moe-router-enable-expert-bias" not in args


def test_z_loss_and_per_layer_logging_optional():
    on = build_megatron_args(_cfg(moe_z_loss_coeff=1e-3, moe_per_layer_logging=True))
    assert _pairs(on)["--moe-z-loss-coeff"] == "0.001"
    assert "--moe-per-layer-logging" in on
    off = build_megatron_args(_cfg())
    assert "--moe-z-loss-coeff" not in off
    assert "--moe-per-layer-logging" not in off


def test_wandb_args_gated_on_project():
    off = build_megatron_args(_cfg(wandb_project=None, wandb_exp_name="x"))
    assert "--wandb-project" not in off
    assert "--wandb-exp-name" not in off  # not emitted without a project
    on = _pairs(
        build_megatron_args(
            _cfg(wandb_project="moe", wandb_exp_name="switch-local", wandb_entity="me")
        )
    )
    assert on["--wandb-project"] == "moe"
    assert on["--wandb-exp-name"] == "switch-local"
    assert on["--wandb-entity"] == "me"


def test_tensorboard_and_throughput_flags():
    args = build_megatron_args(_cfg(tensorboard_dir="/run/tb", log_throughput=True))
    assert _pairs(args)["--tensorboard-dir"] == "/run/tb"
    assert "--log-throughput" in args


def test_exit_interval_emitted_only_when_set():
    # Sliced training. --exit-interval is present when set and absent by default, in which case
    # the run goes straight to train_iters. train_iters itself is unaffected, so the LR-schedule
    # horizon stays the full budget.
    sliced = build_megatron_args(_cfg(exit_interval=5000, train_iters=20000))
    assert _pairs(sliced)["--exit-interval"] == "5000"
    assert _pairs(sliced)["--train-iters"] == "20000"
    assert "--exit-interval" not in build_megatron_args(_cfg())


def test_exit_on_missing_checkpoint_flag():
    # Set by the launcher whenever a load dir is given, so that an explicit resume finding no
    # checkpoint fails loud instead of silently starting from random. Bare flag, only when True.
    assert "--exit-on-missing-checkpoint" in build_megatron_args(
        _cfg(load="/run/checkpoints", exit_on_missing_checkpoint=True)
    )
    assert "--exit-on-missing-checkpoint" not in build_megatron_args(_cfg())


def test_resolved_expands_env_vars_in_paths(tmp_path, monkeypatch):
    # Committed configs reference ${DATA_STORE}, so resolved() has to expand it or Megatron gets a
    # literal "${DATA_STORE}" path. An unset variable must fail loud rather than yield a bad path.
    monkeypatch.setenv("DATA_STORE", "/store")
    r = MoEPretrainConfig(data_path="${DATA_STORE}/climbmix/climbmix").resolved(tmp_path)
    assert r.data_path == "/store/climbmix/climbmix"
    monkeypatch.delenv("DATA_STORE")
    with pytest.raises(ValueError, match="unresolved environment variable"):
        MoEPretrainConfig(data_path="${DATA_STORE}/x").resolved(tmp_path)


def test_resolved_absolutises_logging_dirs(tmp_path):
    r = MoEPretrainConfig(tensorboard_dir="run/tb", wandb_save_dir="run/wandb").resolved(tmp_path)
    assert r.tensorboard_dir == str(tmp_path / "run/tb")
    assert r.wandb_save_dir == str(tmp_path / "run/wandb")
    unset = MoEPretrainConfig().resolved(tmp_path)
    assert unset.tensorboard_dir is None and unset.wandb_save_dir is None


def test_build_megatron_args_requires_a_data_source():
    with pytest.raises(ValueError, match="a data source is required"):
        build_megatron_args(MoEPretrainConfig(train_data_path=None, data_path=None))


def test_build_megatron_args_valid_data_path_optional():
    # valid is emitted only when set (enables train-only runs; None doesn't leak into the args).
    assert "--valid-data-path" not in build_megatron_args(_cfg(valid_data_path=None))
    on = _pairs(build_megatron_args(_cfg(valid_data_path="/data/valid")))
    assert on["--valid-data-path"] == "/data/valid"


def test_single_blob_data_path_emits_data_path_and_split():
    # ClimbMix mode: one blob carved by --split; no per-split paths.
    args = build_megatron_args(
        MoEPretrainConfig(train_data_path=None, data_path="/data/blob", split="99,1,0")
    )
    pairs = _pairs(args)
    assert pairs["--data-path"] == "/data/blob"
    assert pairs["--split"] == "99,1,0"
    assert "--train-data-path" not in args


def test_data_path_requires_split():
    # A blob without --split leaves the valid split empty and eval crashes, so fail loud instead.
    with pytest.raises(ValueError, match="split is required with data_path"):
        build_megatron_args(MoEPretrainConfig(train_data_path=None, data_path="/data/blob"))


def test_split_incompatible_with_train_data_path():
    # Megatron forbids --split alongside per-split paths ("split and blend_per_split incompatible").
    with pytest.raises(ValueError, match="split is incompatible with train_data_path"):
        build_megatron_args(_cfg(split="99,1,0"))


def test_data_path_and_train_data_path_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        build_megatron_args(_cfg(data_path="/data/blob", split="99,1,0"))


def test_pre_split_mode_still_works_without_split():
    # Per-cluster ClimbLab: train/valid prefixes, no --split, no --data-path.
    args = build_megatron_args(_cfg(valid_data_path="/data/valid"))
    assert _pairs(args)["--train-data-path"] == "/data/train"
    assert "--split" not in args
    assert "--data-path" not in args


def test_resolved_absolutises_data_path(tmp_path):
    r = MoEPretrainConfig(data_path="artifacts/blob", split="99,1,0").resolved(tmp_path)
    assert r.data_path == str(tmp_path / "artifacts/blob")


def test_build_megatron_args_carries_moe_and_tokenizer():
    cfg = _cfg(num_experts=8, moe_router_topk=2, moe_aux_loss_coeff=0.01)
    pairs = _pairs(build_megatron_args(cfg))
    assert pairs["--num-experts"] == "8"
    assert pairs["--moe-router-topk"] == "2"
    assert pairs["--moe-router-load-balancing-type"] == "aux_loss"
    assert pairs["--moe-aux-loss-coeff"] == "0.01"
    assert pairs["--tokenizer-type"] == "NullTokenizer"
    assert pairs["--vocab-size"] == "50257"  # NullTokenizer eod = 50256 = <|endoftext|>
    assert pairs["--transformer-impl"] == "transformer_engine"
    assert pairs["--attention-backend"] == "auto"


def test_architecture_flags_always_emitted_at_megatron_defaults():
    # These define the network a checkpoint loads into, so they belong in the frozen launch command
    # even when they match Megatron's own defaults. An omitted flag is an undocumented default.
    pairs = _pairs(build_megatron_args(_cfg()))
    assert pairs["--position-embedding-type"] == "learned_absolute"
    assert pairs["--normalization"] == "LayerNorm"
    assert pairs["--norm-epsilon"] == "1e-05"


def test_architecture_flags_carry_the_reference_architecture():
    # base_cluster.yaml's parity setting is RoPE, which has no learned position table, plus
    # RMSNorm, which has no bias. This bare _cfg() call still gets the dataclass default for
    # norm_epsilon (1e-5), unrelated to what base_cluster.yaml itself sets.
    cfg = _cfg(position_embedding_type="rope", normalization="RMSNorm")
    pairs = _pairs(build_megatron_args(cfg))
    assert pairs["--position-embedding-type"] == "rope"
    assert pairs["--normalization"] == "RMSNorm"
    assert pairs["--norm-epsilon"] == "1e-05"
    assert _pairs(build_megatron_args(_cfg(norm_epsilon=1e-6)))["--norm-epsilon"] == "1e-06"


def test_moe_ffn_hidden_size_emitted_only_when_set():
    # Left unset, Megatron inherits ffn_hidden_size with only a warning, which is ambiguous for
    # reproduction once a dense layer exists. Real runs therefore set it explicitly.
    assert "--moe-ffn-hidden-size" not in build_megatron_args(_cfg())
    on = build_megatron_args(_cfg(moe_ffn_hidden_size=1024))
    assert _pairs(on)["--moe-ffn-hidden-size"] == "1024"


def test_shared_experts_and_dense_layer_off_by_default():
    # Nothing architectural may leak into the args unless a config asks for it.
    off = build_megatron_args(_cfg())
    assert "--moe-shared-expert-intermediate-size" not in off
    assert "--moe-layer-freq" not in off
    assert "--swiglu" not in off


def test_shared_experts_dense_layer_and_swiglu_route_through_when_set():
    # The documented full-FLAME 64/6/2 arm, whose 2 shared experts are expressed as one 2x-wide
    # expert.
    cfg = _cfg(
        moe_router_topk=6,
        moe_shared_expert_intermediate_size=1408,
        moe_layer_freq="[0]*1+[1]*8",
        swiglu=True,
    )
    args = build_megatron_args(cfg)
    pairs = _pairs(args)
    assert pairs["--moe-router-topk"] == "6"
    assert pairs["--moe-shared-expert-intermediate-size"] == "1408"
    assert pairs["--moe-layer-freq"] == "[0]*1+[1]*8"
    assert "--swiglu" in args


def test_untie_embeddings_opt_in():
    # Tied is Megatron's default and stays the default here. Untying adds a second V_pad x d
    # tensor.
    assert "--untie-embeddings-and-output-weights" not in build_megatron_args(_cfg())
    on = build_megatron_args(_cfg(untie_embeddings_and_output_weights=True))
    assert "--untie-embeddings-and-output-weights" in on


def test_wsd_schedule_flags_emitted_together():
    cfg = _cfg(lr_decay_style="WSD", lr_wsd_decay_iters=2000, train_iters=20000)
    pairs = _pairs(build_megatron_args(cfg))
    assert pairs["--lr-decay-style"] == "WSD"
    assert pairs["--lr-wsd-decay-iters"] == "2000"
    assert pairs["--lr-wsd-decay-style"] == "exponential"  # Megatron's default shape
    # Left unset, Megatron defaults lr_decay_iters to train_iters, so it must not be emitted.
    assert "--lr-decay-iters" not in build_megatron_args(cfg)


def test_wsd_without_decay_iters_fails_loud():
    # Megatron asserts deep inside the scheduler build; catch it here instead, at config time.
    with pytest.raises(ValueError, match="lr_wsd_decay_iters"):
        build_megatron_args(_cfg(lr_decay_style="WSD", lr_wsd_decay_iters=None))


def test_branch_anneal_flags():
    # A branch anneal is the one case that legitimately re-sizes the schedule on resume. It has a
    # shorter horizon than the trunk, plus the override that lets Megatron accept it.
    cfg = _cfg(
        load="/trunk/checkpoints",
        train_iters=5500,
        lr_decay_iters=5500,
        lr_decay_style="WSD",
        lr_wsd_decay_iters=500,
        override_opt_param_scheduler=True,
    )
    args = build_megatron_args(cfg)
    pairs = _pairs(args)
    assert pairs["--lr-decay-iters"] == "5500"
    assert pairs["--lr-wsd-decay-iters"] == "500"
    assert "--override-opt-param-scheduler" in args
    # anneal begins at lr_decay_iters - lr_wsd_decay_iters, i.e. exactly the trunk checkpoint
    assert cfg.lr_decay_iters - cfg.lr_wsd_decay_iters == 5000
    assert "--override-opt-param-scheduler" not in build_megatron_args(_cfg())


def test_router_pre_softmax_and_dtype_opt_in():
    off = build_megatron_args(_cfg())
    assert "--moe-router-pre-softmax" not in off
    assert "--moe-router-dtype" not in off
    on = build_megatron_args(_cfg(moe_router_pre_softmax=True, moe_router_dtype="fp32"))
    assert "--moe-router-pre-softmax" in on
    assert _pairs(on)["--moe-router-dtype"] == "fp32"


def test_grouped_gemm_and_distributed_optimizer_opt_in():
    # Both are pure throughput and memory levers with no effect on routing or loss, but grouped
    # GEMM swaps the expert module, so it has to be recorded in the launch command.
    off = build_megatron_args(_cfg())
    assert "--moe-grouped-gemm" not in off
    assert "--use-distributed-optimizer" not in off
    on = build_megatron_args(_cfg(moe_grouped_gemm=True, use_distributed_optimizer=True))
    assert "--moe-grouped-gemm" in on
    assert "--use-distributed-optimizer" in on


def test_base_cluster_config_pins_the_reference_architecture():
    # Guards the committed cluster config rather than only the plumbing. Reverting any of these to
    # a Megatron default changes the model, or its parameter count, without changing the arms.
    cfg = MoEPretrainConfig.from_yaml(_CONFIGS / "base_cluster.yaml")
    assert cfg.position_embedding_type == "rope"  # else +2.10M learned position table
    assert cfg.normalization == "RMSNorm"
    assert cfg.norm_epsilon == 1.0e-6  # matches FLAME's own training epsilon
    assert not cfg.add_bias_linear  # else +1.22M linear biases
    assert cfg.swiglu
    assert cfg.moe_layer_freq == "[0]*1+[1]*8"  # 1 dense + 8 MoE, FLAME's own pattern
    # The two FFN widths are unrelated once layer 0 is dense. 5472 is that layer and 704 is the
    # experts. Left unset, moe_ffn_hidden_size would inherit 5472 and widen every expert 7.8x.
    assert cfg.ffn_hidden_size == 5472
    assert cfg.moe_ffn_hidden_size == 704
    assert cfg.untie_embeddings_and_output_weights
    # Renormalised combine weights, which is Megatron's default and matches OLMoE, the model Exp
    # 2's combine-weight ablation runs on. That keeps one convention throughout the thesis.
    assert not cfg.moe_router_pre_softmax
    assert cfg.moe_router_dtype == "fp32"
    assert cfg.moe_grouped_gemm
    assert cfg.use_distributed_optimizer
    # WSD, so a longer horizon can pick this run's stable phase back up (extended_budget_cluster).
    assert cfg.lr_decay_style == "WSD"
    assert cfg.lr_wsd_decay_iters == 500
    assert cfg.lr_decay_iters is None  # defaults to train_iters; only a re-horizoned run overrides
    assert not cfg.override_opt_param_scheduler  # the primary run never re-sizes its own schedule
    # FLAME's token budget: 5500 * 1024 * 2048 = 11.53B, against their 11.4B.
    assert cfg.train_iters * cfg.global_batch_size * cfg.seq_length == 11_534_336_000
    assert cfg.global_batch_size == 1024  # FLAME's, reached by grad accumulation (not extra memory)
    # Megatron has no unconditional end-of-training save, so this is what guarantees the final
    # annealed checkpoint exists, and that the branch point extended_budget resumes from does too.
    assert cfg.train_iters % cfg.save_interval == 0
    assert (cfg.train_iters - cfg.lr_wsd_decay_iters) % cfg.save_interval == 0
    # Grad accumulation must come out whole: global / (micro * data-parallel ranks).
    assert cfg.global_batch_size % (cfg.micro_batch_size * 4) == 0
    # No shared experts is the one remaining deviation from FLAME, and top-8 is what pays for it.
    assert cfg.moe_shared_expert_intermediate_size is None
    assert cfg.moe_router_topk == 8
    # At EP=1 every rank holds every expert, so per-expert load logging stays exact.
    assert cfg.expert_model_parallel_size == 1
    # The common probe grid is structural: every arm shares one coarse interval so their dumps
    # land on identical steps. Pinning it here guards against a config drift.
    assert cfg.moe_probe_coarse_interval == 250


def test_base_cluster_active_params_match_flame_exactly():
    # topk is 8 rather than FLAME's 6 because dropping their 2 shared experts and spending that
    # capacity on 2 more routed experts is compute-neutral, which makes having no shared experts a
    # pure change of routing structure. This catches a width or topk retuned without the other.
    cfg = MoEPretrainConfig.from_yaml(_CONFIGS / "base_cluster.yaml")
    d, num_moe_layers = cfg.hidden_size, 8
    gated_ffn = 3 * d * cfg.moe_ffn_hidden_size  # fc1 d->2f plus fc2 f->d

    ours = num_moe_layers * cfg.moe_router_topk * gated_ffn
    flame = num_moe_layers * (6 * gated_ffn + 3 * d * 1408)  # 6 routed + one 2x-wide shared
    assert ours == flame

    # Full active-minus-embedding count, which is the number to quote against their 290M.
    common = 9 * 4 * d * d + 3 * d * cfg.ffn_hidden_size + num_moe_layers * d * cfg.num_experts
    assert common + ours + (2 * 9 + 1) * d == 193_514_496


def test_local_config_mirrors_the_cluster_architecture():
    # The local smoke config earns its keep only if it exercises the cluster run's code paths at a
    # size that fits one dev GPU. Scale is free to differ, but the architectural switches that
    # select those code paths are not. If they drift, a green local run stops being evidence about
    # the cluster run.
    local = MoEPretrainConfig.from_yaml(_CONFIGS / "base_local.yaml")
    cluster = MoEPretrainConfig.from_yaml(_CONFIGS / "base_cluster.yaml")
    for field in (
        "position_embedding_type",
        "normalization",
        "swiglu",
        "untie_embeddings_and_output_weights",
        "moe_layer_freq",
        "moe_router_pre_softmax",
        "moe_router_dtype",
        "moe_grouped_gemm",
        "moe_router_score_function",
        "lr_decay_style",
        "use_distributed_optimizer",
        "transformer_impl",
    ):
        assert getattr(local, field) == getattr(cluster, field), field
    # Its own (much shorter) horizon still has to be internally consistent.
    assert local.lr_wsd_decay_iters is not None  # WSD without it trips Megatron's assert
    assert local.train_iters % local.save_interval == 0  # else no final annealed checkpoint
    assert local.moe_layer_freq is None or local.num_layers == 9  # pattern length == num_layers


def test_build_megatron_args_disables_apex_megatron_fusions():
    # Under TE these apex/Megatron fusion paths stay off; TE fuses its own.
    args = build_megatron_args(_cfg())
    assert "--no-persist-layer-norm" in args
    assert "--no-gradient-accumulation-fusion" in args


def test_build_megatron_args_attention_backend_overridable():
    args = build_megatron_args(_cfg(attention_backend="unfused"))
    assert _pairs(args)["--attention-backend"] == "unfused"


def test_build_megatron_args_toggles_optional_flags():
    on = build_megatron_args(_cfg(bf16=True, save="ckpt", save_interval=10))
    assert "--bf16" in on
    assert _pairs(on)["--save"] == "ckpt"
    assert _pairs(on)["--save-interval"] == "10"

    off = build_megatron_args(_cfg(bf16=False, save=None, save_interval=None))
    assert "--bf16" not in off
    assert "--save" not in off
    assert "--save-interval" not in off


def test_build_megatron_args_emits_load_when_set():
    on = build_megatron_args(_cfg(load="/ckpt/dir"))
    assert _pairs(on)["--load"] == "/ckpt/dir"
    assert "--load" not in build_megatron_args(_cfg(load=None))


def test_build_megatron_args_emits_ckpt_step_to_pin_iteration():
    # load points at the checkpoints directory, and ckpt_step selects which iter_<N>/ inside it.
    on = build_megatron_args(_cfg(load="/ckpt/dir", ckpt_step=200))
    assert _pairs(on)["--ckpt-step"] == "200"
    assert "--ckpt-step" not in build_megatron_args(_cfg(load="/ckpt/dir"))


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


def test_resolve_run_dir_fresh_run_uses_the_tag():
    assert resolve_run_dir("/art/e1", run_tag="trunk") == Path("/art/e1/trunk")
    with pytest.raises(ValueError, match="run_tag"):
        resolve_run_dir("/art/e1")


def test_resolve_run_dir_plain_resume_continues_in_place(tmp_path):
    # A sliced run is the same model continuing, so it writes back into its own dir, giving one
    # train.log, one checkpoints dir and, through run_dir.name, one unbroken W&B curve.
    trunk = tmp_path / "trunk"
    (trunk / "checkpoints").mkdir(parents=True)
    assert resolve_run_dir(tmp_path, load=str(trunk / "checkpoints")) == trunk


def test_resolve_run_dir_branch_gets_its_own_dir(tmp_path):
    # A WSD branch reads the trunk's checkpoints but is a different lineage, so it must not land in
    # the trunk's dir, where it would claim the same W&B step numbers with different weights.
    trunk = tmp_path / "trunk"
    (trunk / "checkpoints").mkdir(parents=True)
    branch = resolve_run_dir(
        tmp_path,
        run_dir="flame-budget",
        load=str(trunk / "checkpoints"),
        is_branch=True,
    )
    assert branch == tmp_path / "flame-budget"
    assert branch.name != trunk.name  # distinct WANDB_RUN_ID


def test_resolve_run_dir_rejects_a_branch_writing_into_its_parent(tmp_path):
    # Forgetting --run-dir on a branch would move the trunk's latest_checkpointed_iteration.txt
    # onto annealed, dead-end weights, so the next trunk resume would silently continue from cooled
    # weights. Fail loud instead.
    trunk = tmp_path / "trunk"
    (trunk / "checkpoints").mkdir(parents=True)
    with pytest.raises(ValueError, match="branched from"):
        resolve_run_dir(tmp_path, load=str(trunk / "checkpoints"), is_branch=True)
    # ...but pointing it somewhere else is fine, including via an absolute path.
    out = resolve_run_dir(
        tmp_path,
        run_dir=str(tmp_path / "elsewhere"),
        load=str(trunk / "checkpoints"),
        is_branch=True,
    )
    assert out == tmp_path / "elsewhere"


def test_resolve_run_dir_expands_env_vars(tmp_path, monkeypatch):
    # The same contract as config-file paths: ${VAR} and ~ expand, and an unset variable fails
    # loud. Without it a literal "${SCRATCH}" is a legal directory name, so the run would create it
    # and write a whole training run to the wrong filesystem without erroring.
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    out = resolve_run_dir("/art/e1", run_dir="${SCRATCH}/myrun")
    assert out == tmp_path / "scratch/myrun"
    assert "$" not in str(out)

    trunk = tmp_path / "scratch" / "trunk"
    (trunk / "checkpoints").mkdir(parents=True)
    assert resolve_run_dir("/art/e1", load="${SCRATCH}/trunk/checkpoints") == trunk

    monkeypatch.delenv("SCRATCH")
    with pytest.raises(ValueError, match="unresolved environment variable"):
        resolve_run_dir("/art/e1", run_dir="${SCRATCH}/myrun")
    with pytest.raises(ValueError, match="unresolved environment variable"):
        resolve_run_dir("/art/e1", load="${SCRATCH}/trunk/checkpoints")


def test_resolve_run_dir_expansion_is_idempotent(tmp_path, monkeypatch):
    # The launcher expands --load before passing it, so resolve_run_dir sees an already-expanded
    # path. Expanding again must be a no-op rather than a second round of substitution.
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    once = resolve_run_dir("/art/e1", run_dir="${SCRATCH}/myrun")
    assert resolve_run_dir("/art/e1", run_dir=str(once)) == once


def test_resolve_run_dir_names_a_fresh_run_without_a_load():
    # --run-dir also replaces the timestamp on a fresh run, which is how the trunk gets a stable
    # name for a later --load path to point at.
    assert resolve_run_dir("/art/e1", run_dir="trunk", run_tag="20260101-000000") == Path(
        "/art/e1/trunk"
    )


def test_build_launch_command_wraps_torch_distributed_run():
    cfg = _cfg()
    cmd = build_launch_command(cfg, "/repo/Megatron-LM/pretrain_gpt.py", nproc=1)
    # Launched via `python -m torch.distributed.run` so that workers inherit sys.executable, the
    # venv python on the cluster, rather than the torch-shipped `torchrun` script under the system
    # python.
    assert cmd[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--standalone" in cmd  # single-node default
    assert "--rdzv-endpoint" not in cmd
    assert cmd[_pairs_index(cmd, "--nproc-per-node")] == "1"
    assert "/repo/Megatron-LM/pretrain_gpt.py" in cmd
    assert "--num-experts" in cmd  # megatron args appended after the script


def test_build_launch_command_multinode_uses_c10d_rendezvous():
    cfg = _cfg()
    cmd = build_launch_command(
        cfg, "/repo/Megatron-LM/pretrain_gpt.py", nproc=4, nnodes=2, rdzv_endpoint="nid001:25678"
    )
    assert "--standalone" not in cmd
    assert cmd[_pairs_index(cmd, "--nnodes")] == "2"
    assert cmd[_pairs_index(cmd, "--nproc-per-node")] == "4"
    assert cmd[_pairs_index(cmd, "--rdzv-backend")] == "c10d"
    assert cmd[_pairs_index(cmd, "--rdzv-endpoint")] == "nid001:25678"
    assert cmd[_pairs_index(cmd, "--max-restarts")] == "0"


def test_build_launch_command_multinode_requires_endpoint():
    with pytest.raises(ValueError, match="rdzv_endpoint"):
        build_launch_command(_cfg(), "/repo/Megatron-LM/pretrain_gpt.py", nproc=4, nnodes=2)


def _pairs_index(args: list[str], flag: str) -> int:
    return args.index(flag) + 1


def test_base_cluster_emits_no_rosenthal_flags_and_keeps_its_own_balancing_type():
    # base_cluster.yaml never opts into a rosenthal balancing type, so every rosenthal flag is
    # optional plumbing that has to stay silent for it.
    cfg = MoEPretrainConfig.from_yaml(_CONFIGS / "base_cluster.yaml")
    args = build_megatron_args(cfg)
    pairs = _pairs(args)
    assert not any(flag.startswith("--moe-rosenthal-") for flag in pairs)
    assert "--moe-rosenthal-log-grad-ratio" not in args
    assert pairs["--moe-router-load-balancing-type"] == cfg.moe_router_load_balancing_type


def test_global_rosenthal_emits_the_three_value_flags():
    # variant is "hard" here. test_global_rosenthal_soft_no_longer_rejected below covers "soft"
    # with global_rosenthal, which the retired rule 5 used to reject.
    cfg = _cfg(
        moe_router_load_balancing_type="global_rosenthal",
        moe_rosenthal_variant="hard",
        moe_rosenthal_cost="quadratic",
        moe_rosenthal_lambda=0.25,
    )
    args = build_megatron_args(cfg)
    pairs = _pairs(args)
    assert pairs["--moe-rosenthal-variant"] == "hard"
    assert pairs["--moe-rosenthal-cost"] == "quadratic"
    assert pairs["--moe-rosenthal-lambda"] == "0.25"
    assert "--moe-rosenthal-log-grad-ratio" not in args


def test_rosenthal_log_grad_ratio_flag_only_when_true():
    on = build_megatron_args(
        _cfg(
            moe_router_load_balancing_type="rosenthal",
            moe_rosenthal_log_grad_ratio=True,
        )
    )
    assert "--moe-rosenthal-log-grad-ratio" in on
    off = build_megatron_args(_cfg(moe_router_load_balancing_type="rosenthal"))
    assert "--moe-rosenthal-log-grad-ratio" not in off


def test_rosenthal_flags_absent_for_non_rosenthal_types():
    args = build_megatron_args(_cfg(moe_router_load_balancing_type="aux_loss"))
    assert "--moe-rosenthal-variant" not in args
    assert "--moe-rosenthal-cost" not in args
    assert "--moe-rosenthal-lambda" not in args


def test_rosenthal_rule1_rejects_unknown_variant():
    with pytest.raises(ValueError, match="moe_rosenthal_variant"):
        build_megatron_args(
            _cfg(moe_router_load_balancing_type="rosenthal", moe_rosenthal_variant="medium")
        )


def test_rosenthal_rule2_rejects_unknown_cost_family():
    with pytest.raises(ValueError, match="moe_rosenthal_cost"):
        build_megatron_args(
            _cfg(moe_router_load_balancing_type="rosenthal", moe_rosenthal_cost="cubic")
        )


def test_rosenthal_rule3_rejects_nonpositive_lambda():
    with pytest.raises(ValueError, match="moe_rosenthal_lambda"):
        build_megatron_args(
            _cfg(moe_router_load_balancing_type="rosenthal", moe_rosenthal_lambda=0.0)
        )
    with pytest.raises(ValueError, match="moe_rosenthal_lambda"):
        build_megatron_args(
            _cfg(moe_router_load_balancing_type="rosenthal", moe_rosenthal_lambda=-1.0)
        )


def test_rosenthal_lambda_explicit_value_passes_through_unchanged():
    args = build_megatron_args(
        _cfg(
            moe_router_load_balancing_type="rosenthal",
            moe_rosenthal_cost="quadratic",
            moe_rosenthal_lambda=0.9,
        )
    )
    assert _pairs(args)["--moe-rosenthal-lambda"] == "0.9"


def test_rosenthal_lambda_omitted_resolves_to_the_cost_familys_default():
    # moe_rosenthal_lambda defaults to None, meaning "use this cost family's own default", rather
    # than to a family-independent literal. A config selecting quadratic and omitting lambda must
    # not silently get linear's default, which is twice the slope-matched value.
    for cost_family, default in DEFAULT_LAMBDA.items():
        args = build_megatron_args(
            _cfg(moe_router_load_balancing_type="rosenthal", moe_rosenthal_cost=cost_family)
        )
        assert _pairs(args)["--moe-rosenthal-lambda"] == str(default)


def test_rosenthal_rule4_requires_softmax_score_function():
    with pytest.raises(ValueError, match="moe_router_score_function"):
        build_megatron_args(
            _cfg(
                moe_router_load_balancing_type="rosenthal",
                moe_router_score_function="sigmoid",
            )
        )


def test_global_rosenthal_soft_no_longer_rejected():
    # Rules 5 and 6 are retired. The synced-coefficient construction makes 'soft' correct at any
    # reduce-group size, so global_rosenthal with soft must parse and emit the same three value
    # flags as every other rosenthal arm, with no validation error.
    args = build_megatron_args(
        _cfg(moe_router_load_balancing_type="global_rosenthal", moe_rosenthal_variant="soft")
    )
    assert _pairs(args)["--moe-rosenthal-variant"] == "soft"


def test_rosenthal_soft_no_longer_requires_tensor_model_parallel_size_one():
    args = build_megatron_args(
        _cfg(
            moe_router_load_balancing_type="rosenthal",
            moe_rosenthal_variant="soft",
            tensor_model_parallel_size=2,
        )
    )
    assert _pairs(args)["--moe-rosenthal-variant"] == "soft"


def test_rosenthal_rule7_log_grad_ratio_requires_rosenthal_type():
    with pytest.raises(ValueError, match="moe_rosenthal_log_grad_ratio"):
        build_megatron_args(
            _cfg(
                moe_router_load_balancing_type="aux_loss",
                moe_rosenthal_log_grad_ratio=True,
            )
        )
    # both rosenthal names are accepted
    for balancing_type in ("rosenthal", "global_rosenthal"):
        args = build_megatron_args(
            _cfg(
                moe_router_load_balancing_type=balancing_type,
                moe_rosenthal_log_grad_ratio=True,
            )
        )
        assert "--moe-rosenthal-log-grad-ratio" in args


def test_rosenthal_rule8_warns_rather_than_raises_over_the_pressure_bound():
    # At E=64, K=2 (the _cfg default topk), linear (p=1) and hard, the bound is
    # lambda*alpha*(E/K) = 5*0.01*32 = 1.6, which exceeds 1.
    with pytest.warns(UserWarning, match=r"\(num_experts/moe_router_topk\)"):
        args = build_megatron_args(
            _cfg(
                moe_router_load_balancing_type="rosenthal",
                moe_rosenthal_variant="hard",
                moe_rosenthal_lambda=5.0,
                num_experts=64,
                moe_router_topk=2,
                moe_aux_loss_coeff=0.01,
            )
        )
    # It is a warning rather than an error, so the flags still come through.
    assert _pairs(args)["--moe-rosenthal-lambda"] == "5.0"


def test_rosenthal_rule8_soft_variant_uses_the_soft_bound():
    # The same lambda, alpha, E and K as the hard case above, but soft's bound is E**p rather than
    # (E/K)**p, so it warns at a much smaller lambda and the message must say which bound it used.
    # soft's relative load is softmax mass, uncapped by top-k selection, so printing the hard
    # expression instead would under-warn by K**p.
    with pytest.warns(UserWarning, match=r"num_experts\*\*"):
        args = build_megatron_args(
            _cfg(
                moe_router_load_balancing_type="rosenthal",
                moe_rosenthal_variant="soft",
                moe_rosenthal_lambda=5.0,
                num_experts=64,
                moe_router_topk=2,
                moe_aux_loss_coeff=0.01,
            )
        )
    assert _pairs(args)["--moe-rosenthal-lambda"] == "5.0"


def test_unknown_balancing_type_rejected():
    with pytest.raises(ValueError, match="rosethal"):
        build_megatron_args(_cfg(moe_router_load_balancing_type="rosethal"))


@pytest.mark.parametrize(
    "balancing_type",
    [
        "aux_loss",
        "seq_aux_loss",
        "global_aux_loss",
        "sinkhorn",
        "none",
        "rosenthal",
        "global_rosenthal",
    ],
)
def test_all_seven_allowed_balancing_types_accepted(balancing_type):
    # Emission only. Some of these, sinkhorn for one, would fail Megatron's own aux-loss assert on
    # base_cluster.yaml's other settings, but that is Megatron's concern rather than this schema's:
    # build_megatron_args must not reject a name Megatron accepts.
    args = build_megatron_args(_cfg(moe_router_load_balancing_type=balancing_type))
    assert _pairs(args)["--moe-router-load-balancing-type"] == balancing_type


def test_probe_flags_absent_by_default():
    # moe_probe_coarse_interval defaults to 0, so an existing config's emitted command must not
    # gain any of the six probe flags just by upgrading to a version of this schema that knows
    # about them.
    args = build_megatron_args(_cfg())
    for flag in (
        "--moe-probe-batch",
        "--moe-probe-coarse-interval",
        "--moe-probe-dense-interval",
        "--moe-probe-dense-windows",
        "--moe-probe-seqs",
        "--moe-probe-dir",
    ):
        assert flag not in args


def test_probe_emits_all_flags_when_enabled():
    args = build_megatron_args(
        _cfg(
            moe_probe_batch="assets/probe/dev.npz",
            moe_probe_coarse_interval=250,
            moe_probe_dense_interval=25,
            moe_probe_dense_windows=["0:500"],
            moe_probe_seqs=8,
            moe_probe_dir="/run/probes",
        )
    )
    pairs = _pairs(args)
    assert pairs["--moe-probe-batch"] == "assets/probe/dev.npz"
    assert pairs["--moe-probe-coarse-interval"] == "250"
    assert pairs["--moe-probe-dense-interval"] == "25"
    assert pairs["--moe-probe-seqs"] == "8"
    assert pairs["--moe-probe-dir"] == "/run/probes"
    i = args.index("--moe-probe-dense-windows")
    assert args[i + 1 : i + 2] == ["0:500"]


def test_probe_requires_an_asset_when_enabled():
    with pytest.raises(ValueError, match="moe_probe_batch"):
        build_megatron_args(_cfg(moe_probe_coarse_interval=250))


def test_probe_dense_settings_without_coarse_interval_are_rejected():
    # moe_probe_coarse_interval == 0 skips the whole probe block, so dense settings left on their
    # own would be silently dropped. Refuse the combination outright instead.
    with pytest.raises(ValueError, match="moe_probe_coarse_interval"):
        build_megatron_args(_cfg(moe_probe_dense_interval=25))
    with pytest.raises(ValueError, match="moe_probe_coarse_interval"):
        build_megatron_args(_cfg(moe_probe_dense_windows=["0:500"]))


def test_probe_refuses_context_parallel_size_when_the_field_is_present():
    # context_parallel_size is not a MoEPretrainConfig field today (pinned below), so this
    # simulates a future config that adds it, via object.__setattr__ on the frozen dataclass.
    cfg = _cfg(moe_probe_batch="assets/probe/dev.npz", moe_probe_coarse_interval=250)
    object.__setattr__(cfg, "context_parallel_size", 2)
    with pytest.raises(ValueError, match="context_parallel_size"):
        build_megatron_args(cfg)


def test_probe_refuses_moe_input_jitter_eps_when_the_field_is_present():
    cfg = _cfg(moe_probe_batch="assets/probe/dev.npz", moe_probe_coarse_interval=250)
    object.__setattr__(cfg, "moe_input_jitter_eps", 0.01)
    with pytest.raises(ValueError, match="moe_input_jitter_eps"):
        build_megatron_args(cfg)


def test_probe_refuses_moe_expert_capacity_factor_when_the_field_is_present():
    cfg = _cfg(moe_probe_batch="assets/probe/dev.npz", moe_probe_coarse_interval=250)
    object.__setattr__(cfg, "moe_expert_capacity_factor", 1.5)
    with pytest.raises(ValueError, match="moe_expert_capacity_factor"):
        build_megatron_args(cfg)


def test_probe_refuses_cuda_graphs_when_the_field_is_present():
    cfg = _cfg(moe_probe_batch="assets/probe/dev.npz", moe_probe_coarse_interval=250)
    object.__setattr__(cfg, "cuda_graph_impl", "local")
    with pytest.raises(ValueError, match="cuda_graph_impl"):
        build_megatron_args(cfg)


def test_probe_refusal_fields_are_absent_from_the_schema():
    # The four checks above read these via getattr(cfg, name, default) precisely because none is
    # a real field. This test fires if any of these names is ever added.
    field_names = {f.name for f in dataclasses.fields(MoEPretrainConfig)}
    for absent in (
        "context_parallel_size",
        "moe_input_jitter_eps",
        "moe_expert_capacity_factor",
        "cuda_graph_impl",
    ):
        assert absent not in field_names


def test_probe_requires_tensor_and_pipeline_parallel_size_one():
    with pytest.raises(ValueError, match="tensor_model_parallel_size"):
        build_megatron_args(
            _cfg(
                moe_probe_batch="assets/probe/dev.npz",
                moe_probe_coarse_interval=250,
                tensor_model_parallel_size=2,
            )
        )
    with pytest.raises(ValueError, match="pipeline_model_parallel_size"):
        build_megatron_args(
            _cfg(
                moe_probe_batch="assets/probe/dev.npz",
                moe_probe_coarse_interval=250,
                pipeline_model_parallel_size=2,
            )
        )


def test_probe_seqs_must_divide_micro_batch_size():
    with pytest.raises(ValueError, match="moe_probe_seqs"):
        build_megatron_args(
            _cfg(
                moe_probe_batch="assets/probe/dev.npz",
                moe_probe_coarse_interval=250,
                moe_probe_seqs=7,
                micro_batch_size=4,
            )
        )


def test_probe_dense_interval_required_whenever_a_window_exists():
    with pytest.raises(ValueError, match="moe_probe_dense_interval"):
        build_megatron_args(
            _cfg(
                moe_probe_batch="assets/probe/dev.npz",
                moe_probe_coarse_interval=250,
                moe_probe_dense_windows=["0:500"],
            )
        )


def test_probe_windows_must_be_sorted_and_non_overlapping():
    with pytest.raises(ValueError, match="sorted and non-overlapping"):
        build_megatron_args(
            _cfg(
                moe_probe_batch="assets/probe/dev.npz",
                moe_probe_coarse_interval=250,
                moe_probe_dense_interval=25,
                moe_probe_dense_windows=["500:600", "0:100"],
            )
        )
    with pytest.raises(ValueError, match="sorted and non-overlapping"):
        build_megatron_args(
            _cfg(
                moe_probe_batch="assets/probe/dev.npz",
                moe_probe_coarse_interval=250,
                moe_probe_dense_interval=25,
                moe_probe_dense_windows=["0:500", "400:600"],
            )
        )


def test_probe_windows_malformed_spec_rejected():
    with pytest.raises(ValueError, match="malformed"):
        build_megatron_args(
            _cfg(
                moe_probe_batch="assets/probe/dev.npz",
                moe_probe_coarse_interval=250,
                moe_probe_dense_interval=25,
                moe_probe_dense_windows=["0-500"],
            )
        )


def test_resolved_absolutises_moe_probe_paths(tmp_path):
    r = MoEPretrainConfig(
        train_data_path="/data/train",
        moe_probe_batch="assets/probe/dev.npz",
        moe_probe_dir="probes",
    ).resolved(tmp_path)
    assert r.moe_probe_batch == str(tmp_path / "assets/probe/dev.npz")
    assert r.moe_probe_dir == str(tmp_path / "probes")


def test_pretrain_config_module_imports_no_torch():
    # pretrain_config.py must stay torch-free, because --dry-run runs on a cluster login node
    # where importing torch costs several seconds for no reason. Run in a subprocess, since
    # in-process torch may already have been loaded by another test module in this session.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import moe_congestion_routing.training.pretrain_config; "
            "assert 'torch' not in sys.modules, sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

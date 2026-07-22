"""Run config for a MoE pretraining run through Megatron's ``pretrain_gpt.py``."""

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

# Key a config file uses to name its base config(s). Consumed by the loader (never a
# MoEPretrainConfig field), so it is stripped before the dataclass is constructed.
_EXTENDS_KEY = "extends"


def _load_yaml_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    """Load a yaml mapping, resolving an optional ``extends:`` chain into one merged dict.

    Bases are merged first (in listed order, each recursively resolved), then the current
    file's own keys override them. ``extends`` paths are relative to the file that declares
    them. Cycles raise rather than recurse forever.
    """
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, path))
        raise ValueError(f"circular config extends chain: {chain}")

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a valid yaml mapping, got {type(data).__name__}")

    bases = data.pop(_EXTENDS_KEY, None)
    if bases is None:
        return data

    base_paths = [bases] if isinstance(bases, str) else bases
    merged: dict = {}
    for base in base_paths:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged.update(_load_yaml_with_extends(base_path, (*_seen, path)))
    merged.update(data)  # this file's own keys win over everything it extends
    return merged


@dataclass(frozen=True)
class MoEPretrainConfig:
    """Everything needed to launch one MoE pretraining run, loadable from a yaml file."""

    num_layers: int = 4
    """Number of transformer layers."""

    hidden_size: int = 256
    """Model/embedding hidden dimension."""

    num_attention_heads: int = 8
    """Number of attention heads."""

    ffn_hidden_size: int = 512
    """Inner dimension of each expert's MLP."""

    seq_length: int = 512
    """Training sequence length (also used for max position embeddings)."""

    num_experts: int = 8
    """Number of routed experts."""

    moe_router_topk: int = 2
    """Experts activated per token."""

    moe_router_load_balancing_type: str = "aux_loss"
    """Load-balancing strategy; ``aux_loss`` is the vanilla Switch auxiliary loss."""

    moe_aux_loss_coeff: float = 0.01
    """Aux-loss weight. Megatron's default is 0.0, which makes the aux loss a no-op."""

    moe_router_score_function: str = "softmax"
    """Router scoring: ``softmax`` or ``sigmoid`` (Deepseek-V3 style). Megatron REQUIRES sigmoid
    whenever expert bias is on, so ALF-LB must use ``sigmoid`` (or sqrtsoftplus)."""

    moe_router_enable_expert_bias: bool = False
    """ALF-LB / aux-loss-free load balancing: maintain a per-expert selection bias updated
    from realized load (Megatron's built-in ``sign(mean_load - load)`` rule). Combine weights
    stay unbiased. Requires ``moe_router_score_function`` in {sigmoid, sqrtsoftplus}."""

    moe_router_bias_update_rate: float = 1e-3
    """Step size for the expert-bias update (only used when expert bias is enabled)."""

    moe_z_loss_coeff: float | None = None
    """Router z-loss coefficient (ST-MoE). ``None`` disables it."""

    moe_per_layer_logging: bool = False
    """Also log every MoE metric per layer (``moe/<metric>_layer_<i>``), not just the layer-mean."""

    train_data_path: str | None = None
    """``.bin``/``.idx`` prefix for the training split."""

    valid_data_path: str | None = None
    """``.bin``/``.idx`` prefix for the validation split (pre-split mode only)."""

    data_path: str | None = None
    """Single ``.bin``/``.idx`` prefix - ONE blend that Megatron carves into train/valid/test at
    load time via ``split``. Mutually exclusive with ``train_data_path``."""

    split: str | None = None
    """Train/valid/test ratios for the ``data_path`` blob, e.g. ``"99,1,0"``. REQUIRED with
    ``data_path`` and forbidden with ``train_data_path`` (per-split paths are already split).
    Ensure valid splits are non-empty if you set ``eval_interval`` > 0, else eval crashes."""

    tokenizer_type: str = "NullTokenizer"
    """NullTokenizer(vocab_size) sets eod = vocab_size-1 = 50256 = <|endoftext|>, so no
    vocab/merge files are needed for the pre-tokenized data."""

    vocab_size: int = 50257
    """GPT-2 vocabulary size; drives the NullTokenizer eod id above."""

    # optimisation / schedule
    lr: float = 3.0e-4
    """Peak learning rate."""

    min_lr: float = 3.0e-5
    """Floor learning rate for the decay schedule."""

    lr_decay_style: str = "constant"
    """Learning-rate decay schedule."""

    lr_warmup_iters: int = 5
    """Linear warmup iterations before the decay schedule applies."""

    # batch / iterations
    micro_batch_size: int = 4
    """Samples per micro-batch (one forward/backward)."""

    global_batch_size: int = 8
    """Samples per optimizer step (across gradient accumulation / data parallelism)."""

    train_iters: int = 30
    """Total training iterations (optimizer steps)."""

    seed: int = 1234
    """RNG seed; also part of the dataset index cache key."""

    # eval
    eval_interval: int = 1000
    """Iterations between validation passes."""

    eval_iters: int = 0
    """Batches per validation pass; 0 disables eval entirely."""

    # checkpointing (Megatron semantics). save/load are DIRECTORIES, not single checkpoints: each
    # save drops an iter_<N>/ subdir + a latest_checkpointed_iteration.txt tracker inside.
    save: str | None = None
    """Directory to write checkpoints to; unset => the launcher uses
    ``<output_dir>/<timestamp>/checkpoints``."""

    save_interval: int | None = None
    """Iterations between saves and this harness's on-switch, unsetting it means no checkpoints."""

    load: str | None = None
    """Checkpoint DIRECTORY to resume/infer from; loads the newest ``iter_<N>/`` per the tracker."""

    ckpt_step: int | None = None
    """Load this iteration from ``load`` instead of the newest (200 => ``iter_0000200/``)."""

    # Use Transformer Engine (its fused attention/LayerNorm/Linear speedup the model).
    # Training and inference both need to use the same implementation, so a checkpoint never
    # crosses impls - a `local`-trained checkpoint is NOT loadable into a TE model.
    transformer_impl: str = "transformer_engine"
    """Megatron transformer implementation. ``transformer_engine`` uses TE's fused modules."""

    attention_backend: str = "auto"
    """TE attention backend: flash / fused / unfused / auto / local. ``auto`` lets TE pick."""

    # The Megatron/apex fusion paths below stay OFF: TE supplies its own fused kernels, and the
    # apex/prebuilt kernels these need aren't installed locally. (Harmless no-ops under TE.)
    persist_layer_norm: bool = False
    """Megatron's non-TE fused persistent LayerNorm; off (TE has its own)."""

    gradient_accumulation_fusion: bool = False
    """apex-fused gradient accumulation; off (apex absent locally; TE handles wgrad)."""

    masked_softmax_fusion: bool = False
    """Megatron's fused scaled masked softmax; off (kernel unbuilt; TE fuses attention)."""

    bias_gelu_fusion: bool = False
    """Megatron's fused bias+GELU; off (TE fuses the MLP activation)."""

    add_bias_linear: bool = False
    """Linear-layer bias. Off (modern default + what the reference uses); also sidesteps a Megatron
    in-place-on-view autograd error in the non-fused MoE expert bias path."""

    bf16: bool = True
    """Run in bfloat16."""

    tensor_model_parallel_size: int = 1
    """Tensor-model-parallel world size."""

    pipeline_model_parallel_size: int = 1
    """Pipeline-model-parallel world size."""

    expert_model_parallel_size: int = 1
    """Expert-parallel world size (MoE experts sharded across ranks)."""

    log_interval: int = 1
    """Iterations between training-log lines."""

    log_throughput: bool = False
    """Log Megatron's native per-GPU throughput (TFLOP/s/GPU). The tokens/s/GPU patch adds a
    complementary token-rate line; both are cheap and always worth having."""

    tensorboard_dir: str | None = None
    """TensorBoard log dir. ``None`` => the launcher derives ``<run_dir>/tensorboard``."""

    wandb_project: str | None = None
    """W&B project. Set => W&B on (logs to wandb.ai using WANDB_API_KEY from the env); unset =>
    off. The arm base configs set this, so a run logs to W&B with only WANDB_API_KEY in the env."""

    wandb_exp_name: str | None = None
    """W&B run name. Megatron requires a non-empty name whenever ``wandb_project`` is set; the
    launcher derives it from the config file stem + run timestamp when left unset."""

    wandb_entity: str | None = None
    """W&B entity (team/user). Optional; unset uses your default entity."""

    wandb_save_dir: str | None = None
    """Local dir for W&B run files. ``None`` => the launcher derives ``<run_dir>/wandb``."""

    data_cache_path: str | None = None
    """Dataset sample/shuffle index cache. ``None`` => ``<output_dir>/cache`` (derived in the
    launcher). Shared across runs (keyed by seed/seq_length) so the indices build once."""

    output_dir: str = "artifacts/runs"
    """Root for run artifacts. The launcher writes each run to its own ``<output_dir>/<timestamp>/``
    subdir (train.log, launch_command.txt, checkpoints); the dataset cache above is the one shared
    exception at ``<output_dir>/cache``."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MoEPretrainConfig":
        """Build from a yaml file. Unknown keys raise ``TypeError`` (fail loud).

        A file may declare ``extends: <path>`` (or a list of paths) to inherit from one or
        more base configs: bases are loaded first (recursively) and this file's own keys
        override them. Paths are resolved relative to the file that names them, so an arm
        delta like ``switch_local.yaml`` can carry only its balancing fields on top of a
        shared ``base_local.yaml``.
        """
        data = _load_yaml_with_extends(Path(path))
        return cls(**data)

    def resolved(self, repo_root: Path) -> "MoEPretrainConfig":
        """Absolutise all paths against ``repo_root`` and derive the data cache dir if unset."""

        def absolutise(p: str) -> str:
            path = Path(p)
            return str(path if path.is_absolute() else repo_root / path)

        output_dir = absolutise(self.output_dir)
        return replace(
            self,
            train_data_path=absolutise(self.train_data_path) if self.train_data_path else None,
            valid_data_path=absolutise(self.valid_data_path) if self.valid_data_path else None,
            data_path=absolutise(self.data_path) if self.data_path else None,
            output_dir=output_dir,
            data_cache_path=absolutise(self.data_cache_path)
            if self.data_cache_path
            else str(Path(output_dir) / "cache"),
            save=absolutise(self.save) if self.save else None,
            load=absolutise(self.load) if self.load else None,
            tensorboard_dir=absolutise(self.tensorboard_dir) if self.tensorboard_dir else None,
            wandb_save_dir=absolutise(self.wandb_save_dir) if self.wandb_save_dir else None,
        )


def build_megatron_args(cfg: MoEPretrainConfig) -> list[str]:
    """Map the config to a flat Megatron ``pretrain_gpt.py`` CLI arg list (pure)."""
    args = [
        # model
        "--num-layers",
        str(cfg.num_layers),
        "--hidden-size",
        str(cfg.hidden_size),
        "--num-attention-heads",
        str(cfg.num_attention_heads),
        "--ffn-hidden-size",
        str(cfg.ffn_hidden_size),
        "--seq-length",
        str(cfg.seq_length),
        "--max-position-embeddings",
        str(cfg.seq_length),
        # MoE
        "--num-experts",
        str(cfg.num_experts),
        "--moe-router-topk",
        str(cfg.moe_router_topk),
        "--moe-router-load-balancing-type",
        cfg.moe_router_load_balancing_type,
        "--moe-aux-loss-coeff",
        str(cfg.moe_aux_loss_coeff),
        "--moe-router-score-function",
        cfg.moe_router_score_function,
        # tokenizer / data
        "--tokenizer-type",
        cfg.tokenizer_type,
        "--vocab-size",
        str(cfg.vocab_size),
        # optimisation / schedule
        "--lr",
        str(cfg.lr),
        "--min-lr",
        str(cfg.min_lr),
        "--lr-decay-style",
        cfg.lr_decay_style,
        "--lr-warmup-iters",
        str(cfg.lr_warmup_iters),
        # batch / iterations
        "--micro-batch-size",
        str(cfg.micro_batch_size),
        "--global-batch-size",
        str(cfg.global_batch_size),
        "--train-iters",
        str(cfg.train_iters),
        "--seed",
        str(cfg.seed),
        # eval
        "--eval-interval",
        str(cfg.eval_interval),
        "--eval-iters",
        str(cfg.eval_iters),
        # parallelism / runtime
        "--transformer-impl",
        cfg.transformer_impl,
        "--attention-backend",
        cfg.attention_backend,
        "--tensor-model-parallel-size",
        str(cfg.tensor_model_parallel_size),
        "--pipeline-model-parallel-size",
        str(cfg.pipeline_model_parallel_size),
        "--expert-model-parallel-size",
        str(cfg.expert_model_parallel_size),
        "--log-interval",
        str(cfg.log_interval),
        "--distributed-backend",
        "nccl",
    ]
    if cfg.moe_router_enable_expert_bias:
        args += [
            "--moe-router-enable-expert-bias",
            "--moe-router-bias-update-rate",
            str(cfg.moe_router_bias_update_rate),
        ]
    if cfg.moe_z_loss_coeff is not None:
        args += ["--moe-z-loss-coeff", str(cfg.moe_z_loss_coeff)]
    if cfg.moe_per_layer_logging:
        args += ["--moe-per-layer-logging"]
    if cfg.log_throughput:
        args += ["--log-throughput"]
    if cfg.tensorboard_dir:
        args += ["--tensorboard-dir", cfg.tensorboard_dir]
    if cfg.wandb_project:
        args += ["--wandb-project", cfg.wandb_project]
        if cfg.wandb_exp_name:
            args += ["--wandb-exp-name", cfg.wandb_exp_name]
        if cfg.wandb_entity:
            args += ["--wandb-entity", cfg.wandb_entity]
        if cfg.wandb_save_dir:
            args += ["--wandb-save-dir", cfg.wandb_save_dir]
    # Data source — two mutually exclusive modes Megatron enforces (blend vs blend_per_split):
    #   (1) single blob carved by --split into train/valid/test (ClimbMix), or
    #   (2) pre-split --train-/--valid-data-path prefixes (per-cluster ClimbLab).
    # Mixing them, or a blob without --split, leaves the valid split empty and eval crashes at
    # eval_interval — hence the fail-loud checks here.
    if cfg.data_path and cfg.train_data_path:
        raise ValueError("set either data_path (single blob + split) or train_data_path, not both")
    if cfg.data_path:
        if not cfg.split:
            raise ValueError("split is required with data_path (e.g. '99,1,0')")
        args += ["--data-path", cfg.data_path, "--split", cfg.split]
    elif cfg.train_data_path:
        if cfg.split:
            raise ValueError(
                "split is incompatible with train_data_path (per-split paths are already split); "
                "use data_path + split for a single blob"
            )
        args += ["--train-data-path", cfg.train_data_path]
        if cfg.valid_data_path:
            args += ["--valid-data-path", cfg.valid_data_path]
    else:
        raise ValueError("a data source is required: set data_path (+ split) or train_data_path")
    if not cfg.persist_layer_norm:
        args += ["--no-persist-layer-norm"]
    if not cfg.gradient_accumulation_fusion:
        args += ["--no-gradient-accumulation-fusion"]
    if not cfg.masked_softmax_fusion:
        args += ["--no-masked-softmax-fusion"]
    if not cfg.bias_gelu_fusion:
        args += ["--no-bias-gelu-fusion"]
    if not cfg.add_bias_linear:
        args += ["--disable-bias-linear"]
    if cfg.bf16:
        args += ["--bf16"]
    if cfg.data_cache_path:
        args += ["--data-cache-path", cfg.data_cache_path]
    if cfg.save:
        args += ["--save", cfg.save]
    if cfg.save_interval:
        args += ["--save-interval", str(cfg.save_interval)]
    if cfg.load:
        args += ["--load", cfg.load]
    if cfg.ckpt_step:
        args += ["--ckpt-step", str(cfg.ckpt_step)]
    return args


def build_launch_command(
    cfg: MoEPretrainConfig, megatron_script: str | Path, nproc: int = 1
) -> list[str]:
    """Full ``torchrun`` command: single-node standalone, one process per GPU."""
    return [
        "torchrun",
        "--standalone",
        "--nproc-per-node",
        str(nproc),
        str(megatron_script),
        *build_megatron_args(cfg),
    ]

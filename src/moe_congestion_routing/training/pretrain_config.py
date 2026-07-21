"""Run config for a MoE pretraining run through Megatron's ``pretrain_gpt.py``."""

from dataclasses import dataclass, replace
from pathlib import Path

import yaml


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

    train_data_path: str | None = None
    """``.bin``/``.idx`` prefix for the training split."""

    valid_data_path: str | None = None
    """``.bin``/``.idx`` prefix for the validation split."""

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

    data_cache_path: str | None = None
    """Dataset sample/shuffle index cache. ``None`` => ``<output_dir>/cache`` (derived in the
    launcher). Shared across runs (keyed by seed/seq_length) so the indices build once."""

    output_dir: str = "artifacts/runs"
    """Root for run artifacts. The launcher writes each run to its own ``<output_dir>/<timestamp>/``
    subdir (train.log, launch_command.txt, checkpoints); the dataset cache above is the one shared
    exception at ``<output_dir>/cache``."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MoEPretrainConfig":
        """Build from a yaml file. Unknown keys raise ``TypeError`` (fail loud)."""
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a valid yaml mapping, got {type(data).__name__}")
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
            output_dir=output_dir,
            data_cache_path=absolutise(self.data_cache_path)
            if self.data_cache_path
            else str(Path(output_dir) / "cache"),
            save=absolutise(self.save) if self.save else None,
            load=absolutise(self.load) if self.load else None,
        )


def build_megatron_args(cfg: MoEPretrainConfig) -> list[str]:
    """Map the config to a flat Megatron ``pretrain_gpt.py`` CLI arg list (pure)."""
    if not cfg.train_data_path:
        raise ValueError("train_data_path is required (set it in the run yaml)")
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
        # tokenizer / data
        "--tokenizer-type",
        cfg.tokenizer_type,
        "--vocab-size",
        str(cfg.vocab_size),
        "--train-data-path",
        cfg.train_data_path,
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
    if cfg.valid_data_path:
        args += ["--valid-data-path", cfg.valid_data_path]
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

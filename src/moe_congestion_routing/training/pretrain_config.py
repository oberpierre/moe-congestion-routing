"""Run config for a MoE pretraining run through Megatron's ``pretrain_gpt.py``."""

from dataclasses import dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MoEPretrainConfig:
    """Everything needed to launch one MoE pretraining run, loadable from a yaml file."""

    # model (tiny for the smoke run)
    num_layers: int = 4
    hidden_size: int = 256
    num_attention_heads: int = 8
    ffn_hidden_size: int = 512
    seq_length: int = 512

    # MoE — vanilla Megatron Switch aux-loss load balancing (no patches)
    num_experts: int = 8
    moe_router_topk: int = 2
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float = 0.01  # Megatron default is 0.0 → aux loss would be a no-op

    # data — ClimbLab is pre-tokenized GPT-2; NullTokenizer(vocab_size) sets eod = vocab_size-1
    # = 50256 = <|endoftext|>, so no vocab/merge files are needed.
    train_data_path: str = "artifacts/climblab_local/cluster_1_train"
    valid_data_path: str = "artifacts/climblab_local/cluster_1_valid"
    tokenizer_type: str = "NullTokenizer"
    vocab_size: int = 50257

    # optimisation / schedule
    lr: float = 3.0e-4
    min_lr: float = 3.0e-5
    lr_decay_style: str = "constant"
    lr_warmup_iters: int = 5

    # batch / iterations
    micro_batch_size: int = 4
    global_batch_size: int = 8
    train_iters: int = 30
    seed: int = 1234

    # eval (off by default: eval_iters=0 skips it; a later slice turns it on)
    eval_interval: int = 1000
    eval_iters: int = 0

    # checkpointing (off by default; a later slice turns it on)
    save: str | None = None
    save_interval: int | None = None
    load: str | None = None

    # runtime
    transformer_impl: str = "local"  # avoid Transformer Engine (not installed locally)
    # Fused kernels default ON via argparse but need apex/TE; with transformer_impl=local
    # they must be disabled (torch LayerNorm rejects persist; grad-accum fusion needs apex).
    persist_layer_norm: bool = False
    gradient_accumulation_fusion: bool = False
    masked_softmax_fusion: bool = False  # needs the scaled_masked_softmax_cuda kernel (unbuilt)
    bias_gelu_fusion: bool = False  # fused act path requires swiglu/quick_gelu under MoE probs
    # No linear bias (modern default + what the reference uses). Also sidesteps a Megatron
    # in-place-on-view autograd error in the non-fused MoE expert bias path.
    add_bias_linear: bool = False
    bf16: bool = True
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    log_interval: int = 1
    # Shared across runs (keyed by seed/seq_length), so the sample/shuffle indices build once.
    data_cache_path: str | None = None  # None → <output_dir>/cache (derived in the launcher)

    # The launcher writes each run to its own <output_dir>/<timestamp>/ subdir (train.log,
    # launch_command.txt, and later checkpoints), so repeated/concurrent runs don't interfere
    # with each other; the dataset cache above is the one shared exception at <output_dir>/cache.
    output_dir: str = "artifacts/moe_smoke"

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
            train_data_path=absolutise(self.train_data_path),
            valid_data_path=absolutise(self.valid_data_path),
            output_dir=output_dir,
            data_cache_path=absolutise(self.data_cache_path)
            if self.data_cache_path
            else str(Path(output_dir) / "cache"),
            save=absolutise(self.save) if self.save else None,
            load=absolutise(self.load) if self.load else None,
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
        # tokenizer / data
        "--tokenizer-type",
        cfg.tokenizer_type,
        "--vocab-size",
        str(cfg.vocab_size),
        "--train-data-path",
        cfg.train_data_path,
        "--valid-data-path",
        cfg.valid_data_path,
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

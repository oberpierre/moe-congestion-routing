"""Run config for text-in/text-out inference on a trained MoE checkpoint.

Drives Megatron's shipped static-inference example
(``examples/inference/advanced/gpt_static_inference.py``), which builds the model, loads the
checkpoint, and runs the real ``StaticInferenceEngine`` + ``TextGenerationController``. We only
map this config to that script's CLI args — the actual generation pipeline is Megatron's.

The model-architecture fields MUST match the checkpoint's training config: they define the
network the weights load into. Their defaults match ``configs/train/climblab_moe_smoke.yaml``.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MoEInferConfig:
    """Everything needed to generate text from a trained MoE checkpoint, loadable from yaml."""

    # checkpoint
    load: str | None = None
    """Checkpoint DIRECTORY to load (a ``<run>/checkpoints`` dir). Required at launch — set here
    or via the launcher's ``--load``. Loads the newest ``iter_<N>/`` unless ``ckpt_step`` is set."""

    ckpt_step: int | None = None
    """Load this iteration instead of the newest (200 → ``iter_0000200/``)."""

    # generation
    prompts: list[str] = field(default_factory=lambda: ["The capital of France is"])
    """Input prompt strings; each is encoded, continued, and decoded back to text."""

    num_tokens_to_generate: int = 32
    """New tokens to generate per prompt."""

    temperature: float = 1.0
    """Sampling temperature (only matters when top_k != 1)."""

    top_k: int = 1
    """Top-k sampling; ``1`` = greedy (deterministic), the sane default for a round-trip check."""

    top_p: float = 0.0
    """Top-p (nucleus) sampling; ``0.0`` disables it."""

    # tokenizer — real GPT-2 BPE for text<->ids, loaded offline from the vendored snapshot dir.
    tokenizer_type: str = "HuggingFaceTokenizer"
    """``HuggingFaceTokenizer`` pointed at a local dir loads fully offline (no hub lookup)."""

    tokenizer_model: str = "assets/tokenizer/gpt2"
    """Local gpt2 tokenizer snapshot dir (tokenizer.json + tokenizer_config.json)."""

    vocab_size: int = 50257
    """GPT-2 vocabulary size; must match the checkpoint (NullTokenizer training used 50257 too)."""

    # model architecture — MUST match the checkpoint (defaults match the smoke training config).
    num_layers: int = 4
    """Number of transformer layers."""

    hidden_size: int = 256
    """Model/embedding hidden dimension."""

    num_attention_heads: int = 8
    """Number of attention heads."""

    ffn_hidden_size: int = 512
    """Inner dimension of each expert's MLP."""

    seq_length: int = 512
    """Sequence length the model was trained at (also max position embeddings)."""

    num_experts: int = 8
    """Number of routed experts."""

    moe_router_topk: int = 2
    """Experts activated per token."""

    # runtime — mirror training so the model instantiates the same way on this machine.
    transformer_impl: str = "local"
    """``local`` avoids Transformer Engine (not installed locally)."""

    persist_layer_norm: bool = False
    """Fused persistent LayerNorm; off under transformer_impl=local."""

    gradient_accumulation_fusion: bool = False
    """Fused gradient accumulation; off (needs apex)."""

    masked_softmax_fusion: bool = False
    """Fused scaled masked softmax; off (kernel unbuilt)."""

    bias_gelu_fusion: bool = False
    """Fused bias+GELU; off (fused act path needs swiglu/quick_gelu under MoE probs)."""

    add_bias_linear: bool = False
    """Linear-layer bias; off (matches training / the reference)."""

    bf16: bool = True
    """Run in bfloat16 (matches the saved checkpoint)."""

    tensor_model_parallel_size: int = 1
    """Tensor-model-parallel world size."""

    pipeline_model_parallel_size: int = 1
    """Pipeline-model-parallel world size."""

    expert_model_parallel_size: int = 1
    """Expert-parallel world size."""

    use_legacy_static_engine: bool = True
    """Use the legacy (true static-batching) inference engine. On by default because the modern
    path builds a dynamic-batching context that requires flash-attn >=2.7.3 (not installed here,
    same reason we avoid Transformer Engine); legacy static needs neither."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MoEInferConfig":
        """Build from a yaml file. Unknown keys raise ``TypeError`` (fail loud)."""
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a valid yaml mapping, got {type(data).__name__}")
        return cls(**data)

    def resolved(self, repo_root: Path) -> "MoEInferConfig":
        """Absolutise the checkpoint and tokenizer paths against ``repo_root``."""

        def absolutise(p: str) -> str:
            path = Path(p)
            return str(path if path.is_absolute() else repo_root / path)

        return replace(
            self,
            load=absolutise(self.load) if self.load else None,
            tokenizer_model=absolutise(self.tokenizer_model),
        )


def build_infer_megatron_args(cfg: MoEInferConfig) -> list[str]:
    """Map the config to a flat CLI arg list for ``gpt_static_inference.py`` (pure)."""
    if not cfg.load:
        raise ValueError("load is required for inference (set it in the yaml or via --load)")
    args = [
        # model architecture — must match the checkpoint
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
        # tokenizer / data
        "--tokenizer-type",
        cfg.tokenizer_type,
        "--tokenizer-model",
        cfg.tokenizer_model,
        "--vocab-size",
        str(cfg.vocab_size),
        # checkpoint
        "--load",
        cfg.load,
        # generation
        "--num-tokens-to-generate",
        str(cfg.num_tokens_to_generate),
        "--temperature",
        str(cfg.temperature),
        "--top_k",
        str(cfg.top_k),
        "--top_p",
        str(cfg.top_p),
        # parallelism / runtime
        "--transformer-impl",
        cfg.transformer_impl,
        "--tensor-model-parallel-size",
        str(cfg.tensor_model_parallel_size),
        "--pipeline-model-parallel-size",
        str(cfg.pipeline_model_parallel_size),
        "--expert-model-parallel-size",
        str(cfg.expert_model_parallel_size),
        "--distributed-backend",
        "nccl",
    ]
    if cfg.ckpt_step:
        args += ["--ckpt-step", str(cfg.ckpt_step)]
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
    if cfg.use_legacy_static_engine:
        args += ["--use-legacy-static-engine"]
    # --prompts is nargs='+'; keep it last so the variadic list can't swallow another flag's value.
    args += ["--prompts", *cfg.prompts]
    return args


def build_infer_launch_command(
    cfg: MoEInferConfig, megatron_script: str | Path, nproc: int = 1
) -> list[str]:
    """Full ``torchrun`` command driving Megatron's static-inference script."""
    return [
        "torchrun",
        "--standalone",
        "--nproc-per-node",
        str(nproc),
        str(megatron_script),
        *build_infer_megatron_args(cfg),
    ]

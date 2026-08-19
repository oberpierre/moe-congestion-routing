"""Run config for a MoE pretraining run through Megatron's ``pretrain_gpt.py``."""

import itertools
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

from ..config_extends import load_yaml_with_extends
from ..losses.cost_families import (
    COST_FAMILIES,
    DEFAULT_LAMBDA,
    ROSENTHAL_TYPES,
    VARIANTS,
    cost_exponent,
    pressure_bound,
)
from ..paths import expand_path
from ..probe_windows import parse_windows

# Every name Megatron's own --moe-router-load-balancing-type accepts at the pinned commit, plus
# our two rosenthal names. Checked unconditionally so a typo fails at --dry-run rather than at the
# front of a cluster queue. Without the check an unknown value builds cleanly, skips every
# rosenthal validation rule, and emits no --moe-rosenthal-* flags at all.
_MOE_ROUTER_LOAD_BALANCING_TYPES = (
    "aux_loss",
    "seq_aux_loss",
    "global_aux_loss",
    "sinkhorn",
    "none",
    *ROSENTHAL_TYPES,
)


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
    """Inner dimension of a DENSE layer's MLP. Only live when ``moe_layer_freq`` leaves some layer
    dense; otherwise its sole effect is seeding ``moe_ffn_hidden_size`` when that is unset. Set both
    explicitly -- the two are unrelated widths as soon as a dense layer exists."""

    seq_length: int = 512
    """Training sequence length (also used for max position embeddings)."""

    position_embedding_type: str = "learned_absolute"
    """``learned_absolute`` (Megatron's default: an extra ``seq_length * hidden_size`` parameter
    table) or ``rope`` (no parameters, what every modern reference uses). ARCHITECTURAL: a
    checkpoint cannot be loaded under a different value."""

    normalization: str = "LayerNorm"
    """``LayerNorm`` (Megatron's default; has a bias) or ``RMSNorm`` (scale only). ARCHITECTURAL."""

    untie_embeddings_and_output_weights: bool = False
    """Give the output head its own ``vocab x hidden`` matrix instead of reusing the input
    embedding's. Megatron's default TIES them (one tensor serving both roles). Untying costs a
    second ``V_pad * d`` tensor and zero FLOPs -- the head matmul is the same shape either way.
    Small models tie (the saving outweighs the constraint), large ones untie (the two roles want
    different geometry). ARCHITECTURAL: adds ``output_layer.weight`` to the state dict."""

    norm_epsilon: float = 1.0e-5
    """Epsilon inside LayerNorm/RMSNorm. Defaults to Megatron's own 1e-5 because a config that sets
    nothing should get Megatron's behaviour. The reference architecture (base_cluster.yaml) sets
    1e-6 to match FLAME, which trains at that value."""

    swiglu: bool = False
    """Gated SiLU MLP instead of the default non-gated GELU. ARCHITECTURAL: ``linear_fc1`` emits
    ``2 * ffn_hidden_size``, so a gated FFN costs ``3 * d * f`` against a non-gated ``2 * d * f`` --
    shrink the widths to ~2/3 to hold cost fixed. Also redirects Megatron's activation fusion
    from ``bias_gelu_fusion`` to ``bias_swiglu_fusion``, which is a pure-torch jit path (no apex),
    so it stays on and the ``bias_gelu_fusion`` field below becomes inert."""

    num_experts: int = 8
    """Number of routed experts."""

    moe_router_topk: int = 2
    """Experts activated per token."""

    moe_ffn_hidden_size: int | None = None
    """Inner dimension of each ROUTED expert's MLP. ``None`` silently inherits ``ffn_hidden_size``
    (Megatron warns and assigns), which is ambiguous once a dense layer exists -- set it
    explicitly on any real run."""

    moe_shared_expert_intermediate_size: int | None = None
    """TOTAL width of the always-on shared expert(s), i.e. ``num_shared * width_each``, run for
    every token in addition to the top-k routed experts. ``None`` = no shared expert, our
    deliberate deviation from both baselines. NOTE: shared experts bypass the router entirely, so
    the patched load-balance/SwapGap metrics never see that capacity."""

    moe_layer_freq: str | int | None = None
    """Which layers are MoE. ``None`` leaves Megatron's default (every layer). An int ``N`` means
    one MoE layer per ``N``; a string python list expression gives an explicit pattern, e.g.
    ``"[0]*1+[1]*8"`` for one dense layer followed by 8 MoE layers. The pattern length must equal
    ``num_layers``. Dense layers use ``ffn_hidden_size``, MoE layers ``moe_ffn_hidden_size``."""

    moe_router_pre_softmax: bool = False
    """Softmax over ALL experts before top-k (gates sum to <1 and vary with routing confidence)
    instead of Megatron's default softmax over only the k selected logits (gates sum to 1). Affects
    the combine weights ONLY: the aux loss recomputes its own full-softmax scores, and the patched
    metrics read raw logits + routing_map, so neither changes."""

    moe_router_dtype: str | None = None
    """Upcast router logits / expert-output weighting to ``fp32`` (or ``fp64``). ``None`` keeps
    them in bf16. Recommended at large expert counts, where bf16 logit drift can reorder top-k."""

    moe_grouped_gemm: bool = False
    """Batch the per-expert GEMMs into one grouped kernel instead of looping over experts
    sequentially. Big throughput win at high expert counts; falls back to the sequential path when
    Transformer Engine is too old to provide GroupedLinear. ARCHITECTURAL: changes the expert module
    (``TEGroupedMLP`` vs ``SequentialMLP``) and so the checkpoint's parameter names."""

    moe_router_load_balancing_type: str = "aux_loss"
    """Load-balancing strategy; ``aux_loss`` is the vanilla Switch auxiliary loss."""

    moe_aux_loss_coeff: float = 0.01
    """Aux-loss weight. Megatron's default is 0.0, which makes the aux loss a no-op. Reused
    unchanged as the Rosenthal congestion loss's coefficient (alpha) when
    ``moe_router_load_balancing_type`` is ``rosenthal``/``global_rosenthal``."""

    moe_rosenthal_variant: str = "hard"
    """Rosenthal congestion loss variant, live only under ``moe_router_load_balancing_type in
    (rosenthal, global_rosenthal)``: ``hard`` (default) prices the REALIZED, detached load;
    ``soft`` prices the EXPECTED load. ``soft``'s continuized potential is non-linear in the mean
    gate mass, so it cannot decompose across ranks the way a straight all-reduce of a
    differentiable tensor would -- but the router never does that: it all-reduces the DETACHED
    marginal cost and prices a still-local, still-differentiable carrier against it (the
    synced-coefficient construction), which is exact at any reduce-group size. ``soft`` is
    therefore valid with ``global_rosenthal`` and at any ``tensor_model_parallel_size``, exactly
    like ``hard``."""

    moe_rosenthal_cost: str = "linear"
    """Congestion cost family, live only under a rosenthal balancing type: ``linear`` (default,
    ``c(x) = lambda*x``) or ``quadratic`` (``c(x) = lambda*x**2``)."""

    moe_rosenthal_lambda: float | None = None
    """Congestion cost coefficient (lambda), live only under a rosenthal balancing type. ``None``
    (the default) resolves to the cost family's own slope-matched default from
    ``DEFAULT_LAMBDA`` in ``losses/cost_families.py`` (1.0 for linear, 0.5 for quadratic) when the
    flag is emitted, so paired arms stay slope-matched without the config author restating it. An
    explicit value overrides that and must be > 0."""

    moe_rosenthal_log_grad_ratio: bool = False
    """Log the logit-space gradient norm the congestion loss injects, relative to the task
    gradient at the same point. Requires a rosenthal balancing type. Read by the router-grad-probe
    patch, which places one identity autograd probe on each consumer of the post-z-loss logits."""

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

    moe_probe_batch: str | None = None
    """Path to a frozen probe-batch asset (``assets/probe/*.npz``). Required whenever
    ``moe_probe_coarse_interval`` is nonzero."""

    moe_probe_coarse_interval: int = 0
    """Probe every N iterations. ``0`` (the default) disables the probe entirely."""

    moe_probe_dense_interval: int = 0
    """Probe cadence inside a ``moe_probe_dense_windows`` window. Must be > 0 whenever a window is
    given."""

    moe_probe_dense_windows: list[str] | None = None
    """Inclusive ``"start:end"`` step ranges probed at ``moe_probe_dense_interval``, anchored at
    each window's own start rather than at iteration 0, e.g. ``["0:500"]``."""

    moe_probe_seqs: int = 8
    """Number of sequences from the probe asset forwarded per probe step. ``micro_batch_size``
    must divide it evenly, so the probe batch splits into a whole number of microbatches."""

    moe_probe_dir: str | None = None
    """Directory probe dumps are written to. ``None`` => the launcher derives ``<run_dir>/probes``,
    exactly like ``tensorboard_dir``."""

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

    lr_decay_style: str = "WSD"
    """Learning-rate decay schedule: ``constant``/``linear``/``cosine``/``inverse-square-root``/
    ``WSD``. ``WSD`` holds ``lr`` flat until ``lr_decay_iters - lr_wsd_decay_iters``, then anneals
    to ``min_lr``, which makes ANY stable-phase checkpoint branch-annealable into a finished model
    (see ``configs/train/anneal_short_cluster.yaml``)."""

    lr_decay_iters: int | None = None
    """Horizon the decay curve is sized for. ``None`` => ``train_iters``. Only set it apart from
    ``train_iters`` for a branch anneal, where the run stops at the end of a shortened curve."""

    lr_wsd_decay_iters: int | None = None
    """Length of WSD's annealing phase, in iterations. REQUIRED by Megatron when
    ``lr_decay_style: WSD`` (it asserts, so a missing value fails loud). ~10% of the horizon is the
    usual choice. The anneal occupies the LAST ``lr_wsd_decay_iters`` of ``lr_decay_iters``."""

    lr_wsd_decay_style: str = "exponential"
    """Shape of WSD's annealing phase: ``exponential`` (Megatron's default), ``linear``,
    ``cosine`` or ``minus_sqrt``."""

    override_opt_param_scheduler: bool = False
    """Rebuild the LR schedule from THIS config instead of the checkpoint's. Megatron otherwise
    asserts that lr / min_lr / warmup / horizon / decay style all match the checkpoint, so an
    ordinary resumed slice must leave them alone -- that assert is a feature. Set this ONLY for a
    branch anneal, which deliberately swaps in a shorter horizon. The iteration counter is still
    restored from the checkpoint, so the anneal starts where the trunk left off."""

    lr_warmup_iters: int = 5
    """Linear warmup iterations before the decay schedule applies."""

    # batch / iterations
    micro_batch_size: int = 4
    """Samples per micro-batch (one forward/backward)."""

    global_batch_size: int = 8
    """Samples per optimizer step (across gradient accumulation / data parallelism)."""

    train_iters: int = 30
    """Total training iterations (optimizer steps) -- also fixes the LR-schedule horizon, so keep it
    constant across resumed slices (see ``exit_interval``)."""

    exit_interval: int | None = None
    """Exit cleanly (after a checkpoint) whenever the GLOBAL iteration is a multiple of this. Lets a
    run be trained in slices: keep ``train_iters`` at the full budget so the LR schedule is
    unchanged, set e.g. ``5000`` to stop every quarter of a 20000-iter run. ``None`` runs straight
    to ``train_iters``. To resume a slice, relaunch with ``run_moe_pretrain.py --load
    <run>/checkpoints`` (needs ``save``/``save_interval`` so the slice actually checkpointed)."""

    seed: int = 1234
    """RNG seed; also part of the dataset index cache key."""

    # eval
    eval_interval: int = 1000
    """Iterations between validation passes."""

    eval_iters: int = 0
    """Batches per validation pass; 0 disables eval entirely."""

    # Checkpointing, in Megatron's semantics. save and load are directories rather than single
    # checkpoints: each save drops an iter_<N>/ subdir and a latest_checkpointed_iteration.txt
    # tracker inside.
    save: str | None = None
    """Directory to write checkpoints to; unset => the launcher uses
    ``<output_dir>/<timestamp>/checkpoints``."""

    save_interval: int | None = None
    """Iterations between saves and this harness's on-switch, unsetting it means no checkpoints."""

    load: str | None = None
    """Checkpoint DIRECTORY to resume/infer from; loads the newest ``iter_<N>/`` per the tracker.
    Usually set via ``run_moe_pretrain.py --load`` (autocompletes; also drives the run dir)."""

    exit_on_missing_checkpoint: bool = False
    """Fail loud+fast if ``load`` holds no checkpoint, instead of silently starting from random. The
    launcher sets this whenever a ``load`` is given as an explicit resume that finds nothing is a
    mistake to surface, not paper over."""

    ckpt_step: int | None = None
    """Load this iteration from ``load`` instead of the newest (200 => ``iter_0000200/``)."""

    # Transformer Engine, for its fused attention, LayerNorm and Linear kernels. Training and
    # inference must use the same implementation, because a checkpoint never crosses them: one
    # trained under `local` is not loadable into a TE model.
    transformer_impl: str = "transformer_engine"
    """Megatron transformer implementation. ``transformer_engine`` uses TE's fused modules."""

    attention_backend: str = "auto"
    """TE attention backend: flash / fused / unfused / auto / local. ``auto`` lets TE pick."""

    # The Megatron and apex fusion paths below stay off. TE supplies its own fused kernels, and
    # the prebuilt kernels these would need are not installed locally, so they are no-ops under TE.
    persist_layer_norm: bool = False
    """Megatron's non-TE fused persistent LayerNorm; off (TE has its own)."""

    gradient_accumulation_fusion: bool = False
    """apex-fused gradient accumulation; off (apex absent locally; TE handles wgrad)."""

    masked_softmax_fusion: bool = False
    """Megatron's fused scaled masked softmax; off (kernel unbuilt; TE fuses attention)."""

    bias_gelu_fusion: bool = False
    """Megatron's fused bias+GELU; off (TE fuses the MLP activation). Inert under ``swiglu``, which
    routes the activation fusion through ``bias_swiglu_fusion`` instead."""

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

    use_distributed_optimizer: bool = False
    """Shard optimizer state (fp32 master weights + Adam moments) across the data-parallel ranks
    instead of replicating it on every rank. Pure memory win, no change to the math or to the
    routing -- unlike expert parallelism, which we keep at 1 so every rank sees every expert (no
    all-to-all, exact per-expert logging)."""

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

    wandb_group: str | None = None
    """W&B group tying related runs together in the UI, e.g. a WSD trunk and the arms branched off
    it, or the two sides of an A/B. ``None`` => the launcher derives it from ``output_dir``'s name,
    so runs sharing an output_dir group automatically. Exported as ``WANDB_RUN_GROUP`` rather than
    passed as a Megatron flag -- Megatron's ``wandb.init()`` sets only dir/name/project/config, so
    the env var is what reaches W&B."""

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
        data = load_yaml_with_extends(Path(path))
        return cls(**data)

    def resolved(self, repo_root: Path) -> "MoEPretrainConfig":
        """Absolutise all paths against ``repo_root`` and derive the data cache dir if unset."""

        def absolutise(p: str) -> str:
            # Expand ${VAR} and ~ first, so a committed config can reference ${DATA_STORE} rather
            # than a personal path, and an unset variable fails loud. Then anchor to repo_root.
            path = Path(expand_path(p))
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
            moe_probe_batch=absolutise(self.moe_probe_batch) if self.moe_probe_batch else None,
            moe_probe_dir=absolutise(self.moe_probe_dir) if self.moe_probe_dir else None,
        )


def resolve_run_dir(
    output_dir: str | Path,
    *,
    run_dir: str | None = None,
    load: str | None = None,
    run_tag: str | None = None,
    is_branch: bool = False,
) -> Path:
    """Directory this run writes its OWN artifacts to (train.log, checkpoints, tensorboard, W&B).

    Three cases:
      * ``run_dir`` given: use it, relative names resolving under ``output_dir``. This is how a
        WSD branch reads one run's checkpoints while writing its own, and how a fresh run takes a
        stable name instead of a timestamp.
      * ``load`` only: continue IN the loaded run's dir, so a sliced run appends to one log and
        one W&B curve. This is the plain resume, and the default.
      * neither: a fresh ``<output_dir>/<run_tag>``.

    ``run_dir`` and ``load`` are expanded through ``expand_path`` (``${VAR}``/``~``) exactly as
    config-file paths are, so an unset variable fails loud here instead of being taken literally.
    That matters most for ``run_dir``: a literal ``${SCRATCH}`` is a legal directory name, so
    without this the run would happily create it and write a whole training run to the wrong
    filesystem. Idempotent, so callers that already expanded may pass either form.

    ``is_branch`` (the config's ``override_opt_param_scheduler``) marks a run that deliberately
    re-sizes the LR schedule. Such a run must NOT write into the run it branched from: its
    checkpoints would move that run's ``latest_checkpointed_iteration.txt`` onto a model trained
    under a different schedule -- and, since an annealed model is a dead end, the next resume of the
    trunk would silently continue from cooled weights. Its metrics would also splice into the
    parent's curve at step numbers the parent already owns.
    """
    output_dir = Path(output_dir)
    run_dir = expand_path(run_dir) if run_dir is not None else None
    load = expand_path(load) if load is not None else None
    if run_dir is not None:
        resolved = Path(run_dir)
        if not resolved.is_absolute():
            resolved = output_dir / resolved
    elif load is not None:
        resolved = Path(load).resolve().parent
    elif run_tag is not None:
        resolved = output_dir / run_tag
    else:
        raise ValueError("a fresh run needs a run_tag (or an explicit run_dir)")

    if is_branch and load is not None and resolved.resolve() == Path(load).resolve().parent:
        raise ValueError(
            f"this run re-sizes the LR schedule (override_opt_param_scheduler) but would write "
            f"into the run it branched from ({resolved}) -- its checkpoints would move that run's "
            f"latest_checkpointed_iteration.txt onto differently-scheduled weights. Pass "
            f"--run-dir <name> so the branch gets its own directory."
        )
    return resolved


def build_megatron_args(cfg: MoEPretrainConfig) -> list[str]:
    """Map the config to a flat Megatron ``pretrain_gpt.py`` CLI arg list (pure)."""
    if cfg.moe_router_load_balancing_type not in _MOE_ROUTER_LOAD_BALANCING_TYPES:
        raise ValueError(
            f"moe_router_load_balancing_type must be one of {_MOE_ROUTER_LOAD_BALANCING_TYPES}, "
            f"got {cfg.moe_router_load_balancing_type!r}"
        )
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
        # Architecture, emitted even at Megatron's own defaults. These define the network a
        # checkpoint loads into, so the frozen launch_command.txt has to record them.
        "--position-embedding-type",
        cfg.position_embedding_type,
        "--normalization",
        cfg.normalization,
        "--norm-epsilon",
        str(cfg.norm_epsilon),
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
    if cfg.untie_embeddings_and_output_weights:
        args += ["--untie-embeddings-and-output-weights"]
    if cfg.swiglu:
        args += ["--swiglu"]
    if cfg.lr_decay_iters is not None:
        args += ["--lr-decay-iters", str(cfg.lr_decay_iters)]
    if cfg.lr_decay_style == "WSD" and cfg.lr_wsd_decay_iters is None:
        raise ValueError("lr_decay_style: WSD requires lr_wsd_decay_iters (Megatron asserts on it)")
    if cfg.lr_wsd_decay_iters is not None:
        args += [
            "--lr-wsd-decay-iters",
            str(cfg.lr_wsd_decay_iters),
            "--lr-wsd-decay-style",
            cfg.lr_wsd_decay_style,
        ]
    if cfg.override_opt_param_scheduler:
        args += ["--override-opt-param-scheduler"]
    if cfg.moe_ffn_hidden_size is not None:
        args += ["--moe-ffn-hidden-size", str(cfg.moe_ffn_hidden_size)]
    if cfg.moe_shared_expert_intermediate_size is not None:
        args += [
            "--moe-shared-expert-intermediate-size",
            str(cfg.moe_shared_expert_intermediate_size),
        ]
    if cfg.moe_layer_freq is not None:
        args += ["--moe-layer-freq", str(cfg.moe_layer_freq)]
    if cfg.moe_router_pre_softmax:
        args += ["--moe-router-pre-softmax"]
    if cfg.moe_router_dtype is not None:
        args += ["--moe-router-dtype", cfg.moe_router_dtype]
    if cfg.moe_grouped_gemm:
        args += ["--moe-grouped-gemm"]
    if cfg.use_distributed_optimizer:
        args += ["--use-distributed-optimizer"]
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
    # Router probe. moe_probe_coarse_interval == 0 is the off-switch: no flag below is emitted.
    if not cfg.moe_probe_coarse_interval and (
        cfg.moe_probe_dense_interval or cfg.moe_probe_dense_windows
    ):
        raise ValueError(
            "moe_probe_dense_interval/moe_probe_dense_windows are set but "
            "moe_probe_coarse_interval is 0, which disables the probe entirely: the dense "
            "settings would be silently dropped rather than sampled"
        )
    if cfg.moe_probe_coarse_interval:
        if not cfg.moe_probe_batch:
            raise ValueError("moe_probe_coarse_interval requires moe_probe_batch to be set")
        if cfg.tensor_model_parallel_size != 1 or cfg.pipeline_model_parallel_size != 1:
            raise ValueError(
                "the router probe requires tensor_model_parallel_size == 1 and "
                "pipeline_model_parallel_size == 1 (TP shards tokens across ranks, PP puts MoE "
                f"layers on other ranks), got TP={cfg.tensor_model_parallel_size} "
                f"PP={cfg.pipeline_model_parallel_size}"
            )
        # These four fields are not part of MoEPretrainConfig today, but would invalidate the probe
        # therefore they are pinned by a test below, so it fires loud if they are added.
        context_parallel_size = getattr(cfg, "context_parallel_size", 1)
        if context_parallel_size != 1:
            raise ValueError(
                "the router probe requires context_parallel_size == 1 (CP shards tokens across "
                f"ranks), got {context_parallel_size}"
            )
        if getattr(cfg, "moe_input_jitter_eps", None) is not None:
            raise ValueError(
                "the router probe requires moe_input_jitter_eps unset: apply_input_jitter is not "
                "guarded by self.training, so a probe forward would route on perturbed inputs"
            )
        if getattr(cfg, "moe_expert_capacity_factor", None) is not None:
            raise ValueError(
                "the router probe requires moe_expert_capacity_factor unset: a capacity drop "
                "would misalign the writer's fixed-K combine array against routing_map's bits"
            )
        if getattr(cfg, "cuda_graph_impl", "none") not in (None, "none"):
            raise ValueError(
                "the router probe requires CUDA graphs disabled (cuda_graph_impl='none'): a "
                "captured graph would not re-run the probe's own no-grad forward"
            )
        if cfg.moe_probe_seqs % cfg.micro_batch_size != 0:
            raise ValueError(
                f"moe_probe_seqs {cfg.moe_probe_seqs} must be divisible by micro_batch_size "
                f"{cfg.micro_batch_size}"
            )
        windows = parse_windows(cfg.moe_probe_dense_windows or [])
        if windows and cfg.moe_probe_dense_interval <= 0:
            raise ValueError("moe_probe_dense_windows requires moe_probe_dense_interval > 0")
        for (_, end), (next_start, _) in itertools.pairwise(windows):
            if next_start <= end:
                raise ValueError(
                    "moe_probe_dense_windows must be sorted and non-overlapping, got "
                    f"{cfg.moe_probe_dense_windows}"
                )
        args += [
            "--moe-probe-batch",
            cfg.moe_probe_batch,
            "--moe-probe-coarse-interval",
            str(cfg.moe_probe_coarse_interval),
            "--moe-probe-dense-interval",
            str(cfg.moe_probe_dense_interval),
            "--moe-probe-seqs",
            str(cfg.moe_probe_seqs),
        ]
        if cfg.moe_probe_dense_windows:
            args += ["--moe-probe-dense-windows", *cfg.moe_probe_dense_windows]
        if cfg.moe_probe_dir:
            args += ["--moe-probe-dir", cfg.moe_probe_dir]
    # Rosenthal congestion-loss balancing types. Validated here and again on Megatron's own
    # TransformerConfig. The duplication is deliberate rather than a leftover: this copy saves
    # cluster queue time by failing at --dry-run, and the TransformerConfig copy is the one no
    # launch path can bypass.
    if (
        cfg.moe_rosenthal_log_grad_ratio
        and cfg.moe_router_load_balancing_type not in ROSENTHAL_TYPES
    ):
        raise ValueError(
            "moe_rosenthal_log_grad_ratio requires moe_router_load_balancing_type to be "
            f"'rosenthal' or 'global_rosenthal', got {cfg.moe_router_load_balancing_type!r}"
        )
    if cfg.moe_router_load_balancing_type in ROSENTHAL_TYPES:
        if cfg.moe_rosenthal_variant not in VARIANTS:
            raise ValueError(
                f"moe_rosenthal_variant must be one of {VARIANTS}, got "
                f"{cfg.moe_rosenthal_variant!r}"
            )
        if cfg.moe_rosenthal_cost not in COST_FAMILIES:
            raise ValueError(
                f"moe_rosenthal_cost must be one of {COST_FAMILIES}, got {cfg.moe_rosenthal_cost!r}"
            )
        # None means "use this cost family's own slope-matched default". Resolving it here means
        # the sanity bound below and the emitted flag both see one concrete value.
        _lambda = (
            cfg.moe_rosenthal_lambda
            if cfg.moe_rosenthal_lambda is not None
            else DEFAULT_LAMBDA[cfg.moe_rosenthal_cost]
        )
        if _lambda <= 0:
            raise ValueError(f"moe_rosenthal_lambda must be > 0, got {_lambda}")
        if cfg.moe_router_score_function != "softmax":
            raise ValueError(
                "a rosenthal balancing type requires moe_router_score_function == 'softmax' "
                "(sigmoid needs mean-centering to define the congestion loss, which is out of "
                f"scope), got {cfg.moe_router_score_function!r}"
            )
        # Rules 5 and 6 rejected soft with global_rosenthal, and required
        # tensor_model_parallel_size == 1 for soft with rosenthal. Both are retired: the
        # synced-coefficient construction replaced the potential-form soft loss, which had needed a
        # size-1 reduce group, and is correct at any reduce-group size. The numbers stay retired
        # rather than being reused, because they are pinned into test names here and in
        # megatron_rosenthal_test.py. The survivors are rules 1-4, 7 and 8.
        #
        # Rule 8 is a sanity bound on the pressure at full imbalance rather than a correctness
        # condition, so it warns instead of blocking what may be a deliberate high-lambda
        # experiment. The bound expression lives in losses/cost_families.py, imported both here and
        # by the Megatron patch's copy of this check, so it is written out once. pressure_bound
        # returns the expression string alongside the value so this caller does not decide which
        # branch produced it a second time. Getting that second branch wrong would print a
        # plausible but wrong expression beside a correct number, in a warning string no test
        # asserts on.
        _p = cost_exponent(cfg.moe_rosenthal_cost)
        _bound = pressure_bound(
            cfg.moe_aux_loss_coeff,
            _lambda,
            cfg.num_experts,
            cfg.moe_router_topk,
            cfg.moe_rosenthal_cost,
            cfg.moe_rosenthal_variant,
        )
        if _bound.value > 1:
            warnings.warn(
                f"moe_rosenthal_lambda * moe_aux_loss_coeff * {_bound.expr}**{_p} = "
                f"{_bound.value:.4g} > 1 -- the congestion pressure at full imbalance exceeds the "
                "sanity bound; not an error, but check moe_rosenthal_lambda and moe_aux_loss_coeff "
                "are what you intend.",
                stacklevel=2,
            )
        args += [
            "--moe-rosenthal-variant",
            cfg.moe_rosenthal_variant,
            "--moe-rosenthal-cost",
            cfg.moe_rosenthal_cost,
            "--moe-rosenthal-lambda",
            str(_lambda),
        ]
        if cfg.moe_rosenthal_log_grad_ratio:
            args += ["--moe-rosenthal-log-grad-ratio"]
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
    # Data source. Megatron enforces two mutually exclusive modes, blend and blend_per_split:
    # either a single blob carved by --split into train/valid/test (ClimbMix), or pre-split
    # --train-data-path and --valid-data-path prefixes (per-cluster ClimbLab). Mixing them, or
    # giving a blob without --split, leaves the valid split empty and eval crashes at
    # eval_interval, so the checks below fail loud instead.
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
    if cfg.exit_on_missing_checkpoint:
        args += ["--exit-on-missing-checkpoint"]
    if cfg.exit_interval:
        args += ["--exit-interval", str(cfg.exit_interval)]
    return args


def build_launch_command(
    cfg: MoEPretrainConfig,
    megatron_script: str | Path,
    nproc: int = 1,
    *,
    nnodes: int = 1,
    rdzv_endpoint: str | None = None,
) -> list[str]:
    """Full torch-elastic launch command, one process per GPU.

    Launched as ``python -m torch.distributed.run`` (not the ``torchrun`` console script) so the
    workers inherit *this* interpreter via ``sys.executable`` ensuring that venv's own deps
    and the container's system torch/Transformer Engine are visible.
    """
    launch = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(nproc)]
    if nnodes > 1 or rdzv_endpoint is not None:
        if rdzv_endpoint is None:
            raise ValueError("multi-node launch (nnodes > 1) requires rdzv_endpoint (HOST:PORT)")
        launch += [
            "--nnodes",
            str(nnodes),
            "--rdzv-backend",
            "c10d",
            "--rdzv-endpoint",
            rdzv_endpoint,
            "--max-restarts",
            "0",
        ]
    else:
        launch += ["--standalone"]
    return [*launch, str(megatron_script), *build_megatron_args(cfg)]

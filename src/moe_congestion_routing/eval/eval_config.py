"""Config for one evaluation run through the pinned lm-evaluation-harness's ``megatron_lm``
backend (``lm_eval/models/megatron_lm.py``).
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from ..paths import expand_path
from ..training.pretrain_config import resolve_run_dir

# Key a config file uses to name its base config(s), exactly as MoEPretrainConfig's loader does.
# Consumed by the loader and never an EvalConfig field, so it is stripped before construction.
_EXTENDS_KEY = "extends"

# These two Megatron flags must reach every evaluation of one of our own checkpoints, no matter
# what a config's own extra_args sets:
#
# --no-use-tokenizer-model-from-checkpoint-args: --use-checkpoint-args (which the harness backend
# always passes) restores tokenizer_type/tokenizer_model from the checkpoint's saved args with
# force=True, silently overwriting the HuggingFaceTokenizer/gpt2 pair below with the training
# run's NullTokenizer -- which has no vocabulary and cannot turn text into ids.
#
# --no-gradient-accumulation-fusion: gradient_accumulation_fusion defaults True and needs an apex
# CUDA extension that isn't built on this box; nothing accumulates gradients at eval anyway.
#
# Appended by build_lm_eval_args itself rather than left to a config's own extra_args, so a config
# that sets extra_args cannot accidentally (or deliberately) drop either one.
_MANDATORY_EXTRA_ARGS = (
    "--no-use-tokenizer-model-from-checkpoint-args",
    "--no-gradient-accumulation-fusion",
)


def _load_yaml_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    """Load a yaml mapping, resolving an optional ``extends:`` chain into one merged dict.

    Identical shape to ``training/pretrain_config.py``'s loader: bases are merged first (in
    listed order, each recursively resolved), then the current file's own keys override them.
    ``extends`` paths are relative to the file that declares them. Cycles raise rather than
    recurse forever.
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
class EvalConfig:
    """Everything needed to launch one lm-evaluation-harness run against a Megatron checkpoint,
    loadable from a yaml file."""

    ckpt_step: int
    """Checkpoint iteration to load, and to name the output directory after
    (``evals/iter_{step:07d}``). Required, with no default."""

    load: str | None = None
    """Checkpoint DIRECTORY to evaluate (a ``<run>/checkpoints`` dir, same semantics as
    training's ``load``). Required at launch; kept optional here only so an unrelated config
    error (e.g. a bad task name) is not masked by a missing-load error first."""

    tokenizer_type: str = "HuggingFaceTokenizer"
    """Real tokenizer for evaluating our own checkpoints, NEVER NullTokenizer, which has no
    vocabulary and cannot turn raw eval-task text into ids. NullTokenizer is only used during
    training as the dataset is pre-tokenized. Restoring from checkpoint without this would restore
    the wrong tokenizer and produce plausible but meaningless scores rather than an error."""

    tokenizer_model: str = "assets/tokenizer/gpt2"
    """The real GPT-2 BPE tokenizer, committed in-tree, loaded fully offline. Not the caller's
    choice as our model is trained using a GPT-2 id space because ClimbMix ships pre-tokenized."""

    seq_length: int | None = None
    """Sequence length the checkpoint was trained at (also the harness's max context).
    Passing the wrong one silently truncates or pads incorrectly rather than erroring."""

    micro_batch_size: int | None = None
    """Megatron micro-batch size for the eval forward passes."""

    batch_size: int | None = None
    """lm-eval's own batching (documents per batch). Independent of micro_batch_size."""

    num_fewshot: int | None = None
    """Few-shot example count. ``None`` leaves the harness's per-task default. A real run should
    set this explicitly per the task's fixed split."""

    tasks: list[str] = field(default_factory=list)
    """Task names to evaluate, e.g. ``["arc_easy"]``. Which tasks belong to the suite is a
    decision this config does not make. A config just names the ones it runs."""

    seed: int = 1234
    """Python/numpy/torch RNG seed passed to the harness. Recorded in the config snapshot
    alongside fewshot_seed so a run's provenance is reproducible."""

    fewshot_seed: int = 1234
    """Few-shot example sampling seed, independent of ``seed``. Must be identical across arms
    being compared, or their demonstration sets differ and the comparison stops being paired."""

    limit: float | None = None
    """Cap examples per task (integer count or a 0-1 fraction). ``None`` runs the full task.
    Sanity probes and smoke runs set this. Should not be set for a real score."""

    output_dir: str | None = None
    """Explicit override for where results land, given as the run directory
    (``<output_dir>/iter_{step:07d}``). ``None`` (the normal case) derives the path from the
    checkpoint's own run directory instead, see ``eval_output_dir``. Needed for a checkpoint
    this repo did not produce (no run directory of ours to nest under) or one moved to read-only
    storage, where the derived path would not be writable."""

    extra_args: str = ""
    """Additional Megatron CLI flags, space-separated, forwarded through the harness's own
    ``extra_args`` model_args key. The two mandatory flags above are appended by
    ``build_lm_eval_args`` regardless of what this holds, so a config cannot drop them by setting
    this."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Build from a yaml file. Unknown keys raise ``TypeError`` (fail loud).

        A file may declare ``extends: <path>`` (or a list of paths) to inherit from one or more
        base configs, exactly as ``MoEPretrainConfig.from_yaml`` does.
        """
        data = _load_yaml_with_extends(Path(path))
        return cls(**data)

    def resolved(self, repo_root: Path) -> "EvalConfig":
        """Absolutise every path-valued field against ``repo_root``."""

        def absolutise(p: str) -> str:
            # Expand ${VAR} and ~ first, so a committed config can reference ${DATA_STORE} rather
            # than a personal path, and an unset variable fails loud instead of being taken
            # literally as a directory name.
            path = Path(expand_path(p))
            return str(path if path.is_absolute() else repo_root / path)

        return replace(
            self,
            load=absolutise(self.load) if self.load else None,
            tokenizer_model=absolutise(self.tokenizer_model),
            output_dir=absolutise(self.output_dir) if self.output_dir else None,
        )


def eval_output_dir(cfg: EvalConfig) -> Path:
    """Where this evaluation's results and provenance land.

    Normally the checkpoint's own run directory, mirroring ``checkpoints/`` with ``evals/``:
    ``resolve_run_dir(..., load=cfg.load) / "evals" / f"iter_{cfg.ckpt_step:07d}"``. Reuses
    ``resolve_run_dir`` (imported from ``training.pretrain_config``) rather than re-deriving the
    parent-of-checkpoints relationship a second time, so the two stay in sync by construction.

    An explicit ``cfg.output_dir`` overrides this, giving ``<output_dir>/iter_{step:07d}``
    directly. E.g. for a checkpoint this repo did not produce like the FLAME-MoE checkpoints.
    """
    step_dir = f"iter_{cfg.ckpt_step:07d}"
    if cfg.output_dir is not None:
        return Path(cfg.output_dir) / step_dir
    # The first positional argument is unused whenever `load` is given (resolve_run_dir returns
    # the load path's own parent in that case), which is always true for a real eval run.
    return resolve_run_dir(cfg.output_dir or ".", load=cfg.load) / "evals" / step_dir


def build_lm_eval_args(cfg: EvalConfig) -> list[str]:
    """Map the config to the lm-evaluation-harness CLI arg list (pure)."""
    model_args_parts = []
    if cfg.load is not None:
        model_args_parts.append(f"load={cfg.load}")
    model_args_parts.append(f"ckpt_step={cfg.ckpt_step}")
    model_args_parts.append(f"tokenizer_type={cfg.tokenizer_type}")
    model_args_parts.append(f"tokenizer_model={cfg.tokenizer_model}")
    if cfg.seq_length is not None:
        model_args_parts.append(f"seq_length={cfg.seq_length}")
    if cfg.micro_batch_size is not None:
        model_args_parts.append(f"micro_batch_size={cfg.micro_batch_size}")
    # extra_args' value itself contains spaces (it is a space-separated flag list, shlex-parsed
    # by the harness), so it must be the last comma-joined part: the harness splits model_args on
    # commas first, and a flag list has none, so nothing here can be mistaken for a further key.
    extra = " ".join(part for part in (cfg.extra_args, *_MANDATORY_EXTRA_ARGS) if part)
    model_args_parts.append(f"extra_args={extra}")

    args = [
        "--model",
        "megatron_lm",
        "--model_args",
        ",".join(model_args_parts),
    ]
    if cfg.tasks:
        args += ["--tasks", ",".join(cfg.tasks)]
    if cfg.num_fewshot is not None:
        args += ["--num_fewshot", str(cfg.num_fewshot)]
    if cfg.batch_size is not None:
        args += ["--batch_size", str(cfg.batch_size)]
    if cfg.limit is not None:
        args += ["--limit", str(cfg.limit)]
    # Both seeds always reach the command line explicitly (python,numpy,torch,fewshot), rather
    # than relying on --seed's single-value form to apply the same seed to all four. Arms compared
    # at 10-shot must draw the same demonstration examples, so the few-shot sampling seed has to be
    # trackable independently of the RNG seed rather than tied to it by --seed's shorthand.
    args += ["--seed", f"{cfg.seed},{cfg.seed},{cfg.seed},{cfg.fewshot_seed}"]
    args += ["--output_path", str(eval_output_dir(cfg))]
    return args

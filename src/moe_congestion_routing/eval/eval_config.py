"""Config for one evaluation run through the pinned lm-evaluation-harness's ``megatron_lm``
backend (``lm_eval/models/megatron_lm.py``).
"""

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..checkpoint_args import CHECKPOINT_OVERRIDE_ARGS
from ..config_extends import load_yaml_with_extends
from ..paths import expand_path
from ..training.pretrain_config import resolve_run_dir


@dataclass(frozen=True)
class EvalConfig:
    """Everything needed to launch one lm-evaluation-harness run against a Megatron checkpoint,
    loadable from a yaml file."""

    ckpt_step: int | None = None
    """Checkpoint iteration to load, and to name the output directory after
    (``evals/iter_{step:07d}``). Required at launch via config field or CLI arg override."""

    load: str | None = None
    """Checkpoint DIRECTORY to evaluate (a ``<run>/checkpoints`` dir, same semantics as
    training's ``load``). Required at launch via config field or CLI arg override."""

    tokenizer_type: str = "HuggingFaceTokenizer"
    """Real tokenizer for evaluating our own checkpoints, NEVER NullTokenizer, which has no
    vocabulary and cannot turn raw eval-task text into ids. NullTokenizer is only used during
    training as the dataset is pre-tokenized. Restoring from checkpoint without this would restore
    the wrong tokenizer and produce plausible but meaningless scores rather than an error."""

    tokenizer_model: str = "./assets/tokenizer/gpt2"
    """The real GPT-2 BPE tokenizer, committed in-tree, loaded fully offline. Not the caller's
    choice as our model is trained using a GPT-2 id space because ClimbMix ships pre-tokenized.
    The leading ``./`` is load-bearing, not decoration: it is what tells ``resolved()`` this is a
    filesystem path rather than a hub id like FLAME's ``EleutherAI/pythia-12b``, so it must stay
    even though a bare ``assets/tokenizer/gpt2`` would resolve identically on this machine."""

    seq_length: int | None = None
    """Sequence length the checkpoint was trained at (also the harness's max context).
    Passing the wrong one silently truncates or pads incorrectly rather than erroring."""

    micro_batch_size: int | None = None
    """Megatron micro-batch size for the eval forward passes."""

    batch_size: int | None = None
    """lm-eval's own batching (documents per batch). Independent of micro_batch_size."""

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
    """Explicit override for the run directory results nest under\
    (``<output_dir>/evals/iter_{step:07d}``). ``None`` derives the run directory from the
    checkpoint's own instead. Needed for a checkpoint this repo did not produce, e.g. FLAME-MoE's"""

    include_path: str = "configs/eval/tasks"
    """Directory lm-eval searches for extra task/group yaml files beyond its own built-in
    registry, e.g. ``configs/eval/tasks/flame_suite.yaml``. Without ``--include_path`` pointed
    here, ``tasks: [flame_suite]`` fails with "task not found" because the harness never looks
    outside its own registry on its own."""

    extra_args: str = ""
    """Additional Megatron CLI flags, space-separated, forwarded through the harness's own
    ``extra_args`` model_args key. ``checkpoint_args.CHECKPOINT_OVERRIDE_ARGS`` is appended by
    ``_model_args_parts`` regardless of what this holds, so a config cannot drop them by setting
    this."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Build from a yaml file. Unknown keys raise ``TypeError`` (fail loud).

        A file may declare ``extends: <path>`` (or a list of paths) to inherit from one or more
        base configs, exactly as ``MoEPretrainConfig.from_yaml`` does.
        """
        data = load_yaml_with_extends(Path(path))
        return cls(**data)

    def resolved(self, repo_root: Path) -> "EvalConfig":
        """Absolutise every path-valued field against ``repo_root``, but only absolutise
        ``tokenizer_model`` when it is explicitly marked as a filesystem path, since it may
        instead be a hub identifier such as FLAME's ``EleutherAI/pythia-12b``."""

        def absolutise(p: str) -> str:
            # Expand ${VAR} and ~ first, so a committed config can reference ${DATA_STORE} rather
            # than a personal path, and an unset variable fails loud instead of being taken
            # literally as a directory name.
            path = Path(expand_path(p))
            return str(path if path.is_absolute() else repo_root / path)

        def absolutise_tokenizer(p: str) -> str:
            # AutoTokenizer.from_pretrained takes either a local directory or a hub id, so this
            # cannot be absolutised like load/output_dir: prefixing repo_root onto a hub id such
            # as "EleutherAI/pythia-12b" makes a path that does not exist. Treat the value as a
            # path only when it is marked as one, and pass anything else through as a hub id.
            expanded = expand_path(p)
            if not expanded.startswith(("/", "~", "./", "../")):
                return expanded
            path = Path(expanded)
            return str(path if path.is_absolute() else repo_root / path)

        return replace(
            self,
            load=absolutise(self.load) if self.load else None,
            tokenizer_model=absolutise_tokenizer(self.tokenizer_model),
            output_dir=absolutise(self.output_dir) if self.output_dir else None,
            include_path=absolutise(self.include_path),
        )

    def require_launch_ready(self) -> None:
        """Raise if ``load`` or ``ckpt_step`` is still unset.

        Both are optional on ``EvalConfig`` itself so one committed config can be reused across
        checkpoints from the command line (``scripts/run_lm_eval.py``'s ``--load``/``--ckpt-step``)
        instead of needing one config file per checkpoint. But they are required to load a model
        and launch an evaluation, so this method requires them to be set before building the
        command line.
        """
        missing = []
        if self.load is None:
            missing.append("`load` (set it in the config, or pass --load)")
        if self.ckpt_step is None:
            missing.append("`ckpt_step` (set it in the config, or pass --ckpt-step)")
        if missing:
            raise ValueError("eval config is missing required field(s): " + "; ".join(missing))


def eval_run_dir(cfg: EvalConfig) -> Path:
    """The run directory this evaluation's results nest under, one level above ``evals/``.

    Normally the checkpoint's own run directory: ``resolve_run_dir(..., load=cfg.load)``, reused
    (imported from ``training.pretrain_config``) rather than re-deriving the
    parent-of-checkpoints relationship a second time, so the two stay in sync by construction.

    An explicit ``cfg.output_dir`` overrides this and IS the run directory directly, since an
    externally-produced checkpoint (e.g. FLAME-MoE's) has no per-checkpoint run directory to nest
    an ``evals/`` tail under.
    """
    if cfg.output_dir is not None:
        return Path(cfg.output_dir)
    # The first positional argument is unused whenever `load` is given (resolve_run_dir returns
    # the load path's own parent in that case), which is always true for a real eval run.
    return resolve_run_dir(cfg.output_dir or ".", load=cfg.load)


def eval_output_dir(cfg: EvalConfig) -> Path:
    """Where this evaluation's results land: ``eval_run_dir(cfg) / "evals" /
    f"iter_{cfg.ckpt_step:07d}"``, mirroring ``checkpoints/`` with ``evals/``."""
    return eval_run_dir(cfg) / "evals" / f"iter_{cfg.ckpt_step:07d}"


def eval_arm(cfg: EvalConfig) -> str | None:
    """Which arm's routing rule this evaluation scored, or ``None`` when that cannot be told
    apart from an arbitrary directory name.

    With an explicit ``cfg.output_dir`` override, that directory IS the run directory (see
    ``eval_run_dir``) and the caller chose its name, so its own basename is the arm, e.g.
    ``artifacts/eval/flame_290m`` names the arm ``flame_290m``. Without an override, the run
    directory is ``<output_dir>/<arm>/<run_tag>``, but only if it was produced by our launcher,
    which this checks by probing the filesystem for the ``launch_command.txt`` it writes.
    """
    run_dir = eval_run_dir(cfg)
    if cfg.output_dir is not None:
        return run_dir.name
    if not (run_dir / "launch_command.txt").exists():
        return None
    return run_dir.parent.name


def _model_args_parts(cfg: EvalConfig, devices: int) -> list[str]:
    """Build the comma-joined ``--model_args`` key=value parts, called from ``build_launch_command``
    by way of ``_lm_eval_args``.

    ``devices`` is the backend's own parallelism key (``lm_eval/models/megatron_lm.py``), not an
    ``EvalConfig`` field, and always reflects ``--nproc-per-node`` of the launch command it is
    part of.
    """
    model_args_parts = []
    if cfg.load is not None:
        model_args_parts.append(f"load={cfg.load}")
    if cfg.ckpt_step is not None:
        model_args_parts.append(f"ckpt_step={cfg.ckpt_step}")
    model_args_parts.append(f"devices={devices}")
    model_args_parts.append(f"tokenizer_type={cfg.tokenizer_type}")
    model_args_parts.append(f"tokenizer_model={cfg.tokenizer_model}")
    if cfg.seq_length is not None:
        model_args_parts.append(f"seq_length={cfg.seq_length}")
    if cfg.micro_batch_size is not None:
        model_args_parts.append(f"micro_batch_size={cfg.micro_batch_size}")
    # extra_args' value itself contains spaces (it is a space-separated flag list, shlex-parsed
    # by the harness), so it must be the last comma-joined part: the harness splits model_args on
    # commas first, and a flag list has none, so nothing here can be mistaken for a further key.
    extra = " ".join(part for part in (cfg.extra_args, *CHECKPOINT_OVERRIDE_ARGS) if part)
    model_args_parts.append(f"extra_args={extra}")
    return model_args_parts


def _lm_eval_args(cfg: EvalConfig, devices: int) -> list[str]:
    """The lm-evaluation-harness CLI arg list (pure)"""
    args = [
        "--model",
        "megatron_lm",
        "--model_args",
        ",".join(_model_args_parts(cfg, devices=devices)),
    ]
    # Always emitted, not guarded like the optional flags below: a config naming only built-in
    # tasks is unaffected by pointing the harness at one extra directory, while a config naming
    # a group config (e.g. flame_suite) fails with "task not found" without this.
    args += ["--include_path", cfg.include_path]
    if cfg.tasks:
        args += ["--tasks", ",".join(cfg.tasks)]
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


def build_launch_command(cfg: EvalConfig, nproc: int = 1) -> list[str]:
    """Full torch-elastic launch command that scores one checkpoint data-parallel across
    ``nproc`` GPUs on one node.

    ``--nproc-per-node`` and the backend's own ``devices`` must be in sync and are therefore derived
    from ``nproc``.
    """
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(nproc),
        "-m",
        "lm_eval",
        *_lm_eval_args(cfg, devices=nproc),
    ]

"""Build the ``torchrun`` command that drives ``scripts/moe_generate.py``.

Pure: no ``megatron``, no ``torch``. Mirrors ``eval/eval_config.py``'s command-building shape
(a frozen request dataclass plus a builder function) but deliberately loads no yaml, because a
generation run names a checkpoint and takes its architecture from ``--use-checkpoint-args``
rather than restating it in a config file.
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

from ..checkpoint_args import checkpoint_override_argv
from ..paths import expand_path


@dataclass(frozen=True)
class GenerateRequest:
    """Everything needed to generate text from one of our checkpoints, ours or FLAME's."""

    load: str
    """Checkpoint DIRECTORY to load (a ``<run>/checkpoints`` dir, or FLAME's own layout)."""

    ckpt_step: int | None = None
    """Load this iteration instead of the newest."""

    prompts: tuple[str, ...] = ("The capital of France is",)
    """One-shot prompts. Ignored in interactive mode, where prompts come from stdin instead."""

    interactive: bool = False
    """Read prompts from the terminal until Ctrl-D instead of using ``prompts``."""

    engine: str = "auto"  # auto | static | dynamic
    """Which inference engine to run. ``auto`` picks dynamic where flash-attn clears Megatron's
    ``>= 2.7.3`` gate and static everywhere else, since only static runs without it."""

    max_new_tokens: int = 64
    """New tokens to generate per prompt."""

    temperature: float = 1.0
    """Sampling temperature (only matters when ``top_k`` != 1)."""

    top_k: int = 1
    """Top-k sampling; ``1`` = greedy, so the two engines are comparable."""

    top_p: float = 0.0
    """Top-p (nucleus) sampling; ``0.0`` disables it."""

    tokenizer_type: str = "HuggingFaceTokenizer"
    """Real tokenizer, never ``NullTokenizer``, since raw prompt text needs turning into ids."""

    tokenizer_model: str = "./assets/tokenizer/gpt2"
    """The real GPT-2 BPE tokenizer, committed in-tree. A hub id (e.g. FLAME's
    ``EleutherAI/pythia-12b``) is also valid. See ``resolved`` for the marker that distinguishes
    the two."""

    attention_backend: str = "auto"
    """TE attention backend (flash/fused/unfused/auto/local)."""

    passthrough: tuple[str, ...] = ()
    """Extra Megatron flags forwarded verbatim, positioned before ``--prompts`` so a caller can
    override anything earlier in the arg list."""

    def resolved(self, repo_root: Path) -> "GenerateRequest":
        """Absolutise ``load`` against ``repo_root``, and ``tokenizer_model`` only when it is
        explicitly marked as a filesystem path, mirroring
        ``eval/eval_config.py::EvalConfig.resolved`` so a hub id such as FLAME's
        ``EleutherAI/pythia-12b`` is not mistaken for a nonexistent local directory.
        """

        def absolutise(p: str) -> str:
            path = Path(expand_path(p))
            return str(path if path.is_absolute() else repo_root / path)

        def absolutise_tokenizer(p: str) -> str:
            expanded = expand_path(p)
            if not expanded.startswith(("/", "~", "./", "../")):
                return expanded
            path = Path(expanded)
            return str(path if path.is_absolute() else repo_root / path)

        return replace(
            self,
            load=absolutise(self.load),
            tokenizer_model=absolutise_tokenizer(self.tokenizer_model),
        )


def build_generate_command(
    req: GenerateRequest, generate_script: str | Path, nproc: int = 1
) -> list[str]:
    """Full ``torchrun`` command driving ``moe_generate.py`` (pure).

    ``--prompts`` is Megatron's own ``nargs='+'`` flag, so anything emitted after it is
    swallowed as a prompt: it comes last, and interactive mode omits it entirely because prompts
    then come from stdin instead of argv.
    """
    args: list[str] = [
        "--use-checkpoint-args",
        "--bf16",
        "--transformer-impl",
        "transformer_engine",
        "--distributed-backend",
        "nccl",
        "--load",
        req.load,
    ]
    if req.ckpt_step is not None:
        args += ["--ckpt-step", str(req.ckpt_step)]
    args += [
        "--tokenizer-type",
        req.tokenizer_type,
        "--tokenizer-model",
        req.tokenizer_model,
        "--attention-backend",
        req.attention_backend,
        "--num-tokens-to-generate",
        str(req.max_new_tokens),
        "--temperature",
        str(req.temperature),
        "--top_k",
        str(req.top_k),
        "--top_p",
        str(req.top_p),
        "--engine",
        req.engine,
    ]
    if req.interactive:
        args.append("--interactive")
    args += checkpoint_override_argv()
    args += list(req.passthrough)
    if not req.interactive:
        args += ["--prompts", *req.prompts]
    # `sys.executable -m torch.distributed.run` rather than a bare `torchrun`, because the
    # cluster venv is built with --system-site-packages and `uv sync --no-install-package
    # torch`, so it has no torchrun of its own and PATH finds the container's, which runs the
    # system python and cannot see the venv's transformers. Same form as eval's launcher.
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(nproc),
        str(generate_script),
        *args,
    ]

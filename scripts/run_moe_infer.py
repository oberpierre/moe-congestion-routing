#!/usr/bin/env python
"""Generate text from a trained MoE checkpoint through Megatron's static-inference pipeline.

Reads an inference-config yaml, builds the CLI arg list, and execs ``torchrun`` on Megatron's
shipped ``examples/inference/advanced/gpt_static_inference.py`` — so the model build, checkpoint
load, and generation all run through Megatron's real StaticInferenceEngine.

Usage:
    uv run python scripts/run_moe_infer.py --config configs/infer/climblab_moe_smoke.yaml \
        --load artifacts/moe_smoke/<timestamp>/checkpoints
    uv run python scripts/run_moe_infer.py --config <cfg> --load <ckpt> --prompt "Hello there"
    uv run python scripts/run_moe_infer.py --config <cfg> --load <ckpt> --dry-run
"""

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from moe_congestion_routing.training.infer_config import (
    MoEInferConfig,
    build_infer_launch_command,
)
from moe_congestion_routing.training.megatron_path import megatron_root


def _checkpoint_iters(path: Path) -> list[int]:
    """Iterations saved under a Megatron checkpoint dir (its ``iter_<N>/`` subdirs)."""
    return sorted(
        int(p.name[5:]) for p in path.glob("iter_*") if p.is_dir() and p.name[5:].isdigit()
    )


def _validate_checkpoint(
    load: Path, ckpt_step: int | None, parser: argparse.ArgumentParser
) -> None:
    """Fail fast if --load has no loadable checkpoint. Giving a clear error message incl. hints."""
    if not load.is_dir():
        parser.error(f"--load {load} is not a directory")
    iters = _checkpoint_iters(load)
    # ckpt_step bypasses the tracker and loads iter_<step>/ directly, otherwise retrieved from
    # latest_checkpointed_iteration.txt.
    if ckpt_step is not None:
        if ckpt_step not in iters:
            parser.error(f"--ckpt-step {ckpt_step} not in {load}; available iterations: {iters}")
        return
    if (load / "latest_checkpointed_iteration.txt").is_file():
        return
    nested = load / "checkpoints"
    hint = ""
    if nested.is_dir() and (nested / "latest_checkpointed_iteration.txt").is_file():
        hint = f"\n  did you mean the checkpoints subdir? --load {nested}"
    elif iters:
        hint = f"\n  found untracked iters {iters} (a crashed save?); pick one with --ckpt-step"
    parser.error(
        f"no complete checkpoint under --load {load} (no latest_checkpointed_iteration.txt).{hint}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a MoEInferConfig yaml file")
    parser.add_argument("--load", help="checkpoint dir; overrides the config's `load`")
    parser.add_argument("--ckpt-step", type=int, help="load this iteration instead of the newest")
    parser.add_argument(
        "--prompt", action="append", help="prompt text; repeatable, overrides the config's prompts"
    )
    parser.add_argument("--nproc", type=int, default=1, help="processes (GPUs) per node")
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = parser.parse_args()

    megatron_dir = megatron_root()  # validated vendored path; also the PYTHONPATH source below
    repo_root = megatron_dir.parent

    cfg = MoEInferConfig.from_yaml(args.config)
    # CLI overrides win, so one config can be pointed at any run's checkpoint ad hoc.
    if args.load:
        cfg = replace(cfg, load=args.load)
    if args.ckpt_step is not None:
        cfg = replace(cfg, ckpt_step=args.ckpt_step)
    if args.prompt:
        cfg = replace(cfg, prompts=args.prompt)
    cfg = cfg.resolved(repo_root)

    if not cfg.load:
        parser.error("no checkpoint given: set `load` in the yaml or pass --load <dir>")
    _validate_checkpoint(Path(cfg.load), cfg.ckpt_step, parser)

    infer_script = megatron_dir / "examples" / "inference" / "advanced" / "gpt_static_inference.py"
    cmd = build_infer_launch_command(cfg, infer_script, nproc=args.nproc)

    if args.dry_run:
        print(" ".join(cmd))
        return

    # gpt_static_inference.py imports `megatron` (and `model_provider`/`examples`) in the
    # subprocess, so the Megatron root must be on its PYTHONPATH.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(megatron_dir), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    # Load the vendored gpt2 tokenizer purely from disk — never reach the HF hub (offline cluster).
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    # One CUDA work queue per device (Megatron's recommended default; see run_moe_pretrain.py).
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    print(f"[run_moe_infer] launching:\n  {' '.join(cmd)}\n", flush=True)
    # Inherit the terminal: the generated text is the output we want to see.
    sys.exit(subprocess.run(cmd, env=env, cwd=repo_root).returncode)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate text from a trained MoE checkpoint, ours or FLAME's.

Builds a ``GenerateRequest``, turning it into a ``torchrun`` command driving
with CLI args to launch ``scripts/moe_generate.py``.

Usage:
    uv run python scripts/run_moe_infer.py --load <run>/checkpoints --ckpt-step 100 \
        --prompt "The capital of France is" --max-new-tokens 10
    uv run python scripts/run_moe_infer.py --load <run>/checkpoints --ckpt-step 100 --interactive
    uv run python scripts/run_moe_infer.py --load <run>/checkpoints --ckpt-step 100 --dry-run
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from moe_congestion_routing.infer.launch import GenerateRequest, build_generate_command
from moe_congestion_routing.training.megatron_path import megatron_root, torch_cuda_lib_dirs


def _checkpoint_iters(load: Path) -> list[int]:
    """Iterations saved under a Megatron checkpoint dir (its ``iter_<N>/`` subdirs)."""
    return sorted(
        int(p.name[5:]) for p in load.glob("iter_*") if p.is_dir() and p.name[5:].isdigit()
    )


def _validate_checkpoint(
    load: Path, ckpt_step: int | None, parser: argparse.ArgumentParser
) -> None:
    """Fail fast if ``--load`` has no loadable checkpoint, before spending a minute on the
    ``torchrun``/Transformer Engine imports.
    """
    if not load.is_dir():
        parser.error(f"--load {load} is not a directory")
    iters = _checkpoint_iters(load)
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
    # Split off everything after a bare `--` before argparse sees it, so it reaches Megatron
    # untouched rather than being rejected as an unrecognized argument by this parser.
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1 :]
    else:
        passthrough = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", required=True, help="checkpoints DIR to generate from")
    parser.add_argument(
        "--ckpt-step", type=int, default=None, help="load this iteration instead of the newest"
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        default=None,
        help="repeatable prompt text, overrides the default prompt",
    )
    prompt_group.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="read prompts from the terminal until EOF instead of a fixed --prompt list",
    )
    parser.add_argument("--engine", choices=["auto", "static", "dynamic"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--tokenizer-type", default="HuggingFaceTokenizer")
    parser.add_argument("--tokenizer-model", default="./assets/tokenizer/gpt2")
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--nproc", type=int, default=1, help="processes (GPUs) per node")
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = parser.parse_args(argv)

    # Only rank 0 can read a terminal, so this fails in a second here rather than after a minute
    # of imports inside moe_generate.py, which enforces the same rule as a fallback.
    if args.interactive and args.nproc > 1:
        parser.error("--interactive requires --nproc 1 (only rank 0 can read a terminal)")

    megatron_dir = megatron_root()
    repo_root = megatron_dir.parent

    req_kwargs: dict = {"load": args.load, "passthrough": tuple(passthrough)}
    if args.ckpt_step is not None:
        req_kwargs["ckpt_step"] = args.ckpt_step
    if args.prompts:
        req_kwargs["prompts"] = tuple(args.prompts)
    req = GenerateRequest(
        interactive=args.interactive,
        engine=args.engine,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        tokenizer_type=args.tokenizer_type,
        tokenizer_model=args.tokenizer_model,
        attention_backend=args.attention_backend,
        **req_kwargs,
    ).resolved(repo_root)

    _validate_checkpoint(Path(req.load), req.ckpt_step, parser)

    generate_script = repo_root / "scripts" / "moe_generate.py"
    cmd = build_generate_command(req, generate_script, nproc=args.nproc)

    # shlex.join, not " ".join: checkpoint_override_argv() and passthrough are already split
    # into individual tokens, but this keeps the printed/recorded command pasteable regardless of
    # whether a future flag value carries a space (see scripts/run_lm_eval.py for the bug this
    # avoids).
    if args.dry_run:
        print(shlex.join(cmd))
        return

    # moe_generate.py imports `megatron` in the subprocess, so it needs Megatron on its PYTHONPATH
    env = os.environ.copy()
    env["MEGATRON_PATH"] = str(megatron_dir)
    env["PYTHONPATH"] = os.pathsep.join([str(megatron_dir), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    # Transformer Engine dlopens libnccl/libcudnn at import; with a pip torch these live under
    # site-packages/nvidia/*/lib, off the loader path. Prepend them (no-op on the cluster's
    # system-CUDA container, where torch_cuda_lib_dirs() returns []).
    lib_dirs = torch_cuda_lib_dirs()
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*lib_dirs, env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
    # One CUDA work queue per device, matching run_moe_pretrain.py and run_lm_eval.py.
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    # Load the vendored gpt2 tokenizer purely from disk, instead of the HF hub.
    env.setdefault("HF_HOME", str(repo_root / "hf_cache"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    # --use-checkpoint-args reads common.pt, which pickles an argparse.Namespace rather than
    # only tensors, whereas torch.load now defaults to weights_only=True and rejects that
    # global by name.
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    print(f"[run_moe_infer] launching:\n  {shlex.join(cmd)}\n", flush=True)
    # Inherit the terminal rather than capturing output: the generated text and the REPL are the
    # entire point of this script.
    sys.exit(subprocess.run(cmd, env=env, cwd=repo_root).returncode)


if __name__ == "__main__":
    main()

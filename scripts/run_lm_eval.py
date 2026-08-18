#!/usr/bin/env python
"""Score a Megatron checkpoint with the pinned lm-evaluation-harness's ``megatron_lm`` backend.

Reads an eval-config yaml, builds the harness CLI arg list, and execs
``python -m torch.distributed.run -m lm_eval`` with the environment preamble the harness needs
(``MEGATRON_PATH``, a Transformer-Engine-safe ``LD_LIBRARY_PATH``, the offline HF cache switches).
``run_moe_pretrain.py`` builds the equivalent preamble for training and is the reference this
mirrors.

A config describes the experiment; the command line says which checkpoint to score. So
--load and --ckpt-step are not optional extras, they are how a config is aimed -- omitting
both fails loudly rather than falling back to some previously-named checkpoint.

Usage:
    uv run python scripts/run_lm_eval.py --config configs/eval/arc_easy.yaml \
        --load artifacts/e1_local/20260804-180522/checkpoints --ckpt-step 100
    uv run python scripts/run_lm_eval.py --config configs/eval/arc_easy.yaml \
        --load <run>/checkpoints --ckpt-step 100 --dry-run
"""

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from moe_congestion_routing.eval.eval_config import (
    EvalConfig,
    build_launch_command,
    eval_arm,
    eval_output_dir,
    eval_run_dir,
)
from moe_congestion_routing.training.megatron_path import megatron_root, torch_cuda_lib_dirs


def _harness_root(repo_root: Path) -> Path:
    return repo_root / "lm-evaluation-harness"


def _git(*args: str, cwd: Path) -> str | None:
    """Run a git command in `cwd`, returning stripped stdout or None if it fails.

    Used only for the config snapshot's provenance fields (the harness's tag and SHA), so a
    failure here (e.g. a detached-but-untagged checkout) should not block the run.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to an EvalConfig yaml file")
    parser.add_argument(
        "--load",
        default=None,
        help="checkpoints DIR to evaluate, e.g. artifacts/<run>/checkpoints. Overrides the "
        "config's load, so one committed config can be pointed at any arm/checkpoint from the "
        "command line instead of needing a file per pair.",
    )
    parser.add_argument(
        "--ckpt-step",
        type=int,
        default=None,
        help="checkpoint iteration to load, and to name the output directory after "
        "(evals/iter_{step:07d}). Overrides the config's ckpt_step.",
    )
    parser.add_argument("--nproc", type=int, default=1, help="processes (GPUs) per node")
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = parser.parse_args()

    megatron_dir = megatron_root()  # validated vendored path; also the PYTHONPATH source below
    repo_root = megatron_dir.parent
    harness_dir = _harness_root(repo_root)

    cfg = EvalConfig.from_yaml(args.config)
    # An --load off the command line has not been through expand_path yet, so we apply the override
    # before resolved() so both sources get the identical ${VAR}/~ expansion.
    if args.load is not None:
        cfg = replace(cfg, load=args.load)
    if args.ckpt_step is not None:
        cfg = replace(cfg, ckpt_step=args.ckpt_step)
    cfg = cfg.resolved(repo_root)
    # Require `load`/`ckpt_step` to be configured either via CLI or config
    cfg.require_launch_ready()

    results_dir = eval_output_dir(cfg)
    cmd = build_launch_command(cfg, nproc=args.nproc)

    # shlex.join, not " ".join: --model_args' value carries extra_args' own space-separated
    # flag list (e.g. "...,extra_args=--no-use-tokenizer-model-from-checkpoint-args
    # --no-gradient-accumulation-fusion"), a single argv element with a space inside it. Printed
    # or written unquoted, pasting the command back into a shell splits that element in two and
    # turns the second flag into a separate (and wrong) argument. subprocess.run below takes the
    # list directly, so it never saw this bug -- only the printed/recorded string did.
    if args.dry_run:
        print(shlex.join(cmd))
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    # Provenance: append the exact command, like run_moe_pretrain.py's launch_command.txt. This
    # file is genuinely one-per-iteration and meant to grow, unlike config_snapshot.yaml below.
    with open(results_dir / "eval_command.txt", "a") as f:
        f.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}\n{shlex.join(cmd)}\n\n")

    run_dir = eval_run_dir(cfg)
    # A results file next to no record of what produced it is a number nobody can defend later:
    # the harness submodule's own version, both RNG seeds, the task set, which checkpoint this
    # run scored, and which arm it belongs to.
    snapshot = {
        "harness_tag": _git("describe", "--tags", "--exact-match", cwd=harness_dir),
        "harness_sha": _git("rev-parse", "HEAD", cwd=harness_dir),
        "seed": cfg.seed,
        "fewshot_seed": cfg.fewshot_seed,
        "tasks": cfg.tasks,
        "load": cfg.load,
        "run_dir": str(run_dir),
        "ckpt_step": cfg.ckpt_step,
        "arm": eval_arm(cfg),
        # Not an EvalConfig field: the device count is deployment and doesn't influence the results.
        # Still kept to record what actually ran.
        "devices": args.nproc,
    }

    # lm-eval writes each run's results into <results_dir>/<subdir>/results_<timestamp>.json,
    # where <subdir> is a name the harness makes up for itself (eight random characters for our
    # --model_args, not a hash of them) and unknown until it runs. So there may be more than one
    # results file per iteration in case of a reruns or different task sets, and we have to diff
    # before and after to find the new subdirectory and put the config snapshot next to it.
    results_before = set(results_dir.glob("*/results_*.json"))

    # -m lm_eval imports megatron.core (via the harness's own MEGATRON_PATH lookup) in the
    # subprocess, so both the interpreter's PYTHONPATH and the harness-read MEGATRON_PATH env var
    # must carry it.
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
    # One CUDA work queue per device, matching run_moe_pretrain.py.
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    # Default HF_HOME to the project's gitignored hf_cache/
    env.setdefault("HF_HOME", str(repo_root / "hf_cache"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    # torch >= 2.6 defaults torch.load to weights_only=True; Megatron's common.pt pickles a full
    # argparse Namespace plus RNG state, not just tensors, so it needs this escape hatch.
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    print(f"[run_lm_eval] launching:\n  {shlex.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd, env=env, cwd=repo_root)

    if result.returncode == 0:
        new_results_files = sorted(set(results_dir.glob("*/results_*.json")) - results_before)
        for results_file in new_results_files:
            with open(results_file.parent / "config_snapshot.yaml", "w") as f:
                yaml.safe_dump(snapshot, f, sort_keys=False)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

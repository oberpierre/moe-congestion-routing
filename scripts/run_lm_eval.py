#!/usr/bin/env python
"""Score a Megatron checkpoint with the pinned lm-evaluation-harness's ``megatron_lm`` backend.

Reads an eval-config yaml, builds the harness CLI arg list, and execs
``python -m torch.distributed.run -m lm_eval`` with the environment preamble the harness needs
(``MEGATRON_PATH``, a Transformer-Engine-safe ``LD_LIBRARY_PATH``, the offline HF cache switches).
``run_moe_pretrain.py`` builds the equivalent preamble for training and is the reference this
mirrors.

Usage:
    uv run python scripts/run_lm_eval.py --config configs/eval/ours.yaml
    uv run python scripts/run_lm_eval.py --config configs/eval/ours.yaml --dry-run
"""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from moe_congestion_routing.eval.eval_config import EvalConfig, build_lm_eval_args, eval_output_dir
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
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = parser.parse_args()

    megatron_dir = megatron_root()  # validated vendored path; also the PYTHONPATH source below
    repo_root = megatron_dir.parent
    harness_dir = _harness_root(repo_root)

    cfg = EvalConfig.from_yaml(args.config).resolved(repo_root)

    run_dir = eval_output_dir(cfg)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "1",
        "-m",
        "lm_eval",
        *build_lm_eval_args(cfg),
    ]

    # shlex.join, not " ".join: --model_args' value carries extra_args' own space-separated
    # flag list (e.g. "...,extra_args=--no-use-tokenizer-model-from-checkpoint-args
    # --no-gradient-accumulation-fusion"), a single argv element with a space inside it. Printed
    # or written unquoted, pasting the command back into a shell splits that element in two and
    # turns the second flag into a separate (and wrong) argument. subprocess.run below takes the
    # list directly, so it never saw this bug -- only the printed/recorded string did.
    if args.dry_run:
        print(shlex.join(cmd))
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    # Provenance: append the exact command, like run_moe_pretrain.py's launch_command.txt.
    with open(run_dir / "eval_command.txt", "a") as f:
        f.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}\n{shlex.join(cmd)}\n\n")

    # A results file next to no record of what produced it is a number nobody can defend later:
    # the harness submodule's own version, both RNG seeds, the task set and the checkpoint
    # iteration this run resolved to.
    snapshot = {
        "harness_tag": _git("describe", "--tags", "--exact-match", cwd=harness_dir),
        "harness_sha": _git("rev-parse", "HEAD", cwd=harness_dir),
        "seed": cfg.seed,
        "fewshot_seed": cfg.fewshot_seed,
        "tasks": cfg.tasks,
        "ckpt_step": cfg.ckpt_step,
    }
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)

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
    sys.exit(subprocess.run(cmd, env=env, cwd=repo_root).returncode)


if __name__ == "__main__":
    main()

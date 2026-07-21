#!/usr/bin/env python
"""Launch a small MoE pretraining run through Megatron's ``pretrain_gpt.py``.

Reads a run-config yaml, builds the Megatron CLI arg list, and execs
``torchrun pretrain_gpt.py`` on the local GPU(s). We deliberately reuse Megatron's own
training loop so we get its native logging out of the box.

Usage:
    uv run python scripts/run_moe_pretrain.py --config configs/train/climblab_moe_smoke.yaml
    uv run python scripts/run_moe_pretrain.py --config <cfg> --dry-run   # print the command only
"""

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from moe_congestion_routing.training.megatron_path import megatron_root, torch_cuda_lib_dirs
from moe_congestion_routing.training.pretrain_config import (
    MoEPretrainConfig,
    build_launch_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a MoEPretrainConfig yaml file")
    parser.add_argument("--nproc", type=int, default=1, help="processes (GPUs) per node")
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="inherit the terminal (live tqdm bars) instead of teeing output to "
        "<output_dir>/train.log",
    )
    args = parser.parse_args()

    megatron_dir = megatron_root()  # validated vendored path; also the PYTHONPATH source below
    repo_root = megatron_dir.parent

    cfg = MoEPretrainConfig.from_yaml(args.config).resolved(repo_root)

    # Each invocation gets its own <output_dir>/<timestamp>/ for the log, the frozen command,
    # and checkpoints, so repeated/concurrent runs don't interfere with each other. The dataset
    # cache is deliberately shared at <output_dir>/cache (keyed by seed/seq_length) so the
    # sample/shuffle indices are built once and reused across runs.
    run_dir = Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    # Checkpointing enabled (save_interval set) but no explicit save dir → checkpoint into this
    # run's own dir, keeping each run's weights separate and trivial to locate for inference.
    if cfg.save_interval and not cfg.save:
        cfg = replace(cfg, save=str(run_dir / "checkpoints"))

    cmd = build_launch_command(cfg, megatron_dir / "pretrain_gpt.py", nproc=args.nproc)

    if args.dry_run:
        print(" ".join(cmd))
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.data_cache_path).mkdir(parents=True, exist_ok=True)
    if cfg.save:
        Path(cfg.save).mkdir(parents=True, exist_ok=True)

    # Provenance (the launch.sh "frozen script" equivalent): dump the exact command.
    (run_dir / "launch_command.txt").write_text(" ".join(cmd) + "\n")

    # pretrain_gpt.py imports `megatron` in the subprocess, so Megatron must be on its PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(megatron_dir), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    # Transformer Engine dlopens libnccl/libcudnn at import; with a pip torch these are under
    # site-packages/nvidia/*/lib and not on the loader path. Prepend them (no-op on the cluster's
    # system-CUDA container, where torch_cuda_lib_dirs() returns []).
    lib_dirs = torch_cuda_lib_dirs()
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*lib_dirs, env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
    # One CUDA work queue per device: Megatron needs this so tensor/sequence-parallel comms
    # overlap compute in a correct, deterministic order (recommended default even at TP=1).
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    # Unbuffered: when we capture stdout for the log (below) it becomes a pipe, which Python
    # would otherwise block-buffer (~8 KB) — making output laggy and truncating the log's tail
    # on a crash. Forcing per-write flushes keeps the teed log real-time and complete.
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[run_moe_pretrain] launching:\n  {' '.join(cmd)}\n", flush=True)

    if args.no_capture:
        # Inherit the terminal: preserves the TTY but writes no log file.
        sys.exit(subprocess.run(cmd, env=env, cwd=repo_root).returncode)

    # Tee stdout+stderr to the terminal and <output_dir>/train.log. Capturing turns the child's
    # stdout into a pipe, so tqdm bars won't be live (cosmetic) — use --no-capture if you want them.
    log_path = run_dir / "train.log"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered reads on our side
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()  # complete log even if the run crashes mid-stream
        returncode = proc.wait()
    print(f"\n[run_moe_pretrain] full log: {log_path}", flush=True)
    sys.exit(returncode)


if __name__ == "__main__":
    main()

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
from pathlib import Path

from moe_congestion_routing.training.megatron_path import megatron_root
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
    cmd = build_launch_command(cfg, megatron_dir / "pretrain_gpt.py", nproc=args.nproc)

    # Create output dirs so Megatron can write cache/checkpoints immediately.
    for path in (cfg.output_dir, cfg.data_cache_path, cfg.save):
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)

    # Provenance (the launch.sh "frozen script" equivalent): dump the exact command.
    (Path(cfg.output_dir) / "launch_command.txt").write_text(" ".join(cmd) + "\n")

    if args.dry_run:
        print(" ".join(cmd))
        return

    # pretrain_gpt.py imports `megatron` in the subprocess, so Megatron must be on its PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(megatron_dir), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
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
    log_path = Path(cfg.output_dir) / "train.log"
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

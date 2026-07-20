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

    print(f"[run_moe_pretrain] launching:\n  {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd, env=env, cwd=repo_root)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

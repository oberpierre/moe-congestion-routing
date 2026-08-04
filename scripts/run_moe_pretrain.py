#!/usr/bin/env python
"""Launch a small MoE pretraining run through Megatron's ``pretrain_gpt.py``.

Reads a run-config yaml, builds the Megatron CLI arg list, and execs
``python -m torch.distributed.run pretrain_gpt.py`` across the allocated GPU(s) (single-node
standalone by default; ``--nnodes``/``--rdzv-endpoint`` switch to a multi-node c10d rendezvous).
We deliberately reuse Megatron's own training loop so we get its native logging out of the box.

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

from moe_congestion_routing.paths import expand_path
from moe_congestion_routing.training.megatron_path import megatron_root, torch_cuda_lib_dirs
from moe_congestion_routing.training.pretrain_config import (
    MoEPretrainConfig,
    build_launch_command,
    resolve_run_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a MoEPretrainConfig yaml file")
    parser.add_argument("--nproc", type=int, default=1, help="processes (GPUs) per node")
    parser.add_argument(
        "--nnodes", type=int, default=1, help="number of nodes (>1 switches to a c10d rendezvous)"
    )
    parser.add_argument(
        "--rdzv-endpoint",
        default=None,
        help="HOST:PORT of node 0 for the multi-node rendezvous (required when --nnodes > 1)",
    )
    parser.add_argument(
        "--load",
        default=None,
        help="checkpoints DIR to resume from, e.g. artifacts/<run>/checkpoints. The run "
        "continues in that dir and fails loud if it holds no checkpoint. Omit => fresh timestamped "
        "run dir. Overrides the config's load.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="name (or path) for THIS run's artifact dir, relative to the config's output_dir. "
        "Gives a fresh run a stable name instead of a timestamp, and is REQUIRED for a WSD branch: "
        "with --load alone the run writes into the dir it loaded from, which is right for a "
        "resumed slice (one log, one W&B curve) but wrong for a branch, which is a separate "
        "lineage. e.g. --run-dir flame-budget --load artifacts/e1_cluster/trunk/checkpoints",
    )
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

    # The run dir holds the log, the frozen command, TensorBoard/W&B, and checkpoints.
    #
    # RESUME is EXPLICIT: pass --load <run>/checkpoints (or set `load` in the config). We then
    # continue IN THAT run's own dir (run_dir = the load dir's parent), so a sliced/interrupted run
    # picks up exactly where it left off, and we set --exit-on-missing-checkpoint so a wrong/empty
    # path FAILS LOUD immediately instead of silently restarting from random.
    #
    # BRANCH (WSD): --run-dir <name> alongside --load reads the trunk's checkpoints but writes
    # everything else to its own dir, giving it a distinct W&B run id. See resolve_run_dir.
    #
    # FRESH run (no --load): a new <output_dir>/<timestamp>/ dir, isolated from other runs. Multi-
    # node needs every node to agree on that dir, so the sbatch exports one shared MOE_RUN_TAG
    # (c10d assigns ranks, so rank 0 isn't pinned to a node); locally it falls back to a timestamp.
    # Expand ${VAR}/~ once, here, so both consumers below see the same path. cfg.load already went
    # through resolved(); an --load off the command line has not, and a literal "${SCRATCH}" is a
    # legal directory name that would otherwise be taken at face value.
    load = args.load or cfg.load
    if load:
        load = expand_path(load)
    run_dir = resolve_run_dir(
        cfg.output_dir,
        run_dir=args.run_dir,
        load=load,
        run_tag=os.environ.get("MOE_RUN_TAG") or datetime.now().strftime("%Y%m%d-%H%M%S"),
        is_branch=cfg.override_opt_param_scheduler,
    )
    if load:
        cfg = replace(cfg, load=str(Path(load).resolve()), exit_on_missing_checkpoint=True)
    # Checkpointing enabled (save_interval set) but no explicit save dir → checkpoint into this
    # run's own dir, keeping each run's weights separate and trivial to locate for inference.
    if cfg.save_interval and not cfg.save:
        cfg = replace(cfg, save=str(run_dir / "checkpoints"))
    if cfg.exit_interval and not cfg.save:
        # exit_interval only checkpoints when --save is set (Megatron); with no save dir the job
        # exits with nothing to resume from -- slicing would silently lose all progress.
        print(
            "[run_moe_pretrain] WARNING: exit_interval is set but no checkpoint dir is configured "
            "(set save_interval or save) -- the run will exit without saving and cannot resume.",
            flush=True,
        )

    # Logging sinks default into this run's own dir so every run's TensorBoard/W&B files sit
    # next to its log and checkpoints. W&B is only wired when a project is configured. Megatron
    # also requires a non-empty run name, so derive one from the config + timestamp when unset.
    if not cfg.tensorboard_dir:
        cfg = replace(cfg, tensorboard_dir=str(run_dir / "tensorboard"))
    if cfg.wandb_project:
        updates = {}
        if not cfg.wandb_save_dir:
            updates["wandb_save_dir"] = str(run_dir / "wandb")
        if not cfg.wandb_exp_name:
            updates["wandb_exp_name"] = f"{Path(args.config).stem}-{run_dir.name}"
        if updates:
            cfg = replace(cfg, **updates)

    cmd = build_launch_command(
        cfg,
        megatron_dir / "pretrain_gpt.py",
        nproc=args.nproc,
        nnodes=args.nnodes,
        rdzv_endpoint=args.rdzv_endpoint,
    )

    if args.dry_run:
        print(" ".join(cmd))
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.data_cache_path).mkdir(parents=True, exist_ok=True)
    if cfg.save:
        Path(cfg.save).mkdir(parents=True, exist_ok=True)

    # Provenance (the launch.sh "frozen script" equivalent): APPEND the exact command (node 0 only)
    # so a resumed slice preserves earlier slices' commands instead of clobbering them.
    if os.environ.get("SLURM_NODEID", "0") == "0":
        with open(run_dir / "launch_command.txt", "a") as f:
            f.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}\n{' '.join(cmd)}\n\n")

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
    # W&B: pin a stable run id = the run dir name (constant across slices, since a resume reuses the
    # same run dir) with resume="allow", so sliced/resumed training CONTINUES ONE W&B run -- an
    # unbroken curve -- instead of a fresh run per slice. Megatron's wandb.init() passes no
    # id/resume/group, so these env vars drive it (a user-set value still wins via setdefault);
    # "allow" creates the run on slice 1 and resumes it thereafter.
    #
    # ONE RUN ID PER MODEL LINEAGE, not per job. A resumed slice is the same lineage => same id =>
    # one continuous curve. A WSD branch is a DIFFERENT lineage that happens to share a prefix, so
    # --run-dir gives it its own id; sharing the trunk's would write two different models to the
    # same step numbers. The group ties them back together in the UI: Megatron logs against the
    # global iteration, so overlaying a branch on its trunk lines the curves up at the branch point.
    if cfg.wandb_project:
        env.setdefault("WANDB_RUN_ID", run_dir.name)
        env.setdefault("WANDB_RESUME", "allow")
        env.setdefault("WANDB_RUN_GROUP", cfg.wandb_group or Path(cfg.output_dir).name)

    print(f"[run_moe_pretrain] launching:\n  {' '.join(cmd)}\n", flush=True)

    if args.no_capture:
        # Inherit the terminal: preserves the TTY but writes no log file.
        sys.exit(subprocess.run(cmd, env=env, cwd=repo_root).returncode)

    # Tee stdout+stderr to the terminal and <output_dir>/train.log. Capturing turns the child's
    # stdout into a pipe, so tqdm bars won't be live (cosmetic) — use --no-capture if you want them.
    log_path = run_dir / "train.log"
    with open(log_path, "a") as logf:  # append: a resumed slice keeps earlier slices' log
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

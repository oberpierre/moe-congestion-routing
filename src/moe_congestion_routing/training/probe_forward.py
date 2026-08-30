"""Drives the router probe: the fire/no-fire schedule, startup validation, and the no-grad
forwards that produce one dump per probe asset.

``run_probe`` imports ``megatron`` because it only ever runs inside a Megatron training process.
The import is function-local so this module stays importable, and
``probe_fires``/``validate_probe_setup`` stay unit-testable, without it.
"""

import logging
import os
import subprocess
from pathlib import Path

from moe_congestion_routing.training.probe_batch import load_probe_batch

logger = logging.getLogger(__name__)


def probe_fires(iteration: int, coarse: int, dense: int, windows: list[tuple[int, int]]) -> bool:
    """True iff ``iteration`` sits on the coarse grid or inside a dense window.

    A union rather than two independent checks: a step landing on both the grid and a window's
    anchor must still produce exactly one dump.
    """
    if coarse and iteration % coarse == 0:
        return True
    if dense:
        for start, end in windows:
            if start <= iteration <= end and (iteration - start) % dense == 0:
                return True
    return False


def validate_probe_setup(args) -> None:
    """Fail loud, once at process start, on anything the probe cannot recover from mid-run.

    Needs no ``megatron`` import: the asset check is numpy-only (``load_probe_batch``), the
    tracked-asset check shells out to ``git``, and the directory check is plain ``os``/``pathlib``
    -- all things ``--dry-run`` deliberately does NOT run, because they read the asset and the
    filesystem rather than only the config shape (that split lives in ``pretrain_config.py``).
    Iterates every asset in ``args.moe_probe_batch``: the stem-uniqueness check already ran at
    config-build time, so this only re-runs the per-asset checks that need file/git access.
    """
    for probe_batch in args.moe_probe_batch:
        batch = load_probe_batch(probe_batch)
        if batch.seq_length != args.seq_length:
            raise ValueError(
                f"probe asset {probe_batch!r} has seq_length {batch.seq_length}, "
                f"which disagrees with --seq-length {args.seq_length}"
            )
        if batch.role == "standing":
            result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(probe_batch)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise ValueError(
                    f"probe asset {probe_batch!r} has role 'standing' but git does not "
                    f"track it: a standing probe must be a committed asset so every machine "
                    f"measures the same instrument, so commit it before a reported run fires it "
                    f"({result.stderr.strip()})"
                )

    probe_dir = Path(args.moe_probe_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(probe_dir, os.W_OK):
        raise ValueError(f"probe dir {args.moe_probe_dir!r} is not writable")


def run_probe(model, forward_step_func, iteration: int) -> None:
    """Run one no-grad forward per probe asset and dump each asset's router state separately.

    Mirrors the eval block's own scaffolding, since TP/CP/PP are forced to 1 so every
    data-parallel rank computes the identical loss on the identical batch and only rank 0 writes.
    """
    import torch
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
    from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
    from megatron.training import (
        get_args,
        get_tensorboard_writer,
        get_timers,
        get_tokenizer,
        get_wandb_writer,
    )
    from megatron.training.training import (
        disable_forward_pre_hook,
        enable_forward_pre_hook,
        should_disable_forward_pre_hook,
    )
    from megatron.training.utils import is_last_rank

    from moe_congestion_routing.metrics.router_probe import capturing, write_probe_dump
    from moe_congestion_routing.training.probe_batch import probe_micro_batches

    args = get_args()
    timers = get_timers()

    timers("interval-time").stop()
    is_init_dump = iteration == 0
    pre_hook_disabled = not is_init_dump and should_disable_forward_pre_hook(args)
    if pre_hook_disabled:
        disable_forward_pre_hook(model)
    for model_module in model:
        model_module.eval()
    rerun_state_machine = get_rerun_state_machine()
    rerun_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)

    # RNG neutrality is a guarantee this code makes rather than one that dropout-off and eval()
    # happen to give. Whether the forward path below consumes any RNG is unmeasured, so the state
    # is saved here and the equality check below is what establishes it.
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()

    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    forward_backward_func = get_forward_backward_func()
    tokenizer_eod = get_tokenizer().eod

    for probe_batch_path in args.moe_probe_batch:
        # One forward per asset rather than one forward over the concatenation: the LP unit is one
        # asset's tokens, so mixing assets would price a differently-sized game with a different
        # capacity dual. Splitting ONE asset into microbatches stays legitimate because
        # routing at inference is per-token given the weights and bias, with no capacity feedback
        # inside the forward, so the microbatch chunks can be reassembled afterwards.
        stem = Path(probe_batch_path).stem
        batch = load_probe_batch(probe_batch_path)
        micro_batches = probe_micro_batches(
            batch,
            micro_batch_size=args.micro_batch_size,
            num_sequences=args.moe_probe_seqs,
            seq_length=args.seq_length,
            eod_token=tokenizer_eod,
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=args.create_attention_mask_in_dataloader,
        )
        num_microbatches = args.moe_probe_seqs // args.micro_batch_size

        with (
            torch.no_grad(),
            capturing(iteration, args.micro_batch_size, args.moe_router_topk) as capture,
        ):
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=iter(micro_batches),
                model=model,
                num_microbatches=num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
            )

        total_loss = torch.zeros((), device="cuda")
        total_tokens = torch.zeros((), device="cuda")
        for loss_dict in loss_dicts:
            value = loss_dict["lm loss"]
            total_loss += value[0]
            total_tokens += value[1]
        probe_lm_loss = (total_loss / total_tokens).item()

        if writer:
            writer.add_scalar(f"probe_lm_loss/{stem}", probe_lm_loss, iteration)
        if wandb_writer and is_last_rank():
            wandb_writer.log({f"probe_lm_loss/{stem}": probe_lm_loss}, iteration)

        if torch.distributed.get_rank() == 0:
            meta = {
                "iteration": iteration,
                "moe_probe_batch": probe_batch_path,
                "token_sha256": batch.token_sha256,
                "role": batch.role,
                "seq_length": args.seq_length,
                "moe_router_score_function": args.moe_router_score_function,
                "moe_router_pre_softmax": args.moe_router_pre_softmax,
                "moe_probe_coarse_interval": args.moe_probe_coarse_interval,
                "moe_probe_dense_interval": args.moe_probe_dense_interval,
                "moe_probe_dense_windows": args.moe_probe_dense_windows,
                "moe_probe_seqs": args.moe_probe_seqs,
                "resumed_from_iteration": args.iteration,
                "probe_lm_loss": probe_lm_loss,
                "tensor_model_parallel_size": args.tensor_model_parallel_size,
                "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
                "context_parallel_size": args.context_parallel_size,
                "expert_model_parallel_size": args.expert_model_parallel_size,
                "data_parallel_size": args.data_parallel_size,
                "world_size": args.world_size,
            }
            path = Path(args.moe_probe_dir) / stem / f"iter_{iteration:07d}.npz"
            write_probe_dump(path, capture, meta)

    cpu_rng_state_after = torch.get_rng_state()
    cuda_rng_state_after = torch.cuda.get_rng_state()
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    assert torch.equal(cpu_rng_state, cpu_rng_state_after), (
        "the probe forward consumed CPU RNG: training's own RNG stream would drift depending on "
        "whether a step was probed"
    )
    assert torch.equal(cuda_rng_state, cuda_rng_state_after), (
        "the probe forward consumed CUDA RNG: training's own RNG stream would drift depending on "
        "whether a step was probed"
    )

    for model_module in model:
        model_module.train()
    rerun_state_machine.set_mode(rerun_mode)
    if pre_hook_disabled:
        enable_forward_pre_hook(model)
    timers("interval-time", log_level=0).start(barrier=True)
    get_moe_metrics_tracker().clear()

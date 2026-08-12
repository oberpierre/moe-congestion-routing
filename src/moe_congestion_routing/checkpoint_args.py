"""The Megatron flags that correct ``--use-checkpoint-args`` for one of our checkpoints.

Shared by the evaluation harness and the inference harness, since both load a checkpoint through
``--use-checkpoint-args`` and both need every one of these flags to avoid silently loading the
wrong tokenizer, numerics or epsilon.
"""

# These four Megatron flags must reach every load of one of our own checkpoints, no matter what a
# config's own extra_args sets:
#
# --no-use-tokenizer-model-from-checkpoint-args: --use-checkpoint-args (which every consumer of
# this module passes) restores tokenizer_type/tokenizer_model from the checkpoint's saved args
# with force=True, silently overwriting the HuggingFaceTokenizer/gpt2 pair below with the training
# run's NullTokenizer -- which has no vocabulary and cannot turn text into ids.
#
# --no-gradient-accumulation-fusion: gradient_accumulation_fusion defaults True and needs an apex
# CUDA extension that isn't built on this box; nothing accumulates gradients outside training.
#
# --moe-router-dtype fp32: --use-checkpoint-args restores structure but not numerics, so without
# this every checkpoint this project trains (fp32 routing) is loaded with Megatron's bf16 default.
#
# --norm-epsilon 1e-6: matches the epsilon every checkpoint this project trains and FLAME's own
# checkpoint are built at, whereas Megatron's own default is 1e-5.
CHECKPOINT_OVERRIDE_ARGS: tuple[str, ...] = (
    "--no-use-tokenizer-model-from-checkpoint-args",
    "--no-gradient-accumulation-fusion",
    "--moe-router-dtype fp32",
    "--norm-epsilon 1e-6",
)


def checkpoint_override_argv() -> list[str]:
    """``CHECKPOINT_OVERRIDE_ARGS`` split into argv tokens, e.g. for a subprocess launch that
    does not go through a shell and cannot rely on shell word-splitting."""
    argv: list[str] = []
    for arg in CHECKPOINT_OVERRIDE_ARGS:
        argv.extend(arg.split())
    return argv

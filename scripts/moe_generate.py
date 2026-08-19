#!/usr/bin/env python
"""``torchrun`` target that generates text from one of our checkpoints (ours or FLAME's).

Launched by ``scripts/run_moe_infer.py``, which builds the CLI arg list via
``moe_congestion_routing.infer.launch.build_generate_command``. Not meant to be invoked directly:
it imports ``megatron`` at module level, so it only works with ``PYTHONPATH``/``MEGATRON_PATH``
already pointed at the vendored submodule.
"""

import argparse
import sys

import torch
from megatron.core.inference.apis import MegatronLLM
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines import StaticInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import is_fa_min_version
from megatron.inference.utils import (
    add_inference_args,
    get_inference_config_from_model_and_args,
    get_model_for_inference,
)
from megatron.training import get_args, initialize_megatron, print_rank_0
from megatron.training.arguments import parse_and_validate_args

_FLASH_ATTN_MIN_VERSION = "2.7.3"


def add_generate_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """``--prompts``/``--num-tokens-to-generate``/``--temperature``/``--top_k``/``--top_p`` exist
    only in ``megatron.inference.utils.add_inference_args``, not in Megatron's own
    ``add_megatron_arguments`` (verified against the pinned tree), so this must add it.
    """
    add_inference_args(parser)
    group = parser.add_argument_group(title="moe_generate")
    group.add_argument(
        "--engine",
        choices=["auto", "static", "dynamic"],
        default="auto",
        help=f"auto picks dynamic when flash_attn >= {_FLASH_ATTN_MIN_VERSION} is importable, "
        "else static.",
    )
    group.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="read prompts from the terminal until Ctrl-D instead of using --prompts.",
    )
    return parser


def _has_flash_attn() -> bool:
    """True when flash-attn clears the ``>= 2.7.3`` gate Megatron asserts on before any
    dynamic-batching forward pass. This import is the only reliable signal, because
    ``--use-flash-attn`` is vestigial at this pin and ``--attention-backend auto`` succeeding
    only means Transformer Engine found some backend rather than this package.
    """
    try:
        return is_fa_min_version(_FLASH_ATTN_MIN_VERSION)
    except (ImportError, ModuleNotFoundError):
        return False


def _resolve_engine(requested: str) -> str:
    """Pick "static" or "dynamic", printing the choice on rank 0. Exits before the model is
    built when ``--engine dynamic`` is requested but flash-attn is unusable, since Megatron's
    own failure mode there is a ``ModuleNotFoundError`` raised from inside the first forward
    pass rather than a clear message at startup.
    """
    has_fa = _has_flash_attn()
    engine = requested
    if engine == "auto":
        engine = "dynamic" if has_fa else "static"
        print_rank_0(
            f"[moe_generate] --engine auto -> {engine} "
            f"(flash_attn >= {_FLASH_ATTN_MIN_VERSION} "
            f"{'found' if has_fa else 'not found'})"
        )
    if engine == "dynamic" and not has_fa:
        print_rank_0(
            f"[moe_generate] --engine dynamic requires flash_attn >= {_FLASH_ATTN_MIN_VERSION} "
            "importable in this environment, which it is not."
        )
        sys.exit(1)
    return engine


def _build_static_engine(model, args) -> StaticInferenceEngine:
    """``legacy=True`` selects true static batching, which is the only path that runs without
    flash-attn. The tokenizer is owned by the controller from here on, so it is not returned.
    """
    inference_context = StaticInferenceContext(
        args.inference_max_requests, args.inference_max_seq_length
    )
    wrapped_model = GPTInferenceWrapper(model, inference_context)
    controller = TextGenerationController(
        inference_wrapped_model=wrapped_model, tokenizer=build_tokenizer(args)
    )
    return StaticInferenceEngine(controller, legacy=True)


def _build_dynamic_engine(model, args) -> MegatronLLM:
    """``MegatronLLM`` refuses direct mode above EP 1, so this is single-expert-parallel only
    until someone wires up coordinator mode.
    """
    return MegatronLLM(
        model=model,
        tokenizer=build_tokenizer(args),
        inference_config=get_inference_config_from_model_and_args(model, args),
    )


def _generate(engine, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
    """One printing path for both engines: each exposes ``.generate(prompts, sampling_params)``
    returning objects with ``.generated_text``."""
    results = engine.generate(prompts=prompts, sampling_params=sampling_params)
    return [r.generated_text for r in results]


def _run_interactive(engine, sampling_params: SamplingParams) -> None:
    """Read prompts until Ctrl-D, printing prompt-then-continuation for each. The banner and the
    marker are written only to a terminal, so piped input produces the same output as one-shot
    mode. Only rank 0 reaches this, because the caller has already checked the world size is 1.
    """
    # readline() rather than `for line in sys.stdin`, whose read-ahead buffering can swallow a
    # typed line until more input arrives.
    at_terminal = sys.stdin.isatty()
    if at_terminal:
        print("Enter a prompt and press return. Ctrl-D exits.", flush=True)
    while True:
        if at_terminal:
            print(">>> ", end="", flush=True)
        line = sys.stdin.readline()
        if not line:
            break
        prompt = line.strip()
        if not prompt:
            continue
        [generated] = _generate(engine, [prompt], sampling_params)
        print(f"{prompt} {generated}", flush=True)
    if at_terminal:
        print()


def main() -> None:
    args = parse_and_validate_args(
        extra_args_provider=add_generate_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            # validate_args asserts micro_batch_size is set on every run, whereas nothing on
            # the inference path supplies one, so default it here rather than per engine.
            "micro_batch_size": 1,
            "exit_on_missing_checkpoint": True,
        },
    )
    initialize_megatron()

    if args.interactive and torch.distributed.get_world_size() > 1:
        print_rank_0(
            "[moe_generate] --interactive requires a world size of 1: only rank 0 can read a "
            "terminal. Pass --nproc 1."
        )
        sys.exit(1)

    engine_choice = _resolve_engine(args.engine)

    model = get_model_for_inference()
    args = get_args()

    if engine_choice == "static":
        engine = _build_static_engine(model, args)
    else:
        engine = _build_dynamic_engine(model, args)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_tokens_to_generate=args.num_tokens_to_generate,
    )

    if args.interactive:
        _run_interactive(engine, sampling_params)
        return

    generated = _generate(engine, args.prompts, sampling_params)
    for prompt, text in zip(args.prompts, generated, strict=True):
        print(f"{prompt} {text}")


if __name__ == "__main__":
    main()

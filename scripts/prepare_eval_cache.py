#!/usr/bin/env python
"""Pre-fetch every dataset and tokenizer an lm-eval run needs, into $HF_HOME.

Fetches the datasets for the seven benchmark tasks the evaluation suite runs
(see ``configs/eval/tasks/flame_suite.yaml``) plus the `EleutherAI/pythia-12b` tokenizer
FLAME's half of the evaluation needs. Our own checkpoints are evaluated with the in-tree
`assets/tokenizer/gpt2/`, which needs no network fetch.

Dataset identifiers (HF `dataset_path`/`dataset_name`) are likewise read from the pinned
lm-evaluation-harness submodule's own task registry (`TaskManager`) rather than hand-copied here,
so this script cannot silently drift from what the harness itself will request at eval time.

Usage:
    uv run python scripts/prepare_eval_cache.py --dry-run   # list what would be fetched
    uv run python scripts/prepare_eval_cache.py             # fetch for real (safe to rerun)
"""

import argparse
import os
from pathlib import Path

import yaml

# Only FLAME's checkpoints need this: FLAME trained with the NeoX BPE tokenizer, while our own
# checkpoints are evaluated with the GPT-2 tokenizer already committed at assets/tokenizer/gpt2/,
# which needs no hub fetch.
FLAME_TOKENIZER = "EleutherAI/pythia-12b"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _group_config_path() -> Path:
    return _repo_root() / "configs" / "eval" / "tasks" / "flame_suite.yaml"


def _tasks() -> list[str]:
    """The suite's task names, read from the group config rather than duplicated here -- so
    adding a task to that file is the only edit an eval job needs; this script picks it up on
    its next run with no second edit."""
    data = yaml.safe_load(_group_config_path().read_text())
    return [entry["task"] for entry in data["task"]]


def _ensure_hf_home() -> Path:
    """Resolve $HF_HOME, defaulting to the project's gitignored `hf_cache/` (matching the
    `hf_cache/` entry already in .gitignore and the `HF_HOME=$WORKDIR/hf_cache` the cluster
    launcher sets) so a bare local run doesn't scatter the cache into $HOME."""
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        hf_home = str(_repo_root() / "hf_cache")
        os.environ["HF_HOME"] = hf_home
    path = Path(hf_home)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _task_dataset_ids(tasks: list[str]) -> dict[str, tuple[str, str | None]]:
    """{task_name: (dataset_path, dataset_name)}, resolved through the harness's own
    TaskManager (which follows each task yaml's `include:` chain, e.g. arc_challenge.yaml
    includes arc_easy.yaml) rather than by reading the yaml files by hand."""
    from lm_eval.tasks.manager import TaskManager

    tm = TaskManager()
    ids: dict[str, tuple[str, str | None]] = {}
    for name in tasks:
        entry = tm.task_index.get(name)
        if entry is None:
            raise KeyError(
                f"{name!r} is not a task the pinned lm-evaluation-harness registers; "
                "the pinned harness and configs/eval/tasks/flame_suite.yaml have drifted"
            )
        ids[name] = (entry.cfg["dataset_path"], entry.cfg.get("dataset_name"))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be fetched; fetch nothing"
    )
    args = parser.parse_args()

    hf_home = _ensure_hf_home()
    tasks = _tasks()
    dataset_ids = _task_dataset_ids(tasks)

    print(f"HF_HOME={hf_home}")
    print("tasks:")
    for name, (dataset_path, dataset_name) in dataset_ids.items():
        suffix = f" ({dataset_name})" if dataset_name else ""
        print(f"  {name}: {dataset_path}{suffix}")
    print(f"tokenizer: {FLAME_TOKENIZER}")

    if args.dry_run:
        print("--dry-run: fetched nothing")
        return

    from lm_eval.tasks.manager import TaskManager
    from transformers import AutoTokenizer

    print("fetching task datasets...")
    # Building a ConfigurableTask calls its own .download() eagerly (lm_eval/api/task.py), so
    # this is the exact call the harness itself makes at eval time -- not a reimplementation of
    # it. datasets.load_dataset() is itself cache-aware, so rerunning this is a no-op.
    TaskManager().load(tasks)

    print(f"fetching tokenizer {FLAME_TOKENIZER}...")
    AutoTokenizer.from_pretrained(FLAME_TOKENIZER)

    print("done.")


if __name__ == "__main__":
    main()

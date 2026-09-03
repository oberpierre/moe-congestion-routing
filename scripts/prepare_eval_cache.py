#!/usr/bin/env python
"""Pre-fetch every dataset and tokenizer an lm-eval run needs, into $HF_HOME.

Fetches the datasets for every task named by the suite group configs under
``configs/eval/tasks/`` (flame_suite, extended_suite, ...) plus the `EleutherAI/pythia-12b`
tokenizer FLAME's half of the evaluation needs. Our own checkpoints are evaluated with the
in-tree `assets/tokenizer/gpt2/`, which needs no network fetch.

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


def _suite_paths() -> list[Path]:
    return sorted((_repo_root() / "configs" / "eval" / "tasks").glob("*.yaml"))


def _tasks() -> list[str]:
    """Task names across every suite group config, read from those files rather than duplicated
    here -- so adding a task (or a whole suite file) is the only edit an eval job needs; this
    script picks it up on its next run with no second edit. A member is either a plain string
    (also how a suite references a whole harness group such as ``blimp``) or a
    ``{task: name, ...}`` dict carrying per-suite overrides."""
    names: list[str] = []
    for path in _suite_paths():
        data = yaml.safe_load(path.read_text())
        for entry in data["task"]:
            name = entry if isinstance(entry, str) else entry["task"]
            if name not in names:
                names.append(name)
    return names


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


def _task_dataset_ids(tm, tasks: list[str]) -> dict[str, tuple[str, str | None]]:
    """{leaf_task_name: (dataset_path, dataset_name)}, resolved through the harness's own
    TaskManager (which follows each task yaml's `include:` chain, e.g. arc_challenge.yaml
    includes arc_easy.yaml) rather than by reading the yaml files by hand. A name may also be
    a harness *group* (e.g. ``blimp``, 67 subtasks); its members are resolved recursively so
    every leaf's dataset is accounted for -- including our own suite groups when one suite
    nests another (the TaskManager is built with configs/eval/tasks/ as include_path)."""
    ids: dict[str, tuple[str, str | None]] = {}
    for name in tasks:
        entry = tm.task_index.get(name)
        if entry is None:
            raise KeyError(
                f"{name!r} is not a task the pinned lm-evaluation-harness registers; "
                "the pinned harness and the suite configs under configs/eval/tasks/ have drifted"
            )
        if entry.cfg is not None and "dataset_path" in entry.cfg:
            ids[name] = (entry.cfg["dataset_path"], entry.cfg.get("dataset_name"))
        else:  # a group config: its cfg lists member names instead of a dataset
            members = [m if isinstance(m, str) else m["task"] for m in entry.cfg["task"]]
            ids.update(_task_dataset_ids(tm, members))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be fetched; fetch nothing"
    )
    args = parser.parse_args()

    hf_home = _ensure_hf_home()
    tasks = _tasks()

    print(f"HF_HOME={hf_home}")
    print(f"tasks: {', '.join(tasks)}")
    print(f"tokenizer: {FLAME_TOKENIZER}")
    print("indexing the harness task registry (>13000 task files)...", flush=True)

    from lm_eval.tasks.manager import TaskManager

    # The same include_path the eval run itself gets (EvalConfig.include_path), so a suite
    # referencing a sibling suite as a nested group (extended_suite -> flame_suite) resolves
    # here exactly as it will at eval time.
    tm = TaskManager(include_path=str(_repo_root() / "configs" / "eval" / "tasks"))
    leaf_names: list[str] = []
    for name in tasks:
        leaf_ids = _task_dataset_ids(tm, [name])
        leaf_names += [n for n in leaf_ids if n not in leaf_names]
        if len(leaf_ids) == 1:
            ((dataset_path, dataset_name),) = leaf_ids.values()
            suffix = f" ({dataset_name})" if dataset_name else ""
            print(f"  {name}: {dataset_path}{suffix}")
        else:  # a group, e.g. blimp: one line, not one per subtask
            paths = sorted({p for p, _ in leaf_ids.values()})
            print(f"  {name}: {len(leaf_ids)} subtasks from {', '.join(paths)}")

    if args.dry_run:
        print("--dry-run: fetched nothing")
        return

    from transformers import AutoTokenizer

    print("fetching task datasets...", flush=True)
    # Building a ConfigurableTask calls its own .download() eagerly (lm_eval/api/task.py), so
    # this is the exact call the harness itself makes at eval time, not a reimplementation of
    # it. datasets.load_dataset() is itself cache-aware, so rerunning this is a no-op.
    # Loaded by LEAF name, never by suite/group name: the flat scan above yields both
    # extended_suite's nested `flame_suite` and flame_suite's own members, and load() rejects a
    # task reached through a group and standalone in the same call ("Duplicate task"). Groups
    # have no datasets of their own, so fetching the deduplicated leaves fetches everything.
    tm.load(leaf_names)

    print(f"fetching tokenizer {FLAME_TOKENIZER}...")
    AutoTokenizer.from_pretrained(FLAME_TOKENIZER)

    print("done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Prepare ClimbLab .bin/.idx train / held-out-validation prefixes from a yaml config.

Usage:
    uv run python scripts/prepare_climblab.py configs/data/climblab_local.yaml
    uv run python scripts/prepare_climblab.py --list-clusters      # discover cluster names
    uv run python scripts/prepare_climblab.py --plan-conversions   # show planned conversions

Downloads only the selected clusters/shards (never the whole dataset), converts each
(cluster, split) to its own Megatron IndexedDataset prefix, and writes a manifest.json.
Requires network + Hugging Face access to nvidia/Nemotron-ClimbLab (CC BY-NC).
"""

import argparse
import logging

from moe_congestion_routing.data.climb import available_clusters, plan_conversions
from moe_congestion_routing.data.config import DataPrepConfig
from moe_congestion_routing.data.prepare_dataset import run_preparation


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="path to a DataPrepConfig yaml file")
    parser.add_argument(
        "--list-clusters",
        metavar="REPO",
        nargs="?",
        const="nvidia/Nemotron-ClimbLab",
        help="list available cluster folder names in the dataset repo and exit",
    )
    parser.add_argument(
        "--plan-conversions",
        action="store_true",
        help="show planned conversions (cluster, split) -> prefix and exit",
    )
    args = parser.parse_args()

    if args.list_clusters:
        for cluster in available_clusters(args.list_clusters):
            print(cluster)
        return

    if not args.config:
        parser.error("a config yaml path is required (or use --list-clusters)")

    config = DataPrepConfig.from_yaml(args.config)

    if args.plan_conversions:
        for conversion in plan_conversions(config):
            print(
                f"{conversion.cluster:<12} {conversion.role:<8} "
                f"({len(conversion.shards):<3} shards) -> {conversion.prefix}"
            )
        return

    prepared = run_preparation(config)

    print(f"Prepared {len(prepared)} prefix(es) in {config.output_dir}:")
    for p in prepared:
        print(
            f"  {p.prefix:<28} role={p.role:<8} cluster={p.cluster:<12} "
            f"docs={p.num_documents:>9,} tokens={p.num_tokens:>13,}"
        )


if __name__ == "__main__":
    main()

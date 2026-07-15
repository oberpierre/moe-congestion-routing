#!/usr/bin/env python
"""Prepare ClimbLab .bin/.idx train / held-out-validation prefixes from a yaml config.

Usage:
    uv run python scripts/prepare_climblab.py --list-clusters      # discover cluster names

Downloads only the selected clusters/shards (never the whole dataset), converts each
(cluster, split) to its own Megatron IndexedDataset prefix, and writes a manifest.json.
Requires network + Hugging Face access to nvidia/Nemotron-ClimbLab (CC BY-NC).
"""

import argparse

from moe_congestion_routing.data.climblab import available_clusters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-clusters",
        metavar="REPO",
        nargs="?",
        const="nvidia/Nemotron-ClimbLab",
        help="list available cluster folder names in the dataset repo and exit",
    )
    args = parser.parse_args()

    if args.list_clusters:
        for cluster in available_clusters(args.list_clusters):
            print(cluster)


if __name__ == "__main__":
    main()

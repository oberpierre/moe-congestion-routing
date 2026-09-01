#!/usr/bin/env python
"""Strong-scaling probe: solve the identical LP in N parallel processes, for several N.

Answers one question the phi-gap grid cannot. The grid's per-cell cost mixes three things —
this machine's per-core speed, how hard the particular cells are, and contention between
concurrent solves — and no run of the grid separates them, because a different worker count also
means a different set of cells. Here every worker solves the *same* instance, so the only thing
that changes between legs is how many run at once, and per-solve seconds is directly comparable.

    uv run python scripts/lp_contention_probe.py DUMP.npz --layer 2 --unit u0 \
        --workers 1 --workers 32 --workers 128 --workers 287

Read it as: per-solve seconds flat across legs means no contention and the grid's cost is cells
and hardware. Per-solve seconds rising with the worker count is contention, and the ratio to the
one-worker leg is what a larger fleet would actually be paying.
"""

import argparse
import multiprocessing
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from moe_congestion_routing.game.incremental import solve_incremental
from moe_congestion_routing.losses.cost_families import marginal_cost
from moe_congestion_routing.metrics.phi_gap import arc_schedule_length
from moe_congestion_routing.metrics.probe_series import read_dump

_SHARED: dict = {}


def _init(a: np.ndarray, k: int, prices: np.ndarray) -> None:
    _SHARED["a"], _SHARED["k"], _SHARED["prices"] = a, k, prices


def _solve(_i: int) -> float:
    start = time.perf_counter()
    solve_incremental(_SHARED["a"], _SHARED["k"], _SHARED["prices"])
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--unit", default="u0", choices=("u0", "u1"))
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--cost-family", default="linear")
    parser.add_argument("--workers", type=int, action="append", dest="worker_counts", required=True)
    args = parser.parse_args()

    dump = read_dump(args.dump)
    axis = dump.layer_numbers.index(args.layer)
    scores = dump.router_scores()[axis]
    half = scores.shape[0] // 2 if scores.shape[0] > 16384 else scores.shape[0]
    lo, hi = (0, half) if args.unit == "u0" else (half, 2 * half)
    a = np.array(scores[lo:hi])
    del scores

    n, e = a.shape
    k = dump.topk
    num_arcs = arc_schedule_length(
        n,
        k,
        e,
        float((a.max(axis=1) - a.min(axis=1)).max()),
        lam=args.lam,
        cost_family=args.cost_family,
    )
    prices = marginal_cost(
        np.arange(1, num_arcs + 1), n * k / e, lam=args.lam, cost_family=args.cost_family
    )
    print(f"n={n} e={e} k={k} arcs={num_arcs} lam={args.lam} family={args.cost_family}", flush=True)

    baseline = None
    for workers in args.worker_counts:
        start = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init,
            initargs=(a, k, prices),
        ) as pool:
            per_solve = list(pool.map(_solve, range(workers)))
        wall = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
        mean = sum(per_solve) / len(per_solve)
        baseline = mean if baseline is None else baseline
        print(
            f"workers={workers:>4}  wall={wall:>7.1f}s  per_solve mean={mean:>7.1f}s "
            f"min={min(per_solve):>7.1f}s max={max(per_solve):>7.1f}s  "
            f"slowdown_vs_first_leg={mean / baseline:>5.2f}x  peak_child_rss={rss:>7.0f}MB",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())

"""Runs the ALF-LB-versus-LP comparison grid, serially or across worker processes.

Owns cell execution and row ordering only. What one cell computes lives in :mod:`compare` and
:mod:`ensemble`, whereas what happens to a finished row is the caller's own ``emit``, so this
module never opens a file and never prints, which is what lets a test drive it with a plain
list-appending callback instead of running a grid by hand.
"""

import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import NamedTuple

from moe_congestion_routing.game.compare import Comparison, compare
from moe_congestion_routing.game.ensemble import Instance, affinities


class Cell(NamedTuple):
    """One grid cell: an instance and the run configuration scored against it."""

    instance: Instance
    mode: str  # "annealed" | "deployed"
    eta: float
    steps: int


class Row(NamedTuple):
    """One cell's result. ``index`` is its fixed grid position, which is also emission order."""

    index: int
    cell: Cell
    comparison: Comparison
    elapsed: float


Emit = Callable[[Row], None]
# Called the moment a cell finishes, in real completion order, whereas `Emit` is called in grid
# order once the prefix reaches that cell. A caller wanting a progress counter that only ever
# goes up must use this, because under parallelism the two orders differ.
OnComplete = Callable[[Row], None]


def run_cell(cell: Cell) -> tuple[Comparison, float]:
    """Score one grid cell against the LP oracle.

    Module-level so a `spawn` process pool can import and pickle it.
    """
    start = time.perf_counter()
    a = affinities(cell.instance)
    c = compare(a, cell.instance.k, eta=cell.eta, steps=cell.steps, mode=cell.mode)
    return c, time.perf_counter() - start


def run_grid(
    cells: Sequence[Cell], jobs: int, emit: Emit, on_complete: OnComplete | None = None
) -> None:
    """Run every cell and call `emit(row)` once per cell, in grid order, as each row joins the
    contiguous prefix of the grid whose cells all have a result.

    The guarantee is about what was collected, not what had finished: every row whose cell was
    collected before the run gave up is emitted, up to the first gap, before this function
    returns or raises. A cell still mid-computation when the run gives up cannot be collected,
    and a collected cell stuck behind an unfinished earlier one is held back until the gap
    closes, so neither of those can be in the emitted prefix either.

    Re-raises whatever a cell raised, or a `KeyboardInterrupt`, after emitting everything that
    is collectible, so a caller relying on this function returning to mean success still sees a
    failed or interrupted run as a failure.
    """
    if jobs == 1:
        _run_serial(cells, emit, on_complete)
    else:
        _run_parallel(cells, jobs, emit, on_complete)


def _run_serial(cells: Sequence[Cell], emit: Emit, on_complete: OnComplete | None) -> None:
    """Run every cell in this process, in grid order.

    Every cell finishes in submission order here, so the prefix advances by exactly one on
    every iteration, which is also why there is no pool whose shutdown a failure has to wait on.
    """
    for index, cell in enumerate(cells):
        comparison, elapsed = run_cell(cell)
        row = Row(index=index, cell=cell, comparison=comparison, elapsed=elapsed)
        if on_complete is not None:
            on_complete(row)
        emit(row)


def _run_parallel(
    cells: Sequence[Cell], jobs: int, emit: Emit, on_complete: OnComplete | None
) -> None:
    """Run the grid across `jobs` worker processes, emitting the contiguous prefix as it forms.

    Catches `BaseException` rather than `Exception` because a `KeyboardInterrupt` skips an
    `except Exception:` handler entirely, which would otherwise fall into the pool's own
    `shutdown(wait=True)` and wait out every queued cell, not merely the running ones. On
    failure, futures already running are allowed to finish rather than killed, and their
    results are collected and emitted before the exception propagates, because discarding them
    is exactly the defect this function exists to not have.
    """
    results: list[tuple[Comparison, float] | None] = [None] * len(cells)
    next_to_emit = 0

    def emit_prefix() -> None:
        nonlocal next_to_emit
        while next_to_emit < len(results) and results[next_to_emit] is not None:
            # The while condition above already confirmed this slot is not None.
            comparison, elapsed = results[next_to_emit]
            row = Row(
                index=next_to_emit,
                cell=cells[next_to_emit],
                comparison=comparison,
                elapsed=elapsed,
            )
            # Count the row as emitted BEFORE handing it over, because a failing emit lands in
            # the handler below, which calls this again. Advancing first means a row whose emit
            # raised is skipped rather than written twice, and a duplicate row in a results file
            # is far worse than a missing one in a run that is failing anyway.
            next_to_emit += 1
            emit(row)

    # spawn re-execs a fresh interpreter per worker instead of forking this process, which
    # already has numpy and scipy loaded. Forking a process with C-extension state and whatever
    # threads those libraries keep alive can deadlock the child at the fork boundary.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as executor:
        future_to_index = {executor.submit(run_cell, cell): i for i, cell in enumerate(cells)}
        try:
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                comparison, elapsed = future.result()
                results[i] = (comparison, elapsed)
                if on_complete is not None:
                    on_complete(Row(index=i, cell=cells[i], comparison=comparison, elapsed=elapsed))
                emit_prefix()
        except BaseException:
            # Cancel whatever has not started, then wait out whatever has, so a future already
            # mid-run gets to finish instead of being discarded along with the failure.
            executor.shutdown(wait=True, cancel_futures=True)
            for future, i in future_to_index.items():
                if results[i] is not None or not future.done() or future.cancelled():
                    continue
                try:
                    comparison, elapsed = future.result()
                except BaseException:
                    continue
                results[i] = (comparison, elapsed)
            emit_prefix()
            raise

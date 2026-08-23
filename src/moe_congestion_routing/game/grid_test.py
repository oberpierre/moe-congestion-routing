import subprocess
import sys

import pytest

from moe_congestion_routing.game.ensemble import Instance
from moe_congestion_routing.game.grid import Cell, run_grid

# A large annealed cell whose price iteration takes a couple of real seconds, used wherever a
# test needs a cell still running when a faster one elsewhere in the grid has already finished.
_SLOW_CELL = Cell(Instance(n=512, e=64, k=8, separation=2.0, seed=0), "annealed", 1e-2, 2000)

# A cell that fails almost immediately: e=0 makes `compare`'s own `n * k / e` raise
# `ZeroDivisionError` before it does any real work, which is a real failure through the real
# code path rather than a mock, and is plain data so a `spawn` worker can pickle it.
_POISON_CELL = Cell(Instance(n=4, e=0, k=1, separation=1.0, seed=0), "annealed", 1e-2, 10)

# A cell so small it finishes close to instantly, used as a stand-in for "the cell that would
# have been emitted first if completion order, not grid order, controlled emission."
_FAST_CELL = Cell(Instance(n=8, e=4, k=1, separation=2.0, seed=0), "deployed", 1e-1, 1)


def test_importing_grid_does_not_pull_in_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import moe_congestion_routing.game.grid, sys; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------------------------
# A poison cell that fails fast while its predecessors are still running must not discard them.
# ---------------------------------------------------------------------------------------------


def test_a_fast_failure_keeps_the_rows_that_were_already_collected():
    # Both predecessors are still mid-computation when the poison cell's near-instant
    # ZeroDivisionError lands, at jobs=3, so completion order (poison first) and grid order
    # (poison last) genuinely differ, which is exactly the case the fix has to cover.
    cells = [
        _SLOW_CELL,
        Cell(Instance(n=512, e=64, k=8, separation=0.2, seed=1), "annealed", 1e-2, 2000),
        _POISON_CELL,
    ]

    collected = []
    with pytest.raises(ZeroDivisionError):
        run_grid(cells, jobs=3, emit=collected.append)

    assert [row.index for row in collected] == [0, 1]


# ---------------------------------------------------------------------------------------------
# Emission follows grid order even when real completion order is shuffled, and holds back
# everything until the gap at the front of the grid closes.
# ---------------------------------------------------------------------------------------------


def test_emission_follows_grid_order_under_a_shuffled_completion_order():
    # The slow cell sits at grid position 0, so nothing may be emitted until it finishes, even
    # though the two fast cells behind it finish first in real time.
    cells = [
        _SLOW_CELL,
        _FAST_CELL,
        Cell(Instance(n=8, e=4, k=1, separation=2.0, seed=1), "deployed", 1e-1, 1),
    ]

    # One interleaved log of both callbacks, so hold-back is observed rather than inferred from
    # a rank. A wall-clock threshold would only let a loaded machine fail a behaviour that did
    # not change.
    events: list[tuple[str, int]] = []
    run_grid(
        cells,
        jobs=3,
        emit=lambda row: events.append(("emit", row.index)),
        on_complete=lambda row: events.append(("done", row.index)),
    )

    emitted = [i for kind, i in events if kind == "emit"]
    completed = [i for kind, i in events if kind == "done"]
    assert emitted == [0, 1, 2]
    # The gap at the front is real: the slow cell finished last, so completion order differs
    # from grid order, which is what makes the emission order above a property and not a
    # coincidence of everything finishing in order anyway.
    assert completed[-1] == 0
    assert set(completed) == {0, 1, 2}
    # Nothing may be emitted before the cell blocking the prefix completes, and once it does the
    # whole backlog goes out together with no further completion in between.
    assert events.index(("done", 0)) < events.index(("emit", 0))
    assert [kind for kind, _ in events[-3:]] == ["emit", "emit", "emit"]

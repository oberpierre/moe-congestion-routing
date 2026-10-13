"""Synthetic affinity instances for the offline ALF-LB-versus-LP comparison grid.

Instance names a shape and an affinity draw, and :func:`affinities` turns it into the matrix
:mod:`compare` scores. Kept separate from ``compare.py`` because the grid script needs to
enumerate instances before running anything against them.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Instance:
    n: int
    e: int
    k: int
    separation: float
    seed: int


def affinities(inst: Instance) -> np.ndarray:
    """Return ``sigmoid(separation * N(0, 1))``, float64 ``[n, e]``.

    This is the affinity space a sigmoid-scored router actually ranks, not raw logits, so
    ALF-LB and the LP oracle are competing over the same game. ``separation`` is the pre-sigmoid
    scale: large values push affinities toward 0 or 1 and widen tie margins, small values crowd
    everything near 0.5.
    """
    rng = np.random.default_rng(inst.seed)
    z = inst.separation * rng.standard_normal((inst.n, inst.e))
    return 1.0 / (1.0 + np.exp(-z))


# Named so lp_test.py, alflb_test.py and compare_test.py exercise the same instance instead
# of each re-spelling the same shape, separation and seed as an independently pinned constant
# that can silently drift apart from the other two.
N512_E8_K2_SEP2_SEED1 = Instance(n=512, e=8, k=2, separation=2.0, seed=1)

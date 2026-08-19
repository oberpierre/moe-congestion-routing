"""Offline replica of Megatron's ALF-LB expert-bias update (auxiliary-loss-free load balancing).

Reproduces ``get_updated_expert_bias`` from ``megatron.core.transformer.moe.moe_utils`` on a
fixed affinity matrix, so the fixed-step-size limit cycle and the annealed convergence can both be
studied offline.
"""

import math
from typing import NamedTuple

import numpy as np


class AlfResult(NamedTuple):
    x: np.ndarray  # bool [N, E], the final iterate's assignment
    bias: np.ndarray  # float64 [E], the final bias
    objective: float  # (a * x).sum() at x
    max_load: int  # x.sum(axis=0).max()
    steps: int  # iterations actually run
    cycle_length: int | None  # exact, deployed mode only; None if none found
    band_width: float  # max over experts of peak-to-peak bias over the cycle,
    # or over the last half of the trajectory when no cycle was found
    cycle_objective_mean: float  # objective averaged over one full cycle; NaN if no cycle
    cycle_max_load: int  # max load over one full cycle; final iterate's if no cycle
    settled_at: int | None  # step at which the load hit exact balance and the bias froze
    # None if that never happened, which for annealed mode means the convergence
    # hypothesis was not reached inside the step budget


def bias_update(load: np.ndarray, balanced_load: float, eta: float) -> np.ndarray:
    """The ALF-LB bias delta for one step: ``eta * sign(balanced_load - load)``.

    Split out of :func:`iterate` so the rule exists once and can be checked against Megatron's
    own ``get_updated_expert_bias`` during testing.
    """
    return eta * np.sign(balanced_load - np.asarray(load, dtype=np.float64))


def top_k_map(y: np.ndarray, k: int) -> np.ndarray:
    """Return the column indices of the k largest entries per row of y, shape [N, k].

    Ties break to the lowest expert index, which is this package's rule rather than
    ``torch.topk``'s, so a row whose k-th and (k+1)-th values are close is one where the rule
    and not the data picked the expert. :func:`tie_margins` measures that gap.
    """
    # "stable" is what makes the tie rule true: an unstable sort may return tied columns in any
    # order, so lowest-index-wins would hold only by luck.
    return np.argsort(-y, axis=1, kind="stable")[:, :k]


def tie_margins(y: np.ndarray, k: int) -> np.ndarray:
    """Return, per row of y, the gap between the k-th and (k+1)-th largest values.

    A near-zero margin marks a row where :func:`top_k_map`'s tie rule, not the affinities,
    decided the selection, so any comparison of assignments should be read against these.
    Raw margins rather than a boolean count at some epsilon, because the right epsilon belongs
    to whatever reads the result instead of being baked in here.
    """
    num_experts = y.shape[1]
    # k == num_experts would index one past the last column. iterate accepts that k, so refusing
    # here with the package's ValueError keeps one entry point from raising IndexError instead.
    if not 1 <= k < num_experts:
        raise ValueError(
            f"1 <= k < num_experts required to have a margin, got k={k}, E={num_experts}"
        )
    sorted_desc = np.sort(y, axis=1)[:, ::-1]
    return sorted_desc[:, k - 1] - sorted_desc[:, k]


def _assignment_from_top_k(idx: np.ndarray, num_experts: int) -> np.ndarray:
    n, k = idx.shape
    x = np.zeros((n, num_experts), dtype=bool)
    x[np.repeat(np.arange(n), k), idx.ravel()] = True
    return x


def _step_size(t: int, eta: float, mode: str, eta_schedule) -> float:
    if eta_schedule is not None:
        return eta_schedule(t)
    if mode == "annealed":
        return eta / math.sqrt(t + 1)
    return eta


_MODES = frozenset({"deployed", "annealed"})


def _band_width_over(biases: list[np.ndarray]) -> float:
    if len(biases) < 2:
        return 0.0
    stacked = np.stack(biases)
    return float(np.max(np.ptp(stacked, axis=0)))


def iterate(
    a: np.ndarray,
    k: int,
    *,
    eta: float,
    steps: int,
    mode: str = "deployed",
    eta_schedule=None,
) -> AlfResult:
    """Run ALF-LB's bias update on the fixed affinity matrix ``a``.

    ``mode="deployed"`` holds ``eta`` fixed, matching Megatron's shipped
    ``moe_router_bias_update_rate``, and detects the exact limit cycle by hashing the
    bias's lattice coordinate (see the comment below). ``mode="annealed"`` decays the
    step as ``eta / sqrt(t + 1)``, testing the theorem's convergence hypothesis.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
    if eta_schedule is not None and mode != "annealed":
        raise ValueError("eta_schedule requires mode='annealed'")
    # eta scales the lattice the cycle hash is taken on, so eta <= 0 collapses it: at 0 the
    # coordinate is 0/0 = nan, which casts to a platform-defined int and reports a bogus cycle.
    if not eta > 0:
        raise ValueError(f"eta must be positive, got {eta!r}")

    a = np.asarray(a, dtype=np.float64)
    n, num_experts = a.shape
    # Refused rather than truncated, because argsort()[:, :k] would silently return
    # num_experts columns and hand back an assignment that is not k-hot per row.
    if not 1 <= k <= num_experts:
        raise ValueError(f"k must satisfy 1 <= k <= num_experts, got k={k}, E={num_experts}")
    balanced_load = n * k / num_experts
    bias = np.zeros(num_experts, dtype=np.float64)

    detect_cycle = mode == "deployed"
    seen: dict[tuple[int, ...], int] = {}
    # One (bias, load, objective) triple per completed iteration, read back for the
    # deployed-mode cycle statistics and for the no-cycle band-width fallback.
    trajectory: list[tuple[np.ndarray, np.ndarray, float]] = []

    cycle_start = None
    settled_at = None
    t = 0
    for t in range(steps):
        if detect_cycle:
            # bias moves on the lattice bias_0 + eta * Z^E (bias_0 = 0 here), and the
            # map from bias to the next bias is deterministic, so a repeated lattice
            # coordinate is an exact cycle. Round to prevent float drift preventing
            # the detection of a cycle.
            coord = tuple(np.round(bias / eta).astype(int))
            if coord in seen:
                cycle_start = seen[coord]
                break
            seen[coord] = t

        idx = top_k_map(a + bias, k)
        x = _assignment_from_top_k(idx, num_experts)
        load = x.sum(axis=0)
        objective = float((a * x).sum())
        trajectory.append((bias.copy(), load, objective))

        direction = bias_update(load, balanced_load, 1.0)
        # Every sign zero means every expert carries exactly balanced_load. This is an exact
        # fixed point and running on would change nothing. It can only occur when n*k/E is an
        # integer, because loads are integers, so a non-divisible instance never settles.
        if not direction.any():
            settled_at = t
            break

        bias = bias + _step_size(t, eta, mode, eta_schedule) * direction

    steps_run = len(trajectory)

    # The state where the loop stopped has never evaluated the score inside the loop
    idx = top_k_map(a + bias, k)
    x = _assignment_from_top_k(idx, num_experts)
    load = x.sum(axis=0)
    objective = float((a * x).sum())
    max_load = int(load.max())

    if cycle_start is not None:
        cycle_length = t - cycle_start
        cycle = trajectory[cycle_start:t]
        band_width = _band_width_over([b for b, _, _ in cycle])
        cycle_objective_mean = float(np.mean([o for _, _, o in cycle]))
        cycle_max_load = int(max(load_e.max() for _, load_e, _ in cycle))
    else:
        cycle_length = None
        biases = [b for b, _, _ in trajectory] + [bias]
        band_width = _band_width_over(biases[len(biases) // 2 :])
        cycle_objective_mean = float("nan")
        cycle_max_load = max_load

    return AlfResult(
        x=x,
        bias=bias,
        objective=objective,
        max_load=max_load,
        steps=steps_run,
        cycle_length=cycle_length,
        band_width=band_width,
        cycle_objective_mean=cycle_objective_mean,
        cycle_max_load=cycle_max_load,
        settled_at=settled_at,
    )

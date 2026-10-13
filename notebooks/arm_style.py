"""One visual convention for every thesis figure: colours, dashes, labels and matplotlib style.

Imported by notebooks/phi_gap_trajectories.ipynb (probe diagnostics) and notebooks/wandb_figures.ipynb
(training statistics exported from Weights & Biases), so the same arm looks the same everywhere.

Colour carries the cost family or the method, dash the load variant (solid hard, dashed soft), and the
balancing scope becomes a facet rather than a third visual channel. ALF-Cost has no hard/soft
variant, so it takes its own dash-dot.
"""

import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK_2, MUTED, GRID_INK, GRAY = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#52514e"
LAYER_RAMP = [
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#104281",
]

# (arm, figure label, table label, scope, colour, dash), in the legend order of the thesis figures.
# `scope` is the token population the arm's own balancing statistic is counted over, which decides
# the yardstick it may fairly be judged against: `micro` for one micro-batch of 8192 tokens,
# `global` for the whole global batch, `none` only for the control.
#
# ALF-LB and ALF-Cost are `global`. Neither has a balancing *loss*, but the criterion
# is the scope of the statistic and not the mechanism carrying it. Megatron accumulates
# `local_tokens_per_expert` with `+=` across every micro-batch of a step and all-reduces it over the
# tensor, context and data parallel group inside `get_updated_expert_bias`, once per optimizer step.
# So the bias sees the entire global batch, exactly as a `_global` loss arm's reduction does.
ARM_TABLE = (
    ("switch", "Switch", "Switch", "micro", AQUA, "-"),
    ("rosenthal_linear_hard", "CL-Lin-Hard", "CL-Lin-Hard", "micro", BLUE, "-"),
    ("rosenthal_linear_soft", "CL-Lin-Soft", "CL-Lin-Soft", "micro", BLUE, "--"),
    ("rosenthal_quadratic_hard", "CL-Quad-Hard", "CL-Quad-Hard", "micro", ORANGE, "-"),
    ("rosenthal_quadratic_soft", "CL-Quad-Soft", "CL-Quad-Soft", "micro", ORANGE, "--"),
    ("rosenthal_barrier_hard", "CL-Barrier-Hard", "CL-Barrier-Hard", "micro", VIOLET, "-"),
    ("alflb", "ALF-LB", "ALF-LB$^{\\dagger}$", "global", YELLOW, "-"),
    ("rosenthal_price_barrier", "ALF-Cost", "ALF-Cost$^{\\dagger}$", "global", VIOLET, "-."),
    ("switch_global", "Switch (global)", "Switch", "global", AQUA, "-"),
    ("rosenthal_linear_hard_global", "CL-Lin-Hard (global)", "CL-Lin-Hard", "global", BLUE, "-"),
    ("rosenthal_linear_soft_global", "CL-Lin-Soft (global)", "CL-Lin-Soft", "global", BLUE, "--"),
    (
        "rosenthal_quadratic_hard_global",
        "CL-Quad-Hard (global)",
        "CL-Quad-Hard",
        "global",
        ORANGE,
        "-",
    ),
    (
        "rosenthal_quadratic_soft_global",
        "CL-Quad-Soft (global)",
        "CL-Quad-Soft",
        "global",
        ORANGE,
        "--",
    ),
    ("control", "Control", "Control", "none", GRAY, "-"),
)
# The names that label the fleet runs in the Weights & Biases exports under assets/results/wandb/11B/
# (columns `<name> - <metric>`): the figure labels the arms had when the runs were exported. The
# notebooks map an export column to its arm by this name, so it must not follow the labels above.
# A run exported under its W&B run name `<arm>-<experiment>` (the price-barrier arm) is mapped by
# the part before the first hyphen instead.
WANDB_NAMES = {
    "switch": "Switch",
    "rosenthal_linear_hard": "linear, hard",
    "rosenthal_linear_soft": "linear, soft",
    "rosenthal_quadratic_hard": "quadratic, hard",
    "rosenthal_quadratic_soft": "quadratic, soft",
    "rosenthal_barrier_hard": "softplus barrier, hard",
    "alflb": "ALF-LB",
    "rosenthal_price_barrier": "price-barrier bias",
    "switch_global": "Switch (global)",
    "rosenthal_linear_hard_global": "linear, hard (global)",
    "rosenthal_linear_soft_global": "linear, soft (global)",
    "rosenthal_quadratic_hard_global": "quadratic, hard (global)",
    "rosenthal_quadratic_soft_global": "quadratic, soft (global)",
    "control": "Control",
}
ARM_META = {
    arm: {
        "label": lab,
        "table": tab,
        "scope": scope,
        "color": col,
        "dash": dash,
        "wandb": WANDB_NAMES[arm],
    }
    for arm, lab, tab, scope, col, dash in ARM_TABLE
}
# Arms whose top-K selection adds a learned expert bias to the scores. They score on sigmoid rather
# than softmax, and at zero congestion price their gap is the cost of the bias, not zero.
BIAS_ARMS = {"alflb", "rosenthal_price_barrier"}
# Named groups the text refers to.
LEAD_HARD = [
    "rosenthal_quadratic_hard",
    "switch",
    "rosenthal_linear_hard",
    "rosenthal_barrier_hard",
]  # hard, micro-batch
SOFT_MICRO = ["rosenthal_quadratic_soft", "rosenthal_linear_soft"]
HARD_GLOBAL = ["rosenthal_quadratic_hard_global", "rosenthal_linear_hard_global", "switch_global"]
SOFT_GLOBAL = ["rosenthal_quadratic_soft_global", "rosenthal_linear_soft_global"]
SCOPE_TITLES = {"micro": "micro-batch scope", "global": "global-batch scope"}


def meta(arm):
    """Metadata for an arm, falling back to a neutral entry so an unknown arm still shows up."""
    return ARM_META.get(
        arm, {"label": arm, "table": arm, "scope": "none", "color": MUTED, "dash": ":"}
    )


def flat_label(arm):
    """Table label with the scope folded in, as in the throughput and robustness tables."""
    return meta(arm)["table"] + (" (global)" if arm.endswith("_global") else "")


# The style of the thesis figures.
RC = {
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID_INK,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "figure.dpi": 120,
}


def apply_style():
    plt.rcParams.update(RC)


# ==================================================================================================
# Arms that appear only in the collapse races (drawn by notebooks/wandb_figures.ipynb).
#
# Appended section: nothing above is changed by it, and ARM_TABLE, ARM_META and meta() still describe
# the 14 recipe arms only. A race restarts every arm from one collapsed checkpoint, with a global batch
# of 128 sequences and sigmoid scores, so a race arm is not the fleet arm of the same recipe. It keeps
# that arm's colour, dash and label where the fleet has the recipe at the same scope (look it up with
# race_meta, which falls back to ARM_META). Every loss arm of the races balances at the global-batch
# scope except one linear hard arm at the micro-batch scope, so a race figure has no scope facet; that
# one arm is dotted instead. What the fleet does not have:
#   - the softplus-barrier loss at the global-batch scope, hard and soft (the fleet runs it per micro batch);
#   - variants, which change one setting of a base arm: the base arm's colour, dotted;
#   - softmax carriers, which compute the balancing loss on the softmax of the same logits while the
#     router still selects on sigmoid scores: the base arm's colour, dash-dot-dot (long dashes for the
#     carrier that also raises the z-loss weight to 1e-2).
DASH_DOT_DOT = (0, (5, 1.5, 1, 1.5, 1, 1.5))
LONG_DASH = (0, (8, 2.5))

# (arm, figure label, table label, scope, colour, dash), in the legend order of the race figures.
RACE_ARM_TABLE = (
    (
        "rosenthal_barrier_hard_global",
        "CL-Barrier-Hard (global)",
        "CL-Barrier-Hard (global)",
        "global",
        VIOLET,
        "-",
    ),
    (
        "rosenthal_barrier_soft_global",
        "CL-Barrier-Soft (global)",
        "CL-Barrier-Soft (global)",
        "global",
        VIOLET,
        "--",
    ),
    (
        "rosenthal_linear_hard_micro",
        "CL-Lin-Hard (micro)",
        "CL-Lin-Hard (micro)",
        "micro",
        BLUE,
        ":",
    ),
    (
        "rosenthal_barrier_hard_global_coeff3e-2",
        "CL-Barrier-Hard (global), weight 0.03",
        "CL-Barrier-Hard (global), $\\auxlossweight = 0.03$",
        "global",
        VIOLET,
        ":",
    ),
    (
        "alflb_eta1e-2",
        "ALF-LB, $\\eta = 10^{-2}$",
        "ALF-LB, $\\stepsize = 10^{-2}$",
        "global",
        YELLOW,
        ":",
    ),
    (
        "rosenthal_price_barrier_eta1e-2",
        "ALF-Cost, $\\eta = 10^{-2}$",
        "ALF-Cost, $\\stepsize = 10^{-2}$",
        "global",
        VIOLET,
        ":",
    ),
    (
        "switch_global_carrier",
        "Switch (global), softmax carrier",
        "Switch (global), softmax carrier",
        "global",
        AQUA,
        DASH_DOT_DOT,
    ),
    (
        "switch_global_carrier_z1e-2",
        "Switch (global), softmax carrier, z-loss $10^{-2}$",
        "Switch (global), softmax carrier, z-loss $10^{-2}$",
        "global",
        AQUA,
        LONG_DASH,
    ),
    (
        "rosenthal_barrier_hard_global_carrier",
        "CL-Barrier-Hard (global), softmax carrier",
        "CL-Barrier-Hard (global), softmax carrier",
        "global",
        VIOLET,
        DASH_DOT_DOT,
    ),
)
RACE_ARM_META = {
    arm: {"label": lab, "table": tab, "scope": scope, "color": col, "dash": dash}
    for arm, lab, tab, scope, col, dash in RACE_ARM_TABLE
}
# The arm each variant or carrier changes one setting of (two for the carrier with the larger z-loss).
RACE_BASE = {
    "rosenthal_barrier_hard_global_coeff3e-2": "rosenthal_barrier_hard_global",
    "alflb_eta1e-2": "alflb",
    "rosenthal_price_barrier_eta1e-2": "rosenthal_price_barrier",
    "switch_global_carrier": "switch_global",
    "switch_global_carrier_z1e-2": "switch_global",
    "rosenthal_barrier_hard_global_carrier": "rosenthal_barrier_hard_global",
}


def race_meta(arm):
    """Metadata of a race arm: the fleet's entry where the recipe exists there, else the race entry."""
    return RACE_ARM_META[arm] if arm in RACE_ARM_META else meta(arm)

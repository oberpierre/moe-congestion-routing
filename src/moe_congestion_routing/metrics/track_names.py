"""Names of the MoE metrics the router records, for ``training_log``'s ``track_names`` list."""

from ..losses.cost_families import COST_FAMILIES, ROSENTHAL_TYPES
from .swapgap import SWAPGAP_COSTS


def selected_rosenthal_types(args) -> list:
    """Return the rosenthal balancing types ``args`` selects, in selection order.

    ``moe_router_load_balancing_type`` holds either a string or a list. Argparse's ``nargs='+'``
    always produces a list, and Megatron's ``validate_args`` collapses only a single-element list
    back to a bare string. A plain ``"rosenthal" in value`` test therefore means two different
    things across those shapes. On a string it is substring containment and matches
    ``"global_rosenthal"``; on a list it is exact-element membership, so it misses
    ``["seq_aux_loss", "global_rosenthal"]`` while still matching ``["seq_aux_loss",
    "rosenthal"]``. Normalizing once here gives every caller the same answer for both shapes.
    """
    selected = getattr(args, "moe_router_load_balancing_type", "")
    types = selected if isinstance(selected, list) else [selected]
    return [t for t in types if t in ROSENTHAL_TYPES]


def selects_rosenthal_type(args) -> bool:
    """Whether ``args`` selects any rosenthal balancing type.

    See ``selected_rosenthal_types`` for the normalization this relies on.
    """
    return bool(selected_rosenthal_types(args))


def moe_track_names(args) -> list[str]:
    """Names of the per-step MoE metrics the router records via the metrics tracker."""
    names = [
        "load_cv",
        "maxvio",
        "n_eff_0",
        "n_eff_1",
        "n_eff_2",
        "frac_gate_l1",
        "dead_experts",
        *(f"swapgap_{price}" for price in SWAPGAP_COSTS),
        *(f"phi_cong_{family}" for family in COST_FAMILIES),
    ]
    if selects_rosenthal_type(args):
        names += ["rosenthal_loss", "rosenthal_pressure_max", "u_glob_sum"]
    if getattr(args, "moe_rosenthal_log_grad_ratio", False):
        names += ["rosenthal_grad_norm_task", "rosenthal_grad_norm_cg"]
    return names

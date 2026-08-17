from types import SimpleNamespace

from moe_congestion_routing.losses.cost_families import COST_FAMILIES
from moe_congestion_routing.metrics.swapgap import SWAPGAP_COSTS
from moe_congestion_routing.metrics.track_names import (
    moe_track_names,
    selected_rosenthal_types,
    selects_rosenthal_type,
)


def test_moe_track_names_is_non_empty():
    assert moe_track_names(SimpleNamespace())


def test_moe_track_names_has_no_duplicates():
    names = moe_track_names(SimpleNamespace())
    assert len(names) == len(set(names))


def test_moe_track_names_contains_one_swapgap_name_per_registered_price():
    names = set(moe_track_names(SimpleNamespace()))
    for price in SWAPGAP_COSTS:
        assert f"swapgap_{price}" in names


def test_moe_track_names_contains_one_phi_cong_name_per_cost_family():
    names = set(moe_track_names(SimpleNamespace()))
    for family in COST_FAMILIES:
        assert f"phi_cong_{family}" in names


def test_moe_track_names_reads_only_args_attributes():
    # A namespace with no attributes at all must not raise.
    moe_track_names(SimpleNamespace())


def test_moe_track_names_tolerates_a_namespace_with_only_one_attribute():
    # An args-gated addition has to tolerate some attributes being set and others absent, which
    # the fully empty namespace above does not exercise.
    moe_track_names(SimpleNamespace(moe_router_load_balancing_type="rosenthal"))


def test_moe_track_names_includes_rosenthal_names_for_both_rosenthal_types():
    # Getting this gate wrong is silent in both directions. Drop the names and a rosenthal arm
    # trains fine while its loss series is never plotted. Both the bare-string and the nargs='+'
    # list form are covered, including a rosenthal type combined with a non-rosenthal one, since
    # that is the shape selected_rosenthal_types' normalization exists to handle.
    for balancing_type in (
        "rosenthal",
        "global_rosenthal",
        ["rosenthal"],
        ["seq_aux_loss", "rosenthal"],
        ["seq_aux_loss", "global_rosenthal"],
    ):
        names = moe_track_names(SimpleNamespace(moe_router_load_balancing_type=balancing_type))
        assert "rosenthal_loss" in names
        assert "rosenthal_pressure_max" in names
        assert "u_glob_sum" in names


def test_moe_track_names_excludes_rosenthal_names_for_non_rosenthal_types():
    # The other direction of the same gate. Drop it and every Switch, control and ALF-LB run logs
    # a phantom zero series for a loss it never trains against.
    for balancing_type in ("aux_loss", "seq_aux_loss", "global_aux_loss", "sinkhorn", "none"):
        names = moe_track_names(SimpleNamespace(moe_router_load_balancing_type=balancing_type))
        assert "rosenthal_loss" not in names
        assert "rosenthal_pressure_max" not in names
        # u_glob_sum is gated on being a rosenthal arm rather than on being a global one. At
        # tensor_model_parallel_size == context_parallel_size == 1 a micro arm's reduce group has
        # size 1 and the series is a flat E, but that is a fact about the config rather than the
        # code, and the flat series is the control the global arm's is read against.
        assert "u_glob_sum" not in names


def test_selected_rosenthal_types_filters_and_preserves_order():
    args = SimpleNamespace(
        moe_router_load_balancing_type=["seq_aux_loss", "global_rosenthal", "rosenthal"]
    )
    assert selected_rosenthal_types(args) == ["global_rosenthal", "rosenthal"]


def test_selected_rosenthal_types_is_empty_for_a_non_rosenthal_selection():
    for balancing_type in ("aux_loss", "seq_aux_loss", "global_aux_loss", "sinkhorn", "none"):
        args = SimpleNamespace(moe_router_load_balancing_type=balancing_type)
        assert selected_rosenthal_types(args) == []
    # No attribute at all, which is the same must-not-raise contract moe_track_names relies on.
    assert selected_rosenthal_types(SimpleNamespace()) == []


def test_selects_rosenthal_type_agrees_with_selected_rosenthal_types():
    # selects_rosenthal_type is a thin bool wrapper. Pinning the relationship keeps the two from
    # drifting apart if either is changed alone.
    for balancing_type in (
        "rosenthal",
        "global_rosenthal",
        ["rosenthal"],
        ["seq_aux_loss", "rosenthal"],
        "aux_loss",
        "none",
    ):
        args = SimpleNamespace(moe_router_load_balancing_type=balancing_type)
        assert selects_rosenthal_type(args) == bool(selected_rosenthal_types(args))

import json
from pathlib import Path

import numpy
import pytest

from moe_congestion_routing.metrics.probe_saturation import (
    LayerBaseline,
    read_metadata,
    reduce_dump,
    select_asset_dir,
)
from moe_congestion_routing.metrics.probe_series import IncomparableProbes


def _write_dump(
    tmp_path: Path,
    *,
    run: str,
    asset: str,
    iteration: int,
    logits: numpy.ndarray,
    sel_mask: numpy.ndarray,
    role: str = "standing",
    layer_numbers: list[int] | None = None,
    extra_meta: dict | None = None,
    bitorder: str = "big",
) -> Path:
    """A synthetic dump in the ``0032`` layout: ``<run>/probes/<asset>/iter_%07d.npz``."""
    num_layers = logits.shape[0]
    if layer_numbers is None:
        layer_numbers = list(range(2, 2 + num_layers))
    packed = numpy.packbits(sel_mask, axis=-1, bitorder=bitorder)
    # A real dump's K is constant across tokens, so the per-token selected count stands in for
    # it unless a test overrides K directly via extra_meta.
    topk = int(sel_mask.sum(axis=-1).max())
    meta = {
        "iteration": iteration,
        "role": role,
        "E": logits.shape[2],
        "K": topk,
        "layer_numbers": layer_numbers,
        "routing_map_bitorder": bitorder,
        "moe_router_score_function": "sigmoid",
        "moe_probe_batch": f"/assets/probe/{asset}.npz",
        "token_sha256": f"sha-{asset}",
    }
    if extra_meta:
        meta.update(extra_meta)
    probes_dir = tmp_path / run / "probes" / asset
    probes_dir.mkdir(parents=True, exist_ok=True)
    path = probes_dir / f"iter_{iteration:07d}.npz"
    numpy.savez(
        path,
        logits=logits.astype(numpy.float32),
        routing_map=packed,
        metadata=numpy.array(json.dumps(meta)),
    )
    return path


def _write_flat_dump(
    tmp_path: Path,
    *,
    run: str,
    iteration: int,
    logits: numpy.ndarray,
    sel_mask: numpy.ndarray,
    asset_name: str = "flat_asset",
    role: str = "standing",
    layer_numbers: list[int] | None = None,
    extra_meta: dict | None = None,
) -> Path:
    """A synthetic dump in the legacy flat layout: ``<run>/probes/iter_%07d.npz``."""
    num_layers = logits.shape[0]
    if layer_numbers is None:
        layer_numbers = list(range(2, 2 + num_layers))
    packed = numpy.packbits(sel_mask, axis=-1, bitorder="big")
    topk = int(sel_mask.sum(axis=-1).max())
    meta = {
        "iteration": iteration,
        "role": role,
        "E": logits.shape[2],
        "K": topk,
        "layer_numbers": layer_numbers,
        "routing_map_bitorder": "big",
        "moe_router_score_function": "sigmoid",
        "moe_probe_batch": f"/assets/probe/{asset_name}.npz",
        "token_sha256": f"sha-{asset_name}",
    }
    if extra_meta:
        meta.update(extra_meta)
    probes_dir = tmp_path / run / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    path = probes_dir / f"iter_{iteration:07d}.npz"
    numpy.savez(
        path,
        logits=logits.astype(numpy.float32),
        routing_map=packed,
        metadata=numpy.array(json.dumps(meta)),
    )
    return path


def _sel_mask(winners: list[int], num_tokens: int, num_experts: int) -> numpy.ndarray:
    """``[N, E]`` bool, the same winner set ``winners`` selected by every token."""
    mask = numpy.zeros((num_tokens, num_experts), dtype=bool)
    mask[:, winners] = True
    return mask


def _one_layer_dump(
    tmp_path: Path,
    *,
    run: str,
    asset: str,
    iteration: int,
    winner_logit: float,
    loser_logit: float,
    winners: list[int],
    num_tokens: int = 6,
    num_experts: int = 8,
    extra_meta: dict | None = None,
    bitorder: str = "big",
) -> Path:
    sel_mask = _sel_mask(winners, num_tokens, num_experts)
    logits = numpy.where(sel_mask, winner_logit, loser_logit).astype(numpy.float32)
    logits = logits[numpy.newaxis, :, :]
    sel_mask = sel_mask[numpy.newaxis, :, :]
    return _write_dump(
        tmp_path,
        run=run,
        asset=asset,
        iteration=iteration,
        logits=logits,
        sel_mask=sel_mask,
        layer_numbers=[3],
        extra_meta=extra_meta,
        bitorder=bitorder,
    )


# --- saturated vs healthy direction -------------------------------------------------------------


def test_saturated_router_has_lower_sel_sigp_and_more_frozen_tokens_than_healthy(tmp_path):
    saturated_path = _one_layer_dump(
        tmp_path,
        run="saturated",
        asset="a",
        iteration=0,
        winner_logit=30.0,
        loser_logit=-30.0,
        winners=[0, 1],
    )
    healthy_path = _one_layer_dump(
        tmp_path,
        run="healthy",
        asset="a",
        iteration=0,
        winner_logit=1.0,
        loser_logit=-1.0,
        winners=[0, 1],
    )

    _, _, saturated_rows, _ = reduce_dump(saturated_path, sat=0.99, resp=0.01, baseline=None)
    _, _, healthy_rows, _ = reduce_dump(healthy_path, sat=0.99, resp=0.01, baseline=None)
    saturated_row, healthy_row = saturated_rows[0], healthy_rows[0]

    assert saturated_row.sel_sigp_mean < healthy_row.sel_sigp_mean
    assert saturated_row.frac_tok_frozen > healthy_row.frac_tok_frozen
    assert saturated_row.frac_sel_tied > healthy_row.frac_sel_tied
    # The fixture is deliberately extreme, so these land at the far ends of [0, 1] rather than
    # merely on the right side of each other.
    assert saturated_row.frac_tok_frozen == pytest.approx(1.0)
    assert saturated_row.frac_sel_tied == pytest.approx(1.0)
    assert healthy_row.frac_tok_frozen == pytest.approx(0.0)
    assert healthy_row.frac_sel_tied == pytest.approx(0.0)


def test_frac_tok_frozen_ignores_a_responsive_unselected_expert(tmp_path):
    # Every token's two selected experts sit saturated (logit 30) while every unselected expert
    # sits mid-range (logit 0, sigp = 0.25, well above resp). A max over all experts would call
    # these tokens unfrozen because of the unselected experts, whereas restricted to the K
    # selected ones, as it must be, they are frozen.
    path = _one_layer_dump(
        tmp_path,
        run="mixed",
        asset="a",
        iteration=0,
        winner_logit=30.0,
        loser_logit=0.0,
        winners=[0, 1],
    )
    _, _, rows, _ = reduce_dump(path, sat=0.99, resp=0.01, baseline=None)
    assert rows[0].frac_tok_frozen == pytest.approx(1.0)


# --- n_eff_2 / n_zero_token against a direct routing-map computation ----------------------------


def test_n_eff_2_and_n_zero_token_match_direct_computation_from_routing_map(tmp_path):
    num_tokens, num_experts = 10, 6
    rng = numpy.random.default_rng(0)
    sel_mask = numpy.zeros((num_tokens, num_experts), dtype=bool)
    for token in range(num_tokens):
        # Expert 5 never gets picked, so it is always the zero-token expert, whereas the rest
        # split unevenly because that is what makes n_eff_2 differ from a plain count.
        choices = rng.choice([0, 1, 2, 3, 4], size=2, replace=False)
        sel_mask[token, choices] = True
    logits = rng.normal(size=(num_tokens, num_experts)).astype(numpy.float32)

    path = _write_dump(
        tmp_path,
        run="mixed",
        asset="a",
        iteration=0,
        logits=logits[numpy.newaxis, :, :],
        sel_mask=sel_mask[numpy.newaxis, :, :],
        layer_numbers=[4],
    )
    _, _, rows, _ = reduce_dump(path, sat=0.99, resp=0.01, baseline=None)
    row = rows[0]

    expert_counts = sel_mask.sum(axis=0)
    expected_n_zero_token = int((expert_counts == 0).sum())
    expected_n_eff_2 = float(expert_counts.sum()) ** 2 / float((expert_counts**2).sum())

    assert row.n_zero_token == expected_n_zero_token == 1
    assert row.n_eff_2 == pytest.approx(expected_n_eff_2)


# --- routing_map_bitorder handling ----------------------------------------------------------


def test_bitorder_is_read_from_the_dump_rather_than_assumed(tmp_path):
    """A dump packed ``little`` and saying so must unpack to the same routing map as ``big``.

    Reading the wrong metadata key is invisible while every dump in the tree is ``big``: the
    fallback silently supplies the right answer and the map is correct by luck. Packing one dump
    the other way is what makes the key actually load-bearing.
    """
    # Asymmetric about zero on purpose. sigma' is symmetric, so winner 1.0 against loser -1.0
    # gives the same statistics whichever experts get selected, and the test would pass while
    # unpacking the wrong map.
    winners = [0, 1]
    little = _one_layer_dump(
        tmp_path,
        run="little_bitorder",
        asset="a",
        iteration=0,
        winner_logit=3.0,
        loser_logit=-1.0,
        winners=winners,
        bitorder="little",
    )
    big = _one_layer_dump(
        tmp_path,
        run="big_bitorder",
        asset="a",
        iteration=0,
        winner_logit=3.0,
        loser_logit=-1.0,
        winners=winners,
    )
    _, _, little_rows, _ = reduce_dump(little, sat=0.99, resp=0.01, baseline=None)
    _, _, big_rows, _ = reduce_dump(big, sat=0.99, resp=0.01, baseline=None)
    assert little_rows[0].n_zero_token == big_rows[0].n_zero_token
    assert little_rows[0].n_eff_2 == pytest.approx(big_rows[0].n_eff_2)
    assert little_rows[0].sel_sig_mean == pytest.approx(big_rows[0].sel_sig_mean)
    assert little_rows[0].sel_sigp_mean == pytest.approx(big_rows[0].sel_sigp_mean)


def test_missing_metadata_key_raises():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        probes_dir = run_dir / "probes" / "a"
        probes_dir.mkdir(parents=True)
        path = probes_dir / "iter_0000000.npz"
        numpy.savez(path, logits=numpy.zeros((1, 2, 4), dtype=numpy.float32))
        with pytest.raises(ValueError, match="metadata"):
            read_metadata(path)


# --- run name and baseline threading --------------------------------------------------------


def test_run_name_is_read_off_the_probes_ancestor_directory(tmp_path):
    path = _one_layer_dump(
        tmp_path,
        run="my-run",
        asset="a",
        iteration=0,
        winner_logit=1.0,
        loser_logit=-1.0,
        winners=[0, 1],
    )
    _, _, rows, _ = reduce_dump(path, sat=0.99, resp=0.01, baseline=None)
    assert rows[0].run == "my-run"


def test_asset_and_token_sha256_are_read_off_the_dumps_own_metadata(tmp_path):
    path = _one_layer_dump(
        tmp_path,
        run="my-run",
        asset="standing_climbmix_small_16x2048",
        iteration=0,
        winner_logit=1.0,
        loser_logit=-1.0,
        winners=[0, 1],
    )
    _, _, rows, _ = reduce_dump(path, sat=0.99, resp=0.01, baseline=None)
    assert rows[0].asset == "standing_climbmix_small_16x2048"
    assert rows[0].token_sha256 == "sha-standing_climbmix_small_16x2048"


def test_dead_at_first_is_pinned_across_later_dumps_even_after_revival(tmp_path):
    # Iteration 0: expert 7 gets no tokens at any token, and neither do experts 2-6, so all six
    # start out dead alongside it.
    first = _one_layer_dump(
        tmp_path,
        run="revival",
        asset="a",
        iteration=0,
        winner_logit=1.0,
        loser_logit=-1.0,
        winners=[0, 1],
        num_experts=8,
    )
    _, _, rows0, baseline = reduce_dump(first, sat=0.99, resp=0.01, baseline=None)
    assert rows0[0].d0_size == 6  # experts {2..7} got no tokens at iteration 0
    assert rows0[0].d0_lift == pytest.approx(0.0)  # baseline dump measured against itself

    # Iteration 25: expert 7 now wins for every token, so it is no longer dead, but the d0
    # reference set (fixed at iteration 0) must still report it as size 6.
    second = _one_layer_dump(
        tmp_path,
        run="revival",
        asset="a",
        iteration=25,
        winner_logit=1.0,
        loser_logit=-1.0,
        winners=[0, 7],
        num_experts=8,
    )
    _, _, rows1, baseline = reduce_dump(second, sat=0.99, resp=0.01, baseline=baseline)
    # experts {1..6} are dead now, a different set of the same size.
    assert rows1[0].n_zero_token == 6
    assert rows1[0].d0_size == 6  # but the d0 reference set is still the iteration-0 one
    # Confirms it is the *same* reference set, not merely one of the same size: expert 7 sits in
    # the fixed d0 set and now wins, so d0_logit_mean sees one winner among six dead-at-first
    # experts (5 at loser_logit, 1 at winner_logit) rather than all six at loser_logit.
    assert rows1[0].d0_logit_mean == pytest.approx((5 * -1.0 + 1.0) / 6)


# --- F1: refuse a dump this module cannot interpret ------------------------------------------


def test_reduce_dump_refuses_a_non_sigmoid_dump(tmp_path):
    path = _one_layer_dump(
        tmp_path,
        run="control-trunk",
        asset="a",
        iteration=0,
        winner_logit=1.0,
        loser_logit=-1.0,
        winners=[0, 1],
        extra_meta={"moe_router_score_function": "softmax"},
    )
    with pytest.raises(IncomparableProbes, match="softmax"):
        reduce_dump(path, sat=0.99, resp=0.01, baseline=None)


# --- F2: select the asset by directory stem ---------------------------------------------------


def test_select_asset_dir_returns_the_lone_asset_when_omitted(tmp_path):
    _one_layer_dump(
        tmp_path,
        run="run",
        asset="only_asset",
        iteration=0,
        winner_logit=1.0,
        loser_logit=-1.0,
        winners=[0, 1],
    )
    probes_dir = tmp_path / "run" / "probes"
    assert select_asset_dir(probes_dir, None) == probes_dir / "only_asset"


def test_select_asset_dir_raises_when_several_assets_present_and_none_named(tmp_path):
    for asset in ("standing_climbmix_small_16x2048", "standing_climbmix_small_strided_16x2048"):
        _one_layer_dump(
            tmp_path,
            run="run",
            asset=asset,
            iteration=0,
            winner_logit=1.0,
            loser_logit=-1.0,
            winners=[0, 1],
        )
    probes_dir = tmp_path / "run" / "probes"
    with pytest.raises(
        IncomparableProbes, match="standing_climbmix_small_16x2048.*standing_climbmix_small_strided"
    ):
        select_asset_dir(probes_dir, None)


def test_select_asset_dir_selects_the_named_asset(tmp_path):
    for asset in ("asset_a", "asset_b"):
        _one_layer_dump(
            tmp_path,
            run="run",
            asset=asset,
            iteration=0,
            winner_logit=1.0,
            loser_logit=-1.0,
            winners=[0, 1],
        )
    probes_dir = tmp_path / "run" / "probes"
    assert select_asset_dir(probes_dir, "asset_b") == probes_dir / "asset_b"


def test_select_asset_dir_flat_layout_raises_on_asset_mismatch(tmp_path):
    _write_flat_dump(
        tmp_path,
        run="run",
        iteration=0,
        logits=numpy.zeros((1, 2, 4), dtype=numpy.float32),
        sel_mask=_sel_mask([0], 2, 4)[numpy.newaxis, :, :],
        asset_name="actual_asset",
    )
    probes_dir = tmp_path / "run" / "probes"
    with pytest.raises(IncomparableProbes, match="actual_asset"):
        select_asset_dir(probes_dir, "requested_asset")


def test_select_asset_dir_flat_layout_accepts_the_matching_asset(tmp_path):
    _write_flat_dump(
        tmp_path,
        run="run",
        iteration=0,
        logits=numpy.zeros((1, 2, 4), dtype=numpy.float32),
        sel_mask=_sel_mask([0], 2, 4)[numpy.newaxis, :, :],
        asset_name="the_asset",
    )
    probes_dir = tmp_path / "run" / "probes"
    assert select_asset_dir(probes_dir, "the_asset") == probes_dir


# --- F3: frac_tok_tie_ambiguous needs more than K saturated experts, not just K -----------------


def test_frac_tok_tie_ambiguous_differs_from_frac_sel_tied_when_no_loser_ties_a_winner(tmp_path):
    # K=2, both winners saturated, and every loser is far from saturation, so the winners are
    # pinned but there is no unselected expert saturated enough to contest them.
    winners = [0, 1]
    num_tokens, num_experts = 6, 8
    sel_mask = _sel_mask(winners, num_tokens, num_experts)
    logits = numpy.where(sel_mask, 30.0, -1.0).astype(numpy.float32)
    path = _write_dump(
        tmp_path,
        run="ambiguity",
        asset="a",
        iteration=0,
        logits=logits[numpy.newaxis, :, :],
        sel_mask=sel_mask[numpy.newaxis, :, :],
        layer_numbers=[3],
        extra_meta={"K": 2},
    )
    _, _, rows, _ = reduce_dump(path, sat=0.99, resp=0.01, baseline=None)
    row = rows[0]
    assert row.frac_sel_tied == pytest.approx(1.0)
    assert row.frac_tok_tie_ambiguous == pytest.approx(0.0)


def test_frac_tok_tie_ambiguous_is_positive_when_a_third_expert_ties_the_two_winners(tmp_path):
    # K=2, both winners saturated, and a third, unselected expert is saturated too, so
    # compute_topk's tie-break is genuinely deciding which two of the three saturated experts win.
    num_tokens, num_experts = 6, 8
    sel_mask = numpy.zeros((num_tokens, num_experts), dtype=bool)
    sel_mask[:, [0, 1]] = True
    logits = numpy.full((num_tokens, num_experts), -1.0, dtype=numpy.float32)
    logits[:, [0, 1, 2]] = 30.0  # expert 2 ties the winners but was not selected
    path = _write_dump(
        tmp_path,
        run="ambiguity",
        asset="a",
        iteration=0,
        logits=logits[numpy.newaxis, :, :],
        sel_mask=sel_mask[numpy.newaxis, :, :],
        layer_numbers=[3],
        extra_meta={"K": 2},
    )
    _, _, rows, _ = reduce_dump(path, sat=0.99, resp=0.01, baseline=None)
    row = rows[0]
    assert row.frac_sel_tied == pytest.approx(1.0)
    assert row.frac_tok_tie_ambiguous == pytest.approx(1.0)


# --- F4: d0_lift is measured against the run's first dump, not the previous one -----------------


def test_d0_lift_is_measured_against_the_first_dump_not_the_previous_one(tmp_path):
    winners = [0, 1]
    num_experts = 4
    loser_logits = {0: -1.0, 25: -2.0, 50: -5.0}
    baseline: dict[int, LayerBaseline] | None = None
    lifts = {}
    for iteration, loser_logit in loser_logits.items():
        path = _one_layer_dump(
            tmp_path,
            run="lift",
            asset="a",
            iteration=iteration,
            winner_logit=1.0,
            loser_logit=loser_logit,
            winners=winners,
            num_experts=num_experts,
        )
        _, _, rows, baseline = reduce_dump(path, sat=0.99, resp=0.01, baseline=baseline)
        lifts[iteration] = rows[0].d0_lift

    assert lifts[0] == pytest.approx(0.0)
    assert lifts[25] == pytest.approx(loser_logits[25] - loser_logits[0])
    # Against the previous dump this would be loser_logits[50] - loser_logits[25] == -3.0, not
    # the -4.0 that measuring against the first dump gives.
    assert lifts[50] == pytest.approx(loser_logits[50] - loser_logits[0])
    assert lifts[50] != pytest.approx(loser_logits[50] - loser_logits[25])

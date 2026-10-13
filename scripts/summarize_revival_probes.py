"""Summarize the probe dumps of a recovery race into one small CSV: how the experts that were dead
at the race's first dump come back, per arm, probe step and MoE layer.

The question it answers: does a balancing method revive a dead expert through the router's scores
or only through the selection? A bias arm adds a learned bias to the scores before the top-K, so it
can route tokens to an expert whose score is still close to zero; the expert is then selected, but
its gate (the normalized score that scales its output, and with it its gradient) stays close to
zero. A loss arm can only raise the scores themselves.

    R=artifacts/collapse/recovery
    python scripts/summarize_revival_probes.py --race step150 \
      --run alflb=$R/alflb/negaux-alflb \
      --run rosenthal_price_barrier=$R/rosenthal-price-barrier/negaux-rosenthal-price-barrier-r2 \
      --run rosenthal_barrier_hard_global=$R/rosenthal-barrier-hard/negaux-rosenthal-barrier-hard \
      --out assets/results/revival-probes/race-step150_revival-probes.csv

Reads <run>/probes/<asset>/iter_*.npz (written by the probe hook every 25 steps in the dense
window) and needs only numpy. Per dump and MoE layer, on the probe unit's N tokens (E experts,
top-K):
  formerly_dead          the experts with no token in this run's first dump (step 0 of the race)
  selection_share        share of all N K selections that go to the formerly dead experts
  gate_share             share of the gate weight (combine, which sums to 1 per token) that goes
                         to them
  gate_per_selection_formerly_dead / _survivors
                         mean gate of one selection of a formerly dead / surviving expert
  dead_probe_unit        experts with no token among the N tokens
  dead_per_micro_batch   the same, counted per micro batch of 8192 tokens (4 sequences) and
                         averaged, as training does
  bias_range_formerly_dead   max - min of their expert bias (0 for an arm without a bias)
  score_spread_formerly_dead mean over tokens of max - min of their sigmoid scores
  score_on_own_tokens_formerly_dead  their mean sigmoid score on the tokens that select them
  maxvio                 max load / mean load - 1
  probe_lm_loss          the language-model loss of the probe forward, as the dump records it
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MICRO_BATCH_TOKENS = 8192


def read(path):
    with np.load(path) as d:
        meta = json.loads(str(d["metadata"]))
        selected = np.unpackbits(
            d["routing_map"], axis=-1, count=meta["E"], bitorder=meta["routing_map_bitorder"]
        ).astype(bool)
        logits = d["logits"]
        # [L, N, K]: the gates of each token's K selected experts, in ascending expert order
        gate = d["combine"]
        bias = (
            d["expert_bias"] if meta.get("has_expert_bias") and "expert_bias" in d.files else None
        )
    return meta, selected, logits, gate, bias


def summarize_run(race, arm, run_dir, asset, steps):
    probes = Path(run_dir) / "probes" / asset
    paths = sorted(probes.glob("iter_*.npz"))
    if steps:
        paths = [p for p in paths if int(p.stem.split("_")[1]) in steps]
    rows, formerly_dead = [], None
    for path in paths:
        meta, selected, logits, gate, bias = read(path)
        L, N, E, K = meta["L"], meta["N"], meta["E"], meta["K"]
        load = selected.sum(axis=1)  # [L, E]
        if formerly_dead is None:
            assert meta["iteration"] == 0, f"{path}: the first dump must be the race's step 0"
            formerly_dead = load == 0
        score = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        for layer in range(L):
            fd = formerly_dead[layer]
            # selected expert ids, ascending
            ids = np.argsort(~selected[layer], axis=1, kind="stable")[:, :K]
            fd_slot = fd[ids]
            g = gate[layer].astype(np.float64)
            own = [
                score[layer][selected[layer][:, e], e].mean()
                for e in np.flatnonzero(fd & (load[layer] > 0))
            ]
            chunks = range(0, N, MICRO_BATCH_TOKENS)
            rows.append(
                {
                    "race": race,
                    "arm": arm,
                    "run": Path(run_dir).name,
                    "asset": asset,
                    "step": meta["iteration"],
                    "layer": meta["layer_numbers"][layer],
                    "formerly_dead": int(fd.sum()),
                    "selection_share": fd_slot.mean(),
                    "gate_share": (g * fd_slot).sum() / g.sum(),
                    "gate_per_selection_formerly_dead": g[fd_slot].mean() if fd_slot.any() else "",
                    "gate_per_selection_survivors": g[~fd_slot].mean(),
                    "dead_probe_unit": int((load[layer] == 0).sum()),
                    "dead_per_micro_batch": np.mean(
                        [
                            (selected[layer][c : c + MICRO_BATCH_TOKENS].sum(axis=0) == 0).sum()
                            for c in chunks
                        ]
                    ),
                    "bias_range_formerly_dead": float(np.ptp(bias[layer][fd]))
                    if bias is not None and fd.any()
                    else 0.0,
                    "score_spread_formerly_dead": (
                        score[layer][:, fd].max(axis=1) - score[layer][:, fd].min(axis=1)
                    ).mean()
                    if fd.any()
                    else "",
                    "score_on_own_tokens_formerly_dead": float(np.mean(own)) if own else "",
                    "maxvio": load[layer].max() / (N * K / E) - 1,
                    "probe_lm_loss": meta.get("probe_lm_loss", ""),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--race", required=True, help="name of the race, written to every row")
    parser.add_argument("--run", action="append", required=True, help="ARM=RUN_DIR, once per arm")
    parser.add_argument("--asset", default="standing_climbmix_small_strided_16x2048")
    parser.add_argument(
        "--max-step", type=int, default=500, help="last probe step to read (the dense window)"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for spec in args.run:
        arm, run_dir = spec.split("=", 1)
        steps = set(range(0, args.max_step + 1))
        rows += summarize_run(args.race, arm, run_dir, args.asset, steps)
        print(f"{arm}: {len({r['step'] for r in rows if r['arm'] == arm})} dumps")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}
            )
    print(f"wrote {out}, {len(rows)} rows")


if __name__ == "__main__":
    main()

# Game-Theoretic Routing for Mixture-of-Experts Models

Code, configurations and released results of the MSc thesis *Game-Theoretic Routing for
Mixture-of-Experts Models* (Pierre Obermaier, Department of Informatics, University of Zurich, 2026;
supervised by Paul Friedrich and Prof. Dr. Giorgia Ramponi).

<!-- TODO(author): link the thesis PDF (or DOI / repository record). -->

The thesis models token-choice top-K routing in a Mixture-of-Experts (MoE) layer as a congestion game
with an exact potential. In this view, the gradient of the Switch auxiliary loss is a linear
congestion cost, and the negated expert bias of auxiliary-loss-free load balancing (ALF-LB) is a price
of a barrier cost. The repository implements, as patches to Megatron-LM:

- a family of **congestion losses**: linear, quadratic and softplus-barrier costs, priced at the
  realized ("hard") or the expected ("soft") load, over the micro batch or the global batch;
- **ALF-Cost**, ALF-LB with a *cost step* that moves each expert bias according to a congestion cost
  instead of the sign of the load error;
- **equilibrium diagnostics**: the swap gap and the assignment potential, logged at every step, and
  the potential gap, scored offline on frozen probe batches against an exact LP oracle.

With these, it compares 14 load-balancing arms on a model with the active-parameter count of
FLAME-MoE-290M-1.3B, trained on 11.53B tokens of ClimbMix on the CSCS Clariden cluster (NVIDIA GH200).
It also contains collapse-induction and recovery races and an ALF-LB price-recovery study.

**Without a GPU cluster**, the figures and tables of the thesis can be regenerated from the released
results in `assets/results/` and the training logs (see [Reproducing the figures and tables](#reproducing-the-figures-and-tables)).

## Repository layout

| Path | Contents |
|---|---|
| `src/moe_congestion_routing/` | Python package: congestion losses, diagnostics, LP oracle, Megatron and lm-eval glue ([details](#the-package)) |
| `scripts/` | Entry points (`*.py`) and Slurm jobs (`*.sbatch`) for data, training, probing, scoring and evaluation |
| `configs/train/` | `base_cluster.yaml` (model and schedule), one file per arm, `collapse/` for the collapse experiments |
| `configs/eval/`, `configs/data/` | lm-evaluation-harness and data-preparation configs |
| `patches/` | Patches to the pinned submodules: `Megatron-LM/` (0001–0009) and `lm-evaluation-harness/` (0001) |
| `Megatron-LM/`, `lm-evaluation-harness/` | Submodules pinned to `core_v0.18.0` and `v0.4.12`; changed only through `patches/` |
| `notebooks/` | The notebooks that write every thesis figure and table; `arm_style.py` holds the shared colors and labels |
| `assets/results/` | Released results (CSV, JSON) with `MANIFEST.csv` and a `README.md` |
| `assets/thesis/` | Generated thesis figures (`figures/*.pdf`) and table rows (`tables/*.tex`) |
| `assets/probe/`, `assets/tokenizer/gpt2/` | Frozen probe batches and the GPT-2 tokenizer |

Runs write to `artifacts/` (checkpoints, probe dumps, W&B files), which is not part of the repository.

The notebooks also read the Megatron training logs under `logs/`, which are not in the repository yet.

<!-- TODO(author): commit the logs the notebooks read (or the extracts they need), about 240 MB in total. -->

## Method in brief

All arms extend `configs/train/base_cluster.yaml`: 9 layers (1 dense, 8 MoE), 64 routed experts,
top-8 routing, no shared experts, 296.5M active and 1.265B total parameters, 5,500 steps at a global
batch of 1,024 × 2,048 tokens, a warmup-stable-decay schedule, and router z-loss 10⁻³. The config name
is the arm name in every result file.

| Arm (`configs/train/<arm>.yaml`) | Name in the thesis | Method | Cost | Load | Scope |
|---|---|---|---|---|---|
| `control` | Control | none | — | — | — |
| `switch`, `switch_global` | Switch, Switch (global) | Switch auxiliary loss (Megatron) | linear | hard | micro / global |
| `rosenthal_{linear,quadratic}_{hard,soft}` | CL-{Lin,Quad}-{Hard,Soft} | congestion loss | linear / quadratic | hard / soft | micro |
| `rosenthal_{linear,quadratic}_{hard,soft}_global` | CL-{Lin,Quad}-{Hard,Soft} (global) | congestion loss | linear / quadratic | hard / soft | global |
| `rosenthal_barrier_hard` | CL-Barrier-Hard | congestion loss | softplus barrier | hard | micro |
| `alflb` | ALF-LB | expert bias, sign step, sigmoid scores | — | hard | global |
| `rosenthal_price_barrier` | ALF-Cost | expert bias, cost step, sigmoid scores | softplus barrier | hard | global |

The loss arms use the weight α = 0.01. The cost families and their slope-matched coefficients are
defined in `src/moe_congestion_routing/losses/cost_families.py`.

For a batch of N tokens, E experts and top-K routing, the balanced load is L = NK/E. With the mean
normalized score P_e and the realized count n_e, the soft and hard relative loads are E·P_e and
n_e/L, both 1 at balance. The hard loss with the linear cost and λ = 1 equals Megatron's Switch loss
for every K. The swap gap is the per-token mean gain of the best single swap of one selected expert,
at the joining and staying prices of the cost (`metrics/swapgap.py`). The normalized potential gap is
(Φ(s) − Φ(s\*)) / (Φ(s°) − Φ(s\*)), with s the realized routing, s\* an exact potential minimizer and s°
the affinity-blind round-robin routing (`metrics/phi_gap.py`). A probe *asset* is a frozen set of
16 × 2,048 tokens, a *unit* one half of an asset, and an *instance* one unit at one MoE layer.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```bash
git submodule update --init                # Megatron-LM and lm-evaluation-harness (a path dependency)
uv sync --inexact --group analysis         # environment for the notebooks (numpy, pandas, matplotlib, nbconvert)
```

**Cluster (CSCS Clariden).** The Slurm jobs run in the container environment `alps6`
(`alps6.toml`, copied to `~/.edf/alps6.toml` on the cluster) via `srun --environment=alps6`. They read
user paths from a git-ignored `config.sh`, created from `config.sh.example` (`WORKDIR`, `DATA_STORE`;
export `WANDB_API_KEY` for W&B logging). The first job on a fresh work directory builds the
environment and applies the patches; submit it with `MOE_PREPARE_ENV=1`. Later jobs only verify the
patches.

**Patches.** `Megatron-LM/` is never edited in place:

```bash
./scripts/apply-patches.sh            # reset the submodules to their pinned commits and apply patches/ in order
./scripts/apply-patches.sh --verify   # check that the patches are applied
```

| Patch | Adds |
|---|---|
| `0001` | per-layer load metrics in the router (MaxVio, load CV, dead experts, swap gap, congestion potential) |
| `0002` | throughput logging |
| `0003` | the congestion losses (`rosenthal`, `global_rosenthal`), using `losses/rosenthal.py` |
| `0004` | a logit-space gradient probe of the router |
| `0005` | the router probe: periodic forward passes on the frozen probe assets, dumped as `.npz` |
| `0006` | ALF-LB's token counts kept in fp32 under bf16 training |
| `0007` | the cost step of ALF-Cost (`moe_router_bias_update_rule: rosenthal_price`) |
| `0008` | the balancing loss on softmax scores while the selection uses sigmoid scores |
| `0009` | negative auxiliary-loss weights (collapse induction) |

## Reproducing the figures and tables

Three notebooks write every figure and table file of the thesis to `assets/thesis/`. They read the
committed files in `assets/results/` and the training logs (see the note above). They solve no LP
and load no model. Run them from the repository root:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/phi_gap_trajectories.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/wandb_figures.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/alflb_price_internalization.ipynb
```

Set `THESIS_DIR=<thesis folder>` to also copy the outputs into a thesis checkout, and
`WANDB_FIGURES=<names>` or `ALFLB_FIGURES=<names>` to draw only some figures (every number is
computed either way). The first cell of each notebook defines its terms and rules, and the last cell
prints every number the thesis quotes. Figures are written without a creation date, so a re-run
changes only files whose content changed.

| Output in `assets/thesis/` | Notebook | Content |
|---|---|---|
| `tables/tab_ex_phi_gap_ranking{,_with_loss}.tex` | `phi_gap_trajectories` | potential-gap ranking of the 14 arms |
| `tables/tab_ex_phi_gap_decomposition.tex`, `tab_ex_robustness.tex`, `tab_ex_paired.tex`, `tab_ex_lambda_sweep.tex` | `phi_gap_trajectories` | gap decomposition, robustness, paired comparisons, λ sweep |
| `tables/tab_ex_throughput{,_by_scope}.tex`, `figures/fig_ex_throughput.pdf` | `phi_gap_trajectories` | idealized expert computation time |
| `tables/tab_ex_lm_eval.tex` | `phi_gap_trajectories` | downstream scores |
| `figures/fig_ex_{phi_gap,maxvio,gap_parts}_trajectories.pdf` | `phi_gap_trajectories` | probe statistics over training |
| `figures/fig_app_training_dynamics.pdf`, `fig_app_spike_anatomy.pdf`, `tables/tab_app_spikes.tex` | `wandb_figures` | per-step training statistics and loss spikes |
| `figures/fig_app_online_swapgap{,_zero}.pdf` | `wandb_figures` | logged swap gap and assignment potential |
| `figures/fig_app_collapse_induction.pdf`, `fig_step125_race.pdf`, `fig_step125_variants.pdf`, `fig_zloss_race.pdf`, `fig_app_revival_probes.pdf` | `wandb_figures` | collapse inductions and recovery races |
| `tables/tab_app_alflb_lp.tex`, `tab_app_alflb_synthetic.tex`, `tab_ex_alflb_train.tex`, `figures/fig_ex_alflb_kappa.pdf` | `alflb_price_internalization` | ALF-LB biases against the optimal duals |

Table files hold only LaTeX rows; the thesis reads them with `\importrows{...}`.

## Reproducing the experiments

All commands run from the repository root; cluster commands assume `source config.sh`.

**Data.** Training uses `climbmix_small` of `nvidia/Nemotron-ClimbMix` (GPT-2 token ids), converted
into a Megatron `.bin`/`.idx` prefix under `$DATA_STORE` with a provenance manifest:

```bash
sbatch scripts/prepare_climb.sbatch configs/data/climbmix_small.yaml
```

The probe assets in `assets/probe/` are frozen and must not be regenerated, since every probe dump
is keyed to their tokens (`scripts/make_probe_batch.py --show <asset>` inspects one).

**Training the 14 arms.** `scripts/run_moe_pretrain.py` resolves a config's `extends` chain into a
Megatron command (`--dry-run` prints it). On the cluster, the fleet ran on 4 nodes × 4 GH200:

```bash
MOE_PREPARE_ENV=1 sbatch --nodes=4 scripts/train_moe_climblab.sbatch configs/train/control.yaml   # first job
sbatch --nodes=4 scripts/train_moe_climblab.sbatch configs/train/<arm>.yaml [<checkpoints to resume>]
```

Probe dumps are written every 250 steps and every 25 steps in the windows set by
`moe_probe_dense_windows`; the online diagnostics go to the training log and to W&B.

**Collapse induction and recovery races.** `configs/train/collapse/` holds the inductions
(`induce_negaux.yaml`: negative auxiliary-loss weight; `induce_{softmax,sigmoid}.yaml`: negative
z-loss) and the race arms, which restart from one collapsed checkpoint with identical weights, seed
and data order (`race_base.yaml`):

```bash
sbatch scripts/train_moe_climblab.sbatch configs/train/collapse/induce_negaux.yaml "" induce-negaux
sbatch --nodes=2 scripts/train_moe_climblab.sbatch configs/train/collapse/race/<arm>.yaml \
       artifacts/collapse/induce_negaux/induce-negaux/checkpoints step125-<arm>
```

<!-- TODO(author): the configs at HEAD reproduce the race from step 125. Name the commit or settings for the earlier races (from step 150 and from the negative z-loss checkpoint). -->

**Potential-gap scoring.** Each probe dump is scored against the reference game with an exact LP
(SciPy/HiGHS), one independent solve per instance, on CPUs. The driver resumes from its output CSV:

```bash
sbatch scripts/run_phi_gap_grid.sbatch artifacts/exp1/<arm>/<run> assets/results/phi-gap/<arm>.csv
```

**ALF-LB price recovery.** `scripts/run_price_probe_duals.sbatch` computes the LP capacity duals and
stores the trained bias for every probe cell of a run; `scripts/run_game_comparison.py` runs the
synthetic ALF-LB-against-LP grid. The command behind every committed result file is in the `source`
column of `assets/results/MANIFEST.csv`.

**Downstream evaluation.** Checkpoints and the FLAME-MoE-290M-1.3B reference (iteration 5473) are
scored with the pinned lm-evaluation-harness through its `megatron_lm` backend. Compute nodes run
offline, so the datasets are fetched first:

```bash
sbatch scripts/prepare_eval_cache.sbatch
sbatch --time=02:00:00 scripts/eval_moe.sbatch configs/eval/extended_suite.yaml artifacts/exp1/<arm>/<run>/checkpoints 5500
```

## Released results

`assets/results/` holds every number the notebooks use, apart from the training logs. `MANIFEST.csv`
describes the derived result files with their tier (`headline`, `supporting`, `provenance`,
`superseded`), the question each one answers, and the command that produced it.

| Directory | Contents |
|---|---|
| `phi-gap/` | potential gaps per arm, probe step, asset, unit, layer and reference cost; the λ sweep |
| `eval/` | lm-evaluation-harness results per arm at step 5500, and of FLAME-MoE at iteration 5473 |
| `wandb/` | W&B exports, one CSV per metric: `11B/` (the 14 arms), `collapse-induce/`, `recovery/`, `recovery-null/` |
| `revival-probes/` | summary of the probe dumps of the race from step 150 (`scripts/summarize_revival_probes.py`) |
| `price-recovery/` | optimal duals and trained biases of the probe units, and the ALF-LB studies built on them |
| `barrier/` | the τ calibration of the softplus barrier |

Not included: model checkpoints, raw probe dumps (the dumps of the step-125 race no longer exist),
the tokenized data and the full W&B histories.

<!-- TODO(author): link checkpoints and probe dumps if they are published (e.g. Zenodo, Hugging Face), or state that they are available on request. -->

## The package

| Module | Purpose |
|---|---|
| `losses/rosenthal.py`, `losses/cost_families.py` | congestion losses (soft and hard), discrete potential, cost step; cost families and coefficients |
| `metrics/swapgap.py`, `metrics/phi_gap.py`, `metrics/phi_gap_grid.py` | swap gap; potential gap of one probe unit; resumable grid driver |
| `metrics/dual_store.py`, `metrics/triad.py`, `metrics/probe_comparison.py` | LP capacity duals and stored biases; κ estimators |
| `metrics/router_probe.py`, `metrics/probe_dump_format.py` | probe-dump writer and readers |
| `game/lp.py`, `game/incremental.py`, `game/alflb.py` | capacity LP oracle; incremental-arc LP for the potential minimizer without capacity; offline ALF-LB replica |
| `training/`, `data/`, `eval/`, `infer/` | config-to-Megatron translation, probe schedule, data preparation, lm-eval and generation launchers |

The `losses/`, `metrics/` and `game/` modules contain no Megatron code; the patches import them.
Tests sit next to the code (`src/**/<module>_test.py`): `uv run pytest`, `uv run ruff check .`.

## Citation

```bibtex
@mastersthesis{obermaier2026gametheoretic,
  author  = {Obermaier, Pierre},
  title   = {Game-Theoretic Routing for Mixture-of-Experts Models},
  school  = {University of Zurich},
  type    = {MSc thesis},
  address = {Zurich, Switzerland},
  year    = {2026}
}
```

<!-- TODO(author): add the month and a URL or DOI. -->

## License

<!-- TODO(author): choose a license for the code and the released results. Megatron-LM and lm-evaluation-harness keep their own licenses; ClimbMix and the FLAME-MoE weights are not redistributed here. -->

## Acknowledgments

This work was supervised by Paul Friedrich and Prof. Dr. Giorgia Ramponi. The experiments ran on the
Clariden cluster of the Swiss National Supercomputing Centre (CSCS). The code builds on NVIDIA
Megatron-LM and EleutherAI's lm-evaluation-harness, trains on NVIDIA's Nemotron-ClimbMix, and uses
FLAME-MoE-290M-1.3B as its architecture and evaluation reference. This work was supported as part of the
"Swiss AI initiative" by a grant from the Swiss National Supercomputing Centre (CSCS) under project ID
aa004 on Alps.

<!-- TODO(author): confirm the CSCS project ID. -->

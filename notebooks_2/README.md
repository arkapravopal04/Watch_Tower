# Follow-up round: SVD initialization, protection + replay, a larger protection basis

The main study (notebooks/ NB0–NB6) is **not touched**. Its code, configs and
notebooks are unchanged. This round adds an add-on package (`pclx/`) that
imports `pcl`, three notebooks (NB7–NB9), and its own code Dataset.

## The questions

The main study (dev200, selected checkpoints, seeds 0–1) left three open questions:

1. **SVD initialization.** The Lean proofs (`verification/`) show something precise. Initialize DoRA's
   A from the top-8 right singular vectors of the *projected* skill-B gradient,
   scaled to the spectral norm of the usual Kaiming draw. Then one plain (SGD) gradient
   step on B, with the DoRA row gains held at their initial value 1 (the detached-norm map
   the gradient is taken through), is the best rank-8 approximation of the full projected
   step. It also gives the most negative first-order loss change among all rank-8 A of that norm.
   This is exact-real arithmetic about that one step only. It does not cover AdamW, the magnitude
   step, the row-norm renormalization the forward pass applies after B moves (a first-order
   change, about 1–2% of the first-order loss change in a tiny-model check), later steps, FP16, or
   accuracy; see "What is NOT proved" in `verification/README.md`.
   Does that translate into faster B learning, or less forgetting of A, under AdamW on the real model?
2. **Protection + replay.** Protection cut forgetting of A roughly in half versus
   plain sequential training (−3 to −6.5 pp vs −10 to −12.5 pp). Only replay
   (−1 pp, seed 1) met the −2 pp rule. Does combining them close the gap?
3. **Is k ≤ 128 the bottleneck?** The protection basis hit its cap of 128
   directions in 68–70 of 72 layers, at about 90% captured energy instead of 95%.

Before any long run, NB7 also measures whether the premise of (1) holds. It checks how
concentrated the B gradient is, and how much of it falls inside A's protected subspace.

## Arms

Every arm starts from the same seed's verified M_A. It shares the seed, the data
order, the token budget and the evaluation batches with the main study's runs,
and differs from its pair in one thing.

| Tier | Arm | Paired with | The one difference |
|---|---|---|---|
| 1 | `protected_svd` | `protected` (main study) | A from the projected B gradient instead of Kaiming |
| 1 | `protected_replay` | `protected` (main study) | adds 10% replay of Stage-A memory |
| 2 | `protected_replay_svd` | `protected_replay` | SVD initialization on top of protection + replay |
| 2 | `sequential_svd` | `sequential` (main study) | A from the unprojected B gradient |
| 3 | `protected_k256` | `protected` (main study) | up to 256 protection directions (99% energy target) |

The main study's `sequential`, `protected`, `random_q` and `replay` runs are
reused as baselines and **not retrained**.

## Where the time goes, and what this round changes

A main-study Stage-B run took 3.9–4.3 h of training plus dev scoring, then ~50
minutes of merge checks: about 5 h per run. Only ~62 minutes of that is training.
Each dev-scoring pass (200 B + 200 A + 200 general answers at batch 8) takes ~25
minutes, and the main study scored 7 times per run.

| Change | Saving per run | Why it keeps the comparison fair |
|---|---|---|
| Score dev only at update 35 and the final update | 5 passes, ~125 min | The main study scored these same updates, so baselines are compared on this grid. B goes from ~16% to ~40% by update 35, and every main-study curve is monotone after it. `score_at: [35, 105]` in `configs/next.yaml` restores the mid-run point for +25 min per run. |
| Reuse update-0 predictions | 1 pass, ~25 min | Update 0 is the same model (M_A, B = 0) in every Stage-B run of a seed. They are copied only after model, dev-file and prediction hashes match. |
| No verified merge | ~50 min | These are exploratory comparisons, not promotion candidates. Adapters and selectable checkpoints are kept, so a winner can be merged later. |
| Gradient checkpointing off when it fits | ~15 min, if it fits | It only recomputes activations. A memory probe on the longest example decides, and the replication check covers both modes. On a T4 it probably will not fit (pcl's DoRA forward keeps an FP32 copy of each adapted weight for backward), so the budget below assumes checkpointing stays on. |
| Early learning speed from training losses | no evaluation needed | Logged every update. Paired runs see the same data in the same order. |
| Baselines reused | 12 runs, ~60 h | Same code path. NB8's first item checks that this driver reproduces the main study's protected run loss for loss. |

Also in the code: NB7's two kinds of task are balanced over both GPUs while the
tests run on the CPU; the spectrum uses an FP64 eigendecomposition of the
2560×2560 Gram matrix rather than 144 full SVDs; the replication check runs once
when both checkpointing modes would be the same run; and every queued run frees
its 8 GB model before the next one loads.

**Result: ~1.9 GPU-hours per arm and seed** (3 min loading, ~62 min training,
2 × 25 min scoring) instead of ~5.

## Budget

| Notebook | What runs | Time |
|---|---|---|
| NB7 | add-on tests, gradient spectrum + SVD bases + k ≤ 256 basis, 3 seeds on 2 GPUs | ~35–40 min |
| NB8, session 1 | replication check (~5 min), then tiers 1 and 2 (12 items, about 6 per GPU; a run the clock cuts off resumes next session) | ~12 h |
| NB8, session 2 | tier 3 (3 items) | ~4 h |
| NB9 | report (CPU) | minutes |

That is about 17 GPU-hours for the whole round. Tier 1 alone (6 items) takes ~6 h;
set `TIERS = (1,)` in NB8 to stop there.

This round needs the main study's Stage B runs (NB4) for its pairs. NB8 runs
whatever has its inputs, and NB9 reports whatever exists.

## Running it on Kaggle

1. `python notebooks_2/build_code_zip.py` writes `pcl-next-code.zip` next to the repository
   (the frozen study code plus `notebooks_2/`). Upload it as a **new** private
   Dataset, e.g. `pcl-next`. Keep the main study on its own `pcl-code`, and never
   attach both to one notebook.
2. Import `NB7_spectrum`, `NB8_stage_b_next` and `NB9_next_report` from this folder.
   Run each with **Save Version → Save & Run All (Commit)**. Inputs are found by
   content, so you only attach them.

| Notebook | GPU | Internet | Attach | Run until |
|---|---|---|---|---|
| `NB7_spectrum` | T4 x2 | OFF | pcl-next, model, NB2, all NB3 | `spectrum/spectrum_report.md` is shown |
| `NB8_stage_b_next` | T4 x2 | OFF | pcl-next, model, NB2, all NB3, all NB4, NB7, earlier NB8 runs | it prints "Follow-up arms finished" |
| `NB9_next_report` | none | OFF | pcl-next, NB2, all NB3, all NB4, NB7, all NB8 | it shows `report_next/report.md` |

NB8 resumes interrupted runs exactly, as the main study does. Run it again with
its previous output attached until nothing is left.

## Rules fixed before any follow-up run

- **Go / no-go for the SVD arms** (`configs/next.yaml`, `pclx.spectrum.go_no_go`).
  The SVD arms run only if, in every seed, the median top-8 share of the projected
  gradient's energy is at least 10× what a random 8-dim subspace captures, and the
  predicted AdamW first-step speed-up is at least 1.2×. Otherwise NB8 skips them.
  `FORCE_SVD_ARMS` overrides this, and the override is visible in the notebook.
- **Replication:** the first 6 updates of the main study's seed-0 protected run,
  rerun through this driver, must match its training losses to 0.1% relative. This is
  checked in "auto" mode, the mode the arms use. If "auto" turned checkpointing off, it is
  also checked with checkpointing on. If only "on" passes, set
  `gradient_checkpointing: on` in `configs/next.yaml`.
- **Selection:** the main study's dev rule on the common grid (max B dev200, with A ≥ −2 pp vs
  update 0 and general ≥ −2 pp vs M0; ties to the earliest update).
- **Readings:** update 35 (early), final update (selection-free) and selected checkpoint
  (optimistic). The report gives paired scenario-group bootstrap intervals, pooled over seeds.

## How the SVD initializer is built (NB7, `pclx.spectrum`)

- Take exactly the run's first update: its 16 examples, in the seed's own training order, with the trainer's
  normalization (summed answer-token cross-entropy ÷ that update's answer tokens). The Lean theorems are about one
  plain step on the loss whose gradient is G, so G must be that step's gradient. The replay arms' first update
  mixes in Stage-A memory, so `protected_replay_svd` gets its own bases (`svd_bases_replay.pt`). A run checks the
  example ids, the protection hash and its own Kaiming scale against the bases, and refuses to start if any differ.
- Compute the training loss's gradient G = ∂L/∂W for every adapted q/v weight at M_A.
  The formula is exact (FP16 activations and output gradients, FP32 sums), computed with hooks (Σ δᵀx accumulated straight into FP32 sums), so the FP16 base
  weights need no gradient. It runs with gradient checkpointing (pure recomputation) so it fits on a T4.
- Project it, H = G(I − QQᵀ), and take its right singular vectors (an FP64 eigendecomposition
  of HᵀH, which gives the same vectors as an SVD and is much faster).
- Set A = c·V₈ᵀ, where c is the largest singular value of that layer's own Kaiming draw as the forward pass sees it
  (A(I − QQᵀ)). The Kaiming draw is consumed first, so the random stream is unchanged.
- B = 0, so the model at update 0 is exactly M_A.

NB7 also reports, per layer:
- the share of the B gradient inside A's protected subspace;
- the top-r energy shares, with the random-subspace baseline;
- the effective rank;
- the predicted first-step decrease for SGD (s²‖HAᵀ‖²_F) and for AdamW's sign-like first step (s‖HAᵀ‖₁).

## What this round does not do

- No locked tests. Everything is dev data, which the main study also selected on.
  An arm worth adopting needs a verified merge and a
  locked test on fresh items, generated from a new seed.
- No new skills, no new hyperparameters, and no change to the main study's
  protocol, gates or registry.

## Local development

```bash
python -m pytest -q notebooks_2/tests          # unit tests + the whole round in miniature (~3 min on CPU)
python notebooks_2/build_notebooks.py          # regenerate NB7-NB9 (edit the script, not the .ipynb)
python notebooks_2/build_code_zip.py           # the Kaggle code Dataset
```

The tests include two numerical checks of the Lean theorems against the real
`ProtectedDoRALinear`: B's gradient equals `scale·G·(I − QQᵀ)·Aᵀ`, and A's
gradient is exactly zero at initialization.

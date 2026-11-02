# Watch Tower

An autonomous system for detecting, spawning, fine-tuning, and merging adapters,
designed for seamless integration with the Hive.

Hive is a multi-agent system that should get better at its work without quietly
getting worse at what it already knows. Watch Tower is the part that keeps watch:
it spots where a new skill is needed, spawns an adapter for it, fine-tunes that
adapter while protecting existing skills, and merges it only if independent tests
say nothing was lost. Anything else rolls back.

```
        ┌────────────────────── Watch Tower ──────────────────────┐
        │                                                         │
Hive ──►│ detect ─► spawn ─► fine-tune ─► gate ──pass──► merge ───┼──► Hive
        │                       ▲          │                      │
        │                  protect old     └──fail──► roll back   │
        │                     skills                              │
        └─────────────────────────────────────────────────────────┘
```

Two rules hold throughout. The judge sits outside the loop: gates and test sets are
fixed before any test is seen, and nothing being trained can change them. And every
merge is reversible and verified bit for bit.

## Status

Watch Tower is research code. The fine-tuning and merge core is built and has been
run end to end on GPU; the autonomy around it is not built yet.

| Part of the loop | Status | Where |
|---|---|---|
| Detect a skill gap or regression | Not built | |
| Spawn an adapter | Partial: DoRA adapters are created by the pipeline, started by hand | `pcl/model_setup.py`, `pcl/protected_dora.py` |
| Fine-tune with protection | Built and run | `pcl/train.py`, `pcl/protection.py` |
| Gate | Built and run | `pcl/registry.py`, `configs/gates.yaml` |
| Verified merge and rollback | Built and run | `pcl/merge.py`, `pcl/merge_workflow.py` |
| Autonomous orchestration | Not built: each stage is a notebook run by a person on Kaggle | `notebooks/`, `notebooks_2/` |
| Hive integration | Not built | |

## The study behind it

The first question is whether an adapter can learn something new without damaging
what the model already knows. To answer it, the study teaches Qwen3-4B skill A
(compressible-flow aerodynamics), protects what it learned, teaches skill B
(low-speed aerodynamics), and tests whether A survived:
learn → protect → learn the new skill → dev selection → verified merge →
locked test → accept or roll back. Built for Kaggle's free T4 GPUs.

- The adapter is **DoRA** ([protocol](docs/dora_protocol.md)). The protected method
  projects the complete DoRA update, magnitude included, off the directions skill A
  used.

## Results so far

**The gate rejected the preregistered candidate, so the system kept M_A.** That is
the rollback path doing its job. On the locked test, `protected_s0` learned B
(+28.9 pp) and passed the general and numerical gates, but failed `kept_A`:
A changed by −1.4 pp with a 95% interval of [−3.8, +1.0], and the gate needs the
lower bound to be at least −2. That reads as inconclusive, not as a pass.

| Method (locked test, 3 seeds pooled) | Change in skill A |
|---|---|
| Sequential (no protection) | −7.7 pp |
| Random Q (control) | −6.8 pp |
| **Protected** | **−2.2 pp** |
| Replay | −1.3 pp |

Protection cut forgetting by about 70% against sequential training, and the random
control shows no such benefit. It ties replay within noise. No replay seed's lower
bound reached −2 either.

The follow-up round ([details](notebooks_2/README.md)) used dev data only, so these
readings are not promotion evidence:

- **SVD initialization:** rejected. Early training loss is lower in all 9 seed pairs,
  but final accuracy shows no measurable gain on B or A.
- **Protection + replay, and a larger protection basis (k ≤ 256):** each improves
  final A by about 4 pp over protection alone, intervals [+1.0, +7.7] and
  [+0.8, +7.4]. Both still sit near −2.5 pp. Either one needs a fresh locked test, and
  the two have not been run together.

## Verified

- **Locally:** the full test suite, including a slow test that runs the whole study
  in miniature on a tiny offline Qwen3, and a rehearsal of all seven notebooks
  against a fake `/kaggle` folder. DoRA matches pinned PEFT 0.18.0.
- **On GPU:** the study was run on Kaggle through the notebooks below, with NB0 and
  NB2 checking the GPU environment first.
- **In Lean 4:** machine-checked proofs of complete-update protection, the identity
  initialization, the detached-norm gradients and the first plain-SGD step from the
  SVD initializer. AdamW, FP16 and accuracy claims are explicitly not proved. See
  [`verification/`](verification/README.md).

## Running it on Kaggle

Upload this repository as a private Kaggle Dataset
called e.g. `pcl-code`, and import the seven notebooks from `notebooks/`. Run each
with **Save Version → Save & Run All (Commit)**. Each notebook finds its inputs by
content, so you only attach them.

| Notebook | GPU | Internet | Attach | Run until |
|---|---|---|---|---|
| `NB0_check` | T4 x2 | ON | code | it prints `NB0 PASSED` |
| `NB1_download` | none | ON | code | done; then **make a Dataset from its output** ("model") |
| `NB2_data_and_pilot` | T4 x2 | OFF | code, model | `pilot/pilot_report.md` says `precision_validated: True` and no concerns |
| `NB3_stage_a` | T4 x2 | OFF | code, model, NB2, earlier NB3 runs | it prints "Stage A finished for all seeds" (≈2 runs) |
| `NB4_stage_b` | T4 x2 | OFF | code, model, NB2, all NB3, earlier NB4 runs | it prints "Stage B finished for all seeds" (≈3 runs) |
| `NB5_freeze` | none | OFF | code, model, NB2, all NB3 and NB4 | it prints `NB5 done` |
| `NB6_test_and_decide` | T4 x2 | OFF | code, model, NB0, NB2, NB5, all NB3 and NB4, earlier NB6 runs | it shows the report |

The notebooks pick the next seed automatically. If a GPU session runs out, the
stage reports "incomplete"; run the same notebook again with its previous output
attached. Training resumes bit-exactly, and finished work is reused. Locked testing
only starts a file it can finish, and never rescores one. Set `SESSION_HOURS` in
the first cell of GPU notebooks to your account's session limit.

**Fixed in advance:** the pilot writes `pilot/study.yaml` (merge tolerances, eval
batch size, dev cadence), and each value carries the rule that produced it. The
gates (`configs/gates.yaml`) and the primary candidate (`PRIMARY` in NB5,
default `protected_s0`) are fixed before any test is seen.

NB7–NB9 and the `pclx` add-on package are the follow-up round, with their own code
Dataset; the runbook is in [`notebooks_2/README.md`](notebooks_2/README.md).

## What each step does

- **Data** (`configs/data.yaml`): A = Mach, isentropic, normal shock and
  applicability questions; B = dynamic pressure, lift, drag, inverse speed and
  missing-input questions; plus general controls. Groups are physical scenarios
  with two wordings (scenario-held-out, *not* formula-held-out). Nothing is
  truncated. The **audit** re-derives every answer from the question text with
  separate code: shocks from the Rankine–Hugoniot conservation laws, inverses by
  bisection, and stagnation ratios from the energy equation. It also writes a
  100-item-per-skill review sheet. This stands in for a human reviewer.
- **Pilot:** FP32-on-CPU reference vs the FP16 model, training probes for
  ordinary and protected DoRA (speed, VRAM, GradScaler skips), generation speed
  per batch size (sanity-checked against batch 1; every model is then scored with
  the same length-sorted batches), Q-capture cost, FP16 merge rounding, the M0 baseline with
  a ceiling check, and a GPU-hour budget.
- **Stages A / B / joint:** train with dev scoring at a fixed update cadence
  (update 0 = the untouched starting model). Then select by the dev-only rule:
  maximize the new skill subject to retention ≥ −2 pp (A vs its start, general vs
  M0), with ties going to the earliest update; a run with no eligible checkpoint is
  reported as ineligible, never dropped. Next comes a verified merge: live =
  exported = merged = reloaded, on logits and on decoded answers. Stage A then
  captures the protection basis Q. Paired methods share initialization and the
  token budget; joint gets the combined A+B budget.
- **Merged models are weight patches:** only `q_proj`/`v_proj` change, so each
  stage stores those tensors (~0.9 GB instead of ~8 GB) plus the exact hashes of
  the whole model. Loading rebuilds it bit-identically on the pinned base.
- **Freeze / test / decide:** the registry freezes model hashes, test-file hashes,
  gates, the selection rule and the primary candidate. Each model/test pair is
  scored once. The gates (learn B ≥ +5 pp with a lower bound > 0; keep A and
  general with lower bounds ≥ −2 pp; all numerical checks pass) apply to the
  primary candidate only. A rejection keeps M_A. The report adds per-method
  contrasts and exploratory pooled-seed intervals.

## Repository layout

| Path | What it is |
|---|---|
| `pcl/` | The core package: training, protection, merge, registry, gates, reports |
| `pcl/data/` | Question generator, independent solvers, audit, grader, splits |
| `configs/` | Data, model, training, protection and gate settings |
| `notebooks/` | NB0–NB6, the main study (generated by `scripts/build_notebooks.py`) |
| `notebooks_2/` | NB7–NB9 and `pclx`, the follow-up round |
| `verification/` | The Lean 4 proofs |
| `docs/` | The DoRA protocol |
| `tests/` | Test suite, including the miniature study |

## Local development

```bash
python -m pip install -e ".[model,test,reference]"
python -m pytest -q                 # everything, incl. the ~2.5 min miniature study
python -m pytest -q -m "not slow"   # quick suite
python scripts/build_notebooks.py   # regenerate notebooks/ (edit this script, not the .ipynb)
```

Commands (`python -m pcl <command> --help`): `generate`, `audit`, `train`,
`select`, `merge`, `protect`, `pilot`, `stage-a`, `stage-b`, `stage-joint`,
`register`, `evaluate-locked`, `decide`, `report`, `download`, `check-env`, `smoke`.

## Limits

- FP16 base weights on T4, 512 tokens, 2,000 items per skill: these are protocol
  choices, so report them. If FP16 fails the pilot, the fallback (Qwen3-1.7B in
  FP32) is a new study.
- Dev selection uses dev200 point estimates; promotion uses locked group-bootstrap
  intervals. Three seeds are exploratory, and the gate applies to one preregistered
  model. An interval crossing a margin is inconclusive, not a pass.
- Two skills from one domain do not show how the loop behaves at Hive's scale. A third
  skill, ablations and prior-method baselines (O-LoRA, LoRA-Null, …), which any novelty
  claim would need, are out of scope so far.
- Known bug: a float comparison in `notebooks_2/pclx/report.py` marks exactly −2.0 pp
  as ineligible, which mislabels `protected_k256` seeds 1 and 2 in the follow-up report.

## License

See [LICENSE](LICENSE).

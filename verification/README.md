# Lean verification of protected DoRA and the SVD initializer

**Status (26 October 2026): verified.** `verify.ps1 -IndependentCheck` passes
with Lean 4.35.0 and Mathlib pinned to commit
`56a005855ea17e0b0958f73ec801ae140b0a1679`:

- all 9 proof modules compile, with no `sorry`, `admit` or custom `axiom` in the sources;
- `AxiomAudit.lean` finds every theorem declared in a `ProtectedDoRA.*` module
  automatically (112 written by hand, plus 10 lemmas the compiler generates). Each
  one depends only on Lean's three standard axioms: `propext`,
  `Classical.choice` and `Quot.sound`;
- independent re-checks:
  - `leanchecker` replays all 10 modules through the kernel again;
  - `nanoda`, a separate type checker written in Rust, re-checks the exported
    dependency closure: 36,246 declarations, with only those three axioms allowed.

Two of the theorems are also checked numerically against the real PyTorch
`ProtectedDoRALinear` in `notebooks_2/tests`: the B gradient formula
`gradB = scale·G·(I − QQᵀ)·Aᵀ` (`gradB_spec`) and `gradA = 0` at
initialization (`gradA_at_zero`).

Two negative tests confirm the checks can fail. A throwaway module containing
`sorry` made the audit fail (`sorryAx`, exit 1). Removing `Classical.choice` from
nanoda's whitelist made nanoda fail (exit 1).

## Conventions

Exact real matrices, with PyTorch's `(out, in)` layout, matching
`pcl/protected_dora.py` and `docs/dora_protocol.md`:

| Lean | Meaning | Python |
|---|---|---|
| `allowed Q` | `P = I - Q Qᵀ`, assuming `Qᵀ Q = I` | `residual_x = x - (x @ Q) @ Q.T`; `validate_basis` |
| `direction W B A s` | `V = W + s B A` | `base.weight + (B @ A) * scale` |
| `rowNorm V` | Euclidean norm of each output row | `torch.linalg.vector_norm(direction, dim=1)` |
| `doraGain` | `g = m / rowNorm V` | `gain = magnitude / direction_norm()` |
| `protectedWeight W D Q` | `W + (D - W) P`, with `D = diag(g) V` | the whole update is projected |
| `detachedWeight W B A s g Q` | the same weight with `g` held fixed | gradients with row norms detached (`@torch.no_grad()`) |
| `rowNorm W i ≠ 0` | nonzero base rows | `direction_norm` rejects norms `<= 1e-12` |
| `SpectralData H` | orthonormal right singular vectors `V` of `H`; `eig` holds σᵢ² | `torch.linalg.svd(H).Vh` (exact arithmetic) |
| `svdInit sd s c` | proposed initializer `A = c V_sᵀ` | **not implemented** in `pcl/` |

`frob2 M` is `‖M‖_F²`. `frob2_eq_norm_sq` proves it equals the square of
Mathlib's own Frobenius norm. `IsTopSet e s` says the index set `s` holds the
largest values of `e` (nothing outside beats anything inside), with ties
allowed on either side.

## What is proved

### 1. Protection (`Projection.lean`)
- `P` is symmetric (`allowed_transpose`) and idempotent (`allowed_idempotent`),
  and `P Q = 0` (`allowed_mul_basis`).
- The complete update, magnitude part included, annihilates `Q`
  (`complete_update_annihilates_basis`). Hence `W_eff Q = W Q`
  (`protected_weight_on_basis`), and inputs in `span Q` produce unchanged outputs
  (`protected_input_preserved`).
- `storage_leakage_identity`: if the intended update annihilates `Q`, any leakage
  after storage is exactly `(stored - intended) Q`. This is an identity only. It
  does **not** bound FP16 error.

### 2. Initialization and the forward pass (`Initialization.lean`, `FirstStep.lean`)
- With `B = 0` and `m = rowNorm W`, ordinary and protected DoRA both reproduce
  `W` for **any** `A`, including an SVD-derived one
  (`dora_initial_identity`, `protected_dora_initial_identity`, `svdInit_identity`).
  The gains are then exactly 1 (`doraGain_init`).
- `detachedWeight_mulVec`: entry by entry, the Lean weight applied to an input
  equals the Python forward pass,
  `W x + (g - 1)·W(P x) + g·(s·B(A(P x)))` (frozen bias left out of both sides).

### 3. SVD, Ky Fan and Eckart–Young (`SVD.lean`, `Spectral.lean`, `Frobenius.lean`)

This is the matrix-to-spectral bridge. The optimality
conclusion is never assumed.

- `exists_spectralData`: every real matrix has right singular vectors. Proved
  from Mathlib's spectral theorem for `Hᵀ H`.
- `frob2_mul_eq_weighted`: for any `U`, `‖H U‖² = Σᵢ σᵢ² wᵢ` with weights
  `wᵢ = Σⱼ (Vᵀ U)ᵢⱼ²`. If `U` has orthonormal columns, each `wᵢ ∈ [0, 1]`
  (`weights_le_one`) and `Σ wᵢ = rank` (`weights_sum`).
- **Ky Fan** (`SpectralData.kyFan`): any `U` with `r` orthonormal columns captures
  at most `Σ_{i∈s} σᵢ²`. The top singular vectors attain this (`frob2_mul_cols`).
- **Eckart–Young, Frobenius** (`SpectralData.eckart_young`): for every `X` with
  `rank X ≤ r`, `‖H - H V_s V_sᵀ‖² ≤ ‖H - X‖²`. The optimal error is
  `Σ_{i∉s} σᵢ²` (`truncate_error`, `error_lower_bound`). The factored form `B A`,
  with inner size `≤ r`, is `eckart_young_factored`.
- **Rank deficiency and ties need no extra hypothesis.** Everything holds for
  every orthonormal singular basis and every top set, including when
  `rank H < r` or singular values tie at the cutoff.
- **Feasibility in Q's complement** (`truncate_mul_eq_zero`,
  `protected_truncation_optimal`): if `H Q = 0`, e.g. `H = G P`, the truncation
  also annihilates `Q`. This holds even when some chosen singular vectors have
  σ = 0 and point into `span Q`. So the truncation is also optimal among
  rank-`r` matrices that annihilate `Q`.

### 4. Gradients (`FirstStep.lean`, `Gradient.lean`)
- With detached row norms, the weight is affine in `B` and in `A`. Its exact
  Frobenius gradients are `gradB = s·diag(g)·G·P·Aᵀ` (`gradB_spec`) and
  `gradA = s·Bᵀ·diag(g)·G·P` (`gradA_spec`).
- `loss_hasFDerivAt_B` / `_A` are genuine chain-rule theorems. If the loss is
  Fréchet differentiable at the effective weight with Frobenius gradient `G`,
  then the loss as a function of `B` (or `A`) is differentiable, with gradient
  `gradB` (or `gradA`). Matrices use Mathlib's Frobenius norm here.
- At `B = 0`, `gradA = 0` (`gradA_at_zero`), so `A` does not move on the first
  plain step. Also at `B = 0`, the weight as a function of the magnitudes does
  not depend on `A` (`init_weight_independent_of_A`), so the choice of `A`
  cannot affect the magnitude path at initialization.

### 5. The first step from the SVD initializer (`FirstStep.lean`)
Let `firstStep` be one plain gradient step of size `η` on `B`, from `B = 0`
with unit gains. Write `H = G P` for the projected gradient.
- `firstStep_eq`: for any `A`, the weight change is `-(η s²)·H Aᵀ A P`.
  `firstStep_first_order`: its first-order loss change is `-(η s²)·‖H Aᵀ‖²`.
- `svdInit_firstStep_eq_truncate`: with `A = c V_sᵀ`, the change is **exactly**
  `-(η s² c²)` times the Eckart–Young truncation of `H`. So it is a best
  rank-`r` approximation of the dense projected step `-(η s² c²) H`
  (`svdInit_firstStep_best`).
- `svdInit_minimizes_first_order_opNorm`: take *any* initializer `A` of rank
  at most `r` with spectral norm at most `|c|`, i.e. `‖A x‖ ≤ |c|·‖x‖`. The SVD
  initializer gives the most negative first-order loss change among all of
  them, equal to `-(η s² c²)·Σ_{i∈s} σᵢ²` (`svdInit_first_order`). The class
  contains the SVD initializer itself (`svdInit_opNorm`, via Bessel's
  inequality, and `svdInit_rank_le`). It also contains any Kaiming draw
  rescaled so that its largest singular value is `|c|`. The orthonormal-rows
  case `A = c Uᵀ` is `svdInit_minimizes_first_order`. The matrix fact behind
  this is the operator-norm Ky Fan bound `SpectralData.kyFan_opNorm`.
- `first_order_frobenius_budget`: if only `‖A‖_F² ≤ β` is fixed, the best
  possible first-order change is `-(η s²)·σ_max²·β`, and a rank-one `A` on a top
  singular vector reaches it (`frob2_mul_transpose_le_max`, `rank_one_attains`).
  So under a Frobenius budget alone, rank one wins. The spectral-norm budget
  above is the one under which the rank-`r` SVD initializer is optimal.
- `svdInit_descent` / `svdInit_strict_descent`: assume the smoothness inequality
  `loss(W+Δ) ≤ loss W + ⟪G,Δ⟫ + (L/2)‖Δ‖²` holds along this one step. Then the
  loss drops by at least `η'(1 - Lη'/2)·E`, with `η' = η s² c²` and
  `E = Σ_{i∈s} σᵢ²`. The descent is strict when `0 < η'`, `Lη' < 2` and
  `E > 0`. The two side conditions `Descent.lean` requires,
  `⟪G,Δ⟫ = -η'E` and `‖Δ‖² = η'²E`, are now proved, not assumed.

### 6. Claims in `docs/dora_protocol.md` and the merge (`Protocol.lean`)
- "Empty Q is ordinary DoRA": `protectedWeight_of_isEmpty`.
- "Projecting A alone would leave `(diag(g) - I) W Q` unconstrained":
  `aOnly_update_on_basis` shows the A-only variant acts on `Q` by exactly
  `(diag(g) - I) W Q`. That is why the whole update is projected.
- "`rank(delta_W) <= min(out, in - k)`": `protected_update_rank_le`, using
  `rank(I - Q Qᵀ) + k ≤ in` (`rank_allowed_add_le`).
- Exports and merge: `pcl/merge.py` rebuilds the update as
  `(gain - 1)·(W - W Q Qᵀ) + gain·(B A_eff)·scale`, with
  `A_eff = A - (A Q) Qᵀ` (`effective_a`). This equals the protected update
  (`merge_intended_eq`), so the merged weight is the forward weight
  (`merge_matches_forward`). Also, `A_eff` annihilates `Q`
  (`effective_a_mul_basis`). As everywhere here, this is exact arithmetic: the
  FP32 accumulation and storage cast are not modeled.

## What is NOT proved (read before citing any of this)

- **Nothing about accuracy, forgetting or the gates.** The results concern
  weights and one gradient step. They say nothing about Skill A retention, the
  −2 pp rule, or whether SVD initialization improves final accuracy. No
  probability of improvement follows from them.
- **Exact reals only.** FP16/FP32 rounding, the `atol = 1e-5` tolerance on
  `Qᵀ Q = I`, and the `1e-12` norm floor are not modeled. The numerical
  residuals (~5e-8) are separate, empirical checks.
- **Plain gradient descent, first step only.** AdamW's first step is roughly
  `lr·sign(grad)`, which is *not* proportional to `H V_s`, so its update is not
  the Eckart–Young truncation. Momentum, weight decay, the simultaneous
  magnitude step, and all later steps are outside the model.
- **The comparison is at equal spectral norm.** The optimality result compares
  initializers whose largest singular value is at most `|c|`. A raw Kaiming draw
  has its own spectral norm, so a fair empirical comparison has to rescale it
  (or choose `c`) so the norms match. At equal *Frobenius* norm, rank one wins
  instead (`first_order_frobenius_budget`).
- **The loss is a black box.** That `G` is the true gradient of the training
  loss is the chain-rule premise (`hloss`, `hG`). The smoothness constant `L`
  is assumed, not estimated. Nothing is proved about the transformer.
- **The Python is not extracted or verified.** The correspondence table above
  and `detachedWeight_mulVec` were checked by reading `pcl/protected_dora.py`.
  The initializer is implemented in `notebooks_2/pclx/svd_init.py` (not in `pcl/`) and is
  checked numerically against these statements in `notebooks_2/tests`.
- **Gains held fixed.** `firstStep` changes the weight at fixed row gains (g = 1). The real
  forward recomputes g = m / rowNorm(W + sBA) after B moves, which subtracts each row's
  component along `W_i P` at first order in η. The truncation identity and the first-order
  ranking are about the fixed-gain step, not the renormalized weight.
- **Exact singular vectors.** The theorems apply to whichever orthonormal basis
  an SVD routine returns, assuming the decomposition is exact.
- **Not formalized:** the random-subspace baseline (expected
  captured energy `r/(n-k)` for a uniformly random rank-`r` subspace). That is a
  probabilistic statement.

## Running it

From the repository root:

```powershell
.\verification\verify.ps1                    # build + axiom audit (offline, a few minutes once built)
.\verification\verify.ps1 -FetchCache        # new machine: fetch Mathlib cache for the imported modules
.\verification\verify.ps1 -IndependentCheck  # also leanchecker + nanoda (~15 min, ~200 MB temporary export)
```

The toolchain is the official portable Lean 4.35.0 under
`outputs/lean-tools/lean-4.35.0-windows`, and the Mathlib cache sits in
`outputs/lean-tools/mathlib-cache`. Nothing is installed globally.

## Files

| File | Contents |
|---|---|
| `ProtectedDoRA/Projection.lean` | the input projector and complete-update protection |
| `ProtectedDoRA/Initialization.lean` | row norm, DoRA weight, identity initialization |
| `ProtectedDoRA/Spectral.lean` | scalar top-`r` allocation lemma, top sets |
| `ProtectedDoRA/Descent.lean` | scalar consequence of an assumed smoothness bound |
| `ProtectedDoRA/Frobenius.lean` | `frob2`, `frobInner`, trace forms, orthogonal splitting |
| `ProtectedDoRA/SVD.lean` | singular vectors, Ky Fan, Eckart–Young, Q-constrained optimality |
| `ProtectedDoRA/FirstStep.lean` | Python forward correspondence, gradients, the SVD-initialized first step |
| `ProtectedDoRA/Gradient.lean` | chain-rule derivatives, `frob2` = Mathlib Frobenius norm² |
| `ProtectedDoRA/Protocol.lean` | claims in `docs/dora_protocol.md`; merge = forward weight |
| `AxiomAudit.lean` | enumerates every theorem and rejects non-standard axioms |
| `verify.ps1` | the checks above |

The build's remaining warnings are style lints only. Some theorems carry an
unused instance argument from a shared `variable` line, and Mathlib suggests
the new module system. Neither affects what is proved.

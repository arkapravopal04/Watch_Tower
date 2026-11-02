import ProtectedDoRA.SVD
import ProtectedDoRA.Initialization
import ProtectedDoRA.Descent

/-!
# The first gradient step from the SVD initialization

Model, stated so nothing is hidden:
* exact real arithmetic (no FP16/FP32 rounding);
* row norms detached, as in `ProtectedDoRALinear.forward`: the row gains are
  held fixed while differentiating, so the weight is affine in `B` and in `A`;
* `G` is the gradient of the loss with respect to the effective weight at
  initialization (where the effective weight equals `W`);
* one plain gradient step of size `η` on `B`, from `B = 0` with unit gains.

AdamW (its first step is roughly sign-like, not proportional to the gradient),
momentum, weight decay, the simultaneous magnitude step, later steps and FP16
storage are outside this model.

With `H = G (I - Q Qᵀ)` and the initializer `A = c V_sᵀ` built from `H`'s top
right singular vectors, the results are:
* `firstStep_eq`: the B-path weight change is `-(η scale²) H Aᵀ A (I - Q Qᵀ)`;
* `svdInit_firstStep_eq_truncate`: for the SVD initializer it is
  `-(η scale² c²)` times the Eckart–Young truncation of `H`, so it is a best
  rank-`|s|` approximation of the dense step `-(η scale² c²) H`
  (`svdInit_firstStep_best`);
* `svdInit_minimizes_first_order`: among all `A = c Uᵀ` with orthonormal `U` of
  the same rank, it gives the most negative first-order loss change;
* `svdInit_descent`: under an explicitly assumed smoothness inequality, the
  loss decreases by at least `η'(1 - Lη'/2) E`.
-/

namespace ProtectedDoRA

open Matrix

variable {ο ι κ ρ : Type*} [Fintype ο] [Fintype ι] [Fintype κ] [Fintype ρ]
  [DecidableEq ο] [DecidableEq ι] [DecidableEq κ] [DecidableEq ρ]

/-! ## The forward map with detached row norms -/

/-- DoRA's row gain `m / rowNorm (W + scale • B A)`. -/
noncomputable def doraGain (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ)
    (scale : ℝ) (m : ο → ℝ) : ο → ℝ :=
  fun i => m i / rowNorm (direction W B A scale) i

theorem doraWeight_eq_diagonal_mul (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ)
    (A : Matrix ρ ι ℝ) (scale : ℝ) (m : ο → ℝ) :
    doraWeight W B A scale m = diagonal (doraGain W B A scale m) * direction W B A scale := by
  ext i j
  rw [diagonal_mul]
  rfl

/-- The magnitude initialization `m = rowNorm W` with `B = 0` gives unit gains. -/
theorem doraGain_init (W : Matrix ο ι ℝ) (A : Matrix ρ ι ℝ) (scale : ℝ)
    (hW : ∀ i, rowNorm W i ≠ 0) : doraGain W 0 A scale (rowNorm W) = 1 := by
  funext i
  simp [doraGain, zero_B_direction, hW i]

/-- The protected weight with the row gains `g` held fixed: with detached row
norms, this is the map autograd differentiates. -/
def detachedWeight (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ) (scale : ℝ)
    (g : ο → ℝ) (Q : Matrix ι κ ℝ) : Matrix ο ι ℝ :=
  protectedWeight W (diagonal g * direction W B A scale) Q

/-- At the current gains the detached map is the actual protected weight. -/
theorem protectedWeight_eq_detachedWeight (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ)
    (A : Matrix ρ ι ℝ) (scale : ℝ) (m : ο → ℝ) (Q : Matrix ι κ ℝ) :
    protectedWeight W (doraWeight W B A scale m) Q =
      detachedWeight W B A scale (doraGain W B A scale m) Q := by
  rw [doraWeight_eq_diagonal_mul]
  rfl

theorem detachedWeight_expand (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ)
    (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) :
    detachedWeight W B A scale g Q =
      W + (diagonal g * W - W) * allowed Q + scale • (diagonal g * B * A * allowed Q) := by
  simp only [detachedWeight, protectedWeight, direction, Matrix.mul_add, Matrix.mul_smul,
    Matrix.sub_mul, Matrix.add_mul, Matrix.smul_mul, Matrix.mul_assoc]
  abel

/-- The weight acting on an input is the Python forward pass, entry by entry:
`base(x) + (gain - 1) * base_residual + gain * low_rank`, with
`residual_x = (I - Q Qᵀ) x`, `base_residual = W residual_x` and
`low_rank = scale * B (A residual_x)`. The frozen bias is left out of both sides. -/
theorem detachedWeight_mulVec (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ)
    (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) (x : ι → ℝ) (i : ο) :
    (detachedWeight W B A scale g Q *ᵥ x) i =
      (W *ᵥ x) i + (g i - 1) * (W *ᵥ (allowed Q *ᵥ x)) i +
        g i * (scale * (B *ᵥ (A *ᵥ (allowed Q *ᵥ x))) i) := by
  rw [detachedWeight_expand]
  simp only [add_mulVec, sub_mulVec, smul_mulVec, ← mulVec_mulVec, Pi.add_apply, Pi.sub_apply,
    Pi.smul_apply, smul_eq_mul, mulVec_diagonal]
  ring

theorem detachedWeight_add_B (W : Matrix ο ι ℝ) (B ΔB : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ)
    (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) :
    detachedWeight W (B + ΔB) A scale g Q - detachedWeight W B A scale g Q =
      scale • (diagonal g * ΔB * A * allowed Q) := by
  rw [detachedWeight_expand, detachedWeight_expand]
  simp only [Matrix.mul_add, Matrix.add_mul, smul_add]
  abel

theorem detachedWeight_add_A (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A ΔA : Matrix ρ ι ℝ)
    (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) :
    detachedWeight W B (A + ΔA) scale g Q - detachedWeight W B A scale g Q =
      scale • (diagonal g * B * ΔA * allowed Q) := by
  rw [detachedWeight_expand, detachedWeight_expand]
  simp only [Matrix.mul_add, Matrix.add_mul, smul_add]
  abel

theorem detachedWeight_zero_one (W : Matrix ο ι ℝ) (A : Matrix ρ ι ℝ) (scale : ℝ)
    (Q : Matrix ι κ ℝ) : detachedWeight W (0 : Matrix ο ρ ℝ) A scale 1 Q = W := by
  have hd : diagonal (1 : ο → ℝ) = 1 := diagonal_one
  rw [detachedWeight, zero_B_direction, hd, Matrix.one_mul, protected_identity]

/-! ## Gradients (Frobenius adjoints of the affine forward map) -/

/-- Gradient with respect to `B` of `B ↦ ⟪G, detachedWeight⟫_F`. -/
def gradB (G : Matrix ο ι ℝ) (A : Matrix ρ ι ℝ) (scale : ℝ) (g : ο → ℝ)
    (Q : Matrix ι κ ℝ) : Matrix ο ρ ℝ :=
  scale • (diagonal g * G * allowed Q * Aᵀ)

/-- Gradient with respect to `A` of `A ↦ ⟪G, detachedWeight⟫_F`. -/
def gradA (G : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (scale : ℝ) (g : ο → ℝ)
    (Q : Matrix ι κ ℝ) : Matrix ρ ι ℝ :=
  scale • (Bᵀ * diagonal g * G * allowed Q)

/-- The map is affine in `B`, so this holds exactly for every increment `ΔB`. -/
theorem gradB_spec (W G : Matrix ο ι ℝ) (B ΔB : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ)
    (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) :
    frobInner G (detachedWeight W (B + ΔB) A scale g Q - detachedWeight W B A scale g Q) =
      frobInner (gradB G A scale g Q) ΔB := by
  rw [detachedWeight_add_B, gradB, frobInner_smul_right, frobInner_smul_left,
    frobInner_mul_right, frobInner_mul_right, frobInner_mul_left, diagonal_transpose,
    allowed_transpose]
  simp only [Matrix.mul_assoc]

/-- The map is affine in `A`, so this holds exactly for every increment `ΔA`. -/
theorem gradA_spec (W G : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A ΔA : Matrix ρ ι ℝ)
    (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) :
    frobInner G (detachedWeight W B (A + ΔA) scale g Q - detachedWeight W B A scale g Q) =
      frobInner (gradA G B scale g Q) ΔA := by
  rw [detachedWeight_add_A, gradA, frobInner_smul_right, frobInner_smul_left,
    frobInner_mul_right, frobInner_mul_left, transpose_mul, diagonal_transpose,
    allowed_transpose]
  simp only [Matrix.mul_assoc]

/-- With `B = 0` the gradient for `A` vanishes: `A` does not move on the
first plain gradient step, whatever it was initialized to. -/
theorem gradA_at_zero (G : Matrix ο ι ℝ) (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) :
    gradA G (0 : Matrix ο ρ ℝ) scale g Q = 0 := by
  simp [gradA]

/-- With `B = 0` the weight does not depend on `A` for any magnitudes `m`, so
the magnitude path (value and gradient) at initialization is the same for
every choice of `A`. -/
theorem init_weight_independent_of_A (W : Matrix ο ι ℝ) (A A' : Matrix ρ ι ℝ) (scale : ℝ)
    (m : ο → ℝ) (Q : Matrix ι κ ℝ) :
    protectedWeight W (doraWeight W 0 A scale m) Q =
      protectedWeight W (doraWeight W 0 A' scale m) Q := by
  unfold doraWeight
  simp only [zero_B_direction]

/-! ## The first step on `B` -/

/-- Change of the protected weight after one plain gradient step of size `η` on
`B`, from `B = 0` with unit gains (the identity initialization). -/
def firstStep (W G : Matrix ο ι ℝ) (A : Matrix ρ ι ℝ) (scale η : ℝ) (Q : Matrix ι κ ℝ) :
    Matrix ο ι ℝ :=
  detachedWeight W ((-η) • gradB G A scale 1 Q) A scale 1 Q - W

theorem add_firstStep (W G : Matrix ο ι ℝ) (A : Matrix ρ ι ℝ) (scale η : ℝ)
    (Q : Matrix ι κ ℝ) :
    W + firstStep W G A scale η Q = detachedWeight W ((-η) • gradB G A scale 1 Q) A scale 1 Q := by
  rw [firstStep]
  abel

theorem firstStep_eq (W G : Matrix ο ι ℝ) (A : Matrix ρ ι ℝ) (scale η : ℝ)
    (Q : Matrix ι κ ℝ) :
    firstStep W G A scale η Q = -(η * scale ^ 2) • (G * allowed Q * Aᵀ * A * allowed Q) := by
  have hd : diagonal (1 : ο → ℝ) = 1 := diagonal_one
  have h := detachedWeight_add_B W 0 ((-η) • gradB G A scale 1 Q) A scale 1 Q
  rw [zero_add, detachedWeight_zero_one] at h
  rw [firstStep, h, gradB, hd]
  simp only [Matrix.one_mul, Matrix.smul_mul, smul_smul]
  congr 1
  ring

/-- First-order loss change of the step: `⟪G, Δ⟫ = -(η scale²) ‖H Aᵀ‖²`. -/
theorem firstStep_first_order (W G : Matrix ο ι ℝ) (A : Matrix ρ ι ℝ) (scale η : ℝ)
    (Q : Matrix ι κ ℝ) :
    frobInner G (firstStep W G A scale η Q) = -(η * scale ^ 2) * frob2 (G * allowed Q * Aᵀ) := by
  rw [firstStep_eq, frobInner_smul_right, frobInner_mul_right, allowed_transpose,
    frobInner_mul_right, frobInner_self]

/-! ## The SVD initializer -/

/-- A matrix that already annihilates `Q` is unchanged by the input projection. -/
theorem mul_allowed_of_mul_eq_zero {Q : Matrix ι κ ℝ} (T : Matrix ο ι ℝ) (h : T * Q = 0) :
    T * allowed Q = T := by
  rw [allowed, Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc, h, Matrix.zero_mul, sub_zero]


/-- The proposed initializer `A = c V_sᵀ` from the top right singular vectors
of the projected gradient. -/
def svdInit {H : Matrix ο ι ℝ} (sd : SpectralData H) (s : Finset ι) (c : ℝ) : Matrix s ι ℝ :=
  c • (cols sd.V s)ᵀ

/-- The SVD initializer keeps the identity at initialization (`B = 0`). -/
theorem svdInit_identity {H : Matrix ο ι ℝ} (sd : SpectralData H) (s : Finset ι) (c : ℝ)
    (W : Matrix ο ι ℝ) (scale : ℝ) (Q : Matrix ι κ ℝ) (hW : ∀ i, rowNorm W i ≠ 0) :
    protectedWeight W (doraWeight W 0 (svdInit sd s c) scale (rowNorm W)) Q = W :=
  protected_dora_initial_identity W _ scale Q hW

theorem frob2_mul_smul_transpose (M : Matrix ο ι ℝ) (U : Matrix ι ρ ℝ) (c : ℝ) :
    frob2 (M * (c • Uᵀ)ᵀ) = c ^ 2 * frob2 (M * U) := by
  rw [transpose_smul, transpose_transpose, Matrix.mul_smul, frob2_smul]

theorem svdInit_first_order (W G : Matrix ο ι ℝ) (Q : Matrix ι κ ℝ)
    (sd : SpectralData (G * allowed Q)) (s : Finset ι) (scale η c : ℝ) :
    frobInner G (firstStep W G (svdInit sd s c) scale η Q) =
      -(η * scale ^ 2 * c ^ 2) * ∑ i ∈ s, sd.eig i := by
  rw [firstStep_first_order, svdInit, frob2_mul_smul_transpose, sd.frob2_mul_cols]
  ring

/-- Among all initializers `A = c Uᵀ` with orthonormal `U` of the same rank,
the SVD initializer gives the most negative first-order loss change. -/
theorem svdInit_minimizes_first_order (W G : Matrix ο ι ℝ) (Q : Matrix ι κ ℝ)
    (sd : SpectralData (G * allowed Q)) {s : Finset ι} (hs : IsTopSet sd.eig s)
    (scale c : ℝ) {η : ℝ} (hη : 0 ≤ η) (U : Matrix ι ρ ℝ) (hU : Uᵀ * U = 1)
    (hcard : Fintype.card ρ = s.card) :
    frobInner G (firstStep W G (svdInit sd s c) scale η Q) ≤
      frobInner G (firstStep W G (c • Uᵀ) scale η Q) := by
  rw [svdInit_first_order, firstStep_first_order, frob2_mul_smul_transpose]
  have hky := sd.kyFan U hU hs hcard
  have hk : 0 ≤ η * scale ^ 2 * c ^ 2 :=
    mul_nonneg (mul_nonneg hη (sq_nonneg _)) (sq_nonneg _)
  nlinarith [mul_le_mul_of_nonneg_left hky hk]

/-- The SVD initializer has spectral norm at most `|c|` ... -/
theorem svdInit_opNorm {H : Matrix ο ι ℝ} (sd : SpectralData H) (s : Finset ι) (c : ℝ)
    (x : ι → ℝ) : ∑ j, (svdInit sd s c *ᵥ x) j ^ 2 ≤ c ^ 2 * ∑ k, x k ^ 2 := by
  have hb := bessel (cols sd.V s) (cols_orthonormal sd.V sd.orth_left s) x
  have h : ∑ j, (svdInit sd s c *ᵥ x) j ^ 2 = c ^ 2 * ∑ j, ((cols sd.V s)ᵀ *ᵥ x) j ^ 2 := by
    simp only [svdInit, smul_mulVec, Pi.smul_apply, smul_eq_mul, mul_pow, Finset.mul_sum]
  rw [h]
  exact mul_le_mul_of_nonneg_left hb (sq_nonneg c)

/-- ... and rank at most `|s|`. -/
theorem svdInit_rank_le {H : Matrix ο ι ℝ} (sd : SpectralData H) (s : Finset ι) (c : ℝ) :
    (svdInit sd s c).rank ≤ s.card :=
  (rank_le_card_height _).trans (Fintype.card_coe s).le

/-- **Wider comparison.** Among *all* initializers `A` of rank at most `|s|` with
spectral norm at most `|c|` (`‖A x‖ ≤ |c| ‖x‖`), the SVD initializer gives the
most negative first-order loss change. The class contains the SVD initializer
itself (`svdInit_opNorm`, `svdInit_rank_le`) and, for example, any Kaiming draw
rescaled so that its largest singular value is `|c|`. -/
theorem svdInit_minimizes_first_order_opNorm (W G : Matrix ο ι ℝ) (Q : Matrix ι κ ℝ)
    (sd : SpectralData (G * allowed Q)) {s : Finset ι} (hs : IsTopSet sd.eig s)
    (scale c : ℝ) {η : ℝ} (hη : 0 ≤ η) (A : Matrix ρ ι ℝ)
    (hA : ∀ x : ι → ℝ, ∑ j, (A *ᵥ x) j ^ 2 ≤ c ^ 2 * ∑ k, x k ^ 2) (hrank : A.rank ≤ s.card) :
    frobInner G (firstStep W G (svdInit sd s c) scale η Q) ≤
      frobInner G (firstStep W G A scale η Q) := by
  rw [svdInit_first_order, firstStep_first_order]
  have hky := sd.kyFan_opNorm hs A hA hrank
  have hk : 0 ≤ η * scale ^ 2 := mul_nonneg hη (sq_nonneg _)
  nlinarith [mul_le_mul_of_nonneg_left hky hk]

/-- With only a Frobenius budget `‖A‖_F² ≤ β`, the first-order change is at least
`-(η scale²) σ_max² β`; a rank-one `A` on a top singular vector reaches it
(`SpectralData.rank_one_attains`). -/
theorem first_order_frobenius_budget (W G : Matrix ο ι ℝ) (Q : Matrix ι κ ℝ)
    (sd : SpectralData (G * allowed Q)) {M : ℝ} (hM : ∀ i, sd.eig i ≤ M) (hM0 : 0 ≤ M)
    (scale : ℝ) {η : ℝ} (hη : 0 ≤ η) (A : Matrix ρ ι ℝ) {β : ℝ} (hβ : frob2 A ≤ β) :
    -(η * scale ^ 2) * (M * β) ≤ frobInner G (firstStep W G A scale η Q) := by
  rw [firstStep_first_order]
  have h1 := sd.frob2_mul_transpose_le_max hM A
  have h2 : M * frob2 A ≤ M * β := mul_le_mul_of_nonneg_left hβ hM0
  have hk : 0 ≤ η * scale ^ 2 := mul_nonneg hη (sq_nonneg _)
  nlinarith [mul_le_mul_of_nonneg_left (h1.trans h2) hk]

/-- For the SVD initializer the step is exactly `-(η scale² c²)` times the
Eckart–Young truncation of `H = G (I - Q Qᵀ)`. -/
theorem svdInit_firstStep_eq_truncate (W G : Matrix ο ι ℝ) {Q : Matrix ι κ ℝ}
    (hQ : Qᵀ * Q = 1) (sd : SpectralData (G * allowed Q)) (s : Finset ι) (scale η c : ℝ) :
    firstStep W G (svdInit sd s c) scale η Q = -(η * scale ^ 2 * c ^ 2) • sd.truncate s := by
  have hzero : sd.truncate s * Q = 0 :=
    sd.truncate_mul_eq_zero (by rw [Matrix.mul_assoc, allowed_mul_basis Q hQ, Matrix.mul_zero]) s
  have hP : sd.truncate s * allowed Q = sd.truncate s := mul_allowed_of_mul_eq_zero _ hzero
  have hM : G * allowed Q * cols sd.V s * (cols sd.V s)ᵀ * allowed Q =
      sd.truncate s * allowed Q := by
    simp only [SpectralData.truncate, Matrix.mul_assoc]
  rw [firstStep_eq, svdInit, transpose_smul, transpose_transpose]
  simp only [Matrix.mul_smul, Matrix.smul_mul, smul_smul]
  rw [hM, hP]
  congr 1
  ring

/-- The step is a best rank-`|s|` approximation of the dense projected step
`-(η scale² c²) H`. -/
theorem svdInit_firstStep_best (W G : Matrix ο ι ℝ) {Q : Matrix ι κ ℝ} (hQ : Qᵀ * Q = 1)
    (sd : SpectralData (G * allowed Q)) {s : Finset ι} (hs : IsTopSet sd.eig s)
    (scale η c : ℝ) (X : Matrix ο ι ℝ) (hX : X.rank ≤ s.card) :
    frob2 ((-(η * scale ^ 2 * c ^ 2)) • (G * allowed Q) - firstStep W G (svdInit sd s c) scale η Q)
      ≤ frob2 ((-(η * scale ^ 2 * c ^ 2)) • (G * allowed Q) - X) := by
  rw [svdInit_firstStep_eq_truncate W G hQ, ← sd.truncate_smul]
  exact (sd.smul _).eckart_young (sd.isTopSet_smul hs _) X hX

theorem svdInit_firstStep_energy (W G : Matrix ο ι ℝ) {Q : Matrix ι κ ℝ} (hQ : Qᵀ * Q = 1)
    (sd : SpectralData (G * allowed Q)) (s : Finset ι) (scale η c : ℝ) :
    frob2 (firstStep W G (svdInit sd s c) scale η Q) =
      (η * scale ^ 2 * c ^ 2) ^ 2 * ∑ i ∈ s, sd.eig i := by
  rw [svdInit_firstStep_eq_truncate W G hQ, frob2_smul, sd.frob2_truncate]
  ring

/-- Conditional descent. The smoothness inequality along this one step is an
explicit premise about the loss, not proved here. With `η' = η scale² c²` and
`E = ∑ i ∈ s, eig i` (the captured squared singular values), the loss after
the step is at most `loss W - η' (1 - L η' / 2) E`. -/
theorem svdInit_descent (loss : Matrix ο ι ℝ → ℝ) (W G : Matrix ο ι ℝ) {Q : Matrix ι κ ℝ}
    (hQ : Qᵀ * Q = 1) (sd : SpectralData (G * allowed Q)) (s : Finset ι) (scale η c L : ℝ)
    (hsmooth : loss (W + firstStep W G (svdInit sd s c) scale η Q) ≤
      loss W + frobInner G (firstStep W G (svdInit sd s c) scale η Q) +
        (L / 2) * frob2 (firstStep W G (svdInit sd s c) scale η Q)) :
    loss (W + firstStep W G (svdInit sd s c) scale η Q) ≤
      loss W - (η * scale ^ 2 * c ^ 2) * (1 - L * (η * scale ^ 2 * c ^ 2) / 2) *
        ∑ i ∈ s, sd.eig i := by
  rw [svdInit_first_order, svdInit_firstStep_energy W G hQ] at hsmooth
  exact descent_bound _ _ L _ _ (by linarith)

theorem svdInit_strict_descent (loss : Matrix ο ι ℝ → ℝ) (W G : Matrix ο ι ℝ)
    {Q : Matrix ι κ ℝ} (hQ : Qᵀ * Q = 1) (sd : SpectralData (G * allowed Q)) (s : Finset ι)
    (scale η c L : ℝ) (hstep : 0 < η * scale ^ 2 * c ^ 2)
    (hL : L * (η * scale ^ 2 * c ^ 2) < 2) (hE : 0 < ∑ i ∈ s, sd.eig i)
    (hsmooth : loss (W + firstStep W G (svdInit sd s c) scale η Q) ≤
      loss W + frobInner G (firstStep W G (svdInit sd s c) scale η Q) +
        (L / 2) * frob2 (firstStep W G (svdInit sd s c) scale η Q)) :
    loss (W + firstStep W G (svdInit sd s c) scale η Q) < loss W := by
  rw [svdInit_first_order, svdInit_firstStep_energy W G hQ] at hsmooth
  exact strict_descent _ _ L _ _ hstep hL hE (by linarith)

end ProtectedDoRA

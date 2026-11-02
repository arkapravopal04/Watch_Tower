import Mathlib.LinearAlgebra.Matrix.Rank
import ProtectedDoRA.FirstStep

/-!
# Claims made in `docs/dora_protocol.md`

* "Empty Q is ordinary DoRA": `protectedWeight_of_isEmpty`.
* "Projecting A alone would leave `(diag(g) - I) W Q` unconstrained":
  `aOnly_update_on_basis` computes exactly that term.
* "`rank(delta_W) <= min(out, in - k)`": `protected_update_rank_le`.
* "Exports contain `A_eff`, `B`, scale, `Q` and row gains": the update that
  `pcl/merge.py` (`_merged_weight`) rebuilds from them,
  `(gain - 1) (W - W Q Qᵀ) + gain (B A_eff) scale` with `A_eff = A - (A Q) Qᵀ`,
  is exactly the protected update (`merge_intended_eq`, `merge_matches_forward`),
  and `A_eff` annihilates `Q` (`effective_a_mul_basis`).
-/

namespace ProtectedDoRA

open Matrix

variable {ο ι κ ρ : Type*} [Fintype ο] [Fintype ι] [Fintype κ] [Fintype ρ]
  [DecidableEq ο] [DecidableEq ι] [DecidableEq κ] [DecidableEq ρ]

theorem allowed_of_isEmpty [IsEmpty κ] (Q : Matrix ι κ ℝ) : allowed Q = 1 := by
  ext i j
  simp [allowed, mul_apply]

theorem protectedWeight_of_isEmpty [IsEmpty κ] (W D : Matrix ο ι ℝ) (Q : Matrix ι κ ℝ) :
    protectedWeight W D Q = D := by
  rw [protectedWeight, allowed_of_isEmpty, Matrix.mul_one]
  abel

/-- Projecting only `A` leaves the magnitude part acting on `Q`. -/
theorem aOnly_update_on_basis (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ)
    (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) (hQ : Qᵀ * Q = 1) :
    (diagonal g * direction W B (A * allowed Q) scale - W) * Q = (diagonal g - 1) * W * Q := by
  simp only [direction, Matrix.mul_add, Matrix.mul_smul, Matrix.sub_mul, Matrix.add_mul,
    Matrix.smul_mul, Matrix.mul_assoc, allowed_mul_basis Q hQ, Matrix.mul_zero, smul_zero,
    add_zero, Matrix.one_mul]

/-- `rank (I - Q Qᵀ) + k ≤ in`: the span of `Q` lies in the projector's kernel. -/
theorem rank_allowed_add_le (Q : Matrix ι κ ℝ) (hQ : Qᵀ * Q = 1) :
    (allowed Q).rank + Fintype.card κ ≤ Fintype.card ι := by
  have hker : LinearMap.range Q.mulVecLin ≤ LinearMap.ker (allowed Q).mulVecLin := by
    rintro _ ⟨x, rfl⟩
    rw [LinearMap.mem_ker, mulVecLin_apply, mulVecLin_apply, mulVec_mulVec,
      allowed_mul_basis Q hQ, zero_mulVec]
  have hQrank : Q.rank = Fintype.card κ := by
    rw [← rank_transpose_mul_self Q, hQ, rank_one]
  have hrn := LinearMap.finrank_range_add_finrank_ker (allowed Q).mulVecLin
  rw [Module.finrank_fintype_fun_eq_card] at hrn
  have hmono := Submodule.finrank_mono hker
  have h1 : Q.rank = Module.finrank ℝ (LinearMap.range Q.mulVecLin) := rfl
  have h2 : (allowed Q).rank = Module.finrank ℝ (LinearMap.range (allowed Q).mulVecLin) := rfl
  omega

/-- The complete update has rank at most `min(out, in - k)`. -/
theorem protected_update_rank_le (W D : Matrix ο ι ℝ) (Q : Matrix ι κ ℝ) (hQ : Qᵀ * Q = 1) :
    (protectedWeight W D Q - W).rank ≤ Fintype.card ι - Fintype.card κ ∧
      (protectedWeight W D Q - W).rank ≤ Fintype.card ο := by
  have h : protectedWeight W D Q - W = (D - W) * allowed Q := by
    rw [protectedWeight]
    abel
  refine ⟨?_, rank_le_card_height _⟩
  have h1 := rank_allowed_add_le Q hQ
  have h2 := rank_mul_le_right (D - W) (allowed Q)
  rw [h]
  omega

/-- `A_eff = A - (A Q) Qᵀ`, as exported by `effective_a`, annihilates `Q`. -/
theorem effective_a_mul_basis (A : Matrix ρ ι ℝ) (Q : Matrix ι κ ℝ) (hQ : Qᵀ * Q = 1) :
    (A - A * Q * Qᵀ) * Q = 0 := by
  rw [Matrix.sub_mul, Matrix.mul_assoc (A * Q), hQ, Matrix.mul_one, sub_self]

/-- The merge's `intended` update equals the protected update (gains fixed). -/
theorem merge_intended_eq (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ)
    (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) :
    diagonal (fun i => g i - 1) * (W - W * Q * Qᵀ) +
        scale • (diagonal g * (B * (A - A * Q * Qᵀ))) =
      detachedWeight W B A scale g Q - W := by
  have hW : W - W * Q * Qᵀ = W * allowed Q := by
    rw [allowed, Matrix.mul_sub, Matrix.mul_one, Matrix.mul_assoc]
  have hA : A - A * Q * Qᵀ = A * allowed Q := by
    rw [allowed, Matrix.mul_sub, Matrix.mul_one, Matrix.mul_assoc]
  have hd : diagonal (fun i => g i - 1) = diagonal g - 1 := by
    ext i j
    by_cases h : i = j
    · subst h
      simp
    · simp [diagonal_apply_ne _ h, one_apply_ne h]
  rw [hW, hA, hd, detachedWeight_expand]
  simp only [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc]
  abel

/-- With the gains DoRA actually uses, the merged weight is the forward weight. -/
theorem merge_matches_forward (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ)
    (scale : ℝ) (m : ο → ℝ) (Q : Matrix ι κ ℝ) :
    W + (diagonal (fun i => doraGain W B A scale m i - 1) * (W - W * Q * Qᵀ) +
        scale • (diagonal (doraGain W B A scale m) * (B * (A - A * Q * Qᵀ)))) =
      protectedWeight W (doraWeight W B A scale m) Q := by
  rw [merge_intended_eq, protectedWeight_eq_detachedWeight]
  abel

end ProtectedDoRA

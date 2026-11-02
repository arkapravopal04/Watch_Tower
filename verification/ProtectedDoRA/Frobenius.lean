import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Frobenius energy and inner product

Both are plain sums over entries, so no norm instance on matrices is
involved. `frob2 M` is the squared Frobenius norm and `frobInner M N` the
matching inner product. The projector lemmas at the end are the orthogonal
splitting used by the Eckart–Young argument.
-/

namespace ProtectedDoRA

open Matrix

variable {m n p ρ : Type*} [Fintype m] [Fintype n] [Fintype p] [Fintype ρ]

/-- Squared Frobenius norm, `‖M‖_F²`. -/
def frob2 (M : Matrix m n ℝ) : ℝ := ∑ i, ∑ j, M i j ^ 2

/-- Frobenius inner product, `⟪M, N⟫_F`. -/
def frobInner (M N : Matrix m n ℝ) : ℝ := ∑ i, ∑ j, M i j * N i j

theorem frob2_nonneg (M : Matrix m n ℝ) : 0 ≤ frob2 M :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _

theorem frobInner_self (M : Matrix m n ℝ) : frobInner M M = frob2 M := by
  simp only [frobInner, frob2, sq]

theorem frobInner_eq_trace (M N : Matrix m n ℝ) : frobInner M N = trace (Mᵀ * N) := by
  simp only [frobInner, trace, diag_apply, mul_apply, transpose_apply]
  exact Finset.sum_comm

theorem frob2_eq_trace (M : Matrix m n ℝ) : frob2 M = trace (Mᵀ * M) := by
  rw [← frobInner_self, frobInner_eq_trace]

theorem frob2_transpose (M : Matrix m n ℝ) : frob2 Mᵀ = frob2 M := by
  simp only [frob2, transpose_apply]
  exact Finset.sum_comm

theorem frob2_add (M N : Matrix m n ℝ) :
    frob2 (M + N) = frob2 M + frob2 N + 2 * frobInner M N := by
  simp only [frob2, frobInner, add_apply, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  ring

theorem frob2_smul (c : ℝ) (M : Matrix m n ℝ) : frob2 (c • M) = c ^ 2 * frob2 M := by
  simp only [frob2, smul_apply, smul_eq_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  ring

theorem frobInner_smul_left (c : ℝ) (M N : Matrix m n ℝ) :
    frobInner (c • M) N = c * frobInner M N := by
  simp only [frobInner, smul_apply, smul_eq_mul, Finset.mul_sum, mul_assoc]

theorem frobInner_smul_right (c : ℝ) (M N : Matrix m n ℝ) :
    frobInner M (c • N) = c * frobInner M N := by
  simp only [frobInner, smul_apply, smul_eq_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  ring

/-- Move a right factor across the inner product: `⟪M, N R⟫ = ⟪M Rᵀ, N⟫`. -/
theorem frobInner_mul_right (M : Matrix m p ℝ) (N : Matrix m n ℝ) (R : Matrix n p ℝ) :
    frobInner M (N * R) = frobInner (M * Rᵀ) N := by
  rw [frobInner_eq_trace, frobInner_eq_trace, transpose_mul, transpose_transpose,
    ← Matrix.mul_assoc, trace_mul_comm (Mᵀ * N) R, Matrix.mul_assoc]

/-- Move a left factor across the inner product: `⟪M, L N⟫ = ⟪Lᵀ M, N⟫`. -/
theorem frobInner_mul_left (M : Matrix p n ℝ) (L : Matrix p m ℝ) (N : Matrix m n ℝ) :
    frobInner M (L * N) = frobInner (Lᵀ * M) N := by
  rw [frobInner_eq_trace, frobInner_eq_trace, transpose_mul, transpose_transpose,
    Matrix.mul_assoc]

/-- Right multiplication by a matrix with orthonormal rows keeps the energy. -/
theorem frob2_mul_orthonormal_rows (M : Matrix m n ℝ) (R : Matrix n p ℝ)
    [DecidableEq n] (hR : R * Rᵀ = 1) : frob2 (M * R) = frob2 M := by
  calc frob2 (M * R) = trace (Rᵀ * (Mᵀ * M * R)) := by
        rw [frob2_eq_trace, transpose_mul]
        simp only [Matrix.mul_assoc]
    _ = trace (Mᵀ * M * R * Rᵀ) := trace_mul_comm _ _
    _ = frob2 M := by rw [Matrix.mul_assoc (Mᵀ * M), hR, Matrix.mul_one, frob2_eq_trace]

section Projector

variable [DecidableEq n]

/-- Splitting along a symmetric idempotent `P` is Frobenius-orthogonal. -/
theorem frob2_split (M : Matrix m n ℝ) (P : Matrix n n ℝ) (hsym : Pᵀ = P)
    (hidem : P * P = P) : frob2 M = frob2 (M * (1 - P)) + frob2 (M * P) := by
  have hcross : frobInner (M * (1 - P)) (M * P) = 0 := by
    rw [frobInner_mul_right, hsym, Matrix.mul_assoc, Matrix.sub_mul, Matrix.one_mul, hidem,
      sub_self, Matrix.mul_zero]
    simp [frobInner]
  calc frob2 M = frob2 (M * (1 - P) + M * P) := by
        rw [← Matrix.mul_add, sub_add_cancel, Matrix.mul_one]
    _ = frob2 (M * (1 - P)) + frob2 (M * P) := by rw [frob2_add, hcross]; ring

theorem proj_transpose (U : Matrix n ρ ℝ) : (U * Uᵀ)ᵀ = U * Uᵀ := by
  rw [transpose_mul, transpose_transpose]

theorem proj_idem [DecidableEq ρ] (U : Matrix n ρ ℝ) (hU : Uᵀ * U = 1) :
    U * Uᵀ * (U * Uᵀ) = U * Uᵀ := by
  rw [Matrix.mul_assoc, ← Matrix.mul_assoc Uᵀ, hU, Matrix.one_mul]

/-- Energy left after projecting the rows of `M` onto the columns of `U`. -/
theorem frob2_residual [DecidableEq ρ] (M : Matrix m n ℝ) (U : Matrix n ρ ℝ)
    (hU : Uᵀ * U = 1) : frob2 (M - M * (U * Uᵀ)) = frob2 M - frob2 (M * U) := by
  have hsplit := frob2_split M (U * Uᵀ) (proj_transpose U) (proj_idem U hU)
  have hP : frob2 (M * (U * Uᵀ)) = frob2 (M * U) := by
    rw [← Matrix.mul_assoc]
    exact frob2_mul_orthonormal_rows (M * U) Uᵀ (by rw [transpose_transpose, hU])
  rw [Matrix.mul_sub, Matrix.mul_one] at hsplit
  linarith

/-- If every row of `X` lies in the column span of `U`, then `X` is no closer
to `M` than the projection of `M` onto that span. -/
theorem frob2_residual_le [DecidableEq ρ] (M X : Matrix m n ℝ) (U : Matrix n ρ ℝ)
    (hU : Uᵀ * U = 1) (hX : X * (U * Uᵀ) = X) :
    frob2 (M - M * (U * Uᵀ)) ≤ frob2 (M - X) := by
  have hsplit := frob2_split (M - X) (U * Uᵀ) (proj_transpose U) (proj_idem U hU)
  have hres : (M - X) * (1 - U * Uᵀ) = M - M * (U * Uᵀ) := by
    rw [Matrix.mul_sub, Matrix.mul_one, Matrix.sub_mul, hX]
    abel
  rw [hres] at hsplit
  linarith [frob2_nonneg ((M - X) * (U * Uᵀ))]

end Projector

end ProtectedDoRA

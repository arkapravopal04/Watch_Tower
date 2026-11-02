import Mathlib.Analysis.Matrix.Normed
import Mathlib.Analysis.Calculus.FDeriv.Comp
import Mathlib.Analysis.Calculus.FDeriv.Add
import Mathlib.Analysis.Calculus.FDeriv.Linear
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Topology.Algebra.Module.FiniteDimension
import Mathlib.LinearAlgebra.Matrix.FiniteDimensional
import ProtectedDoRA.FirstStep

/-!
# `gradB` and `gradA` are derivatives, by the chain rule

Matrices carry Mathlib's Frobenius norm in this file, and `frob2` is shown to
be its square. Assume the loss has a Fréchet derivative `L'` at the effective
weight and that `L'` is `Δ ↦ ⟪G, Δ⟫_F`, i.e. `G` is its Frobenius gradient.
Then, with the row gains held fixed (detached row norms), the loss has a
derivative in `B` whose Frobenius gradient is `gradB`, and likewise for `A`.
So `firstStep` is one gradient-descent step on `B` for the loss itself.
-/

namespace ProtectedDoRA

open Matrix

attribute [local instance] Matrix.frobeniusNormedAddCommGroup Matrix.frobeniusNormedSpace

variable {ο ι κ ρ : Type*} [Fintype ο] [Fintype ι] [Fintype κ] [Fintype ρ]
  [DecidableEq ο] [DecidableEq ι] [DecidableEq κ] [DecidableEq ρ]

/-- `frob2` is the square of Mathlib's Frobenius norm. -/
theorem frob2_eq_norm_sq (M : Matrix ο ι ℝ) : frob2 M = ‖M‖ ^ 2 := by
  have hS : (0 : ℝ) ≤ ∑ i, ∑ j, ‖M i j‖ ^ (2 : ℝ) :=
    Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => Real.rpow_nonneg (norm_nonneg _) _
  rw [frobenius_norm_def, ← Real.sqrt_eq_rpow, Real.sq_sqrt hS]
  simp only [frob2, Real.rpow_two, Real.norm_eq_abs, sq_abs]

/-- The `B`-path increment `ΔB ↦ scale • (diag g ΔB A (I - Q Qᵀ))`. -/
noncomputable def bPath (A : Matrix ρ ι ℝ) (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) :
    Matrix ο ρ ℝ →L[ℝ] Matrix ο ι ℝ :=
  LinearMap.toContinuousLinearMap
    { toFun := fun ΔB => scale • (diagonal g * ΔB * A * allowed Q)
      map_add' := fun X Y => by simp only [Matrix.mul_add, Matrix.add_mul, smul_add]
      map_smul' := fun c X => by
        simp only [Matrix.mul_smul, Matrix.smul_mul, RingHom.id_apply]
        exact smul_comm scale c _ }

/-- The `A`-path increment `ΔA ↦ scale • (diag g B ΔA (I - Q Qᵀ))`. -/
noncomputable def aPath (B : Matrix ο ρ ℝ) (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) :
    Matrix ρ ι ℝ →L[ℝ] Matrix ο ι ℝ :=
  LinearMap.toContinuousLinearMap
    { toFun := fun ΔA => scale • (diagonal g * B * ΔA * allowed Q)
      map_add' := fun X Y => by simp only [Matrix.mul_add, Matrix.add_mul, smul_add]
      map_smul' := fun c X => by
        simp only [Matrix.mul_smul, Matrix.smul_mul, RingHom.id_apply]
        exact smul_comm scale c _ }

theorem hasFDerivAt_detached_B (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ)
    (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) :
    HasFDerivAt (fun B' => detachedWeight W B' A scale g Q) (bPath A scale g Q) B := by
  have haff : (fun B' => detachedWeight W B' A scale g Q) =
      fun B' => (detachedWeight W B A scale g Q - bPath A scale g Q B) + bPath A scale g Q B' := by
    funext B'
    have h := detachedWeight_add_B W B (B' - B) A scale g Q
    rw [show B + (B' - B) = B' by abel] at h
    have hT : bPath A scale g Q B' - bPath A scale g Q B =
        scale • (diagonal g * (B' - B) * A * allowed Q) := by
      rw [← map_sub]
      rfl
    calc detachedWeight W B' A scale g Q
        = detachedWeight W B A scale g Q +
            (detachedWeight W B' A scale g Q - detachedWeight W B A scale g Q) := by abel
      _ = detachedWeight W B A scale g Q + (bPath A scale g Q B' - bPath A scale g Q B) := by
            rw [h, hT]
      _ = _ := by abel
  rw [haff]
  exact (bPath A scale g Q).hasFDerivAt.const_add _

theorem hasFDerivAt_detached_A (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ)
    (scale : ℝ) (g : ο → ℝ) (Q : Matrix ι κ ℝ) :
    HasFDerivAt (fun A' => detachedWeight W B A' scale g Q) (aPath B scale g Q) A := by
  have haff : (fun A' => detachedWeight W B A' scale g Q) =
      fun A' => (detachedWeight W B A scale g Q - aPath B scale g Q A) + aPath B scale g Q A' := by
    funext A'
    have h := detachedWeight_add_A W B A (A' - A) scale g Q
    rw [show A + (A' - A) = A' by abel] at h
    have hT : aPath B scale g Q A' - aPath B scale g Q A =
        scale • (diagonal g * B * (A' - A) * allowed Q) := by
      rw [← map_sub]
      rfl
    calc detachedWeight W B A' scale g Q
        = detachedWeight W B A scale g Q +
            (detachedWeight W B A' scale g Q - detachedWeight W B A scale g Q) := by abel
      _ = detachedWeight W B A scale g Q + (aPath B scale g Q A' - aPath B scale g Q A) := by
            rw [h, hT]
      _ = _ := by abel
  rw [haff]
  exact (aPath B scale g Q).hasFDerivAt.const_add _

/-- **Chain rule for `B`.** If `G` is the Frobenius gradient of the loss at the
effective weight, then `gradB` is the Frobenius gradient of the loss in `B`. -/
theorem loss_hasFDerivAt_B (loss : Matrix ο ι ℝ → ℝ) (L' : Matrix ο ι ℝ →L[ℝ] ℝ)
    (W G : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ) (scale : ℝ) (g : ο → ℝ)
    (Q : Matrix ι κ ℝ) (hloss : HasFDerivAt loss L' (detachedWeight W B A scale g Q))
    (hG : ∀ Δ, L' Δ = frobInner G Δ) :
    HasFDerivAt (fun B' => loss (detachedWeight W B' A scale g Q))
        (L'.comp (bPath A scale g Q)) B ∧
      ∀ ΔB, L'.comp (bPath A scale g Q) ΔB = frobInner (gradB G A scale g Q) ΔB := by
  refine ⟨hloss.comp B (hasFDerivAt_detached_B W B A scale g Q), fun ΔB => ?_⟩
  rw [ContinuousLinearMap.comp_apply, hG, ← gradB_spec W G B ΔB A scale g Q,
    detachedWeight_add_B]
  rfl

/-- **Chain rule for `A`.** Same statement with `gradA`. -/
theorem loss_hasFDerivAt_A (loss : Matrix ο ι ℝ → ℝ) (L' : Matrix ο ι ℝ →L[ℝ] ℝ)
    (W G : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ) (scale : ℝ) (g : ο → ℝ)
    (Q : Matrix ι κ ℝ) (hloss : HasFDerivAt loss L' (detachedWeight W B A scale g Q))
    (hG : ∀ Δ, L' Δ = frobInner G Δ) :
    HasFDerivAt (fun A' => loss (detachedWeight W B A' scale g Q))
        (L'.comp (aPath B scale g Q)) A ∧
      ∀ ΔA, L'.comp (aPath B scale g Q) ΔA = frobInner (gradA G B scale g Q) ΔA := by
  refine ⟨hloss.comp A (hasFDerivAt_detached_A W B A scale g Q), fun ΔA => ?_⟩
  rw [ContinuousLinearMap.comp_apply, hG, ← gradA_spec W G B A ΔA scale g Q,
    detachedWeight_add_A]
  rfl

end ProtectedDoRA

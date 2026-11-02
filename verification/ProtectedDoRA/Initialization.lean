import ProtectedDoRA.Projection
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-! # DoRA's forward equation and identity initialization

The row norm below is the Euclidean norm used in the Python implementation.
The nonzero-row hypothesis matches the implementation's direction_norm check.
This file concerns the forward equation; it does not model stop-gradient.
-/

namespace ProtectedDoRA

open Matrix

variable {ι κ ο ρ : Type*} [Fintype ι] [Fintype κ] [Fintype ο] [Fintype ρ]
  [DecidableEq ι] [DecidableEq κ]

noncomputable def rowNorm (V : Matrix ο ι ℝ) (i : ο) : ℝ :=
  Real.sqrt (∑ j, V i j ^ 2)

def direction (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ)
    (A : Matrix ρ ι ℝ) (scale : ℝ) : Matrix ο ι ℝ := W + scale • (B * A)

noncomputable def doraWeight (W : Matrix ο ι ℝ) (B : Matrix ο ρ ℝ)
    (A : Matrix ρ ι ℝ) (scale : ℝ) (m : ο → ℝ) : Matrix ο ι ℝ :=
  fun i j => (m i / rowNorm (direction W B A scale) i) * direction W B A scale i j

theorem zero_B_direction (W : Matrix ο ι ℝ) (A : Matrix ρ ι ℝ) (scale : ℝ) :
    direction W 0 A scale = W := by
  simp [direction]

/-- A may be arbitrary: in particular it can be initialized from singular
vectors. B=0 keeps the complete forward weight unchanged. -/
theorem dora_initial_identity (W : Matrix ο ι ℝ) (A : Matrix ρ ι ℝ)
    (scale : ℝ) (hW : ∀ i, rowNorm W i ≠ 0) :
    doraWeight W 0 A scale (rowNorm W) = W := by
  ext i j
  simp [doraWeight, zero_B_direction, hW i]

theorem protected_dora_initial_identity (W : Matrix ο ι ℝ) (A : Matrix ρ ι ℝ)
    (scale : ℝ) (Q : Matrix ι κ ℝ) (hW : ∀ i, rowNorm W i ≠ 0) :
    protectedWeight W (doraWeight W 0 A scale (rowNorm W)) Q = W := by
  rw [dora_initial_identity W A scale hW, protected_identity]

end ProtectedDoRA

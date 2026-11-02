import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Matrix.Basic
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ext

/-!
# Complete-update input protection

Exact real matrices, with PyTorch's (output, input) convention.
The protection hypothesis is Qᵀ Q = I. No statement about model accuracy,
FP16 arithmetic, or end-to-end transformer behavior is assumed or concluded.
-/

namespace ProtectedDoRA

open Matrix

variable {ι κ ο : Type*} [Fintype ι] [Fintype κ] [Fintype ο]
  [DecidableEq ι] [DecidableEq κ]

/-- Project inputs onto the complement of the protected columns of Q. -/
def allowed (Q : Matrix ι κ ℝ) : Matrix ι ι ℝ := 1 - Q * Q.transpose

theorem allowed_mul_basis (Q : Matrix ι κ ℝ)
    (hQ : Q.transpose * Q = 1) : allowed Q * Q = 0 := by
  simp [allowed, Matrix.sub_mul, Matrix.mul_assoc, hQ]

theorem allowed_transpose (Q : Matrix ι κ ℝ) :
    (allowed Q).transpose = allowed Q := by
  simp [allowed, Matrix.transpose_mul]

theorem allowed_idempotent (Q : Matrix ι κ ℝ)
    (hQ : Q.transpose * Q = 1) : allowed Q * allowed Q = allowed Q := by
  have h : allowed Q * (Q * Q.transpose) = 0 := by
    rw [← Matrix.mul_assoc, allowed_mul_basis Q hQ, Matrix.zero_mul]
  calc
    allowed Q * allowed Q = allowed Q * (1 - Q * Q.transpose) := rfl
    _ = allowed Q := by rw [mul_sub, Matrix.mul_one, h, sub_zero]

/-- Project the entire DoRA update, including its magnitude contribution. -/
def protectedWeight (W D : Matrix ο ι ℝ) (Q : Matrix ι κ ℝ) : Matrix ο ι ℝ :=
  W + (D - W) * allowed Q

theorem complete_update_annihilates_basis (W D : Matrix ο ι ℝ)
    (Q : Matrix ι κ ℝ) (hQ : Q.transpose * Q = 1) :
    (protectedWeight W D Q - W) * Q = 0 := by
  simp [protectedWeight, Matrix.mul_assoc, allowed_mul_basis Q hQ]

theorem protected_weight_on_basis (W D : Matrix ο ι ℝ)
    (Q : Matrix ι κ ℝ) (hQ : Q.transpose * Q = 1) :
    protectedWeight W D Q * Q = W * Q := by
  simp [protectedWeight, Matrix.add_mul, Matrix.mul_assoc, allowed_mul_basis Q hQ]

theorem protected_input_preserved (W D : Matrix ο ι ℝ)
    (Q : Matrix ι κ ℝ) (hQ : Q.transpose * Q = 1) (c : κ → ℝ) :
    (protectedWeight W D Q).mulVec (Q.mulVec c) = W.mulVec (Q.mulVec c) := by
  rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec,
    protected_weight_on_basis W D Q hQ]

theorem protected_identity (W : Matrix ο ι ℝ) (Q : Matrix ι κ ℝ) :
    protectedWeight W W Q = W := by
  simp [protectedWeight]

/-- Any stored leakage is exactly the storage error acting on Q, if the
intended real-valued update annihilates Q. This does not bound that error. -/
theorem storage_leakage_identity (intended stored : Matrix ο ι ℝ)
    (Q : Matrix ι κ ℝ) (h : intended * Q = 0) :
    stored * Q = (stored - intended) * Q := by
  rw [Matrix.sub_mul, h, sub_zero]

end ProtectedDoRA

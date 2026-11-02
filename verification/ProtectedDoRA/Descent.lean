import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith

/-!
# Conditional descent arithmetic

The smoothness inequality is an explicit premise, not a theorem about the
transformer. The first-order term must equal -eta*energy and the squared
step norm must equal eta^2*energy before this scalar lemma applies.
-/

namespace ProtectedDoRA

theorem descent_bound (before after L eta energy : ℝ)
    (hsmooth : after ≤ before - eta * energy + (L / 2) * (eta ^ 2 * energy)) :
    after ≤ before - eta * (1 - L * eta / 2) * energy := by
  nlinarith [hsmooth]

theorem strict_descent (before after L eta energy : ℝ)
    (heta : 0 < eta) (hstep : L * eta < 2) (henergy : 0 < energy)
    (hsmooth : after ≤ before - eta * energy + (L / 2) * (eta ^ 2 * energy)) :
    after < before := by
  have hfactor : 0 < 1 - L * eta / 2 := by linarith
  have hdecrease := mul_pos (mul_pos heta hfactor) henergy
  have hbound := descent_bound before after L eta energy hsmooth
  linarith

end ProtectedDoRA

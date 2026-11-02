import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith

/-!
# Spectral allocation lemma

This proves the finite-dimensional scalar inequality behind top-r subspace
selection. It does NOT assume optimality as a hypothesis. The matrix bridge
(squared overlaps of orthonormal vectors satisfy these weight conditions,
and captured Frobenius energy equals the weighted spectral sum) is separate.
-/

namespace ProtectedDoRA

open Finset

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Put one unit of weight on each selected coordinate, and zero elsewhere. -/
def selectedWeight (s : Finset ι) (i : ι) : ℝ := if i ∈ s then 1 else 0

theorem selectedWeight_sum (s : Finset ι) :
    ∑ i, selectedWeight s i = (s.card : ℝ) := by
  simp [selectedWeight]

/-- Among weights in [0,1] with total mass |s|, selecting the largest
coordinates maximizes weighted energy. A cutoff describes the ordering;
ties are allowed. -/
theorem top_spectral_allocation (s : Finset ι) (e w : ι → ℝ) (cutoff : ℝ)
    (hlo : ∀ i, 0 ≤ w i) (hhi : ∀ i, w i ≤ 1)
    (hmass : ∑ i, w i = (s.card : ℝ))
    (hselected : ∀ i ∈ s, cutoff ≤ e i)
    (hrest : ∀ i ∉ s, e i ≤ cutoff) :
    ∑ i, e i * w i ≤ ∑ i ∈ s, e i := by
  have hpoint : ∀ i, e i * w i ≤
      e i * selectedWeight s i + cutoff * (w i - selectedWeight s i) := by
    intro i
    by_cases hi : i ∈ s
    · have he := hselected i hi
      have hw := hhi i
      simp only [selectedWeight, ite_eq_left hi]
      nlinarith
    · have he := hrest i hi
      have hw := hlo i
      simp only [selectedWeight, ite_eq_right hi]
      nlinarith
  have hmass0 : ∑ i, (w i - selectedWeight s i) = 0 := by
    rw [Finset.sum_sub_distrib, hmass, selectedWeight_sum, sub_self]
  calc
    ∑ i, e i * w i ≤ ∑ i,
        (e i * selectedWeight s i + cutoff * (w i - selectedWeight s i)) :=
      Finset.sum_le_sum (fun i _ => hpoint i)
    _ = ∑ i, e i * selectedWeight s i := by
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, hmass0, mul_zero, add_zero]
    _ = ∑ i ∈ s, e i := by simp [selectedWeight]

/-- `s` holds the largest values of `e`: nothing outside beats anything inside.
Ties may fall on either side. -/
def IsTopSet (e : ι → ℝ) (s : Finset ι) : Prop := ∀ i ∈ s, ∀ j ∉ s, e j ≤ e i

theorem IsTopSet.exists_cutoff {e : ι → ℝ} {s : Finset ι} (h : IsTopSet e s) :
    ∃ cutoff, (∀ i ∈ s, cutoff ≤ e i) ∧ ∀ i ∉ s, e i ≤ cutoff := by
  by_cases hs : s.Nonempty
  · exact ⟨s.inf' hs e, fun i hi => Finset.inf'_le e hi,
      fun j hj => Finset.le_inf' hs e fun i hi => h i hi j hj⟩
  · refine ⟨∑ i, |e i|, fun i hi => absurd ⟨i, hi⟩ hs, fun j _ => ?_⟩
    exact (le_abs_self (e j)).trans
      (Finset.single_le_sum (f := fun i => |e i|) (fun i _ => abs_nonneg (e i))
        (Finset.mem_univ j))

/-- `top_spectral_allocation` with the ordering stated as a top set. -/
theorem top_allocation (s : Finset ι) (e w : ι → ℝ)
    (hlo : ∀ i, 0 ≤ w i) (hhi : ∀ i, w i ≤ 1)
    (hmass : ∑ i, w i = (s.card : ℝ)) (htop : IsTopSet e s) :
    ∑ i, e i * w i ≤ ∑ i ∈ s, e i := by
  obtain ⟨cutoff, hsel, hrest⟩ := htop.exists_cutoff
  exact top_spectral_allocation s e w cutoff hlo hhi hmass hsel hrest

/-- Every size up to the dimension has a top set (repeatedly add a largest
remaining coordinate). -/
theorem exists_isTopSet (e : ι → ℝ) {r : ℕ} (hr : r ≤ Fintype.card ι) :
    ∃ s : Finset ι, s.card = r ∧ IsTopSet e s := by
  induction r with
  | zero => exact ⟨∅, rfl, fun i hi => absurd hi (Finset.notMem_empty i)⟩
  | succ r ih =>
    obtain ⟨s, hcard, htop⟩ := ih (Nat.le_of_succ_le hr)
    have hne : sᶜ.Nonempty := by
      rw [← Finset.card_pos, Finset.card_compl]
      omega
    obtain ⟨j, hj, hmax⟩ := Finset.exists_max_image sᶜ e hne
    have hjs : j ∉ s := Finset.mem_compl.mp hj
    refine ⟨insert j s, by rw [Finset.card_insert_of_notMem hjs, hcard], ?_⟩
    intro i hi k hk
    have hks : k ∉ s := fun h => hk (Finset.mem_insert_of_mem h)
    rcases Finset.mem_insert.mp hi with rfl | his
    · exact hmax k (Finset.mem_compl.mpr hks)
    · exact htop i his k hks

/-- Scaled form with spare mass: weights in `[0, κ]` of total at most `κ |s|`,
and nonnegative values. With `κ = 1` and full mass this is `top_allocation`. -/
theorem top_allocation_scaled (s : Finset ι) (e w : ι → ℝ) (κ : ℝ)
    (he : ∀ i, 0 ≤ e i) (hlo : ∀ i, 0 ≤ w i) (hhi : ∀ i, w i ≤ κ)
    (hmass : ∑ i, w i ≤ κ * s.card) (htop : IsTopSet e s) :
    ∑ i, e i * w i ≤ κ * ∑ i ∈ s, e i := by
  obtain ⟨cutoff, hsel, hrest⟩ := htop.exists_cutoff
  have hnn : 0 ≤ max cutoff 0 := le_max_right _ _
  have hsel' : ∀ i ∈ s, max cutoff 0 ≤ e i := fun i hi => max_le (hsel i hi) (he i)
  have hrest' : ∀ i ∉ s, e i ≤ max cutoff 0 := fun i hi => (hrest i hi).trans (le_max_left _ _)
  have hpoint : ∀ i, e i * w i ≤ κ * (e i * selectedWeight s i) +
      max cutoff 0 * (w i - κ * selectedWeight s i) := by
    intro i
    by_cases hi : i ∈ s
    · have h1 := hsel' i hi
      have h2 := hhi i
      simp only [selectedWeight, ite_eq_left hi, mul_one]
      nlinarith
    · have h1 := hrest' i hi
      have h2 := hlo i
      simp only [selectedWeight, ite_eq_right hi, mul_zero, sub_zero]
      nlinarith
  have hspare : ∑ i, (w i - κ * selectedWeight s i) ≤ 0 := by
    rw [Finset.sum_sub_distrib, ← Finset.mul_sum, selectedWeight_sum]
    linarith
  calc ∑ i, e i * w i ≤ ∑ i, (κ * (e i * selectedWeight s i) +
        max cutoff 0 * (w i - κ * selectedWeight s i)) :=
      Finset.sum_le_sum fun i _ => hpoint i
    _ = κ * ∑ i, e i * selectedWeight s i +
        max cutoff 0 * ∑ i, (w i - κ * selectedWeight s i) := by
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    _ ≤ κ * ∑ i, e i * selectedWeight s i := by
      linarith [mul_nonpos_of_nonneg_of_nonpos hnn hspare]
    _ = κ * ∑ i ∈ s, e i := by simp [selectedWeight]

end ProtectedDoRA

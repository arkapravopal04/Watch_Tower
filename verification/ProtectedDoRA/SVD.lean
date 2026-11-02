import Mathlib.Analysis.Matrix.Spectrum
import Mathlib.LinearAlgebra.Matrix.Rank
import ProtectedDoRA.Frobenius
import ProtectedDoRA.Projection
import ProtectedDoRA.Spectral

/-!
# Right singular vectors, Ky Fan and Eckart–Young (Frobenius)

`SpectralData H` is an orthonormal basis `V` of right singular vectors of `H`:
`V` is orthogonal and the columns of `H * V` are pairwise orthogonal, with
squared lengths `eig` (the squared singular values). One exists for every real
matrix (Mathlib's spectral theorem applied to `Hᵀ * H`).

Every theorem below holds for *every* such basis. Ties between singular values
and rank deficiency (zero singular values, `rank H < r`) need no extra
hypothesis, so the results apply to whichever basis a numerical SVD returns,
provided it is exact. Nothing here assumes the optimality being proved.

Main results, with `V_s` the columns of `V` indexed by a top set `s` of `eig`:
* `SpectralData.kyFan`: every orthonormal `U` with `|s|` columns captures at most
  `∑ i ∈ s, eig i` of `H`'s energy, and `V_s` attains it.
* `SpectralData.eckart_young`: `H V_s V_sᵀ` is a best Frobenius approximation of
  `H` among all matrices of rank at most `|s|`; its error is `∑ i ∉ s, eig i`.
* `SpectralData.truncate_mul_eq_zero`: if `H Q = 0` (e.g. `H = G (I - Q Qᵀ)`),
  the truncation also satisfies `X Q = 0`, so it is optimal under that
  constraint too, even when some chosen singular vectors have zero singular value.
-/

namespace ProtectedDoRA

open Matrix

variable {ο ι κ ρ : Type*} [Fintype ο] [Fintype ι] [Fintype κ] [Fintype ρ]
  [DecidableEq ι] [DecidableEq ρ]

/-- An orthonormal basis of right singular vectors of `H`, with the squared
singular values `eig`. -/
structure SpectralData (H : Matrix ο ι ℝ) where
  V : Matrix ι ι ℝ
  eig : ι → ℝ
  orth_left : Vᵀ * V = 1
  orth_right : V * Vᵀ = 1
  gram : (H * V)ᵀ * (H * V) = diagonal eig

/-- Every real matrix has right singular vectors (spectral theorem for `Hᵀ H`). -/
theorem exists_spectralData (H : Matrix ο ι ℝ) : Nonempty (SpectralData H) := by
  have hS : (Hᵀ * H).IsHermitian := by
    rw [IsHermitian, conjTranspose_eq_transpose_of_trivial, transpose_mul, transpose_transpose]
  have hstar : star (hS.eigenvectorUnitary : Matrix ι ι ℝ) =
      (hS.eigenvectorUnitary : Matrix ι ι ℝ)ᵀ := by
    rw [star_eq_conjTranspose, conjTranspose_eq_transpose_of_trivial]
  have hdiag := hS.conjStarAlgAut_star_eigenvectorUnitary
  rw [Unitary.conjStarAlgAut_star_apply, hstar, RCLike.ofReal_real_eq_id,
    Function.id_comp] at hdiag
  refine ⟨⟨hS.eigenvectorUnitary, hS.eigenvalues, ?_, ?_, ?_⟩⟩
  · rw [← hstar]
    exact Unitary.coe_star_mul_self _
  · rw [← hstar]
    exact Unitary.coe_mul_star_self _
  · rw [transpose_mul, ← hdiag]
    simp only [Matrix.mul_assoc]

/-- Columns of `V` indexed by `s`. -/
def cols (V : Matrix ι ι ℝ) (s : Finset ι) : Matrix ι s ℝ := V.submatrix id Subtype.val

theorem mul_cols_apply (M : Matrix ο ι ℝ) (V : Matrix ι ι ℝ) (s : Finset ι) (a : ο) (j : s) :
    (M * cols V s) a j = (M * V) a j := rfl

theorem cols_orthonormal (V : Matrix ι ι ℝ) (hV : Vᵀ * V = 1) (s : Finset ι) :
    (cols V s)ᵀ * cols V s = 1 := by
  ext a b
  have h := congrFun (congrFun hV a.1) b.1
  simp only [cols, mul_apply, transpose_apply, submatrix_apply, id] at h ⊢
  rw [h]
  simp only [one_apply, Subtype.ext_iff]

/-- Orthonormal columns give weights in `[0, 1]`: each row of `W` has squared
length at most one. -/
theorem weights_le_one (W : Matrix ι ρ ℝ) (hW : Wᵀ * W = 1) (i : ι) :
    ∑ j, W i j ^ 2 ≤ 1 := by
  have hPP : W * Wᵀ * (W * Wᵀ) = W * Wᵀ := proj_idem W hW
  have hdiag : (W * Wᵀ) i i = ∑ j, W i j ^ 2 := by
    simp only [mul_apply, transpose_apply, sq]
  have hsymm : ∀ k, (W * Wᵀ) k i = (W * Wᵀ) i k := fun k => by
    simp only [mul_apply, transpose_apply, mul_comm]
  have hsq : (W * Wᵀ) i i = ∑ k, (W * Wᵀ) i k ^ 2 := by
    conv_lhs => rw [← hPP]
    rw [mul_apply]
    exact Finset.sum_congr rfl fun k _ => by rw [hsymm k, sq]
  have hge : (W * Wᵀ) i i ^ 2 ≤ ∑ k, (W * Wᵀ) i k ^ 2 :=
    Finset.single_le_sum (f := fun k => (W * Wᵀ) i k ^ 2) (fun k _ => sq_nonneg _)
      (Finset.mem_univ i)
  have hnn : 0 ≤ (W * Wᵀ) i i := by
    rw [hdiag]
    exact Finset.sum_nonneg fun _ _ => sq_nonneg _
  rw [← hdiag]
  nlinarith [hsq, hge, hnn]

theorem trace_transpose_diagonal_mul (W : Matrix ι ρ ℝ) (e : ι → ℝ) :
    trace (Wᵀ * diagonal e * W) = ∑ i, e i * ∑ j, W i j ^ 2 := by
  have hdiag : ∀ j, (Wᵀ * diagonal e * W) j j = ∑ i, W i j * e i * W i j := fun j => by
    rw [mul_apply]
    simp only [mul_diagonal, transpose_apply]
  simp only [trace, diag_apply, hdiag]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  ring

/-- ... and the weights sum to the number of columns. -/
theorem weights_sum (W : Matrix ι ρ ℝ) (hW : Wᵀ * W = 1) :
    ∑ i, ∑ j, W i j ^ 2 = Fintype.card ρ := by
  have h := frob2_eq_trace W
  rw [hW, trace_one] at h
  exact h

/-- Bessel's inequality: coordinates on orthonormal columns carry at most the
vector's energy. -/
theorem bessel (U : Matrix ι ρ ℝ) (hU : Uᵀ * U = 1) (x : ι → ℝ) :
    ∑ j, (Uᵀ *ᵥ x) j ^ 2 ≤ ∑ k, x k ^ 2 := by
  let X : Matrix Unit ι ℝ := Matrix.of fun _ k => x k
  have hsplit := frob2_split X (U * Uᵀ) (proj_transpose U) (proj_idem U hU)
  have hXP : frob2 (X * (U * Uᵀ)) = frob2 (X * U) := by
    rw [← Matrix.mul_assoc]
    exact frob2_mul_orthonormal_rows _ _ (by rw [transpose_transpose, hU])
  have hl : ∑ j, (Uᵀ *ᵥ x) j ^ 2 = frob2 (X * U) := by
    simp only [frob2, Fintype.sum_unique]
    refine Finset.sum_congr rfl fun j _ => ?_
    congr 1
    simp only [mulVec, dotProduct, transpose_apply, mul_apply, X, of_apply]
    exact Finset.sum_congr rfl fun k _ => mul_comm _ _
  have hr : ∑ k, x k ^ 2 = frob2 X := by
    simp only [frob2, Fintype.sum_unique, X, of_apply]
  rw [hl, hr, hsplit, hXP]
  linarith [frob2_nonneg (X * (1 - U * Uᵀ))]

namespace SpectralData

variable {H : Matrix ο ι ℝ} (sd : SpectralData H)

theorem eig_eq (j : ι) : sd.eig j = ∑ a, (H * sd.V) a j ^ 2 := by
  have h := congrFun (congrFun sd.gram j) j
  rw [diagonal_apply_eq, mul_apply] at h
  rw [← h]
  simp only [transpose_apply, sq]

theorem eig_nonneg (j : ι) : 0 ≤ sd.eig j := by
  rw [sd.eig_eq]
  exact Finset.sum_nonneg fun _ _ => sq_nonneg _

/-- A zero singular value means `H` kills that singular vector. -/
theorem col_eq_zero {j : ι} (h : sd.eig j = 0) (a : ο) : (H * sd.V) a j = 0 := by
  rw [sd.eig_eq] at h
  have h' := (Finset.sum_eq_zero_iff_of_nonneg fun b _ => sq_nonneg ((H * sd.V) b j)).mp h a
    (Finset.mem_univ a)
  exact (pow_eq_zero_iff two_ne_zero).mp h'

theorem frob2_eq_sum_eig : frob2 H = ∑ j, sd.eig j := by
  rw [← frob2_mul_orthonormal_rows H sd.V sd.orth_right]
  simp only [frob2, sd.eig_eq]
  exact Finset.sum_comm

theorem col_sq_sum (i : ι) : ∑ k, sd.V k i ^ 2 = 1 := by
  have h := congrFun (congrFun sd.orth_left i) i
  rw [mul_apply, one_apply_eq] at h
  simpa only [transpose_apply, sq] using h

/-- At most `rank H` squared singular values are nonzero. -/
theorem card_nonzero_eig_le_rank :
    (Finset.univ.filter fun i => sd.eig i ≠ 0).card ≤ H.rank := by
  have h1 := rank_diagonal sd.eig
  rw [← sd.gram, rank_transpose_mul_self, Fintype.card_subtype] at h1
  rw [← h1]
  exact rank_mul_le_left H sd.V

/-- An operator-norm bound `‖H x‖ ≤ |c| ‖x‖` caps every squared singular value. -/
theorem eig_le_of_opNorm {c : ℝ} (hH : ∀ x : ι → ℝ, ∑ a, (H *ᵥ x) a ^ 2 ≤ c ^ 2 * ∑ k, x k ^ 2)
    (i : ι) : sd.eig i ≤ c ^ 2 := by
  have h := hH fun k => sd.V k i
  rw [sd.col_sq_sum, mul_one] at h
  rw [sd.eig_eq]
  exact h

/-- ... so the Frobenius energy is at most `c² rank H`. -/
theorem frob2_le_of_opNorm {c : ℝ}
    (hH : ∀ x : ι → ℝ, ∑ a, (H *ᵥ x) a ^ 2 ≤ c ^ 2 * ∑ k, x k ^ 2) :
    frob2 H ≤ c ^ 2 * H.rank := by
  obtain ⟨sdH⟩ := exists_spectralData H
  have hsum : ∑ i ∈ Finset.univ.filter (fun i => sdH.eig i ≠ 0), sdH.eig i ≤
      (Finset.univ.filter fun i => sdH.eig i ≠ 0).card • c ^ 2 :=
    Finset.sum_le_card_nsmul _ _ _ fun i _ => sdH.eig_le_of_opNorm hH i
  rw [Finset.sum_filter_ne_zero, nsmul_eq_mul] at hsum
  have hcard : ((Finset.univ.filter fun i => sdH.eig i ≠ 0).card : ℝ) ≤ H.rank := by
    exact_mod_cast sdH.card_nonzero_eig_le_rank
  rw [sdH.frob2_eq_sum_eig]
  nlinarith [sq_nonneg c]

/-- The matrix-to-spectral bridge: the energy `H` keeps on any `U` is the
squared singular values weighted by the squared overlaps `(Vᵀ U)ᵢⱼ²`. -/
theorem frob2_mul_eq_weighted (U : Matrix ι ρ ℝ) :
    frob2 (H * U) = ∑ i, sd.eig i * ∑ j, (sd.Vᵀ * U) i j ^ 2 := by
  have hU : H * U = H * sd.V * (sd.Vᵀ * U) := by
    rw [Matrix.mul_assoc, ← Matrix.mul_assoc sd.V, sd.orth_right, Matrix.one_mul]
  have hgram : (H * sd.V * (sd.Vᵀ * U))ᵀ * (H * sd.V * (sd.Vᵀ * U)) =
      (sd.Vᵀ * U)ᵀ * diagonal sd.eig * (sd.Vᵀ * U) := by
    rw [← sd.gram, transpose_mul (H * sd.V)]
    simp only [Matrix.mul_assoc]
  rw [hU, frob2_eq_trace, hgram, trace_transpose_diagonal_mul]

/-- **Ky Fan maximum principle** (Frobenius form). -/
theorem kyFan (U : Matrix ι ρ ℝ) (hU : Uᵀ * U = 1) {s : Finset ι}
    (hs : IsTopSet sd.eig s) (hcard : Fintype.card ρ = s.card) :
    frob2 (H * U) ≤ ∑ i ∈ s, sd.eig i := by
  have hW : (sd.Vᵀ * U)ᵀ * (sd.Vᵀ * U) = 1 := by
    rw [transpose_mul, transpose_transpose, Matrix.mul_assoc, ← Matrix.mul_assoc sd.V,
      sd.orth_right, Matrix.one_mul, hU]
  rw [sd.frob2_mul_eq_weighted]
  refine top_allocation s sd.eig _ (fun i => Finset.sum_nonneg fun _ _ => sq_nonneg _)
    (weights_le_one _ hW) ?_ hs
  rw [weights_sum _ hW, hcard]

/-- **Ky Fan, operator-norm form.** Any `A` with rank at most `|s|` and
`‖A x‖ ≤ |c| ‖x‖` keeps at most `c²` times the top `|s|` squared singular values
of `H` (`H Aᵀ` is `H` read through the rows of `A`). Orthonormal rows scaled by
`c` are the case `A Aᵀ = c² I`. -/
theorem kyFan_opNorm {s : Finset ι} (hs : IsTopSet sd.eig s) (A : Matrix ρ ι ℝ) {c : ℝ}
    (hA : ∀ x : ι → ℝ, ∑ j, (A *ᵥ x) j ^ 2 ≤ c ^ 2 * ∑ k, x k ^ 2) (hrank : A.rank ≤ s.card) :
    frob2 (H * Aᵀ) ≤ c ^ 2 * ∑ i ∈ s, sd.eig i := by
  obtain ⟨sdA⟩ := exists_spectralData A
  rw [sd.frob2_mul_eq_weighted, (transpose_mul A sd.V).symm]
  refine top_allocation_scaled s sd.eig _ (c ^ 2) sd.eig_nonneg
    (fun i => Finset.sum_nonneg fun _ _ => sq_nonneg _) (fun i => ?_) ?_ hs
  · have h := hA fun k => sd.V k i
    rw [sd.col_sq_sum, mul_one] at h
    exact h
  · have hF : ∑ i, ∑ j, (A * sd.V)ᵀ i j ^ 2 = frob2 A := by
      rw [← frob2_mul_orthonormal_rows A sd.V sd.orth_right, ← frob2_transpose (A * sd.V)]
      rfl
    rw [hF]
    have h1 := frob2_le_of_opNorm hA
    have h2 : (A.rank : ℝ) ≤ s.card := by exact_mod_cast hrank
    nlinarith [sq_nonneg c]

/-- With only a Frobenius budget, `‖H Aᵀ‖² ≤ σ_max² ‖A‖_F²`; one top singular
vector already attains it (`rank_one_attains`). -/
theorem frob2_mul_transpose_le_max {M : ℝ} (hM : ∀ i, sd.eig i ≤ M) (A : Matrix ρ ι ℝ) :
    frob2 (H * Aᵀ) ≤ M * frob2 A := by
  rw [sd.frob2_mul_eq_weighted, (transpose_mul A sd.V).symm]
  calc ∑ i, sd.eig i * ∑ j, (A * sd.V)ᵀ i j ^ 2
      ≤ ∑ i, M * ∑ j, (A * sd.V)ᵀ i j ^ 2 :=
        Finset.sum_le_sum fun i _ =>
          mul_le_mul_of_nonneg_right (hM i) (Finset.sum_nonneg fun _ _ => sq_nonneg _)
    _ = M * frob2 (A * sd.V)ᵀ := by rw [← Finset.mul_sum]; rfl
    _ = M * frob2 A := by
        rw [frob2_transpose, frob2_mul_orthonormal_rows A sd.V sd.orth_right]

/-- The top singular vectors attain the Ky Fan bound. -/
theorem frob2_mul_cols (s : Finset ι) : frob2 (H * cols sd.V s) = ∑ i ∈ s, sd.eig i := by
  have h : frob2 (H * cols sd.V s) = ∑ j : s, ∑ a, (H * sd.V) a j ^ 2 := by
    simp only [frob2, mul_cols_apply]
    exact Finset.sum_comm
  rw [h, Finset.sum_coe_sort s (fun i => ∑ a, (H * sd.V) a i ^ 2)]
  exact Finset.sum_congr rfl fun i _ => (sd.eig_eq i).symm

/-- A single top singular vector, a rank-one `A` with `‖A‖_F = 1`, keeps `σᵢ²`. -/
theorem rank_one_attains (i : ι) :
    frob2 (H * ((cols sd.V {i})ᵀ)ᵀ) = sd.eig i ∧ frob2 (cols sd.V {i})ᵀ = 1 := by
  refine ⟨by rw [transpose_transpose, sd.frob2_mul_cols, Finset.sum_singleton], ?_⟩
  rw [frob2_transpose, frob2_eq_trace, cols_orthonormal sd.V sd.orth_left, trace_one]
  simp

/-- Rank-`|s|` truncation `H V_s V_sᵀ`. -/
def truncate (s : Finset ι) : Matrix ο ι ℝ := H * (cols sd.V s * (cols sd.V s)ᵀ)

theorem truncate_rank_le (s : Finset ι) : (sd.truncate s).rank ≤ s.card := by
  rw [truncate, ← Matrix.mul_assoc]
  exact (rank_mul_le_right _ _).trans ((rank_le_card_height _).trans (Fintype.card_coe s).le)

theorem frob2_truncate (s : Finset ι) : frob2 (sd.truncate s) = ∑ i ∈ s, sd.eig i := by
  rw [truncate, ← Matrix.mul_assoc, frob2_mul_orthonormal_rows _ _
    (by rw [transpose_transpose, cols_orthonormal sd.V sd.orth_left s]), sd.frob2_mul_cols]

theorem truncate_error (s : Finset ι) :
    frob2 (H - sd.truncate s) = ∑ i ∈ sᶜ, sd.eig i := by
  rw [truncate, frob2_residual H _ (cols_orthonormal sd.V sd.orth_left s),
    sd.frob2_eq_sum_eig, sd.frob2_mul_cols, ← Finset.sum_add_sum_compl s]
  ring

/-- If every singular value outside `t` is zero, `V_t` spans the row space. -/
theorem mul_cols_mul_transpose {t : Finset ι} (hzero : ∀ j ∉ t, sd.eig j = 0) :
    H * (cols sd.V t * (cols sd.V t)ᵀ) = H := by
  ext a b
  have hrest : ∑ j ∈ tᶜ, (H * sd.V) a j * sd.V b j = 0 :=
    Finset.sum_eq_zero fun j hj => by
      rw [sd.col_eq_zero (hzero j (Finset.mem_compl.mp hj)) a, zero_mul]
  calc (H * (cols sd.V t * (cols sd.V t)ᵀ)) a b
      = ∑ j ∈ t, (H * sd.V) a j * sd.V b j := by
        rw [← Matrix.mul_assoc, mul_apply, ← Finset.sum_coe_sort t]
        rfl
    _ = ∑ j, (H * sd.V) a j * sd.V b j := by
        rw [← Finset.sum_add_sum_compl t, hrest, add_zero]
    _ = (H * sd.V * sd.Vᵀ) a b := by
        rw [mul_apply]
        rfl
    _ = H a b := by rw [Matrix.mul_assoc, sd.orth_right, Matrix.mul_one]

/-- A matrix of rank at most `|t|` has at most `|t|` nonzero singular values,
so they all lie in any top set of that size. -/
theorem eig_eq_zero_off_top {t : Finset ι} (ht : IsTopSet sd.eig t) (hrank : H.rank ≤ t.card) :
    ∀ j ∉ t, sd.eig j = 0 := by
  intro j hj
  by_contra hne
  have hpos : 0 < sd.eig j := lt_of_le_of_ne (sd.eig_nonneg j) (Ne.symm hne)
  have h1 := rank_diagonal sd.eig
  rw [← sd.gram, rank_transpose_mul_self, Fintype.card_subtype] at h1
  have h2 : (H * sd.V).rank ≤ H.rank := rank_mul_le_left H sd.V
  have hsub : insert j t ⊆ Finset.univ.filter fun i => sd.eig i ≠ 0 := by
    intro i hi
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ i, ?_⟩
    rcases Finset.mem_insert.mp hi with rfl | hi
    · exact hne
    · exact ne_of_gt (lt_of_lt_of_le hpos (ht i hi j hj))
  have h3 := Finset.card_le_card hsub
  rw [Finset.card_insert_of_notMem hj] at h3
  omega

end SpectralData

/-- Any matrix of rank at most `r ≤ dim` has its rows inside the span of some
`r` orthonormal columns. -/
theorem exists_orthonormal_cover (X : Matrix ο ι ℝ) {r : ℕ} (hX : X.rank ≤ r)
    (hr : r ≤ Fintype.card ι) :
    ∃ t : Finset ι, t.card = r ∧ ∃ U : Matrix ι t ℝ, Uᵀ * U = 1 ∧ X * (U * Uᵀ) = X := by
  obtain ⟨sd⟩ := exists_spectralData X
  obtain ⟨t, ht, htop⟩ := exists_isTopSet sd.eig hr
  have hzero := sd.eig_eq_zero_off_top htop (ht ▸ hX)
  exact ⟨t, ht, cols sd.V t, cols_orthonormal sd.V sd.orth_left t,
    sd.mul_cols_mul_transpose hzero⟩

namespace SpectralData

variable {H : Matrix ο ι ℝ} (sd : SpectralData H)

/-- **Eckart–Young** (Frobenius, lower bound): no matrix of rank at most `|s|`
gets closer to `H` than the discarded squared singular values. -/
theorem error_lower_bound {s : Finset ι} (hs : IsTopSet sd.eig s) (X : Matrix ο ι ℝ)
    (hX : X.rank ≤ s.card) : ∑ i ∈ sᶜ, sd.eig i ≤ frob2 (H - X) := by
  obtain ⟨t, ht, U, hU, hXU⟩ := exists_orthonormal_cover X hX (Finset.card_le_univ s)
  have hky := sd.kyFan U hU hs (by rw [Fintype.card_coe, ht])
  have hres := frob2_residual H U hU
  have hle := frob2_residual_le H X U hU hXU
  have htot := sd.frob2_eq_sum_eig
  rw [← Finset.sum_add_sum_compl s] at htot
  linarith

/-- **Eckart–Young** (Frobenius): the SVD truncation is a best approximation of
rank at most `|s|`. -/
theorem eckart_young {s : Finset ι} (hs : IsTopSet sd.eig s) (X : Matrix ο ι ℝ)
    (hX : X.rank ≤ s.card) : frob2 (H - sd.truncate s) ≤ frob2 (H - X) := by
  rw [sd.truncate_error]
  exact sd.error_lower_bound hs X hX

/-- Factored form, as an adapter produces it: `B A` with inner size at most `|s|`. -/
theorem eckart_young_factored {s : Finset ι} (hs : IsTopSet sd.eig s)
    (B : Matrix ο ρ ℝ) (A : Matrix ρ ι ℝ) (hρ : Fintype.card ρ ≤ s.card) :
    frob2 (H - sd.truncate s) ≤ frob2 (H - B * A) :=
  sd.eckart_young hs (B * A) ((rank_mul_le_right B A).trans ((rank_le_card_height A).trans hρ))

/-- If `H` kills `Q`, so does its truncation, whatever singular vectors were
chosen for zero singular values. -/
theorem truncate_mul_eq_zero {Q : Matrix ι κ ℝ} (hHQ : H * Q = 0) (s : Finset ι) :
    sd.truncate s * Q = 0 := by
  have hdiag : diagonal sd.eig * (sd.Vᵀ * Q) = 0 := by
    rw [← sd.gram, transpose_mul]
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc sd.V sd.Vᵀ Q, sd.orth_right, Matrix.one_mul, hHQ, Matrix.mul_zero,
      Matrix.mul_zero]
  have hterm : ∀ j c, sd.eig j * (sd.Vᵀ * Q) j c = 0 := fun j c => by
    have h := congrFun (congrFun hdiag j) c
    rwa [diagonal_mul] at h
  have hsplit : sd.truncate s * Q = H * cols sd.V s * ((cols sd.V s)ᵀ * Q) := by
    simp only [truncate, Matrix.mul_assoc]
  rw [hsplit]
  ext a c
  rw [mul_apply, Matrix.zero_apply]
  refine Finset.sum_eq_zero fun j _ => ?_
  have h2 : ((cols sd.V s)ᵀ * Q) j c = (sd.Vᵀ * Q) j c := rfl
  rw [mul_cols_apply, h2]
  by_cases hj : sd.eig j = 0
  · rw [sd.col_eq_zero hj, zero_mul]
  · rw [(mul_eq_zero.mp (hterm j c)).resolve_left hj, mul_zero]

/-- Constrained Eckart–Young for protected updates: with `H = G (I - Q Qᵀ)`, the
truncation has rank at most `|s|`, annihilates `Q`, and is a best rank-`|s|`
approximation of `H` even among matrices that also annihilate `Q`. -/
theorem protected_truncation_optimal [DecidableEq κ] {G : Matrix ο ι ℝ} {Q : Matrix ι κ ℝ}
    (hQ : Qᵀ * Q = 1) (sd : SpectralData (G * allowed Q)) {s : Finset ι}
    (hs : IsTopSet sd.eig s) :
    (sd.truncate s).rank ≤ s.card ∧ sd.truncate s * Q = 0 ∧
      ∀ X : Matrix ο ι ℝ, X.rank ≤ s.card → X * Q = 0 →
        frob2 (G * allowed Q - sd.truncate s) ≤ frob2 (G * allowed Q - X) := by
  refine ⟨sd.truncate_rank_le s, sd.truncate_mul_eq_zero ?_ s, fun X hX _ => sd.eckart_young hs X hX⟩
  rw [Matrix.mul_assoc, allowed_mul_basis Q hQ, Matrix.mul_zero]

/-- Scaling `H` keeps the singular vectors and scales the squared singular values. -/
def smul (c : ℝ) : SpectralData (c • H) where
  V := sd.V
  eig := fun i => c ^ 2 * sd.eig i
  orth_left := sd.orth_left
  orth_right := sd.orth_right
  gram := by
    rw [Matrix.smul_mul, transpose_smul, Matrix.smul_mul, Matrix.mul_smul, sd.gram, smul_smul]
    ext i j
    by_cases h : i = j
    · subst h
      simp [sq]
    · simp [diagonal_apply_ne _ h]

theorem isTopSet_smul {s : Finset ι} (hs : IsTopSet sd.eig s) (c : ℝ) :
    IsTopSet (sd.smul c).eig s := fun i hi j hj =>
  mul_le_mul_of_nonneg_left (hs i hi j hj) (sq_nonneg c)

theorem truncate_smul (c : ℝ) (s : Finset ι) : (sd.smul c).truncate s = c • sd.truncate s := by
  simp only [truncate, smul, Matrix.smul_mul]

end SpectralData

end ProtectedDoRA

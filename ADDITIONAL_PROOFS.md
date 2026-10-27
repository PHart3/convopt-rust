# Additional proofs: five post-hoc statements

Five statements proved in Verus on top of the verified solver (layers L21 and L22 of
`VERIFICATION_PROGRESS.md`). They add no trusted items and reuse existing lemmas.
Statements 2–4 live in `src/feas_spec.rs`; its header comment gives the overview.
Statement 5 lives in `src/lin_alg/resid_spec.rs` (with `src/lin_alg/norm_spec.rs`).

## 1. Gap identity

**Statement.** Use the setup's operator `A` (dense rows `s.mat_dense`, `s.c_len` of them, zero
list `s.zeros`), objective `C_F = s.obj_frob`, and right-hand side `b = s.point`. For packed
`Z`, `S` of order `d = s.var_dim` and `y` with `y.len() == b.len()`:

    <C_F, Z>_F - b . y = <Z, S>_F + <Z, C_F - A*(y) - S>_F + y . (A(Z) - b)

Corollary: if `A(Z) = b` and `C_F - A*(y) = S`, then `pobj - dobj = <Z, S>_F`.

**Proved by** `lemma_gap_identity` and `lemma_gap_complementarity` (`sdpad_spec.rs`), stated
with `prim_val_r`, `dual_val_r`, `setup_action_r` and `setup_slack_r` (`C_F - A*(y)`).

**Ingredients.** The adjoint identity `lemma_constraint_adjoint` with `lemma_adjoint_r_ok`
(`lin_alg/constr_spec.rs`); `lemma_frob_sub_right` (`lin_alg/mat_spec.rs`) and
`lemma_dot_sub_right` (`sdp_spec.rs`).

## 2. Soundness: a feasible standard-form point gives a feasible user point

**Statement.** Assume the case with LMIs. Let `Z` be packed of order `d`, solve every row of the
unreduced system `aug` of `std_rows_ok`, vanish on the zero pattern, and satisfy
`is_psd(sym_view(Z), d)`. Let `X_j = zvar_r(Z, var_off(..), r_j)` (`vars_of`), the variables
`read_solution` returns. Then (`standard_point_feasible`):

1. `Z` is block diagonal (`zeros_ok`).
2. `sum_j A_j(X_j) = b`.
3. Each LMI block `B_i = block_view(Z, block_off(bd, i))` equals `H_i(X) - M_i` and is PSD.
4. `<C_F, Z>_F = sum_j C_j(X_j)` (`lemma_objective_value`).

Without `check`, `A(Z) = b` for the setup's operator already means `Z` solves `aug`. Without
LMIs, `X = Z`.

With `check`, `A(Z) = b` only says that `Z` solves the kept rows of the reduced system; the
dropped rows are controlled only by the residual bound of statement 5, so the theorem is stated
for the unreduced system.

**Proved by** `lemma_standard_feasible_sound` (unreduced system), `lemma_action_feasible_sound`
(`A(Z) = b`, no `check`), and `lemma_action_feasible_sound_single` (no LMIs).

**Ingredients.** `lemma_psd_shifted_block` (`lin_alg/mat_spec.rs`): a diagonal block of a matrix
with `A >= -eps I` satisfies the same bound (pad the test vector with zeros); `is_psd` is the
case `eps = 0`.

## 3. Converse: a feasible user point lifts to a feasible standard-form point

**Statement.** Assume the case with LMIs, with or without `check`, and let `X_j` be packed of
order `r_j = vd[j]` and feasible for the user's problem (`user_feasible`): the equality
constraints hold and each `W_i = H_i(X) - M_i` is symmetric PSD. Then `Z = lift_z(..)`, a packed
matrix of order `d`, satisfies:

1. `zvar_r(Z, var_off(bsum, vd, j), r_j) == X_j` for every `j`;
2. `is_psd(sym_view(Z), d)`;
3. `Z` solves every row of `aug` and vanishes on the zero pattern; the returned rows are those
   of `aug` (without `check`) or their reduction, which `Z` solves as well, so `A(Z) = b`;
4. `<C_F, Z>_F = sum_j C_j(X_j)`, by `lemma_objective_value`.

Without LMIs, a feasible `X` (`user_feasible_single`: `A(X) = b` and `X ⪰ 0`) lifts as
`Z = X`; here `Z ⪰ 0` is the hypothesis `X ⪰ 0`.

**Construction.** `Z` is block diagonal with blocks `B_i = W_i`, `P_j = X_j + t_j I` and
`N_j = t_j I`, where `t_j = sum_{r, c} |X_j[r, c]|`; it is defined on packed indices via
`pk_unpack` and is zero off the blocks (`lift_entry`, `lift_z`).

**Proved by** `lemma_user_feasible_lifts` and `lemma_user_feasible_lifts_single` (no LMIs).

**Ingredients.**
- Shift bound `lemma_psd_shifted_abs_sum` (`lin_alg/mat_spec.rs`):
  `psd_shifted(sym_view(X), r, sum |X_rc|)`, i.e. `X + tI ⪰ 0`.
- Block-diagonal PSD `lemma_block_diag_psd_shifted` (`feas_spec.rs`): a matrix that vanishes
  off the blocks and has blocks `B_i >= -eps I` satisfies `Z >= -eps I`, via
  `lemma_psd_shifted_two_blocks` (`lin_alg/mat_spec.rs`).
- The row identities of `std_rows_ok`, instantiated at `Z`.
- With `check`, a solution of the unreduced rows solves the reduced rows
  (`lemma_solution_reduces`, `lemma_input_solution_reduced` in `lin_alg/reduce_spec.rs`): the
  echelon form has the same solutions, and the kept rows are its first `rank` rows rescaled.

**Not in Verus.** Without `check`, statements 2 and 3 together imply that the two problems
have the same optimal value; this is not proved in Verus. With `check`, statement 3 lifts every
feasible user point, but statement 2 needs a solution of the unreduced rows, which `A(Z) = b`
no longer provides: neither the feasibility of the user's problem at a returned point nor the
equality of optimal values is certified after the tolerance-based row removal.

## 4. Approximate soundness

**Statement.** Assume the case with LMIs and let `Z` be any packed matrix of order `d`
(`z.len() == pk_len(d)`). Let `res = A(Z) - b` (`action_residual_r`) and
`X_j = zvar_r(Z, var_off(..), r_j)`. For every `Z` (`lemma_user_residuals`):

    sum_j A_j(X_j)_e - b_e            = residual of unreduced equality row e at Z
    (H_i(X) - M_i)[r, c] - B_i[r, c]  = residual of unreduced LMI row c k_i + r at Z

where `B_i = block_view(Z, block_off(bd, i))`. Hence:

1. Without `check` (`lemma_approx_sound_unreduced`): both residuals are the entries of `res`
   at the corresponding row, `Z_rc = res[c_len + q]` at listed position `q`, and
   `<C_F, Z>_F = sum_j C_j(X_j)`.
2. With `check` (`lemma_approx_sound_reduced`), `res` measures the reduced system only. Let
   `[A_0 | b_0] = setup_aug` be the unreduced system, `N = ||A_0 Z - b_0||` its residual norm
   at `Z` (`std_resid_vec_r`), `phi(Z) = ||A_0||_F ||Z|| + ||b_0||` (`setup_resid_scale`), and
   `C`, `||U||_F` the constant and recovery norm of statement 5 for this reduction
   (`setup_red_const`, `setup_recovery_norm`). Then:
   - `N <= C tau phi(Z) + ||U||_F pinf (1 + ||b||)`, and `phi(Z) = 0` forces `N = 0`
     (`lemma_unreduced_residual_pinf`): the kept residual `||A_K Z - b_K||` is the norm of the
     dense part of `res`, at most `||res|| = pinf (1 + ||b||)`;
   - `|sum_j A_j(X_j)_e - b_e| <= N` for every equality row `e`;
   - the linking residual `R_i = W_i - B_i` of every LMI, with `W_i = H_i(X) - M_i` the actual
     slack (`lmi_slack_r`, `lmi_link_residual`), satisfies `||R_i||_F <= N`;
   - if `B_i >= -eps_i I`, then `x^T W_i x >= -(eps_i + ||R_i||_F) |x|^2` for every `x`
     (`lemma_lmi_slack_shifted`). `W_i` need not be symmetric at an approximate point, so this
     is a bound on its quadratic form (`psd_shifted`), not a semidefinite-order statement.

   The zero-pattern entries of `Z` are still entries of `res` and
   `<C_F, Z>_F = sum_j C_j(X_j)`. Nothing is claimed about the size of `C` and `||U||_F`, so
   these bounds do not make the LMI violations small in absolute terms.
3. A successful PSD check (`psd_blocks_ok`) gives `B_i >= -eps_i I` with the implementation's
   tolerance (`lemma_lmi_blocks_checked`), and `|res_l| <= pinf * (1 + ||b||)`
   (`lemma_residual_entry_pinf`).

**Ingredients.** The row identities of `std_rows_ok` give the residual identities above for
every `Z`; `lemma_result_aug_residual` identifies the residuals of the returned rows with the
entries of `res`. For the reduced case: the window bound `||x[off .. off + len]|| <= ||x||`
(`lemma_window_norm_le`), the grid sum `sum_c sum_r f(c k + r) = sum_t f(t)`
(`lemma_sum_fn_grid`) identifying `||R_i||_F` with the norm of a window of `A_0 Z - b_0`, and
the perturbation bound `|x^T R x| <= ||R||_F ||x||^2` on quadratic forms
(`lemma_quad_frob_bound`, `lemma_psd_shifted_perturbed` in `lin_alg/norm_spec.rs`).

## 5. Relative residual bound of the constraint reduction

**Statement.** Let `[A | b]` be the dense rows the reduction receives (`rd` rows, `n` columns of
`A`), and `[A_K | b_K]` the returned system (`rank` rows, `q = rd - rank` rows dropped). Let `D`
be the diagonal matrix of the first-stage divisors (the Frobenius norms of the rows, 1 for rows
of norm zero), `T = [T_K | T_D]` the row transformation of the elimination (normalized input
`= T *` echelon form), split at the kept rows, `S` the diagonal matrix of the second-stage
divisors of the kept rows and `H_D` the dropped echelon rows. With

    U = D T_K S,        [E | f] = D T_D H_D,

the input is recovered as `[A | b] = U [A_K | b_K] + [E | f]`. Let `alpha` and `beta` be the
largest magnitudes of the normalized coefficients and right-hand sides (0 if empty),
`L = ||D T_D||_F`, `tau = TOL`, and

    e_A = 2^rank sqrt(q n) alpha L,    e_b = 2^rank sqrt(q) beta L,
    C   = max(e_A / ||A||_F, e_b / ||b||)   (a zero denominator has a zero numerator; it counts 0),
    d(z) = ||A||_F ||z|| + ||b||.

Then `||E||_F <= tau e_A`, `||f|| <= tau e_b`, and for every `z`

    ||A z - b|| <= C tau d(z) + ||U||_F ||A_K z - b_K||,

which divided by `d(z) > 0` is the relative bound
`||A z - b|| / d(z) <= C tau + ||U||_F ||A_K z - b_K|| / d(z)`. If `d(z) = 0` then
`A z - b = 0`; if `q = 0` then `C = 0`. `U` and `C` depend on the data only, not on `z`;
nothing is claimed about the size of `C`.

**Proved by** `lemma_residual_bound` and `lemma_relative_residual_bound` (explicit pivots and
transformation), `lemma_reduction_residual_bound` (for `reduced_system`, witnesses
`red_pivots`, `red_transform`) and `lemma_reduced_from_residual_bound` in `sdpad_spec.rs` (for
the dense rows of a setup with `check`, `reduced_from`, witness `red_echelon`), the last two
stating both forms and the zero case; the zero cases are `lemma_residual_zero_scale` and
`lemma_red_const_no_drop`.

**Ingredients.**
- The ghost transformation `T` tracked through `gauss_elim` (`row_comb`, `lemma_row_comb_swap`,
  `lemma_row_comb_sub` in `gauss_spec.rs`), exported by `reduced_with`.
- The dropped rows are small (`lemma_dropped_row_bounded`): zero in the pivot columns, at most
  `2^rank * eps` in the other coefficient columns (`skipped_bounded`), and at most
  `TOL * 2^rank * beta` in the right-hand side (`rhs_consistent`, which records the per-row
  scales of the consistency test).
- The recovery identity `lemma_recovery_identity` (both normalization stages), the residual
  decomposition `A z - b = U (A_K z - b_K) + E z - f` (`lemma_residual_decomp`), and the bounds
  `lemma_err_bounds` and `lemma_red_const_dominates`.
- Rectangular norms (`norm_spec.rs`): Cauchy-Schwarz (`lemma_cauchy_schwarz`), the triangle
  inequality (`lemma_norm_triangle`), `||X z|| <= ||X||_F ||z||` (`lemma_mat_vec_rc_bound`),
  `||X Y||_F <= ||X||_F ||Y||_F` (`lemma_mmul_frob_rc`) and the entry bounds
  (`lemma_frob_rc_entry_bound`, `lemma_norm_entry_bound`).

# Additional proofs: four post-hoc statements

Four statements proved in Verus on top of the verified solver (layer L21 of
`VERIFICATION_PROGRESS.md`). They add no trusted items and reuse existing lemmas.
Statements 2–4 live in `src/feas_spec.rs`; its header comment gives the overview.

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

With `check`, a solution of the reduced system satisfies the dropped rows of the echelon form
only up to the bound of `lemma_reduced_solution_echelon`, so the theorem is stated for the
unreduced system; statement 4 gives the quantitative version.

**Proved by** `lemma_standard_feasible_sound` (unreduced system), `lemma_action_feasible_sound`
(`A(Z) = b`, no `check`), and `lemma_action_feasible_sound_single` (no LMIs).

**Ingredients.** `lemma_psd_shifted_block` (`lin_alg/mat_spec.rs`): a diagonal block of a matrix
with `A >= -eps I` satisfies the same bound (pad the test vector with zeros); `is_psd` is the
case `eps = 0`.

## 3. Converse: a feasible user point lifts to a feasible standard-form point

**Statement.** Assume the case with LMIs and let `X_j` be packed of order `r_j = vd[j]` and
feasible for the user's problem (`user_feasible`): the equality constraints hold and each
`W_i = H_i(X) - M_i` is symmetric PSD. Then `Z = lift_z(..)`, a packed matrix of order `d`,
satisfies:

1. `zvar_r(Z, var_off(bsum, vd, j), r_j) == X_j` for every `j`;
2. `is_psd(sym_view(Z), d)`;
3. `Z` solves every row of `aug` and vanishes on the zero pattern; by
   `lemma_reduced_of_input_solution` it also solves the reduced system, so `A(Z) = b` with
   and without `check`;
4. `<C_F, Z>_F = sum_j C_j(X_j)`, by `lemma_objective_value`.

Without LMIs, `Z = X`.

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

**Not in Verus.** Statements 2 and 3 together imply that the two problems have the same
optimal value. This is not proved in Verus.

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
2. With `check` (`lemma_approx_sound_reduced`): for the echelon form `E` of the unreduced
   system (`reduced_with`), both residuals are bounded by `reduction_bound_r`,

       rows * sum_{l < rank} ||E_l||_F |res_l| + (rows - rank) * 2^rank * eps_G * (1 + ||vec Z||_1),

   with `rows` the number of unreduced rows, `rank = c_len`, `||E_l||_F` the scale of kept
   echelon row `l` (`row_scale_r`), and `eps_G = TOL * max |entry|` the elimination tolerance;
   the zero-pattern entries and the objective are as in case 1.
3. A successful PSD check (`psd_blocks_ok`) gives `B_i >= -eps_i I` with the implementation's
   tolerance (`lemma_lmi_blocks_checked`), and `|res_l| <= pinf * (1 + ||b||)`
   (`lemma_residual_entry_pinf`).

**Ingredients.** The `check` case needs the original rows as a bounded combination of the
echelon rows, which solution-set equivalence (`same_solutions`) does not provide.
- **Transformation**, in `lin_alg/gauss_spec.rs` and `gauss_elim`: a ghost `T` with
  `A_0 = T E` (`row_comb`); a row swap exchanges two columns of `T`, a row subtraction adds
  `piv` times a column (`|piv| <= 1` by partial pivoting), so the absolute column sums of `T`
  stay at most `rows` on the pivot columns and `1` elsewhere (`cols_bounded`). Residuals
  transfer along `T` (`lemma_row_comb_residual_bound`).
- **Reduction bound**, in `lin_alg/reduce_spec.rs`: `reduced_with` records the transformation
  as the opaque `reduction_transform`; `lemma_reduced_residual_bound` combines it with the
  scaling of the kept echelon rows and `lemma_dropped_row_residual`, which holds for every `x`.
- The row identities of `std_rows_ok` give the residual identities above for every `Z`.

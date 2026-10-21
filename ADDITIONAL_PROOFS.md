# Additional proofs: four statements to prove in Verus

This file gives four new statements for the agent to prove in Verus. Prove each as a lemma.
Follow `CLAUDE.md`: add no trusted items, and reuse existing lemmas.

## 1. Gap identity

**Statement.** Use the setup's operator `A` (dense rows `s.mat_dense`, `s.c_len` of them, zero
list `s.zeros`), objective `C_F = s.obj_frob`, and right-hand side `b = s.point`. For packed
`Z`, `S` of order `d = s.var_dim` and `y` with `y.len() == b.len()`:

    <C_F, Z>_F - b . y = <Z, S>_F + <Z, C_F - A*(y) - S>_F + y . (A(Z) - b)

Corollary: if `A(Z) = b` and `C_F - A*(y) = S`, then `pobj - dobj = <Z, S>_F`.

**Ingredients.**
- `lemma_constraint_adjoint` (`lin_alg/constr_spec.rs`) proves `<A(X), y> = <X, A*(y)>_F`.
  Write `A*(y)` as `adjoint_r` (see `lemma_adjoint_r_ok`) and `A(Z)` as `setup_action_r`.
- Distributivity of `frob_r` and `dot_r` over `vsub_r`. Check the existing sum lemmas first,
  e.g. `lemma_dot_sub_right`, `lemma_frob_sub_right` (`mat_spec.rs`, where the packed
  `frob_r` is `frob_f` on `sym_view`).
- Place the lemma in `sdpad_spec.rs` and state it with `prim_val_r` and `dual_val_r`.

## 2. Soundness: a feasible standard-form point gives a feasible user point

**Statement.** Assume the case with LMIs and without `check` (`sdpad_setup_ok`). Let `Z` be
packed of order `d` with `A(Z) = b` and `is_psd(sym_view(Z), d)`. Let
`X_j = zvar_r(Z, var_off(..), r_j)`, the variables `read_solution` returns. Then:

1. `Z_rc = 0` at every listed position, so `Z` is block diagonal (`zeros_ok`).
2. `sum_j A_j(X_j) = b`: `maps_row_r(constraint, .., e, Z) == b_e` for every `e`.
3. Each LMI block `B_i = blk_r(Z, block_off(bd, i), k_i)` satisfies
   `B_i[r, c] = H_i(X)[r, c] - M_i[r, c]` (from `std_rows_ok`) and is PSD. Hence
   `H_i(X) - M_i` is symmetric PSD.
4. `<C_F, Z>_F = sum_j C_j(X_j)`. This is `lemma_objective_value`.

Also cover the case without LMIs: there `X = Z`, so `A(Z) = b` and `Z ⪰ 0` make `X` feasible.

**Missing lemma.** A diagonal block of a PSD matrix is PSD. Prove it in `lin_alg/mat_spec.rs`
for `block_view`: pad the test vector with zeros outside the block and use `psd_shifted` and
`quad_r`. Stating it for `psd_shifted` with a general `eps` covers `is_psd` as the case
`eps = 0`.

**With `check`.** The rows are reduced, and a solution of the reduced system satisfies the
dropped rows of the echelon form only up to the bound of `lemma_reduced_solution_echelon`.
State the theorem for `Z` solving the unreduced system `aug` of `std_rows_ok`. Do not claim
exact feasibility from the reduced system; the quantitative statement is 4.

## 3. Converse: a feasible user point lifts to a feasible standard-form point

**Statement.** Assume the case with LMIs. Let `X_j` be packed of order `r_j = vd[j]`, and
suppose `X` is feasible for the user's problem:
- for every `e`: `sum_j dot_r(symrow_r(constraint[j]@, e, r_j), X_j) == b_e`;
- for every `i`, the `k_i x k_i` matrix
  `W_i[r, c] = sum_j dot_r(symrow_r(lmi[i].0[j]@, c * k_i + r, r_j), X_j) - M_i[c * k_i + r]`
  is symmetric and PSD.

Then some packed `Z` of order `d` satisfies:
1. `zvar_r(Z, var_off(bsum, vd, j), r_j) == X_j` for every `j`;
2. `is_psd(sym_view(Z), d)`;
3. `Z` solves the unreduced system: every row of `aug` (`std_rows_ok`), and `Z_rc = 0` at
   every listed position. With `check`, `lemma_reduced_of_input_solution` shows that `Z` also
   solves the reduced system, so this statement holds in both cases;
4. `<C_F, Z>_F = sum_j C_j(X_j)`, by `lemma_objective_value`.

Without LMIs, take `Z = X`.

**Construction.** `Z` is block diagonal, with blocks `B_i = W_i`, `P_j = X_j + t_j I` and
`N_j = t_j I`, where `t_j = sum_{r, c} |X_j[r, c]|`. Define it as a spec function on packed
indices (via `pk_unpack`), zero off the blocks.

**Ingredients.**
- **Shift bound**, in `mat_spec.rs`: `psd_shifted(sym_view(X), r, t)` with
  `t = sum |X_rc|`, so `X + tI ⪰ 0`. Proof: `x^T X x >= -sum |X_rc| |x_r| |x_c|`, and
  `2 |x_r x_c| <= x_r^2 + x_c^2`. For the step from the shift to PSD, see
  `lemma_psd_shifted_meq` and `lemma_shifted_factorization_psd`.
- **Block-diagonal PSD**, in `mat_spec.rs`: a block-diagonal matrix with PSD blocks is PSD,
  because the quadratic form splits by block. State it against the same block view as the
  lemma of statement 2.
- The row identities of `std_rows_ok` hold for every `z`; instantiate them at `Z`.

**Not in Verus.** Statements 2 and 3 together imply that the two problems have the same
optimal value. Do not prove this in Verus.

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
- **Transformation**, in `gauss_spec.rs` and `gauss_elim`: a ghost `T` with `A_0 = T E`
  (`row_comb`); a row swap exchanges two columns of `T`, a row subtraction adds `piv` times a
  column (`|piv| <= 1` by partial pivoting), so the absolute column sums of `T` stay at most
  `rows` on the pivot columns and `1` elsewhere (`cols_bounded`). Residuals transfer along
  `T` (`lemma_row_comb_residual_bound`).
- **Reduction bound**, in `reduce_spec.rs`: `reduced_with` records the transformation as the
  opaque `reduction_transform`; `lemma_reduced_residual_bound` combines it with the scaling
  of the kept echelon rows and `lemma_dropped_row_residual`, which holds for every `x`.
- The row identities of `std_rows_ok` give the residual identities above for every `Z`.

## Done when

- All lemmas verify, and
  `verus src/lib.rs --crate-type=lib --edition=2024 -V no-solver-version-check` passes.
- `VERIFICATION_PROGRESS.md` is updated and records the new verified count.


# Verification Progress

Read at session start; updated at session end. Layer DAG: `DAG.md`.
Trusted items: `TRUSTED_BASE.md`.

## Commands

Full verification (the Z3 shipped with this Verus build reports a version
string with a build hashcode, which fails Verus's exact version check, so
the extra flag is required):

    verus src/lib.rs --crate-type=lib --edition=2024 -V no-solver-version-check

Runtime checks: `cargo build`, `cargo test`.

## Layer status

| Layer | Status | Notes |
|-------|--------|-------|
| L00 scaffolding and trusted exact-real model | done | `src/trusted.rs`, `src/real_spec.rs`, `src/lin_alg/comparison.rs` in `verus!`. |
| L01 vector views and basic vector/matrix operations | done | `src/lin_alg/vec_spec.rs` (new spec module), 14 functions of `src/lin_alg/matrix.rs` in `verus!`; `gauss_elim` stays outside (L13). |
| L02 packed symmetric view and basic symmetric-matrix operations | done | `src/lin_alg/sym_spec.rs` (new spec module), 15 functions of `src/lin_alg/sym_matrix.rs` in `verus!`; the rest of that file stays outside (L04-L07, L14, L15). |
| L03 spec-level matrix algebra | done | `src/lin_alg/mat_spec.rs` (new spec/proof module: 21 spec functions, 54 lemmas) plus 7 generic sum lemmas in `sym_spec.rs`; no executable code touched. |
| L04 constraint operator and adjoint | done | `src/lin_alg/constr_spec.rs` (new spec/proof module: operator `A`, adjoint `A*`, Gram matrix, triangle sums, adjoint identity, Gram-action lemma); `constraint_action`, `constraint_lin_comb`, `constraint_gram`, `diag_scale_gram` in a second `verus!` block of `src/lin_alg/sym_matrix.rs`. |
| L05 triangular and diagonal solvers | done | `src/lin_alg/solve_spec.rs` (new spec/proof module: partial row sums, pivot scale, `mat_vec` of the triangular and diagonal views); `diag_solver`, `forw_subst`, `back_subst` plus the helper `pivot_scale` in a third `verus!` block of `src/lin_alg/sym_matrix.rs`; `TOL` moved into the `verus!` block of `src/lin_alg/matrix.rs`. |
| L06 LDLT factorization and positive-definite solvers | done | `src/lin_alg/ldlt_spec.rs` (new spec/proof module: LDLT recurrence, locality and preservation lemmas, `L D L^T = A`, block-diagonal solve); `ldlt_decomp`, `pos_def_solver`, `pos_def_solver_upperleft` in the third `verus!` block of `src/lin_alg/sym_matrix.rs`; `psd_block_check` stays outside (L07). |
| L07 PSD block check | done | `psd_block_check` in the third `verus!` block of `src/lin_alg/sym_matrix.rs`; block-check specs in `src/lin_alg/ldlt_spec.rs`; reusable PSD lemmas in `src/lin_alg/mat_spec.rs`. |
| L08 shared Jacobi rotation core | done | `src/lin_alg/jacobi_spec.rs` (new spec/proof module); `jacobi_rot` in a `verus!` block of `src/lin_alg/jacobi.rs`; the eigensolvers and spectral operations of that file stay outside (L10-L12). |
| L09 common eigensolver specification | done | `src/lin_alg/eigen_spec.rs` (new spec/proof module: invariant, stopping criterion, off-diagonal Frobenius bound, shared contract `eigensolver_ok`); no executable code touched. |
| L14 constraint reduction | done | `src/lin_alg/reduce_spec.rs` (new spec/proof module); `linear_remove_redundant_sym` in a `verus!` block of `src/lin_alg/sym_matrix.rs`. **Two executable corrections** (packed-column start `start = diag + 1`; off-diagonal factor `sqrt(2)` instead of `1/sqrt(2)`, confirmed by the user), see the L14 architecture decisions. |
| L13 Gaussian elimination | done | `src/lin_alg/gauss_spec.rs` (new spec/proof module); `gauss_elim` in a second `verus!` block of `src/lin_alg/matrix.rs`. |
| L15 user-facing symmetric-map representation | done | `matrix_diagonal_mult_sym`, `mat_sym_diagonal_mult_sym` and the six partitioned helpers in two further `verus!` blocks of `src/lin_alg/sym_matrix.rs`; `matrixsym_flatten`, `mk_symmap`, `zero_symmap`, `ident_symmap` in two `verus!` blocks of `src/sdp.rs` (`sdp_to_standard` stays outside, L16). |
| L10 cached Jacobi | done | `index_max_in_col`, `cached_pivot_col` (new helper), `jacobi_eigen_cach` in the `verus!` block of `src/lin_alg/jacobi.rs`; cache specs there, `offdiag_max_at` in `eigen_spec.rs`, `same_outside` in `jacobi_spec.rs`. **Two executable corrections** (stale `biggest`, out-of-column cache read), see the L10 architecture decisions and `EXPOSED_BUGS.md`. |
| L11 column-cyclic Jacobi | done | `jacobi_eigen_cyc` in the `verus!` block of `src/lin_alg/jacobi.rs`; column-cyclic order `cyc_order`/`visits_each_pair_once` in `eigen_spec.rs`. |
| L16 standard-form conversion | done | `src/sdp_spec.rs` (new spec/proof module); `SDP` struct, `sdp_to_standard` and seven helpers in the first `verus!` block of `src/sdp.rs`. Public API: `SDP` fields are now `pub`; `sdp_to_standard` has shape preconditions (`sdp_single_input_ok` / `sdp_lmi_input_ok`). |
| L12 PSD spectral operations | done | `src/lin_alg/spectral_spec.rs` (new spec/proof module); `JacobiVariant`, the helper `select_eigenpairs`, `poseigendecomp`, `negeigendecomp` in the `verus!` block of `src/lin_alg/jacobi.rs`; `pivot_scale` (L05) is now `pub(crate)`. |
| L17 solver linear-system setup | done | `src/sdpad_spec.rs` (new spec/proof module); `SdpadSetup`, `sdpad_setup`, `psd_block_list`, `dual_solve`, `initial_iterates` in a `verus!` block of `src/sdp_altdir.rs`; `sdpad` itself stays outside `verus!` and calls them (L18/L19). **One executable correction** (empty PSD block list for single-variable problems), see the L17 architecture decisions and `EXPOSED_BUGS.md`. |
| L18 .. L20 | not started | |

## Verified-function count

Last full run: **825 verified, 0 errors** (Verus `--time`: 22 s SMT time; no slow
proofs; the whole run takes 26 s).

- 1 public and 3 private executable functions (`src/sdp_altdir.rs`, `verus!` block) (L17):
  `sdpad_setup(sdp, check)` requires `sdpad_input_ok` (the `sdp_to_standard` preconditions plus
  `dim_fits(total_rows)`, the number of stacked constraint rows) and returns the `SdpadSetup`
  record with `sdpad_setup_ok`: for some packed objective `obj0`, the record's data is the
  standard form (`std_form_ok`, L16), `obj_frob` is `obj0` with halved off-diagonal entries
  (`half_off_diag`), `c_len` is the number of dense rows (the zero-constraint rows follow, with
  zero right-hand sides), `mat_dense` has `c_len * pk_len(d)` entries, `gram_small` is the
  packed Gram matrix of the dense rows (`is_constraint_gram`, L04; empty when `c_len == 0`),
  `block_size_sum` is the total LMI block size, and `psd_blocks` is the block list of the
  partition `psd_partition` of the variable (`block_list_ok`, sizes summing to `d`,
  `blocks_fit`); also `d >= 1`, `dim_fits(d)`, `dim_fits(c_len)`. `psd_block_list` builds that
  list; `dual_solve(gram, rhs, c_len, check)` ensures `dual_solve_ok`; `initial_iterates`
  ensures `initial_iterates_ok` (`Z = I`, `S = 0`).
- 11 spec functions and 9 lemmas in `src/sdpad_spec.rs` (L17): `std_form_ok` (dispatch on
  the two conversion contracts), `total_rows`, `sdpad_input_ok`, `lemma_reduced_rows` (the
  reduced system has at most `total_rows` rows: the result length is `rank * (cols + 1)` and
  the echelon form has `rank <= rows`), `lemma_std_form_facts` (the facts of the standard form
  the setup relies on); `half_off_diag` with `lemma_frob_half_dot` (`<cf, z>_F = c . z`,
  through `lemma_frob_tri` and `lemma_tri_sum_packed` of L04); `psd_partition`,
  `block_list_ok`, `lemma_block_sizes_entry`, `lemma_block_list_fits`; `reg_shift_r` (`0`
  with reduced constraints, else `TOL * max(max_i |G_ii|, 1)`), `residual_ok`,
  `dual_solve_ok` (`(G + s I) y_top = b_top`, `y_bottom = b_bottom`, and for the regularized
  solve `||G y_top - b_top|| / (||G||_F ||y_top|| + ||b_top||) < 1e-9` whenever the denominator
  is nonzero), `lemma_block_ident_solve_inv` (the converse of L06's block solve lemma),
  `lemma_shift_zero_meq`, `diag_shifted` with `lemma_diag_shifted_view`; `initial_iterates_ok`.

- 3 executable functions (`src/lin_alg/jacobi.rs`, `verus!` block) (L12): `poseigendecomp` and
  `negeigendecomp` require the eigensolver preconditions (`dim >= 1`, `dim_fits(dim)`, a packed
  input of length `>= pk_len(dim)`) and ensure `exists q: spectral_part_ok(A, B, q, sel, vals,
  vects, dim)`: `eigensolver_ok(A, B, q, diag(B), dim)` (L09) for the final content `B` of the
  input and some accumulated `q`, and `selected_pairs`: the returned values and columns are the
  pairs `(B_kk, column k of q)` at the indices `sel_indices(sel, dim)` (increasing order), with
  `sel = pos_sel(diag(B))` (`B_kk > 0`) resp. `neg_sel(diag(B), pivot_scale_r(A))` (`B_kk <
  -1e-8 * max(max_t |A_t|, 1)`, the scale of the *input*). The shared helper
  `select_eigenpairs(vals, vects, dim, negative, scale)` runs the one selection loop (invariant
  `selected_pairs(.., i, ..)` over the processed indices).
- 10 spec functions and 7 lemmas in `src/lin_alg/spectral_spec.rs` (L12): the index list
  `sel_indices(sel, n)` with `lemma_sel_indices` (exactly the selected `k < n`, increasing) and
  `lemma_sel_indices_empty`; the tests `pos_sel`, `neg_sel`, `neg_tol_r = 1e-8 * scale`; the masked
  values `masked_r(vals, sel)` (unselected entries 0); `lemma_sum_selected` (a sum over the
  selected indices is the masked sum over all indices, via `sel_terms`/`masked_terms`); the
  predicates `selected_pairs`, `spectral_part_ok`, `sel_count`; `lemma_selected_reconstruction`
  (`Q_sel D_sel Q_sel^T = Q diag(masked vals) Q^T`, the matrix `diag_scale_gram` computes from the
  returned pairs); `lemma_pos_part_psd` (that matrix is PSD for the positive part);
  `lemma_no_negative_part` (an empty negative part gives `B_kk >= -1e-8 * scale` for all `k` and
  `Q diag(B) Q^T >= -1e-8 * scale I`); `lemma_rows_of_product` (the `diag_scale_gram`
  preconditions for `m > 0` selected columns). Plus, in `mat_spec.rs`, `lemma_orthogonal_norm`
  (`|Q^T x| = |x|`) and `lemma_diag_congruence_psd_shifted` (`Q D Q^T >= -eps I` for orthogonal
  `Q` and `d_k >= -eps`).

- 1 public and 9 private executable functions (`src/sdp.rs`, first `verus!` block) (L16):
  `sdp_to_standard` dispatches to `standard_single` (no LMI) and `standard_lmi`; it requires
  `sdp_single_input_ok` (one objective map of order `d >= 1`, at most one constraint map of the
  same order, sizes fitting; with `check_constraints` a nonzero constraint system,
  `single_constraint_nonzero`) resp. `sdp_lmi_input_ok` (`lmi_shapes_ok`: one map per objective
  entry in every LMI, nonempty square constant matrices, map `n` of every LMI a MatrixSym of the
  order `vd[n]` read from the first LMI with `k_i^2` rows; objective and constraint maps of the
  same orders; `dim_fits` and `(pk_len(sd) + 1) * (rows + 1) <= usize::MAX`), and ensures
  `std_form_single_ok` resp. `std_form_lmi_ok` (see the L16 architecture decisions). Helpers:
  `part_concat` (a partitioned matrix as a packed vector, `part_r`), `lmi_dims` (block sizes and
  variable orders with their sums), `maps_sum` (the embedded maps of one LMI or of the
  constraints: row `e` reads `sum_n maps[n][e](Z_n)`, `maps_row_r`), `lmi_block` (`maps_sum`
  minus the block entry `B_i[e]`, with the entry `-1` exposed), `objective_sum` (`<obj, Z> =
  obj_r`), `zero_positions` (`zeros_ok` and the length `zeros_len`), `stack_blocks` (entrywise
  stacking), plus the `SDP` constructor and accessors. `matrixsym_flatten` (L15) now requires
  `mat.len() * pk_len(dim) <= usize::MAX` and computes its capacity as `len * pk_len`.
- 30 spec functions and 24 lemmas in `src/sdp_spec.rs` (L16): packed-index unpacking
  (`pk_unpack`), the packed views `part_r`/`symrow_r`, the block extraction `blk_r` and the user
  variable `zvar_r = blk(st) - blk(st + d)`; the embedding identity `lemma_embed_dot`
  (`<embed(w, st), z> = <w, blk(z, st, d)>`, through `lemma_tri_sum_block`) and its row form
  `lemma_embed_row_dot`; linearity of row dot products (`lemma_row_dot_add_sub`,
  `lemma_row_dot_zero`, `lemma_row_dot_update`, `lemma_dot_add_sub`, `lemma_dot_sub_right`,
  `lemma_stacked_row`); the integer square root (`isqrt_r`, existence and uniqueness); the
  layout arithmetic (`sum_upto`, `sq_sum_upto`, `block_sizes`, `block_off`, `var_off`,
  `std_dim`, `row_off`, `lmi_rows`, `rows_upto` with monotonicity, `lemma_block_sizes_sum`,
  `lemma_pk_len_diff`, `lemma_block_triangles_le`); the zero pattern (`in_zero_region`,
  `listed`, `zeros_ok`, `zeros_len`); the row predicates (`maps_row_r`, `obj_r`, `std_rows_ok`,
  `single_rows_ok` with `lemma_single_rows`, `lmi_row_lo/hi` with `lemma_lmi_row_bounds`,
  `lemma_lmi_row_of`).

- 1 executable function (`src/lin_alg/jacobi.rs`, `verus!` block) (L11): `jacobi_eigen_cyc` has
  the same contract as `jacobi_eigen_cach` (`eigensolver_ok`). Each sweep carries the ghost
  sequence `visited` of examined pivot positions with the invariant `visited == cyc_order(k) +
  [(0, k), .., (j - 1, k)]`, so that after the sweep `visited == cyc_order(dim)` and
  `visits_each_pair_once(visited, dim)` are asserted; every rotation is the shared `jacobi_rot`
  (applied iff the entry is nonzero) under `eigen_decomp_inv`; the scan of the off-diagonal
  entries establishes `|B_rc| <= max_offdiag` for all `r < c` and `max_diag ==
  max_abs_diag_r`, so the stopping test gives `jacobi_converged`. Termination:
  `decreases MAX_SWEEPS - count`.
- 3 spec functions and 5 lemmas in `src/lin_alg/eigen_spec.rs` (L11): `cyc_order(dim)` (the
  pairs `(j, k)` of a sweep, column by column), `cyc_pos(j, k) = k (k - 1) / 2 + j`,
  `visits_each_pair_once` (each pair `j < k < dim` is listed at `cyc_pos(j, k)` and every
  listed position is a pair listed at its own `cyc_pos`), with `lemma_cyc_order_len`,
  `lemma_cyc_order_entries`, `lemma_cyc_order_at`, `lemma_cyc_order_visits_each_pair_once`.

- 3 executable functions (`src/lin_alg/jacobi.rs`, `verus!` block) (L10): `index_max_in_col`
  ensures `col_max_at(A, col, r)` (the returned row `r < col` has maximal magnitude in column
  `col`); the new helper `cached_pivot_col` turns a valid cache (`cache_ok`) into a position of
  maximal off-diagonal magnitude (`offdiag_max_at(A, dim, maxima[k - 1], k)`);
  `jacobi_eigen_cach` requires `dim >= 1`, `dim_fits(dim)` and a packed input of length
  `>= pk_len(dim)`, and ensures the shared contract `eigensolver_ok(A, B, Q, vals, dim)` of L09
  (`vals = diag(B)`, `A = Q B Q^T`, `Q` orthogonal, `||B - diag(B)||_F <= jacobi_eps_r`), with
  `B` the final content of the input vector. Its main loop carries `eigen_decomp_inv`, the
  cache invariant (for the current matrix before the first rotation, for the matrix before the
  last rotation afterwards), the pivot position, the diagonal scale and `biggest`; the cache
  update loop proves `col_max_at` for every column through the six branches; termination is
  proved with `decreases max_rotations - count` (the non-convergence exit aborts, T8).
- 2 spec functions and 1 lemma in `jacobi.rs` (`col_max_at`, `cache_ok`, `lemma_col_unrotated`:
  the unrotated entries of a column are bounded by the old cached entry and by the old
  maximum), `offdiag_max_at` with `lemma_offdiag_max_below` in `eigen_spec.rs` (a maximal
  off-diagonal magnitude below `thr` gives `offdiag_below`), `same_outside` with
  `lemma_jacobi_step_same_outside` in `jacobi_spec.rs` (a `jacobi_step` at `(j, k)` changes only
  rows and columns `j`, `k`).

- 1 executable function (`src/lin_alg/sym_matrix.rs`, own `verus!` block) (L14):
  `linear_remove_redundant_sym` requires `dim >= 1`, `dim_fits(dim)`, an input length that
  is a multiple of `red_col_dim(dim) = pk_len(dim) + 1`, and a nonzero input
  (`abs_max_r(input) > 0`, so that the elimination tolerance `eps = TOL * max|entry|` is
  positive); it ensures `reduced_system(input, final(mat), result, dim)`: for the pivot
  sequence of `gauss_elim`, the echelon form `e` (left in `mat`) satisfies
  `same_solutions(input, e)`, `echelon_form`, `skipped_bounded`, every pivot column lies in
  the packed part (the right-hand side column holds no pivot), every pivot row has a
  positive Frobenius norm, and `result` (`rank` rows, column-major) is row `i` of `e` divided
  by `row_scale_r(e, rd, dim, i)` for `i < rank`. Both data-dependent exits abort (T8).
- 12 spec functions and 15 lemmas in `src/lin_alg/reduce_spec.rs` (L14; Verus reports 28
  items): the Frobenius row norm `row_scale_r = sqrt(sum_{r <= c} pk_wt(r, c) M_rc^2)` with
  its column-wise partial sums (`row_sq_term_r`, `col_sq_r`, `row_sq_upto_r`, step lemmas)
  and `lemma_row_sq_frob` (it is `<M_i, M_i>_F` of the packed row `packed_row_r`); positivity of the norm of
  a pivot row (`lemma_row_scale_pos`, via `lemma_pk_idx_surjective`, `lemma_sum_fn_ge_term`,
  `lemma_tri_sum_ge_term`); the result predicates `reduced_with`/`reduced_system`;
  `lemma_reduced_of_input_solution` (input solutions solve the reduced system);
  `lemma_reduced_solution_echelon` (reduced solutions satisfy the kept echelon rows exactly
  and every dropped row up to the residual `2^rank * eps * (1 + ||x||_1)`, through
  `lemma_dropped_row_entry`, `lemma_dropped_row_residual`, `lemma_abs_sum_fn_le`,
  `lemma_pow2_monotone`, with `one_norm_r` and `row_residual_r`).

- 10 spec functions and 7 lemmas in `src/lin_alg/eigen_spec.rs` (L09; Verus reports 14
  items): the invariant `eigen_decomp_inv(a, q, b, dim)` (`Q` orthogonal, `A = Q B Q^T` on
  the block) with `lemma_eigen_inv_init` (`Q = I`, `B = A`), `lemma_eigen_inv_step` (one
  `jacobi_rot` call, given by its existential `jacobi_step` postcondition, preserves it) and
  `lemma_eigen_inv_eigenvalues` (`A` and `B` have the same eigenvalues); the stopping
  criterion `offdiag_below(b, dim, thr)`, the threshold `jacobi_thr_r = 1e-12 * max(1, max_i
  |B_ii|)` and `jacobi_converged`; the diagonal part `mdiag`, `moffdiag = B - diag(B)`,
  `offdiag_norm_r = ||B - diag(B)||_F`, `offdiag_count_r = dim (dim - 1)` with
  `lemma_mdiag_view`, `lemma_offdiag_frob_bound` (`||B - diag(B)||_F^2 <= dim (dim - 1) thr^2`)
  and `lemma_offdiag_norm_bound`; the contract `eigen_decomp_approx(a, q, b, dim, eps)`,
  the bound `jacobi_eps_r = sqrt(dim (dim - 1)) * jacobi_thr_r`, `lemma_converged_approx`
  (invariant + stopping criterion give the contract with that bound) and the shared
  eigensolver postcondition `eigensolver_ok(a, b, q, vals, dim)` (`vals = diag(B)`, `Q` of
  length `dim * dim`, `eigen_decomp_approx` with `jacobi_eps_r`). Plus `lemma_sqrt_mul` in
  `src/real_spec.rs`, `lemma_sum_fn_const` in `sym_spec.rs` and `lemma_frob_f_self_nonneg`
  in `mat_spec.rs`.

- 12 executable functions (L15): in `src/lin_alg/sym_matrix.rs`, `matrix_diagonal_mult_sym`
  (column `pk_idx(start + a, start + b)` of the result is column `pk_idx(a, b)` of the dense
  input, every other column is zero) and `mat_sym_diagonal_mult_sym` (the same for one
  packed matrix: the block embedding), and the partitioned helpers `mk_symmap_real`
  (`result[c][r] == vect[pk_idx(r, c)]`), `sym_matrix_trace` (identity), `sym_matrix_scale`,
  `sym_matrix_zero`, `sym_matrix_proj` (single 1 at `(row, col)`), `sym_matrix_add`; in
  `src/sdp.rs`, `matrixsym_flatten` (entry `(r, pk_idx(j, i))` of the column-major result is
  `mat[r][i][j]`), `mk_symmap` (row `k` is the packed matrix at `entries[k * pk_len ..]`),
  `zero_symmap`, `ident_symmap` (row `n * dim + i` picks entry `(min(n, i), max(n, i))`).
- 8 spec functions and 7 lemmas (L15): `part_shape`, `symmap_shape`, `is_block_idx`,
  `block_prev_end` with `lemma_pk_idx_order` (packed indices order pairs lexicographically
  by (column, row)), `lemma_block_idx_gap`, `lemma_block_prev_end_le`,
  `lemma_block_prev_end_bound`, `lemma_block_idx_max` in `src/lin_alg/sym_spec.rs`;
  `lemma_cm_idx_bound` in `vec_spec.rs`; `symmap_dim`, `ident_symmap_entry` in `src/sdp.rs`.

- 1 executable function (`src/lin_alg/matrix.rs`, second `verus!` block) (L13): `gauss_elim`
  requires `col_dim >= 1` and `mat.len() == row_dim * col_dim`, and ensures
  `same_solutions(input, output)` (the row operations preserve the solution set of the
  augmented system `[A | b]`) and the existence of a pivot sequence `pivots` with
  `echelon_form(output, pivots, eps)` (pivot columns strictly increasing, pivot `i` in row
  `i` of magnitude `>= eps = TOL * max |input entry|`, exact zeros below every pivot),
  `skipped_bounded(output, pivots, eps)` (the tolerance-based skip rule: a column without a
  pivot is below `2^k * eps` in every row from its examination row on, `k` = pivots found
  later), rank = `pivots.len()`, and consistency flag = "no pivot in the last column".
  Pivot-selection correctness (the chosen row maximizes the magnitude over the remaining
  rows) and the skip rule at examination time are verified assertions inside the function.
- 11 spec functions and 19 lemmas in `src/lin_alg/gauss_spec.rs` (L13): the augmented
  system and its solution set (`aug_row_ok`, `solves`, `same_solutions`), the row operations
  `rows_swapped`/`row_subtracted` with their solution-set preservation lemmas, the echelon
  predicates (`pivots_ok`, `echelon_form`) with preservation under swaps and row
  subtractions below the pivot rows and the push of a new pivot, and the skip rule
  (`pow2_r`, `pivots_below`, `is_pivot_col`, `skip_bound`, `skipped_bounded`) with its
  preservation lemmas. Plus `lemma_abs_mul`, `lemma_abs_sub_triangle`,
  `lemma_abs_quotient_le_one` in `src/real_spec.rs`; `lemma_cm_entry_write_other` moved to
  `src/lin_alg/vec_spec.rs` and generalized to rectangular matrices.

- 1 executable function (`src/lin_alg/jacobi.rs`, `verus!` block) (L08): `jacobi_rot`
  requires a nonzero pivot entry `(j, k)`, `j < k < dim`, and ensures the existence of
  `c, s` with `jacobi_step(A, Q, A', Q', dim, j, k, c, s)`: `c^2 + s^2 = 1`,
  `A' = J^T A J` and `Q' = Q J` for `J = rot_mat(j, k, c, s)` (block equalities of the
  packed and column-major views), `A'_jk = 0`, and the returned value `|A_jk|`; the
  returned references are linked to the parameters by `final(mat)@ == final(res.0)@`.
- 7 spec functions and 19 lemmas in `src/lin_alg/jacobi_spec.rs` (L08; Verus reports 62
  items): the rotation `rot_mat`, the explicit products `rot_target` (`J^T A J`) and
  `rot_cols` (`Q J`) with their product lemmas, the two-term sum, `J^T J = I` and
  `is_orthogonal(J)`, the scalar identities of the executable choice of `t, c, s`
  (`lemma_jacobi_t`, `lemma_jacobi_cs`, `lemma_jacobi_scalars`: the rotated 2 x 2 block is
  `diag(a_jj - t a_jk, a_kk + t a_jk)`, so the pivot entry is annihilated), locality of
  single-entry writes to packed and column-major matrices, `is_orthogonal`/`is_symmetric`
  congruence under `meq`, the step invariant `lemma_rotation_step` (`A = Q B Q^T` with `Q`
  orthogonal is preserved by `B' = J^T B J`, `Q' = Q J`) with its stored-representation form
  `lemma_jacobi_step_preserves`, and `lemma_jacobi_step_similar` (symmetry and eigenvalues of
  `B` are preserved).

- 1 executable function (`src/lin_alg/sym_matrix.rs`, third `verus!` block) (L07):
  `psd_block_check` requires `blocks_fit` (every listed block lies inside the packed matrix
  and its index arithmetic fits in `usize`) and ensures that a `true` result gives
  `block_psd_ok(A, start, dim)` for every listed block, i.e. `B >= -eps I` with
  `eps = 1e-9 * max(max_i |B_ii|, 1)` for the block `B = block_view(A, start)`.
- 4 spec functions and 1 lemma in `src/lin_alg/ldlt_spec.rs` (L07): `block_diag_abs_r`,
  `block_max_abs_diag_r`, `psd_check_eps_r`, `block_psd_ok`, `blocks_fit`, and
  `lemma_block_max_abs_diag` (the diagonal scale of the extracted block equals that of the
  block inside the full matrix). Plus, in `mat_spec.rs`, the view `block_view` and the lemmas
  `lemma_psd_shifted_zero_dim`, `lemma_diag_congruence_psd` (`Q D Q^T >= 0` for `d >= 0`, also
  the spectral PSD argument of L12) and `lemma_shifted_factorization_psd`
  (`Q D Q^T = A + eps I` with `d >= 0` gives `A >= -eps I`); and `lemma_max0_fn_ext` in
  `sym_spec.rs`.

- 3 executable LDLT functions (`src/lin_alg/sym_matrix.rs`, third `verus!` block) (L06):
  `ldlt_decomp` returns `None` or `Some((l, d))` with `l` unit lower-triangular, every pivot
  `d_k >= TOL * max(max_i |A_ii|, 1)`, the entrywise recurrence predicate `ldlt_ok`, and
  `L D L^T = A` as a block equality of views; `pos_def_solver` returns `x` with
  `mat_vec(sym_view(A), dim, x) == b` (aborts when the factorization fails);
  `pos_def_solver_upperleft` returns `x` with `(A, I) x = b` over `block_ident_view`.
- 10 proved lemmas (and 3 spec functions) in `src/lin_alg/ldlt_spec.rs` (L06): step,
  locality and symmetry of the weighted row product `ldlt_sum_r`, preservation of the
  recurrence under a write to `L` and under a pivot push, entry (r, c) of `L D L^T` for unit
  lower-triangular `L`, `L D L^T = A` from the recurrences, and the block-diagonal solve.
  Plus `lemma_lt_idx_injective` in `vec_spec.rs` and `lemma_mul_diag_transpose_symmetric`
  (`Q D Q^T` is symmetric) and the view `block_ident_view` in `mat_spec.rs`.

- 3 executable solvers and 1 helper (`src/lin_alg/sym_matrix.rs`, third `verus!` block) (L05):
  `diag_solver` returns `x` with `x_i = b_i / d_i` and `D x = b` (`mat_vec(diag_view(d))`),
  `forw_subst` returns `x` with `L x = b` (`mat_vec(lt_view(L))`), `back_subst` returns `x`
  with `L^T x = b` (`mat_vec(mtr(lt_view(L)))`); each also ensures the division-safety
  contract that every pivot satisfies `|pivot| >= TOL * max(max_t |m_t|, 1)` on return
  (otherwise the function aborts). `pivot_scale` computes that scale.
- 10 proved lemmas (and 5 spec functions) in `src/lin_alg/solve_spec.rs` (L05): fold step of
  the prefix maximum, step and extensionality of prefix and range row sums, and
  `(A x)_i` as a full, lower-triangular (prefix), upper-triangular (suffix) or diagonal
  (single-term) row sum. Plus `lemma_sum_fn_tail_zero` and `lemma_lt_idx_fits` in
  `sym_spec.rs`.

- 4 executable constraint-operator functions (`src/lin_alg/sym_matrix.rs`, second `verus!`
  block) with exact-real postconditions (L04): `constraint_action` returns `A(X)`,
  `constraint_lin_comb` returns the packed adjoint `A*(y)` (entrywise, with the off-diagonal
  factors 1/2), `constraint_gram` returns the packed Gram matrix of the dense rows (entrywise
  triangle sums), `diag_scale_gram` returns `Q D Q^T` as a block equality of views.
- 14 proved lemmas (and 19 spec functions) in `src/lin_alg/constr_spec.rs` (L04): sum splitting, triangle sums
  (extensionality, additivity, scaling, single term, exchange with an outer sum, decoding of
  flat packed sums), the Frobenius product as a weighted triangle sum, packed-index
  injectivity, column-slice dot products, the adjoint identity
  `<A(X), y> = <X, A*(y)>_F`, symmetry of the Gram entries, and `G y = A(A*(y))` (no
  zero-pattern rows).

- 54 proved lemmas in `src/lin_alg/mat_spec.rs` (L03; Verus reports 66 verified items for the
  module): product algebra (associativity,
  transpose of a product, identity, block-equality congruence, cancellation of inverses),
  orthogonality (identity, transpose, products, inverse pairs), symmetry under congruence,
  matrix-vector products (composition, scaling, identity, zero, shift by `s I`),
  eigenvalues (block invariance, similarity invariance, orthogonal similarity both ways),
  quadratic forms (`<x, A y> = <A^T x, y>`, congruence, shift, diagonal form, diagonal lower
  bound), the semidefinite order (`A >= -eps I` iff `A + eps I >= 0`, congruence, diagonal
  PSD), products with a diagonal matrix (`Q D Q^T` as a sum of scaled outer products), the
  Frobenius product (commutativity, linearity, block invariance, agreement with `frob_r`),
  and the connection lemmas for the stored representations (identity views, symmetric view,
  packed matrix-vector product).
- 30 proved lemmas in `src/lin_alg/sym_spec.rs` (23 from L02 plus, from L03, zero sum,
  single-term sum, monotone sum, exchange of double sums, and extensionality, additivity
  and scaling of square double sums).

- 15 executable symmetric-matrix functions (`src/lin_alg/sym_matrix.rs`), each with an
  exact-real postcondition (L02).
- 14 executable vector/matrix functions (`src/lin_alg/matrix.rs`) (L01).
- 6 executable comparison functions (`src/lin_alg/comparison.rs`) (L00).
- 13 proved lemmas in `src/lin_alg/vec_spec.rs` (L01 lemmas plus column-major index
  injectivity).
- 3 proved lemmas about `sqrt_r` (`src/real_spec.rs`).
- 5 `external_body` wrappers in `src/trusted.rs` (trusted, see TRUSTED_BASE.md).
- The remainder are spec functions and loop bodies, which Verus counts separately.

## Architecture decisions (L00)

- Spec view of a float: `rv(x): real` (uninterpreted) in `src/trusted.rs`. Reals are
  Verus's native `real` type (SMT reals), so ordered-field reasoning is built in; only
  `sqrt_r` is axiomatized.
- Executable `f64` operators keep their syntax. vstd's result hooks (`add_ensures`,
  `lt_ensures`, ...) are given exact-real meaning by broadcast axioms; consumers write
  `broadcast use crate::trusted::group_real_model;` at module level.
- std float methods get `assume_specification` contracts instead of wrappers, so call
  sites are unchanged and there is no runtime overhead.
- Spec vocabulary: `abs_r`, `max_r`, `min_r` (`src/real_spec.rs`), `approx_equal`,
  `near_zero` (`src/lin_alg/comparison.rs`).
- Code outside `verus!` blocks is external by default; functions are moved into `verus!`
  layer by layer.

## Architecture decisions (L01)

- Vector view: `rvs(v@): Seq<real>` for both `Vec<f64>` and `&[f64]` (`src/lin_alg/vec_spec.rs`).
  Vector algebra at spec level: `vadd_r`, `vsub_r`, `vscale_r`, `vmul_r`, `sum_r`, `dot_r`,
  `norm_r`, `dist_r`, `std_basis_r`, `const_r`.
- Matrix index maps (used by L02+): column-major `cm_idx`/`cm_entry`/`is_ident_r`; packed
  lower-triangular column-major `lt_col_start`/`lt_idx`/`lt_entry`/`is_lt_ident_r`.
- Binary vector operations require equal lengths (the mathematical reading of `zip`); the
  original code silently truncated. Callers in later layers must establish the lengths.
- `ident_mat` and `zero_mat` require their `dim * dim` / `dim_c * dim_r` capacity to fit in
  `usize` (Verus checks the executable multiplication for overflow).
- Every function returning a vector also ensures its length explicitly: sequence-length facts
  do not follow automatically from `rvs(r@) == ...` equalities (no term triggers the
  `Seq::new` length axiom).

## Architecture decisions (L02)

- Packed symmetric view (`src/lin_alg/sym_spec.rs`): entry (r, c), r <= c, of a dim x dim
  symmetric matrix sits at `pk_idx(r, c) = c * (c + 1) / 2 + r`; `sym_entry(m, r, c)` extends
  this to all index pairs by symmetry; `pk_len(dim)` is the packed length.
- Index-function sums: `sum_fn_r(f, n) = f(0) + .. + f(n-1)` over `spec_fn(int) -> real`
  closures, with `sum2_fn_r` for square double sums and `max0_fn_r` for the fold-from-zero
  maximum. Closures with captured variables are matched syntactically by Verus, so lemma
  conclusions about closures carry over to call sites without extensionality proofs.
- Mathematical specs: `frob_r(a, b, dim)` is the full double sum over `sym_entry` (so the
  factor 2 on off-diagonal packed entries is proved, not defined, via `lemma_frob_step`);
  `sym_mat_vec_r`, `sym_one_norm_r`, `max_abs_diag_r`, `sym_diag_r`, `is_sym_ident_r`.
- Sweep bookkeeping: loops that visit the packed triangle once while updating rows r and
  c of each entry (`sym_mat_act`, `sym_mat_one_norm`) use the invariant
  `row_k == partial_row(t, dim, k, col, row)`, where `swept` says which unordered pairs
  have been visited; four lemmas (off-diagonal step, diagonal step, next column, done)
  cover every loop event. Reuse this for any similar traversal (e.g. `constraint_gram`).
- `usize` fit: `dim_fits(dim) := (dim + 2)^2 <= usize::MAX` is the single precondition
  covering all packed index arithmetic (`c * (c + 1)`, `i * i + 3 * i`, `k * (2 * dim - k + 1)`).
  Functions indexing a block at `start` require `dim_fits(start + dim)`. `sym_matrix_diagonal`
  additionally requires `pk_len(dim) * dim <= usize::MAX` (its output is cubic in dim).
- Outer products, identity, diagonal blocks and the block map are specified entry-wise
  (`forall r <= c < dim: result[pk_idx(r, c)] == ...`), since no flat-index decoding is needed.
- `max_diag_sym(mat, 0)` returns `|mat[0]|` or `0`; the spec only claims nonnegativity for
  `dim == 0` and the exact maximum for `dim >= 1`.

## Architecture decisions (L03)

- Matrices in the algebra are index functions `MatF = spec_fn(int, int) -> real`
  (`src/lin_alg/mat_spec.rs`); every operation and predicate takes the dimensions it needs
  (`mmul(a, b, inner)`, `meq(a, b, rows, cols)`, `is_orthogonal(q, dim)`, ...). Stored
  matrices enter through views: `sym_view` (packed symmetric), `cm_view` (column-major),
  `lt_view` (packed lower-triangular, zero above the diagonal), `diag_view` (vector as a
  diagonal matrix). No executable representation changes.
- Equality is the block equality `meq`, since index functions are unconstrained outside
  the block. Each entrywise lemma (`lemma_mul_assoc`, `lemma_transpose_mul`, identity laws)
  has a `_meq` form so that proofs are chains of `lemma_mul_meq`/`lemma_meq_trans` steps;
  `lemma_cancel_inverse` packages the `A (B C) = C` step used by every orthogonality and
  similarity argument.
- `is_orthogonal(q, dim)` requires both `Q^T Q = I` and `Q Q^T = I`, so that no
  "left inverse implies right inverse" theorem is needed: rotations satisfy both directly
  and `lemma_orthogonal_mul` propagates both through products (L08/L09).
- `is_eigenvalue(a, dim, lambda)` is the existence of a nonzero `x` with `A x = lambda x`;
  `lemma_similarity_preserves_eigenvalues` is stated for any inverse pair `(P, Q)` and
  specialized to orthogonal similarity in both directions.
- The semidefinite order is the quadratic-form predicate `psd_shifted(a, dim, eps)`
  (`x^T A x + eps |x|^2 >= 0`), with `is_psd` the case `eps = 0`;
  `lemma_psd_shifted_iff` relates it to `A + eps I >= 0`, and `lemma_quad_congruence` plus
  `lemma_quad_diag_lower_bound` give the LDLT argument of L07 and the spectral PSD
  argument of L12.
- `lemma_mul_diag_transpose` gives `(Q D Q^T)_rc = sum_k d_k Q_rk Q_ck`, the entrywise
  spec for `diag_scale_gram` (L04) and the eigen-reconstruction (L09, L12).
- The Frobenius product `frob_f` over index functions agrees with the packed `frob_r`
  (`lemma_frob_sym_view`); linearity lemmas serve the adjoint identity of L04.

## Architecture decisions (L04)

- Constraint representation (`src/lin_alg/constr_spec.rs`): the dense constraint matrix is
  column-major with `row_dim` rows and `pk_len(dim)` columns (`constr_entry(m, row_dim, i, n)`
  is entry `n` of the packed matrix `M_i`); the zero-pattern list holds pairs `(r, c)` with
  `r < c < dim` (`valid_zero_list`), as built by `sdp_to_standard`. `row_dim` is recovered
  from the lengths (`rows_of(len, col_dim) = len / col_dim`), with `len % col_dim == 0` as a
  precondition; `constraint_action` has no `dim` parameter, so it takes the existential
  `zero_list_fits(list, var.len())`.
- Operator specs: `constraint_action_r` (dense rows are packed dot products `row_dot_r`, then
  the listed entries of `X`), `adjoint_entry_r(r, c) = adj_wt(r, c) * col_comb_r(.., pk_idx(r, c))
  + 1/2 * zero_comb_r(r, c)` with `adj_wt = 1` on the diagonal and `1/2` off it, stated as the
  entrywise predicate `is_constraint_adjoint`; `gram_entry_r(a, b)` is the triangle sum of
  `adj_wt * M_a[n] * M_b[n]` (`is_constraint_gram`). The adjoint identity
  `lemma_constraint_adjoint` needs the strict `r < c` of the list: a diagonal pair would get
  weight 1/2 instead of 1.
- Empty dense matrix: both `constraint_action` and `constraint_lin_comb` return early (`[]`,
  resp. zeros) and ignore the zero-pattern list, as the original code does; the postconditions
  state exactly this (see open issues).
- Triangle sums `tri_sum_r(f, dim) = sum_{c < dim} sum_{r <= c} f(r, c)` are the bridge between
  packed loops (which visit the triangle column by column) and the flat sums over packed
  indices (`lemma_tri_sum_packed`) and the Frobenius product (`lemma_frob_tri`, weights
  `pk_wt`). Reuse them for every packed accumulation loop.
- Accumulation loops that add a whole packed matrix (`result[t] += op[t]` for all `t`) are
  verified with a ghost snapshot `prev` and the two-part invariant "updated below `t`,
  unchanged from `t` on"; the per-entry spec is recovered afterwards with `lemma_pk_idx_bound`.
- `lemma_gram_action` (`G y = A(A*(y))` for the dense rows) is the hook for the regularized
  Gram solve of L17.

## Architecture decisions (L05)

- Solver specs are stated as `mat_vec(view, dim, x) == b` over the L03 views (`lt_view`,
  `mtr(lt_view)`, `diag_view`), so L06 can compose `L D L^T x = b` with `lemma_mat_vec_mul`
  without unfolding any solver. `diag_solver` additionally states the entrywise quotient.
- The substitution loops are specified with partial row sums (`src/lin_alg/solve_spec.rs`):
  `row_prefix_r(a, x, i, n) = sum_{k < n} a(i, k) x_k` for forward substitution (column `j`
  processed means the right-hand side `vect[i]` equals `b_i - row_prefix(i, j)` for `i >= j`
  and row `i` is solved, `row_prefix(i, i + 1) == b_i`, for `i < j`) and
  `row_range_r(a, x, i, lo, hi)` for back substitution (suffix sums from `lo = j + 1`).
  The extensionality lemmas discharge the fact that pushing or overwriting `solution[j]`
  leaves sums over other index ranges unchanged.
- The pivot scale `max(max_t |m_t|, 1)` ranges over the *whole* stored vector (as the
  original `fold`), stated as `pivot_scale_r(rvs(mat@))`. The division-safety contract is
  reported as a postcondition (every pivot is at least `TOL` times the scale), which also
  yields `rv(d) != 0` for the T3 division axiom.
- `back_subst` keeps its `(0..dim).rev()` loop using the named-iterator form
  `for j in it: (0..dim).rev()`; invariants refer to `it.index()` (columns processed) and
  `it.seq()` (the reversed range), and the body asserts `j == dim - 1 - it.index()`.
- Preconditions: `dim_fits(dim)` (covers the packed lower-triangular offsets
  `k * (2 * dim - k + 1) / 2`, see `lemma_lt_idx_fits`) and `mat.len() >= dim * (dim + 1) / 2`,
  the length `ident_mat_low_tri`/`ldlt_decomp` produce.

## Architecture decisions (L06)

- `ldlt_decomp` is specified by one entry predicate, `ldlt_entry_ok(a, l, d, dim, r, c)`
  (`src/lin_alg/ldlt_spec.rs`): `d_c L_rc + sum_{k < c} d_k L_rk L_ck == A_rc` for `c <= r`.
  Off the diagonal it is the division step of the algorithm (`d_c != 0` from the pivot
  condition), on the diagonal (with `L_rr = 1`) it is the pivot formula, and with `L_cc = 1`
  it is entry (r, c) of `L D L^T` (`lemma_ldlt_product_entry`). The executable loops carry
  `rv(sum) == ldlt_sum_r(.., k)` and the set of pairs whose recurrence already holds.
- Writes to `L` are handled by locality: `lemma_lt_entry_write_other` (distinct packed
  indices, via `lemma_lt_idx_injective`) and `lemma_ldlt_entry_ok_write` (an entry whose
  recurrence does not read position `(i, j)` keeps it); pushes to `d` by
  `lemma_ldlt_entry_ok_push`. The "sequence agrees except at one index" hypothesis is stated
  directly on `Seq<real>`, so no `update`-based reasoning about `rvs` is needed at call sites.
- The first loop of `ldlt_decomp` rewrites the diagonal ones that `ident_mat_low_tri` already
  produced; its invariant is `is_lt_ident_r` and the write is a no-op on the real view
  (`rvs(l@) =~= prev`).
- `ldlt_decomp` returns `Option`; the postcondition is stated with
  `result matches Some((l, d)) ==> {..}`, so nothing is claimed for `None`. The pivot
  threshold `TOL * max(max_i |A_ii|, 1)` reuses `max_abs_diag_r` (`max_diag_sym` is exact for
  `dim >= 1`, and the loop only checks pivots when `dim >= 1`).
- `pos_def_solver` composes the L05 solver contracts with `lemma_mat_vec_mul` and
  `lemma_mul_assoc_meq`: `A x = L (D (L^T x)) = b`. The block-diagonal solver is specified with
  `block_ident_view(A, n)` (`mat_spec.rs`) and `lemma_block_ident_solve`; the head of `vect`
  is passed as `&mut vect.as_mut_slice()[..ul_len]` (vstd specifies `index_mut` with a
  `RangeTo` on slices, not on `Vec`), which is free at runtime.

## Architecture decisions (L07)

- The guarantee of a successful check is stated per block as `block_psd_ok(A, start, dim)`
  (`src/lin_alg/ldlt_spec.rs`): `psd_shifted(block_view(A, start), dim, eps)` with
  `eps = psd_check_eps_r(A, start, dim) = 1e-9 * max(block_max_abs_diag_r, 1)`, all in terms
  of the full packed matrix, so L18 can use it without reference to the temporary block
  copies. The block list is constrained by `blocks_fit(blocks@, mat@.len())`, which the
  `psd_blocks` construction of L17 must establish.
- The proof composes the existing contracts: `sym_matrix_diag_block` (block entries),
  `max_diag_sym` (the scale, exact for `dim >= 1`), the packed shift loop (entry invariants:
  off-diagonal entries copied, diagonal entries plus `eps`), and `ldlt_decomp`
  (`L D L^T = B + eps I` with pivots `>= TOL > 0`); the mathematical step is
  `lemma_shifted_factorization_psd` in `mat_spec.rs`, built from `lemma_diag_psd`,
  `lemma_psd_congruence` and `lemma_psd_shifted_iff` (L03). For `dim == 0` the scale is
  unspecified and the guarantee is `lemma_psd_shifted_zero_dim`.
- `block_max_abs_diag_r` is defined through the named per-term function `block_diag_abs_r`
  so that its `max0_fn_r` closure can be matched from proofs (closure-matching rule of L04).

## Architecture decisions (L08)

- The rotation is specified by its matrix: `rot_mat(j, k, c, s)` is the identity except for
  the 2 x 2 block `[[c, s], [-s, c]]` at rows/columns `j < k` (`src/lin_alg/jacobi_spec.rs`).
  `jacobi_rot` ensures `exists c, s: jacobi_step(..)`, i.e. `c^2 + s^2 = 1`, `A' = J^T A J`
  and `Q' = Q J` as block equalities (`meq`) of `sym_view`/`cm_view`. The existential is the
  honest statement: `signum(0)` is unspecified in the model (T6), so `c, s` are not
  functions of the input. Everything the eigensolvers need follows from `jacobi_step`
  (`lemma_jacobi_step_preserves`, `lemma_jacobi_step_similar`), so L09-L11 never unfold the
  rotation.
- `rot_entry` writes entry `(r, l)` of `J^T A J` exactly as the products
  `(J^T (A J))(r, l)` compute it (nested, not expanded), so `lemma_rot_congruence` is pure
  sum manipulation (`lemma_sum_fn_two`, `lemma_sum_fn_single`) and all ring reasoning is
  concentrated in `lemma_jacobi_scalars`.
- The executable `t = sign(tau) / (|tau| + sqrt(1 + tau^2))` is characterized by the
  quadratic `t^2 + 2 tau t - 1 = 0` (`lemma_jacobi_t`); multiplied by `a_jk` it becomes
  `a_jk t^2 + (a_kk - a_jj) t - a_jk = 0`, the hypothesis of `lemma_jacobi_scalars`, which
  gives the diagonal updates `a_jj - t a_jk`, `a_kk + t a_jk` and `A'_jk = 0`. The division
  by `2 a_jk` is the reason for the nonzero-pivot precondition (both callers check it).
- Sweep invariants are stated on `sym_entry`/`cm_entry` for all index pairs (not on packed
  positions): `rot_done(j, k, m, r, l)` says which entries already hold `rot_target` after
  rows `< m` are processed; single-entry writes are propagated with
  `lemma_sym_entry_write_other`/`lemma_cm_entry_write_other` (packed/column-major index
  injectivity). Reuse this pattern for the eigensolver bookkeeping loops.
- The tuple-of-`&mut` return type of `jacobi_rot` is kept; the callers' reborrow loop
  (`(a, q, _) = jacobi_rot(a, q, ..)`) is verifiable with the invariant
  `final(x)@ == final(p)@` linking the moving reference to the parameter (probed).

## Architecture decisions (L09)

- The eigensolver loops are specified by one invariant on the stored representations,
  `eigen_decomp_inv(a, q, b, dim)` (`src/lin_alg/eigen_spec.rs`): `cm_view(q)` orthogonal and
  `sym_view(a) = Q sym_view(b) Q^T` on the block. Its three lemmas are the only facts L10 and
  L11 need about the rotation sequence: `lemma_eigen_inv_init` for `ident_mat` and the input,
  `lemma_eigen_inv_step` consuming the existential postcondition of `jacobi_rot` directly
  (`choose` on `c, s`), and `lemma_eigen_inv_eigenvalues` for the similarity invariance.
  Symmetry of `B` needs no bookkeeping (`sym_view` is symmetric by construction).
- The stopping criterion is the predicate `offdiag_below(b, dim, thr)` (every off-diagonal
  magnitude `< thr`), not a fold-computed maximum: the cached solver justifies it through
  its cache invariant (the pivot is the global maximum), the cyclic solver through its final
  scan, and both then use `lemma_converged_approx`. The threshold `jacobi_thr_r` reads the
  executable check `scaled_near_zero(x, 1.0.max(max_diag), 1e-12)` as `|x| < 1e-12 *
  max_r(1, max_abs_diag_r(b, dim))`; `max_abs_diag_r` is the fold the solvers compute (exact
  for `dim >= 1`).
- The approximate-eigendecomposition bound is stated as `offdiag_norm_r(b, dim) =
  sqrt(frob_f(B - diag(B), B - diag(B))) <= jacobi_eps_r = sqrt(dim (dim - 1)) * thr`, with
  `B - diag(B)` the index function `moffdiag(sym_view(b)) = msub(B, mdiag(B))` so that L12 can
  write `B = diag(B) + (B - diag(B))` in the matrix algebra; `lemma_mdiag_view` identifies
  `mdiag(B)` with `diag_view(sym_diag_r(b, dim))`, the eigenvalue vector the solvers return.
  Nothing claims that `B` is diagonal. `dim * (dim - 1)` enters as the real
  `offdiag_count_r(dim)` (`int as real` is supported by this Verus build and interpreted as
  the embedding).
- `eigensolver_ok(a, b, q, vals, dim)` is the postcondition both `jacobi_eigen_cach` and
  `jacobi_eigen_cyc` must ensure (with `a = rvs(old(a)@)`, `b = rvs(final(a)@)`): it is the
  common eigensolver interface used by L12.

## Architecture decisions (L17)

- **Bug found and fixed** (details in `EXPOSED_BUGS.md`): for a problem without LMIs the
  block list `psd_blocks` was empty, so the termination test never checked `Z >= 0`. The
  partition is now `psd_partition(bd, vd, d)`: `block_sizes(bd, vd)` (L16) when the total LMI
  block size is positive, else the single block `[d]` (the test `block_size_sum == 0` matches
  the author's fix on `main`); `psd_block_list` lists `(sum_upto(bs, j), bs[j])` and is
  computed once before the loop (the original rebuilt the same list every iteration with an
  iterator chain).
- `sdpad` is split along the DAG layers: the setup is the verified `sdpad_setup`, returning the
  public record `SdpadSetup` (the former locals), and the equality-system solve of each
  iteration is the verified `dual_solve`; `sdpad` itself remains outside `verus!` until the
  iteration (L18) and termination (L19) are verified, and calls the verified helpers. The
  contract `sdpad_setup_ok` carries the standard form through an existential packed objective
  `obj0` (the vector before `scale_off_diag` rewrote it in place), so that L18/L19 can use the
  objective identity `<obj_frob, z>_F = <obj0, z> = sum_n C_n(Z_n)` via `lemma_frob_half_dot`
  and `std_form_ok`.
- The solver's precondition `sdpad_input_ok` adds `dim_fits(total_rows(sdp))` to the
  conversion's requirements: `constraint_gram` and the solvers need `dim_fits(c_len)`, and
  `c_len <= total_rows` is all the reduction contract gives (`lemma_reduced_rows`); the bound
  is not derivable from the `usize` fit of the stacked matrix alone.
- `dual_solve` states what both branches establish uniformly as `(G + s I) y_top = b_top`
  with the shift `s = reg_shift_r` (`0` or `TOL * max(max_i |G_ii|, 1)`), extracted from the
  block-solve contract of `pos_def_solver_upperleft` by `lemma_block_ident_solve_inv`; the
  residual test is reported exactly as the executable check reads
  (`relative_residual_is_small`), i.e. only when the denominator is nonzero. The right-hand
  side is saved before the solve (`slice_to_vec(&rhs[..c_len])`), as the original did, because
  the forward substitution overwrites it.
- **Second executable change:** without constraint reduction and with `c_len == 0`, the
  original left `dual_y` empty although `point` may hold zero-constraint rows; `dual_solve`
  now runs the block solve in that case too (it returns `b_bottom`), so `dual_y.len() ==
  point.len()` always holds (needed by `dot_prod(&point, &dual_y)` in L19). The case cannot
  arise for the standard forms produced (LMI rows are never all eliminated; without LMIs there
  are no zero constraints), so no test is affected.
- The regularization loop writes `gram_reg[(i * i + 3 * i) / 2] += shift` with the invariant
  "diagonal entries `< i` shifted, everything else equal to `G`" (`diag_shifted` after the
  loop), and `lemma_diag_shifted_view` turns it into `G + s I` on the block.

## Architecture decisions (L12)

- The spectral operations are specified through the eigensolver interface only
  (`src/lin_alg/spectral_spec.rs`): `spectral_part_ok(A, B, q, sel, vals, vects, dim)` is
  `eigensolver_ok` (L09) plus the selection `selected_pairs`, which says that `(vals, vects)` lists
  the pairs `(B_kk, q e_k)` for `k` in `sel_indices(sel, dim)`, the increasing list of indices
  passing the test `sel`. The accumulated `q` is not returned by the executable functions, so it
  is existential in the postcondition; `B` is the final content of the input vector, hence
  `diag(B)`, the eigenvalue vector, is nameable by the caller (L18 needs the thresholds on it).
- The selection tests are the spec closures `pos_sel(vals) = (vals[k] > 0)` and
  `neg_sel(vals, scale) = (vals[k] < -1e-8 * scale)`, returned by named spec functions so that
  they match across call sites (closure-matching rule of L04); the implementation-specific
  `eig_sel(vals, negative, scale)` of `jacobi.rs` dispatches between them for the shared loop.
  The scale of the negative part is `pivot_scale_r` of the *input* matrix (the original fold
  over `a` before the eigensolver overwrites it), computed by the L05 helper `pivot_scale`.
- What the reconstruction is: `lemma_selected_reconstruction` identifies the matrix
  `Q_sel D_sel Q_sel^T` with inner dimension `m` (the `diag_scale_gram` contract of L04) with
  `Q diag(masked_r(vals, sel)) Q^T` with inner dimension `dim`, where `masked_r` zeroes the
  unselected values; the proof is one sum identity (`lemma_sum_selected`) applied entrywise
  through `lemma_mul_diag_transpose`. This is the "spectral operation actually computed":
  `S = Q diag(max-selected B_kk) Q^T` with `A = Q B Q^T` and `||B - diag(B)||_F <= eps`; no claim
  relates it to the exact PSD projection of `A` (CLAUDE.md, section 7).
- PSD facts: `lemma_pos_part_psd` (positive part, via `lemma_diag_congruence_psd`); for PSD
  detection (`negeigendecomp(..).0.is_empty()` in the tests), `lemma_no_negative_part` gives
  `B_kk >= -1e-8 * scale` for every `k` and `Q diag(B) Q^T >= -1e-8 * scale I` (the new
  `lemma_diag_congruence_psd_shifted`, proved through `|Q^T x| = |x|`); `A` itself differs from
  `Q diag(B) Q^T` by `Q (B - diag(B)) Q^T`, which is only bounded in Frobenius norm.
- The two near-identical selection loops of the original are one helper `select_eigenpairs`
  with a `negative` flag (one branch per entry, same pushes); column `k` of `vects` is appended
  with `extend_from_slice(&vects[lo..lo + dim])`, `lo = k * dim`, as before.

## Architecture decisions (L16)

- The contract is stated on the user data directly (`SDP` fields made `pub`; Verus treats a
  datatype with private fields as opaque outside its module, so public specs cannot mention
  them). All user maps are read through `part_r`/`symrow_r`: the packed vector of a partitioned
  matrix (entry `t` is `v[c][r]` for `(r, c) = pk_unpack(t)`), applied by the packed dot product
  `dot_r`. No conversion of the user data is defined beyond this view.
- Standard-form facts are functional, quantified over the variable `z` (a packed `sd x sd`
  matrix): `<obj, z> = sum_n C_n(Z_n)` (`obj_r`), LMI row `(i, e)` reads
  `sum_n L(i)_n[e](Z_n) - z[bo_i + lo(e), bo_i + hi(e)]` with right-hand side `M_i[e]`, and
  equality row `e` reads `sum_n A_n[e](Z_n)` with right-hand side `b[e]` (`std_rows_ok`), where
  `Z_n = zvar_r(z, var_off(n), vd[n])` is the difference of the two blocks of variable `n`.
  These follow from the entrywise contracts of L15's block embeddings through one identity,
  `lemma_embed_dot`, carried through the accumulation loops with the linearity lemmas (no
  entrywise description of the final matrices is exported).
- The layout is a function of the block sizes `bd` (from `isqrt_r` of the constant-matrix
  lengths) and the variable orders `vd` (columns of the first row of the first LMI's maps):
  `block_off`, `var_off(bsum, vd, n) = bsum + 2 sum(vd[..n])`, `std_dim`, `row_off`,
  `lmi_rows`; the returned `block_dims`/`symm_dims` equal them (`dims_of`).
- The zero pattern is specified as a set with positions: every listed pair is upper
  off-block-diagonal (`in_zero_region`, with the block list `block_sizes(bd, vd)`), every such
  pair is listed (`listed`), no pair twice, and the length is `pk_len(sd) - sum_j pk_len(bs[j])`
  (`zeros_len`, by `lemma_pk_len_diff` block by block). The original runtime `assert_eq!`s on
  `start` and `zeros.len()` became static assertions.
- The constraint reduction is exported existentially: the result together with the head of
  `point` (`result_aug`) is `reduced_from(aug, ..)` for an augmented matrix `aug` satisfying
  `std_rows_ok` (resp. `single_rows_ok`); without `check_constraints` it equals `aug`. The
  nonzero-input precondition of `linear_remove_redundant_sym` is discharged from the `-1` entry
  of the first LMI row (always present with LMIs); without LMIs it is the API precondition
  `single_constraint_nonzero` (an all-zero constraint system is rejected, as in L14).
- Helpers were split out of the 200-line original for separate proofs (`part_concat`,
  `lmi_dims`, `maps_sum`, `lmi_block`, `objective_sum`, `zero_positions`, `stack_blocks`,
  `standard_single`, `standard_lmi`); the executable work is unchanged except that
  `isqrt` keeps its documented contract (T9), the objective/constraint accumulation and the two
  pivot-free loops are index loops, and `with_capacity(len * (d (d + 1)) / 2)` halves first.
- `maps_sum` handles maps with zero rows (an empty right-hand side): `symmap_dim` is then 0 and
  every row fact is vacuous; the flatten and embedding calls still type-check through
  `lemma_mod_multiples_basic`/`lemma_div_multiples_vanish` (vstd `div_mod`).

## Architecture decisions (L11)

- "Every pair visited exactly once in column-cyclic order" is stated on a ghost sequence of the
  examined positions (`visited`, pushed before the zero test, so skipped and rotated positions
  are both visits): a sweep produces exactly `cyc_order(dim)` (`eigen_spec.rs`), and
  `visits_each_pair_once(cyc_order(dim), dim)` says that each pair `j < k < dim` sits at
  position `cyc_pos(j, k)` and that nothing else is listed. The order is fixed by the
  positions, not just by membership.
- The sweep's rotations reuse the L09/L10 machinery unchanged (`lemma_eigen_inv_step` per
  rotation); no cache reasoning is needed. The convergence scan is specified directly
  (`|B_rc| <= max_offdiag` for `r < c`), and `max_diag`, which the scan folds from `|B_00|`
  over all `c`, is `max_r(f(0), max0_fn_r(f, c))` during the scan and `max_abs_diag_r` at the
  end (`lemma_max0_fn_bound`).
- The two eigensolvers share the loop skeleton of L10 (`loop` with `ensures`, owned `q0`
  returned after `break`, `abort` on non-convergence).

## Architecture decisions (L10)

- **Two bugs found and fixed** (details in `EXPOSED_BUGS.md`). (1) `biggest`, the bound used
  by the cache update, lagged one rotation behind: it is now the value returned by the last
  `jacobi_rot` call (`biggest_new` and the `eigenvals`/`eigenvects` swap pair are gone; the
  loop reassigns `vals`, `vects`, `biggest` from the returned tuple). (2) The last update
  branch read `A[m(m+1)/2 + k]` also when `k > m`, an index outside column `m`, and could
  store a row index `>= m` (and later call `jacobi_rot` with `j > k`, which crashes the
  original rotation). The branch is now guarded: `k < m` compares the two rotated entries as
  before; `j < m < k` replaces the cached row by `j` only if it now exceeds the cached entry;
  `m < j` keeps the cache.
- Cache specification (`jacobi.rs`): `col_max_at(B, c, i)` (row `i < c` has maximal magnitude
  in column `c`) and `cache_ok(B, maxima, dim)` (`maxima[t]` is such a row of column `t + 1`,
  trigger on `maxima[t]`, since `maxima[c - 1]` contains arithmetic). The pivot is specified by
  `offdiag_max_at(B, dim, j, k)` (`eigen_spec.rs`), from which the stopping test gives
  `offdiag_below` (`lemma_offdiag_max_below`) and then `lemma_converged_approx` of L09.
- The two identical pivot searches of the original (before the loop and inside it) are one
  helper `cached_pivot_col` (first column attaining the maximal cached magnitude; the in-loop
  `m2 == 1 ||` initialization is replaced by starting from column 1, same result). The `first1`
  and `first2` flags are kept; the loop invariant records `first1 == first2`.
- Main loop invariant: the cache, the pivot `(j, k)` and `max_diag` describe `rvs(vals@)` while
  `first1` holds, and the ghost matrix `prev` (the matrix before the last rotation) afterwards,
  together with `biggest == |prev_jk|` and `same_outside(prev, cur, dim, j, k)` from
  `lemma_jacobi_step_same_outside`. These conditional clauses are `invariant_except_break`;
  the `ensures` of the loop exports `jacobi_converged` and the decomposition at `break`.
- The cache update loop proves `col_max_at(cur, m, maxima[m - 1])` column by column with
  `lemma_col_unrotated` (entries of column `m` outside rows `j`, `k` are unchanged and bounded by
  the old cached entry, itself bounded by `biggest`); each of the six branches closes with one
  `assert forall` over the rows of the column.
- The eigenvector matrix is an owned local `q0` borrowed by `vects` for the loop; after the
  loop `q0` is returned directly (the original copied it with `to_vec()`), using the ghost
  `qf == final(vects)@` carried through the reborrow chain.
- `max_rotations = (dim (dim - 1) / 2).saturating_mul(MAX_SWEEPS)` (vstd contract) replaces
  `MAX_SWEEPS * dim * (dim - 1) / 2`, which could overflow for large `dim`; the same number of
  rotations is allowed whenever the product fits.

## Architecture decisions (L14)

- **Bug found and fixed.** The row-norm loop of `linear_remove_redundant_sym` walks the packed
  columns `k = 1..dim`, adding the off-diagonal entries `start..diag` with weight 1/2 and the
  diagonal entry `diag = pk_idx(k, k)` with weight 1, and then advanced `start += k`. The
  correct start of column `k + 1` is `pk_col_start(k + 1) = diag + 1 = start + k + 1`, so from
  `k = 2` on the loop restarted `k - 1` entries too early and counted the last `k - 1` packed
  entries of column `k - 1` (its diagonal entry and all but its first off-diagonal entry) a
  second time, with weight 1/2. The computed scale was therefore not the intended packed
  Frobenius norm (`dim = 3`: `e_2^2` entered as `1.5 e_2^2`). The fix is the one statement
  `start = diag + 1`; the solution-set guarantees hold for any positive scale, so only the
  conditioning of the reduced system changes. `cargo test` passes with the fix.
- **Second change (made by the author, 2026-10-18; not a verification finding).** The off-diagonal entries were
  multiplied by `FRAC_1_SQRT_2` before the `hypot` fold, halving their squares; the intended
  scale is the Frobenius norm of the symmetric matrix, in which off-diagonal packed entries
  count twice. The factor is now `2.0_f64.sqrt()` (bitwise equal to `std::f64::consts::SQRT_2`,
  and covered by the existing `sqrt` contract instead of a trusted constant wrapper).
- The row norm is specified as `row_scale_r(e, rd, dim, i) = sqrt(row_sq_r)` with
  `row_sq_r = tri_sum_r(pk_wt(r, c) * M_i[r, c]^2, dim)`, i.e. `frob_r(M_i, M_i, dim)` for the
  packed row `packed_row_r` (`lemma_row_sq_frob`, via `lemma_frob_tri` of L04). The `hypot`
  fold is tracked by `rv(scale)^2 == row_sq_upto_r(k) + col_sq_r(k, r)` (columns `< k`
  complete, rows `< r` of column `k`), with `(x f)^2 = 2 x^2` for `f = sqrt(2)`.
- The result predicate `reduced_with(a0, e, res, dim, pivots)` (`reduce_spec.rs`) re-exports
  the `gauss_elim` facts on the echelon form `e` together with the row scaling, under one
  existential pivot sequence (`reduced_system`, triggered on `pivots.len()`). What the
  reduction guarantees is proved as lemmas over this predicate, not as postconditions:
  input solutions solve the reduced system exactly, and reduced solutions satisfy the kept
  echelon rows exactly and every dropped echelon row `i >= rank` up to `|row_i . x - b_i|
  <= 2^rank * eps * (1 + ||x||_1)` (dropped rows are exactly zero in pivot columns and below
  `skip_bound <= 2^rank * eps` elsewhere, including the right-hand side). No claim relates
  numerical rank to exact rank, and no bound is claimed for the dropped rows of the *input*
  (that would need the inverse row operations).
- Positivity of the scale (and hence the division contract T3) follows from the echelon
  form: the pivot of row `i < rank` has magnitude `>= eps > 0` and sits in the packed part,
  because the consistency flag of `gauss_elim` says the last pivot column is not the
  right-hand side and pivot columns increase. Hence the precondition `abs_max_r(input) > 0`
  (an all-zero input gives `eps = 0`, zero pivots and `0 / 0` rows in the original code).
- The `assert!(scale.is_finite(), ..)` of the original code is kept as a data-dependent abort
  (T8): `is_finite` has no exact-real meaning (T6).

## Architecture decisions (L13)

- `gauss_elim` is specified by two exact facts and one tolerance fact
  (`src/lin_alg/gauss_spec.rs`). Exact: `same_solutions(input, output)`, i.e. for every `x`
  of length `col_dim - 1`, `A x = b` holds for the input iff it holds for the output (each
  swap and each row subtraction is proved to preserve it); and `echelon_form(output, pivots,
  eps)` for a ghost pivot sequence (exported existentially). Tolerance: `skipped_bounded`
  records what the skip rule `|max remaining entry| < TOL * scale` implies for the output:
  the entries of a skipped column grow by at most a factor 2 per later elimination step,
  because partial pivoting keeps every multiplier `|A_jn / A_rn| <= 1`. Numerical rank is
  not related to exact rank (CLAUDE.md), and rows `>= rank` are not claimed to be zero:
  they are exactly zero in pivot columns and bounded by `2^k * eps` in skipped columns.
- The pivot sequence is ghost state (`let ghost mut pivots: Seq<int>`, pushed in proof
  blocks); `pivots_below(pivots, c)` recovers the row index at which column `c` was
  examined, so no second ghost map is needed. The consistency flag is specified through the
  pivot sequence: `coeff_rank == row_search - [last pivot is the last column]` is the loop
  invariant.
- Loop invariants describe whole-matrix states with `cm_entry` and an index-conditional
  (`if c < m3 { swapped } else { unchanged }`); single writes are propagated with
  `lemma_cm_entry_write_other`, then the row-operation predicate is asserted once and the
  three preservation lemmas (solution set, echelon, skip bound) are applied.
- When the pivot is zero (`scale == 0`, all-zero input, never skipped) the multiplier is
  `0 / 0`, unspecified in the model; the annihilation `A_jn - piv * A_rn = 0` still holds
  because `A_jn = 0` by the maximality of the pivot, and the skip bound is vacuous.

## Architecture decisions (L15)

- Partitioned symmetric matrices (`Vec<Vector>`, column `c` of the upper triangle stored
  as a vector of length `c + 1`) are specified entrywise: `part_shape(v, dim)` fixes the
  lengths and postconditions state `rv(v@[c]@[r])` for `r <= c < dim`; `symmap_shape`
  lifts this to the rows of a `MatrixSym`. No conversion function between the partitioned
  and packed forms is defined; the bridge is the index `pk_idx(r, c)` in the postconditions
  of `mk_symmap_real`, `mk_symmap` and `matrixsym_flatten`.
- The block compositions are specified by the packed index map of the block:
  `is_block_idx(t, start, dim)` says that `t = pk_idx(start + a, start + b)` for some block
  entry `(a, b)`; the loops over `sym_matrix_diag_block_map` carry the ghost pair `(ga, gb)`
  of the current block entry (`i == pk_idx(ga, gb)`), and the zero-fill between consecutive
  block columns is justified by `lemma_block_idx_gap` through `block_prev_end` (the packed
  index following the previous block entry). `matrix_diagonal_mult_sym` keeps its early
  return for `dim == 0` (empty result, stated as such); for `dim >= 1` it requires the dense
  input length to be a multiple of `pk_len(dim)` and the result length to fit in `usize`.
- `sym_matrix_proj` keeps its runtime check `row > col` (user input) as an `abort` (T8) and
  requires `col < dim` (otherwise the original code returns a vector of the wrong length).

## Constructs rewritten for Verus compatibility

| Construct | Replacement | Where |
|-----------|-------------|-------|
| `-(expr)` on `f64` (unsupported by Verus) | `crate::trusted::neg(expr)` (inline wrapper) | `is_significantly_negative`, `negscal_vect` |
| `iter().zip(..)` / `for v in vect` loops | index loops `for i in 0..n` (same cost) | `dot_prod`, `euclid_distance`, `vect_add`, `vect_subt`, `scal_vect`, `negscal_vect`, `vect_add_scaled`, `matrix_add_diff`, `vector_equality` |
| `Vec::extend(vec)` (no vstd spec) | `Vec::append(&mut vec)` (same memcpy) | `ident_mat` |
| imports of spec-only items / `vstd::std_specs` (erased under `cargo`) | gated with `#[cfg(verus_keep_ghost)]` | `trusted.rs`, `real_spec.rs`, `comparison.rs`, `matrix.rs`, `sym_matrix.rs` |
| `x += y`, `v[i] += y`, `v[i] *= y` on `f64` (compiler crash) | `x = x + y`, `v[i] = v[i] + y`, `v[i] = v[i] * y` | `frob_prod_sym`, `sym_mat_one_norm`, `sym_mat_act`, `scale_off_diag` |
| `iter().fold(0.0, max)` | index loop | `sym_mat_one_norm` |
| tuple pattern in a parameter `(mat, list): &(&Matrix, &Vec<..>)` (rejected by the `verus!` macro) | `constr: &(..)` plus `let (mat, list) = constr;` (call sites unchanged) | `constraint_action`, `constraint_lin_comb` |
| `slice.to_vec()` (no vstd spec) | `vstd::slice::slice_to_vec(slice)` (vstd's wrapper around `to_vec`) | `constraint_action`, `constraint_gram`, `diag_scale_gram` |
| `chunks_exact(..).enumerate().skip(1)`, `iter_mut().zip(..)`, `d.iter().zip(q.chunks_exact(len))`, `for (r, c) in *list` | index loops over the same slices (same cost) | `constraint_action`, `constraint_lin_comb`, `constraint_gram`, `diag_scale_gram` |
| `dot_prod(col, vect)` with `vect` longer than `col` (silent `zip` truncation) | `dot_prod(col, &vect[..row_dim])` (free slice) | `constraint_lin_comb` |
| `pub const TOL` declared outside `verus!` (Verus cannot read an external const) | same declaration moved inside the `verus!` block of `matrix.rs` | `TOL` |
| `iter().map(abs).fold(0.0, f64::max).max(1.0)` | helper `pivot_scale` (index loop with `max.max(..)`) | `diag_solver`, `forw_subst`, `back_subst` |
| `for (v, d) in vect.iter().zip(diag.iter())` | `for i in 0..vect.len()` with equal-length precondition | `diag_solver` |
| `panic!("...: division by zero")` | `crate::trusted::abort(..)` (T8) | `diag_solver`, `forw_subst`, `back_subst` |
| `vect[i] -= ..` on `&mut [f64]` | `vect[i] = vect[i] - ..` | `forw_subst`, `back_subst` |
| `let Some(..) = .. else { panic!(..) }` | same let-else with `crate::trusted::abort(..)` (T8) | `pos_def_solver` |
| `&mut vect[..ul_len]` on a `Vec` (no vstd spec for `Vec` range `index_mut`) | `&mut vect.as_mut_slice()[..ul_len]` (same pointer arithmetic) | `pos_def_solver_upperleft` |
| `for v in &vect[ul_len..] { result.push(*v) }` | `for t in ul_len..vect.len() { result.push(vect[t]) }` | `pos_def_solver_upperleft` |
| `for (start, dim) in blocks` over `&Vec<(usize, usize)>` | `for b in 0..blocks.len() { let (start, dim) = blocks[b]; .. }` | `psd_block_check` |
| `for _ in 0..c` | `for _r in 0..c` (the counter appears only in the invariant) | `psd_block_check` |
| `let (mut eps, mut idx): (f64, usize);` declared before the loop | `let eps = ..; let mut idx = 0;` inside the loop body (same cost) | `psd_block_check` |
| `mat[i] += x`, `mat[i] -= x` on `f64` | `mat[i] = mat[i] + x`, `mat[i] = mat[i] - x` | `jacobi_rot` |
| `tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt())` and `(1.0 + t * t).sqrt().recip()` | the same expressions with the intermediate results bound by `let` (`sigma`, `sq`, `sq2`), so the method contracts can be named in the proof | `jacobi_rot` |
| `if m == j \|\| m == k { continue; }` (`continue` is unsupported in `for` loops) | `if m != j && m != k { .. }` | `jacobi_rot` |
| `let (mut val_old_k, ..)` declared before the loop | `let` inside the loop body | `jacobi_rot` |
| `orth.split_at_mut(k * dim)`, `&mut left[j * dim..(j + 1) * dim]`, `iter_mut().zip(..)` | index loop over `orth[j * dim + m]`, `orth[k * dim + m]` (same cost) | `jacobi_rot` |
| `for n in 0..col_dim { if row_search >= row_dim { break; } .. continue; .. }` | `while n < col_dim && row_search < row_dim { .. if !skip { .. } n += 1; }` (same iterations; `continue` is unsupported) | `gauss_elim` |
| `mat.swap(i1, i2)` (no vstd spec) | `let tmp = mat[i1]; mat[i1] = mat[i2]; mat[i2] = tmp;` | `gauss_elim` |
| `mat.iter().map(abs).fold(0.0, f64::max)` | index loop with `max.max(mat[t].abs())` | `gauss_elim` |
| `mat[i] -= piv * mat[k]` | `mat[i] = mat[i] - piv * mat[k]` | `gauss_elim` |
| `let (mut row_search, mut coeff_rank, mut pivot_row);` | separate initialized `let mut` bindings | `gauss_elim` |
| `(mat, row_search, row_search == coeff_rank)` returned directly | bound to `let res = ..` first, so the final proof block can name `res.1`, `res.2` | `gauss_elim` |
| `for c in ones { .. }`, `for v in vect`, `for row in mat`, `iter().zip(..)` | index loops | `matrix_diagonal_mult_sym`, `mat_sym_diagonal_mult_sym`, `sym_matrix_scale`, `sym_matrix_add`, `matrixsym_flatten` |
| `result.extend(vec![0.0; row_dim])` (no vstd spec for `extend`) | `result.append(&mut vec![0.0; row_dim])` (same memcpy) | `matrix_diagonal_mult_sym` |
| `v[a..=b].to_vec()` | `slice_to_vec(&v[a..=b])` (vstd's wrapper; `..=` ranges are supported) | `mk_symmap_real`, `mk_symmap` |
| `for _ in offset..c`, `for _ in 0..output_size` | `for _z in ..`, `for _k in ..` (the counter appears in the invariant) | block compositions, `mk_symmap`, `zero_symmap` |
| `panic!("sym_matrix_proj: ..")` | `crate::trusted::abort(..)` (T8) | `sym_matrix_proj` |
| `use std::f64::consts::FRAC_1_SQRT_2` and `x * FRAC_1_SQRT_2` (wrong weight, see L14 decisions) | `let f = 2.0_f64.sqrt();` once per call (`sqrt` contract, T6), `x * f` | `linear_remove_redundant_sym` |
| `panic!("..inconsistent")`, `assert!(scale.is_finite(), ..)` | `abort(..)` (T8), `if !scale.is_finite() { abort(..) }` | `linear_remove_redundant_sym` |
| `let (mut start, mut scale): (usize, f64);` before the loop | `let mut scale: f64 = 0.0;` and `let mut start = 1;` inside the loop body | `linear_remove_redundant_sym` |
| `scale.hypot(echelon[..] * ..)` | `let v = echelon[..]; let y = v * f; scale = scale.hypot(y);` (named intermediates for the proof) | `linear_remove_redundant_sym` |
| `start += k` (bug, see L14 decisions) | `start = diag + 1` | `linear_remove_redundant_sym` |
| `let (mut diff, mut offset);` | initialized `let` bindings inside the loop | `mk_symmap` |
| `panic!("index_max_in_col : ..")`, `panic!("jacobi_eigen_cach failed to converge after {} sweeps", ..)` | `abort(..)` (T8; the first is unreachable under the precondition `col >= 1`) | `index_max_in_col`, `jacobi_eigen_cach` |
| `for i in &maxima[1..]`, `for i in &mut maxima { .. *i = .. }`, `for i in &maxima` with hand-maintained counters `m1`, `m2` | index loops `for m in 1..dim { let i = maxima[m - 1]; .. maxima[m - 1] = ..; }`; the two pivot searches are the helper `cached_pivot_col` | `jacobi_eigen_cach` |
| `(eigenvals, eigenvects, biggest) = (..)` and `(eigenvals_new, ..) = jacobi_rot(..)` (destructuring assignment is unsupported) | `let (v, q, b) = jacobi_rot(vals, vects, dim, k, j); vals = v; vects = q; biggest = b;` (single reference pair; also the fix of the stale `biggest`) | `jacobi_eigen_cach` |
| `&mut ident_mat(dim)` (borrowed temporary) and `eigenvects_new.to_vec()` | `let mut q0 = ident_mat(dim); let mut vects = &mut q0;` and returning `q0` after the loop (no copy) | `jacobi_eigen_cach` |
| `loop { .. return (..) .. }` | `loop { .. break; .. }` with `ensures`, result built after the loop (the owned `q0` is only available once the borrow has ended) | `jacobi_eigen_cach` |
| `MAX_SWEEPS * dim * (dim - 1) / 2` | `(dim * (dim - 1) / 2).saturating_mul(MAX_SWEEPS)` | `jacobi_eigen_cach` |
| `let mut biggest = f64::NEG_INFINITY` (never read before the first rotation) | `let mut biggest = 0.0` | `jacobi_eigen_cach` |
| `const MAX_SWEEPS` outside `verus!` | moved inside the `verus!` block (still used by `jacobi_eigen_cyc` outside) | `jacobi.rs` |
| `let mut index: usize;` declared before the loop | `let index = ..` inside the loop body | `index_max_in_col` |
| `if vals[..] == 0.0 { continue; }` (unsupported in `for`) | `if vals[..] != 0.0 { .. }` | `jacobi_eigen_cyc` |
| `panic!("jacobi_eigen_cyc failed .."), `return` inside `loop`, `&mut ident_mat(dim)`, `.to_vec()`, destructuring assignments | as for `jacobi_eigen_cach` | `jacobi_eigen_cyc` |
| scan variables `n`, `m` and double reads `vals[..].abs()` in both branches | `c`, `r` (`n` is the ghost dimension) and one `let off = ..` / `let diag = ..` per entry | `jacobi_eigen_cyc` |
| `block_dims.iter().copied().scan(..).chain(symm_dims.iter().copied().scan(..).flatten()).collect()` (per iteration) | the helper `psd_block_list` (two index loops), called once before the loop | `sdpad` |
| `block_dims.iter().sum()` | index loop | `sdpad_setup` |
| `let (c_len, block_size_sum) = (..)`, `(mut prim_z, mut dual_s) = (ident_sym_mat(var_dim), vec![0.0; var_dim_tot])` | the `SdpadSetup` record and the helper `initial_iterates` | `sdpad` |
| `inverse_y_old = (inverse_y[..c_len]).to_vec()`, `gram_small_reg[k] += x`, `panic!(..)` | `slice_to_vec(&rhs[..c_len])`, `gram_reg[k] = old_entry + shift`, `abort(..)` (T8), all inside the helper `dual_solve` | `sdpad` |
| `for (val, vect) in eigenvals.iter().zip(eigenvects.chunks_exact(dim))` (two copies) | the helper `select_eigenpairs` with an index loop `for i in 0..dim` and `extend_from_slice(&vects[lo..lo + dim])` | `poseigendecomp`, `negeigendecomp` |
| `a.iter().map(abs).fold(0.0, f64::max).max(1.0)` | `pivot_scale(a)` (the L05 helper, now `pub(crate)`) | `negeigendecomp` |
| `match var { Cached => .., Cyclic => .. }` with `use JacobiVariant::*` | qualified patterns `JacobiVariant::Cached`; the enum moved inside `verus!` | `poseigendecomp`, `negeigendecomp` |
| `struct SDP` with private fields outside `verus!` | the struct and its `impl` inside `verus!`, fields `pub` | `sdp.rs` |
| `assert!(..)`, `assert_eq!(..)`, `panic!(..)`, `expect(..)`, `unwrap_or_else(|| panic!(..))` | `abort(..)` (T8) on the data-dependent checks; the internal consistency checks (`start == total_dim`, `zeros.len() == zeros_len`) are static `assert`s | `sdp_to_standard` |
| `obj.concat()` (no vstd spec) | `part_concat(obj, dim)` (same push loop) | `sdp_to_standard` |
| `obj.last().expect(..).len()`, `map[0].last().unwrap_or_else(..).len()` | `obj[obj.len() - 1].len()` after an emptiness abort | `sdp_to_standard`, `lmi_dims` |
| `for (maps, mat) in lmis`, `for (n, map) in maps.iter().enumerate()`, `for (n, (obj, constr)) in ..zip(..).enumerate()`, `iter().fold(..)`, `iter().chain(..)`, `iter().map(..).sum()`, `for (mat, vect) in &blocks` | index loops | `sdp_to_standard` and helpers |
| `for ((s, v1), v2) in obj_sum.iter_mut().zip(db1.iter()).zip(db2.iter()) { *s += *v1 - *v2 }` | `for t in 0..total_size { obj_sum[t] = obj_sum[t] + (db1[t] - db2[t]); }` | `objective_sum` |
| `sum[idx] -= 1.0` | `sum[idx] = sum[idx] - 1.0` | `lmi_block` |
| `for _ in 0..2 { .. if count < sequence_len { break; } }` over `block_dims.chain(symm_dims)` | one loop over the block list `j in 0..s + 2 nv` with `k = if j < s { block_dims[j] } else { symm_dims[(j - s) / 2] }` (same pushes in the same order) | `zero_positions` |
| `let (mut block_size, mut symm_size);` and other uninitialized tuple `let`s | initialized bindings inside the loops | `lmi_dims`, `sdp_to_standard` |
| `first` flag and `map_len_old: Option<usize>` | `i == 0` and a `usize` | `lmi_dims` |
| `Vec::with_capacity(mat.len() * (dim_c * (dim_c + 1)) / 2)` | `Vec::with_capacity(mat.len() * ((dim_c * (dim_c + 1)) / 2))` (weaker overflow precondition) | `matrixsym_flatten` |

Known constructs that later layers must rewrite (found during L00 experiments):

- `x += y`, `x -= y`, `x *= y` on `f64` (including `v[i] *= y`) crash this Verus build
  (`get_range Float(64)`); write `x = x + y` (`v[i] = v[i] * y` verifies).
- `f64::INFINITY`, `f64::NEG_INFINITY`: use the wrappers in `src/trusted.rs`; std float
  constants such as `consts::SQRT_2` are unsupported, write `2.0_f64.sqrt()` (same bits).
- `panic!`/`assert!` on data-dependent conditions: use `crate::trusted::abort`.
- Postconditions on `&mut` parameters use the new style: `old(v)` / `final(v)`.
- Iterator chains (`zip`, `map`, `fold`, `chunks_exact`, `scan`, ...) are not
  verifiable; replace with index loops of identical cost when a function is moved
  into `verus!`.
- `vec![x; n]`, `&v[a..b]` slicing, `v.first()`, `Option::unwrap_or`, `for j in 0..=i`
  and `for j in it: (0..n).rev()` loops are supported by vstd (probed 2026-10-17); keep
  such code as written.
- Before rewriting a construct, probe it in a scratch file: every entry in the table above
  was re-confirmed by a probe on 2026-10-17 (unary `-x`, `+=` on `f64`, `INFINITY`,
  `FRAC_1_SQRT_2`, `Vec::extend`, `zip`, `fold` all fail in this build).

Proof-engineering notes (L17; probed 2026-10-19):

- `let ghost obj0: SymMatrix = obj;` snapshots an exec `Vec` as a spec value before an in-place
  update; a tuple of such snapshots can be passed to a predicate expecting the function's
  result tuple (`std_form_ok(sdp, res0, check)`), and tuple projections of the constructed
  tuple reduce.
- A returned `&mut` that is only read in ghost code (`let _obj_ref = scale_off_diag(..); let
  ghost cf = rvs(_obj_ref@);`) must carry a leading underscore, or `cargo` (where ghost code is
  erased) warns about an unused variable; the value of `obj` after the borrow ends is `cf`
  without any extra step.
- `v[k] = v[k] + x` on a `Vec<f64>`: bind the read first (`let old_entry = v[k];`) and state
  `rvs(v@)[k] == rv(old_entry) + rv(x)` explicitly; the untouched entries follow from
  `rvs(v@)[j] == rv(v@[j])` plus the `update` axiom when `j != k` is known via
  `lemma_pk_idx_injective`.
- `sum_upto(s, j + 2)` needs both unfolding steps asserted (`sum_upto(s, j + 1) == .. + s[j]`,
  `sum_upto(s, j + 2) == .. + s[j + 1]`); one level is unfolded automatically.
- A conjunction required by a callee (`part_shape(..) && symm_dims[n] >= 1` for
  `objective_sum`) that was previously found by luck in the long `standard_lmi` body stopped
  being found when unrelated modules were added to the crate; proving it with an explicit
  `assert forall` before the call made the proof stable.

Proof-engineering notes (L12; probed 2026-10-19):

- `-eps * d` and `eps * d` are different monomials for the linear solver; bridge with
  `assert(-eps * d == -(eps * d)) by (nonlinear_arith)` (function applications such as
  `dot_r(x, x)` may be written inside the `nonlinear_arith` assertion).
- After `v.extend_from_slice(&w[lo..lo + n])`, the entries of the real view are related to the
  snapshot `prev = rvs(old(v)@)` by stating `rvs(v@)[j] == prev[j]` (old part) and
  `rvs(v@)[j] == rv(w@[lo + r])` (new part) directly; an f64 equality `v@[j] == w@[..]` is not
  derivable from a snapshot of the real view.
- `match` on a `&Enum` parameter with qualified variant patterns verifies inside `verus!`; the
  enum must be declared inside `verus!` to be used in verified code.
- A spec function returning a closure (`pos_sel(vals)`) is applied as `pos_sel(vals)(k)` and
  reduces; `#[trigger] sel(k)` with `sel` a closure parameter is matched by such applications.

Proof-engineering notes (L16; probed 2026-10-18):

- A `pub fn` contract cannot mention the fields of a struct with private fields ("field
  expression for an opaque datatype"); make the fields `pub` or go through `closed spec`
  views.
- Closures whose body applies arithmetic to the bound variable (`|j| vd[j / 2]`) do not match
  across call sites; wrap the body in a named spec function (`var_block_size`).
- A quantified fact with several conjuncts under one `#[trigger]` is fine to *state*, but when a
  callee's `requires` repeats the same conjunction, prove it as one named predicate
  (`maps_shape_ok`) asserted from separately proved single-conjunct `assert forall`s; the
  conjunction form was not re-derived at the call site.
- Facts about a `Vec` taken from a function's postcondition refer to that SSA version: prove
  what is needed about `mat@` *before* a later `extend_from_slice`, or state it on a ghost
  snapshot (`aug0`), since the old value is no longer nameable afterwards.
- `nonlinear_arith` with integer `/` or `%` hits the rlimit even for `(a * p) % p == 0`; use
  vstd's `arithmetic::div_mod` lemmas (`lemma_mod_multiples_basic`, `lemma_div_multiples_vanish`,
  `lemma_fundamental_div_mod(_converse)`, `lemma_mod_bound`).
- `assert(x) by (nonlinear_arith)` without `requires` sees nothing, not even facts about the
  variables in `x`; always pass the needed hypotheses.
- Very long function bodies exceed the rlimit as a single query; split by control flow
  (`standard_single`/`standard_lmi`) rather than raising the limit.
- `Ghost<T>` parameters are erased by Verus but not by `cargo`, which then reports them unused;
  derive ghost quantities from the data instead (`n_maps`).
- `usize::isqrt` has no vstd spec (T9); `saturating_mul`/`checked_mul` do.

Proof-engineering notes (L11; probed 2026-10-18):

- A ghost `Seq<(int, int)>` of tuples works as loop bookkeeping (`let ghost mut visited`,
  `visited = visited.push((ji, ki))` in a proof block); equality with a `Seq::new` closure
  needs `=~=` (the pushed sequence and `Seq::new(j + 1, ..)` are extensionally equal), and the
  unfolding `cyc_order(k + 1) == cyc_order(k) + Seq::new(k, |j| (j, k + 1 - 1))` must be
  asserted with the definition's own closure body before `visited =~= cyc_order(k + 1)`.
- Facts proved in an outer loop body (`lemma_dim_fits_idx(n, k)`, `pk_len(n) >= 1`) must be
  re-proved inside the inner loop body (loop isolation also applies between nesting levels).

Proof-engineering notes (L10; probed 2026-10-18):

- Destructuring assignment `(a, b, c) = f(..)` is unsupported ("does not yet support
  destructuring assignment"); bind `let (x, y, z) = f(..)` and assign the variables.
  `panic!("..")` with a message is unsupported (`core::panicking::panic_fmt`); use `abort`.
- `loop` requires a `decreases` clause (or `#[verifier::exec_allows_no_decreases_clause]`);
  `loop invariant_except_break .. invariant .. ensures .. decreases ..` is the clause order,
  and the `ensures` facts are available after a `break`.
- An owned local borrowed for the loop (`let mut vects = &mut q0;`) cannot be mentioned in
  ghost code while the borrow is live (E0502 on "(Verus spec q0)"). Carry
  `let ghost qf = final(vects)@;` and the invariant `final(vects)@ == qf` through the
  reborrow chain (`vects = q` after `let (v, q, b) = jacobi_rot(vals, vects, ..)` keeps
  `final(vects)@ == final(q)@`); after the last use of `vects`, `q0@ == qf` holds by
  resolution. Returning `q0` inside the loop does not work (the borrow is still live there);
  `break` and return after the loop.
- `usize::saturating_mul` has a vstd contract (`std_specs/num.rs`); `checked_mul` too.

Proof-engineering notes (L15; probed 2026-10-17):

- In `for x in a..b`, the loop variable is the range iterator's `peek` (vstd
  `std_specs/range.rs`), which is `Some(a + index)` only for `0 <= index <= b - a`; for an
  empty range (`a > b`) it is `None` and `x` is an unconstrained value. So an invariant
  that mentions `x` is checked at entry for arbitrary `x` unless `a <= b` is known: even
  `a <= x` alone fails "before loop" (probed 2026-10-18). Either prove `a <= b` before the
  loop (`lemma_block_prev_end_le`), guard the invariant with `a <= b ==> ..`, or state it
  through the named iterator (`for x in it: a..b`, `it.index()` is 0 at entry) or without
  `x`.
- `extend_from_slice` is specified through `cloned` (vstd), which resolves to equality for
  `f64`; `Vec::append` gives `v@ == old + other@` directly. Entries of the old prefix need the
  index bound (`lemma_cm_idx_bound`) before the `Seq` push/add axioms apply.
- `mat.first().unwrap_or(&vec![]).len()` verifies as written (spec:
  `if len == 0 { 0 } else { v@[0]@.len() }`).
- `x.len()` for a `Vec` must appear inside each loop body that needs `len <= usize::MAX`.

Proof-engineering notes (L14; probed 2026-10-18):

- A private `use crate::lin_alg::matrix::TOL;` in `sym_matrix.rs` shadows the name coming
  from its `pub use crate::lin_alg::matrix::*;` re-export, so downstream modules importing
  `sym_matrix::*` lose `TOL` (a Rust visibility effect, reported by Verus as E0425). Rely on
  the glob re-export instead.
- `let (echelon, rank, solvable) = gauss_elim(mat, ..)` with `echelon: &mut Matrix` only read
  afterwards: `final(mat)@ == final(echelon)@` resolves to the current `echelon@` at the end
  of the function without any extra step (probed), so the postcondition can be stated on
  `rvs(final(mat)@)`. Restate `e == rvs(echelon@)` and `echelon@.len() == len` as loop
  invariants.
- `v[j * n + i]` with `v@.len() == n * m` known only as an int equality reports overflow in
  every loop; carry `len <= usize::MAX` (and `n * m <= len`) as invariants.
- A `choose` over the existential postcondition of `gauss_elim` is found when its body is
  written with exactly the postcondition's argument terms (`rvs(echelon@)`, `row_dim as int`,
  `rv(TOL) * abs_max_r(rvs(old(mat)@))`), not with the ghost aliases.

Proof-engineering notes (L09; probed 2026-10-18):

- `n as real` for `n: int`/`nat` is supported and linear facts transfer (`n >= 2` gives
  `n as real >= 2.0real`); but `(n as real) * x` with `n as real == 0.0real` known only by
  equality is a nonlinear term: close it with `by (nonlinear_arith) requires n as real ==
  0.0real`. The cast of `(n - 1) as nat` must be bridged by an explicit
  `assert((m as real) == (n as real) - 1.0real)` before use in `nonlinear_arith`.

Proof-engineering notes (L13; probed 2026-10-17):

- An `exists` postcondition needs a ground witness term with the *same* function symbol
  and arguments as its trigger, including the result: for a tuple result, bind
  `let res = (..)` before the final proof block and assert the trigger term on `res.0@`,
  `res.1`; `res.k` inside the postcondition is not a ground term, so a witness stated on the
  local variables is not found (probed: fails for `res.1`, passes for `res.0@` by luck).
  Alternatively trigger on a term without the result (`#![trigger p.len()]`).
- `break` is supported in `for` loops (only `continue` is not; `while` loops support
  both), but after a `for` with `break` nothing is known about the exit condition; a
  `while` with the combined condition gives it for free.
- `Vec::swap` has no vstd specification; a three-line swap through a temporary verifies.
- Quantified facts proved before a loop must be restated as invariants with the same
  trigger (`abs_r(cm_entry(swapped, ..)) >= eps` was lost inside the elimination loop).

Proof-engineering notes (L08; probed 2026-10-17):

- `nonlinear_arith` proves a *single* distributive, associative or commutative step
  instantly (`c * (x - y) == c * x - c * y`, `c * (s * x) == c * c * (t * x)` with `s` written
  as `c * t`), but a two-level identity (`c * (c * a - s * b) - s * (c * b - s * d) == ...`)
  or an identity with substitution hypotheses (`requires s == c * t, ..`) runs for minutes.
  Decompose into one-step facts and let the linear solver combine them: it treats
  monomials as atoms, so write the same monomial with the same bracketing everywhere,
  and substitute `s == c * t` by congruence (state the step on `c * (c * t) * x`).
- `let`-bound names are invisible inside `assert .. by (nonlinear_arith)`; write the
  products out (or pass them in `requires`).
- `continue` is not supported in `for` loops (`for-loops do not yet support continue`,
  a front-end error that aborts the whole crate) but is supported in `while` loops
  (probed 2026-10-18). A `for` loop with `continue` can therefore be kept as a `while` loop
  with an explicit counter, or rewritten with an `if` block as in `jacobi_rot`.
- A `Vec` length is only known to fit in `usize` after the term `v.len()` appears
  (`assert(orth.len() == dim * dim)`); otherwise `j * dim` reports possible overflow even
  with `orth@.len() == dim * dim`.
- Reads that precede a write must be related to the loop invariant *before* the write, with
  a ghost snapshot (`assert(sym_entry(prev, m, k) == ..)`), since the invariant's trigger is
  on the entry state.
- Returning `(&'a mut A, &'a mut B, f64)` is supported; the vstd-style postconditions are
  `res.0@ == ..` (value at return) plus `final(mat)@ == final(res.0)@`.

Proof-engineering notes (L07; probed 2026-10-17):

- `ldlt_decomp(..).is_none()` keeps its vstd spec, and the `Some` payload is available in the
  following proof block as `fact->Some_0` (a tuple, destructured with `let (l, d) = ..`),
  so the original control flow needs no let-else (which would leave `l`, `d` unused under
  `cargo` and warn).
- Two facts quantified over the same indices but with different triggers must be stated in
  separate `assert forall` blocks: in a conjunction only the first conjunct's trigger is
  used, and a `meq` goal over the second pair is never instantiated.
- Multiplying by `mscale(eps, mident())` produces `eps * 1.0real` and `eps * 0.0real`; assert
  the entry with the literal product written out and the linear solver closes it.

Proof-engineering notes (L06; probed 2026-10-17):

- `let Some((l, d)) = f() else { abort(..) }` (let-else with a diverging `!` call) and the
  spec form `result matches Some((l, d)) ==> {..}` with a tuple pattern are both supported.
- `&mut v[..k]` on `v: &mut Vec<T>` type-checks but has no vstd postcondition; both
  `&mut v.as_mut_slice()[..k]` and `&mut v[..k]` with `v: &mut [T]` get vstd's `RangeTo`
  `index_mut` contract (`final(v)@ == final(r)@ + old(v)@.subrange(k, len)`).
- Products with a literal factor (`u * v * 1.0real`, `u * v * 0.0real`) and products of three
  factors are not simplified by the linear solver; bind the factors and use
  `by (nonlinear_arith)`.
- `--time` prints no per-function list when no proof exceeds its threshold.

Proof-engineering notes (L05; probed 2026-10-17):

- `for x in it: (0..n).rev()` is supported: `it.index()` counts processed elements and
  `it.seq()` is the reversed range (`it.seq()[k] == n - 1 - k`), both usable in invariants;
  a plain `for j in (0..n).rev()` verifies but gives no exit information. `&mut [f64]`
  parameters support `v[i] = ..` (vstd `index_mut`), `old(v)` in loop invariants, and
  `&Vec<f64>` arguments coerce to `&[f64]` parameters.
- A `const` used inside `verus!` must be declared inside `verus!`; its value is then known
  to the solver (`rv(TOL)` triggers the `1e-10` literal axiom).
- After `v[i] = v[i] - s * m`, matching the invariant's `a(i, j) * x[j]` needs the explicit
  commutativity step `assert(p * q == q * p) by (nonlinear_arith)` on the two `rv` factors.
- Inner loops need the ghost facts `b == rvs(old(vect)@)` and `dim == old(vect)@.len()`
  restated, or `b[i]` has no known length (loop isolation).

Proof-engineering notes (L04, closure matching; probed 2026-10-17):

- A closure literal written in a proof matches the closure inside a spec function's body
  only if the bodies are built from applications of named spec functions, sequence indexing
  and closure applications. A body that compares or combines the bound variable with a
  *captured* scalar (`if k == r {..}`, `if i < row_dim {..}`) does not match across the
  spec/proof boundary (it does match between two closures of the same proof). Hence the
  per-term spec functions `action_entry_r`, `zero_term_r` in `constr_spec.rs`.
- `let`-bound closures are opaque constants: `F(r, c)` inside another closure is not reduced
  to `F`'s body. Unfold a `tri_sum_r(F, n + 1)` as `tri_sum_r(F, n) + sum_fn_r(|r| F(r, n), n + 1)`
  (closure applied to the let-bound name), not with the body written out; bridge to the body
  with `lemma_sum_fn_ext` and the apply fact.
- When instantiating a lemma whose conclusion contains nested closures
  (`lemma_sum_fn_swap`, `lemma_tri_sum_swap`), write the local closures with the same bound
  variable names and the same nesting as the lemma's statement.
- Prove exchange-of-summation lemmas by induction on the outer count with the additivity
  lemmas, not by instantiating an existing swap lemma with a lambda (the instantiated nested
  closures do not match).
- Slices: `&v[lo..lo + n]` with `lo` an executable variable is simpler to verify than
  `&v[j * n..(j + 1) * n]` (one nonlinear fact `j * n + n == (j + 1) * n` suffices, and the
  `subrange` indices are linear).

Proof-engineering notes (L03):

- Associativity/commutativity of real products inside closure applications is not found by
  the linear solver: bind the factors to variables and `assert(x * y * z == z * (x * y)) by
  (nonlinear_arith)`.
- A term such as `mtr(b)(i, k) * mtr(a)(k, j)` is not reduced automatically when it
  appears inside a product; assert the reduced closure applications separately first.
- Two `#[trigger]` marks in one `forall` body form a single conjunctive multi-pattern (both
  terms must be present). For alternatives write `#![trigger a(i, j)] #![trigger b(i, j)]`
  (as in `meq`).
- `dim as nat` for a possibly negative `int` is unspecified, so a `Seq::new(dim as nat, ..)`
  extensionality proof needs `dim >= 0` as a precondition.
- Spec closures returned by spec functions (`mmul(a, b, n)(i, j)`) beta-reduce as expected;
  alpha-equivalent closures at lemma call sites match without extensionality.

Proof-engineering notes (L02):

- A loop whose index arithmetic is quadratic needs `dim_fits` restated in every inner
  invariant and `lemma_dim_fits_idx` called in the body; otherwise Verus reports
  "possible arithmetic overflow" on `start + i`, `idx += 1`.
- `pk_col_start(0) == 0` is not derived automatically before loops (`0 * (0 + 1) / 2`
  stays a nonlinear term); call `lemma_pk_col_start_zero()`.
- Recursive spec functions unfold one level; base cases like `max0_fn_r(f, 1)` need an
  explicit `assert(max0_fn_r(f, 0) == 0)` first.
- `old(..)` cannot appear inside `assert .. by (nonlinear_arith)`; bind the value to a
  ghost variable first.
- A returned `r: &mut T` is specified as in vstd (`vec_index_mut`, `ref_mut_array_unsizing_coercion`):
  `r@` is the value at return time and `final(param)@ == final(r)@` links the prophesied
  final values; `r@ == final(param)@` is wrong (the caller may mutate through `r`).

Proof-engineering notes (L01):

- Loop bodies are verified in isolation: facts about `let ghost` variables bound before a
  loop are invisible inside it unless restated as invariants (or the term is inlined).
- `Vec::append(&mut other)` empties `other`; snapshot `other@` in a ghost variable first.
- `assert(..) by (nonlinear_arith)` sees only its `requires` clause, not `let` definitions.
- `i * dim` with `i == 0` is not simplified automatically; assert `0 * dim == 0` by
  `nonlinear_arith` before loops whose invariant has such products.

## Open issues

- L16 changed the public API: the `SDP` fields are `pub`, and `sdp_to_standard` now has the
  preconditions `sdp_single_input_ok`/`sdp_lmi_input_ok` (shapes and sizes of the user data,
  one map per objective entry, a nonzero constraint system when `check_constraints` is set
  without LMIs). `sdpad` (L17/L18) inherits them; the test problems satisfy them.
- The standard-form contract fixes the row layout of the stacked matrix (LMI rows in LMI
  order, entry `e = c k + r` of block `i` at row `row_off(i) + e`, then the equality rows) and
  the zero list order (block by block, column-major within a block); L17 relies on
  `std_rows_ok`, `zeros_ok` and `reduced_from`.
- L10 changed the executable cache update of `jacobi_eigen_cach` (two corrections, see
  `EXPOSED_BUGS.md`); the cached pivot is now provably the position of maximal off-diagonal
  magnitude. `src/sdp_tests.rs` gained the regression test `test_jacobi_cach_cache_update`.
- `poseigendecomp`/`negeigendecomp` (L12) inherit the eigensolver preconditions `dim >= 1`,
  `dim_fits(dim)` and an input of length `>= pk_len(dim)`; `sdpad_setup_ok` (L17) provides
  `d >= 1` and `dim_fits(d)` for L18. The L18 specification of the dual update should use `spectral_part_ok` with
  `pos_sel`, `lemma_selected_reconstruction` for the `diag_scale_gram` result (with
  `lemma_rows_of_product` for its preconditions when the positive part is nonempty) and
  `lemma_pos_part_psd`.
- L14 changed the executable row scaling of `linear_remove_redundant_sym` twice (packed-column
  start, and the off-diagonal factor `sqrt(2)` requested by the user); the scale is now the
  Frobenius norm of the row's symmetric matrix. Both changes are documented in the L14
  architecture decisions.

- L17 changed the solver interface: `sdpad` now requires (informally, it is still outside
  `verus!`) `sdpad_input_ok`, i.e. additionally `dim_fits(total_rows(sdp))`; `sdpad_setup`,
  `SdpadSetup` and `sdpad_setup_ok` are public. Two executable changes: the single-variable
  PSD block (`EXPOSED_BUGS.md`) and the always-full-length `dual_y` (L17 architecture
  decisions). L18 should verify the loop body as a helper taking `&SdpadSetup` and the iteration
  state, reusing `dual_solve_ok`, `spectral_part_ok` (L12) and the L04 operator contracts.
- The full-verification command in CLAUDE.md needs `-V no-solver-version-check` on
  this machine (Z3 4.16.0 with build hashcode); without it Verus panics as soon as any
  function is verified.
- `cargo build` and `cargo test` currently report no warnings.

# Verification Progress

All layers are done. Layer DAG: `DAG.md`. Trusted items: `TRUSTED_BASE.md`.
Bugs exposed by verification: `EXPOSED_BUGS.md`.

## Commands

    verus src/lib.rs --crate-type=lib --edition=2024 -V no-solver-version-check
    cargo build && cargo test

The flag is required because the bundled Z3 reports a version string with a
build hash, which fails Verus's exact version check. `cargo build` and
`cargo test` pass without warnings.

## Layer status

All 23 layers (L00-L22) are done; L21 holds the post-hoc proofs of `ADDITIONAL_PROOFS.md`
(statements 1-4), L22 the residual bound of the reduction (statement 5).

| Layer | Where |
|-------|-------|
| L00 trusted exact-real model | `src/trusted.rs`, `src/real_spec.rs`, `src/lin_alg/comparison.rs` |
| L01 vector views and operations | `src/lin_alg/vec_spec.rs`; `src/lin_alg/matrix.rs` (block 1) |
| L02 packed symmetric view and operations | `src/lin_alg/sym_spec.rs`; `src/lin_alg/sym_matrix.rs` (block 1) |
| L03 matrix algebra | `src/lin_alg/mat_spec.rs` |
| L04 constraint operator and adjoint | `src/lin_alg/constr_spec.rs`; `sym_matrix.rs` (block 2) |
| L05 triangular and diagonal solvers | `src/lin_alg/solve_spec.rs`; `sym_matrix.rs` (block 3) |
| L06 LDLT and positive-definite solvers | `src/lin_alg/ldlt_spec.rs`; `sym_matrix.rs` (block 3) |
| L07 PSD block check | `ldlt_spec.rs`, `mat_spec.rs`; `psd_block_check` |
| L08 shared Jacobi rotation | `src/lin_alg/jacobi_spec.rs`; `jacobi_rot` in `src/lin_alg/jacobi.rs` |
| L09 common eigensolver specification | `src/lin_alg/eigen_spec.rs` |
| L10 cached Jacobi | `jacobi_eigen_cach`, `index_max_in_col`, `cached_pivot_col` |
| L11 column-cyclic Jacobi | `jacobi_eigen_cyc`; `cyc_order` in `eigen_spec.rs` |
| L12 spectral operations | `src/lin_alg/spectral_spec.rs`; `poseigendecomp`, `negeigendecomp`, `select_eigenpairs` |
| L13 Gaussian elimination | `src/lin_alg/gauss_spec.rs`; `gauss_elim` in `matrix.rs` (block 2), with the ghost row transformation `row_comb` |
| L14 constraint reduction | `src/lin_alg/reduce_spec.rs`; `linear_remove_redundant_sym` |
| L15 symmetric maps and block embeddings | `sym_matrix.rs` (blocks 5-6); `src/sdp.rs` (block 2) |
| L16 standard-form conversion | `src/sdp_spec.rs`; `SDP`, `sdp_to_standard` and helpers in `sdp.rs` (block 1) |
| L17 solver setup | `src/sdpad_spec.rs`; `sdpad_setup`, `dual_solve`, `psd_block_list`, `initial_iterates` in `src/sdp_altdir.rs` |
| L18 Wen-Goldfarb-Yin iteration | `wgy_update`, `penalty_update`, `WgyState` |
| L19 residuals and KKT quantities | `sdpad_loop`, `kkt_quantities`, `read_solution`, `SdpadOutcome` |
| L20 final audit | this file, `TRUSTED_BASE.md`, `src/trusted.rs` |
| L21 post-hoc proofs (gap identity, feasibility correspondence, approximate soundness) | `src/feas_spec.rs`; `lemma_gap_identity` in `sdpad_spec.rs`; block PSD lemmas in `mat_spec.rs` |
| L22 residual bound of the reduction | `src/lin_alg/norm_spec.rs` (rectangular norms), `src/lin_alg/resid_spec.rs` (recovery matrix, error bounds, the theorem); `lemma_reduced_from_residual_bound` in `sdpad_spec.rs` |

## Verified-function count

Last full run: **1046 verified, 0 errors** (about 28 s). `--time` reports no
slow proof, and no `rlimit` is raised. The `verus!` blocks contain about
18,000 lines of code, not counting blank and comment lines.

Verus counts executable functions, lemmas, recursive spec functions, loop
bodies and `assert .. by (nonlinear_arith)` sub-queries; `external_body`
items are not counted. The count therefore drops when inline
`nonlinear_arith` steps are folded into shared lemmas or when proofs leave
the scope; such decreases are not lost proofs.

| Module | Verified | Layers |
|--------|---------:|--------|
| `trusted` | 0 | trusted items only |
| `real_spec` | 38 | L00 |
| `lin_alg::vec_spec` | 20 | L01, L21 |
| `lin_alg::sym_spec` | 66 | L02, L03, L21, L22 |
| `lin_alg::mat_spec` | 87 | L03, L07, L12, L21 |
| `lin_alg::constr_spec` | 23 | L04 |
| `lin_alg::solve_spec` | 11 | L05 |
| `lin_alg::ldlt_spec` | 9 | L06, L07 |
| `lin_alg::jacobi_spec` | 57 | L08 |
| `lin_alg::eigen_spec` | 20 | L09-L11 |
| `lin_alg::spectral_spec` | 8 | L12 |
| `lin_alg::gauss_spec` | 45 | L13, L22 |
| `lin_alg::reduce_spec` | 14 | L14, L21 |
| `lin_alg::norm_spec` | 46 | L21, L22 |
| `lin_alg::resid_spec` | 60 | L22 |
| `lin_alg::matrix` | 53 | L01, L13 |
| `lin_alg::comparison` | 6 | L00 |
| `lin_alg::sym_matrix` | 154 | L02, L04-L07, L14, L15 |
| `lin_alg::jacobi` | 41 | L08, L10-L12 |
| `sdp_spec` | 58 | L16 |
| `sdpad_spec` | 38 | L17-L19, L21, L22 |
| `feas_spec` | 50 | L21 |
| `sdp` | 100 | L15, L16 |
| `sdp_altdir` | 42 | L17-L19 |
| **total** | **1046** | |

## Tolerances

Every tolerance is a named default, and the specifications and lemmas take
the tolerance as a parameter (`gauss_eps_r(tol, ..)`, `rhs_consistent(tol, ..)`,
`reduced_with(tol, ..)`, `reduced_system(tol, ..)`, the `resid_spec` lemmas,
`psd_check_eps_r(tol, ..)`, `block_psd_ok(tol, ..)`, `jacobi_thr_r(tol, ..)`,
`jacobi_converged(tol, ..)`, `jacobi_eps_r(tol, ..)`, `eigensolver_ok(tol, ..)`,
`spectral_part_ok(tol, ..)`, `neg_sel(tol, ..)`, `reg_shift_r(tol, ..)`,
`residual_ok(tol, ..)`). Each executable contract instantiates the parameter
with its named default (`reduced_from`, `dual_solve_ok`, `psd_blocks_ok`,
`pos_part_of`, `stagnation_stop` and `outcome_ok` refer to the defaults
directly). The conditions a proof needs are its explicit hypotheses, and one
lemma per default proves them from the literal reading T5:

| Default | Value | Condition (lemma) |
|---------|-------|-------------------|
| `TOL` (`matrix.rs`) | `1e-10` | `0 < TOL < 1` (`lemma_tol_default`): pivots passing their test are nonzero; the largest coefficient of a nonzero matrix passes the pivot test, so a nonzero system has rank `>= 1` |
| `PSD_TOL` (`sym_matrix.rs`) | `1e-9` | `PSD_TOL > 0` (`lemma_psd_tol_default`) |
| `JACOBI_TOL` (`jacobi.rs`) | `1e-12` | `JACOBI_TOL > 0` (`lemma_jacobi_tol_default`): a pivot failing the stopping test is nonzero; the stopping bound is nonnegative |
| `NEG_EIG_TOL` (`jacobi.rs`) | `1e-8` | none |
| `GRAM_RESID_TOL` (`sdp_altdir.rs`) | `1e-9` | none |
| `STAG_TOL1..3` (`sdp_altdir.rs`) | `1e-5`, `1e-4`, `1e-3` | none |
| `PEN_MIN`, `PEN_INIT`, `PEN_MAX`, `PEN_FACT` | `1e-4`, `5.0`, `1e4`, `0.5` | `0 < PEN_MIN <= PEN_INIT <= PEN_MAX`, `0 < PEN_FACT <= 1` (`lemma_penalty_defaults`) |

No relation between different tolerances is needed. The real-arithmetic
helper `lemma_tol_scale_ge` (`tol >= 0, s >= 1 ==> tol * s >= tol`) turns
each threshold `tol * max(.., 1)` into a positive lower bound.

**Audit.** The full crate verifies when the tolerance entries of
`axiom_literals` (T5) are replaced by exactly these conditions (`0.0`, `0.5`,
`1.0`, `2.0` keep their values; `1e-10` in `(0, 1)`; `1e-9`, `1e-12`
positive; `0 < 1e-4 <= 5.0 <= 1e4`; nothing about `1.6`, `1e-3`, `1e-5`,
`1e-8`). Re-run this audit after touching a tolerance proof: with the exact
values available, the linear solver proves facts such as `TOL * s >= 0` or
`|b| < TOL |b| ==> b = 0` silently, and the audit is what exposes them.

## Main contracts

- **Comparisons** (`comparison.rs`): the exact-real reading of each tolerance
  test, e.g. `scaled_near_zero(x, s, tol)` iff `|x| < tol * s`.
- **Dense matrices** (`matrix.rs`): `gauss_elim` preserves the solution set,
  keeps zero columns, and produces echelon form with nonzero pivots in the
  coefficient columns under the tolerance-based pivot/skip rules
  (`echelon_form`, `skipped_bounded`, tolerance `TOL * max |A_ij|`). A
  consistent result has right-hand sides zero or below
  `TOL * 2^rank * max |b_k|` in the rows without a pivot (`rhs_consistent`),
  and comes from a zero input if its rank is 0. The input is a row
  combination `T` of the result (`row_comb`): the identity, with two columns
  exchanged at each swap and a multiple of one column added to another at
  each subtraction.
- **Constraint operator** (`sym_matrix.rs`): `constraint_action` is `A(X)`;
  `constraint_lin_comb` is `A*(y)` with the off-diagonal factors 1/2 and
  satisfies `<A(X), y> = <X, A*(y)>_F`; `diag_scale_gram` is `Q D Q^T`.
- **Solvers** (`sym_matrix.rs`): `forw_subst`, `back_subst`, `diag_solver`
  solve `L x = b`, `L^T x = b`, `D x = b`; `ldlt_decomp` gives `L D L^T = A`
  with pivots `>= TOL * max(max_i |A_ii|, 1)`; `pos_def_solver_upperleft`
  solves the block system `(A, I/2) x = b`.
- **PSD check**: success of `psd_block_check` gives `B >= -eps I` for each
  block, `eps = PSD_TOL * max(max_i |B_ii|, 1)` (`block_psd_ok(rv(PSD_TOL), ..)`).
- **Reduction**: `linear_remove_redundant_sym` divides the input rows by
  their Frobenius norms (`normalized_r`), then returns the first `rank`
  echelon rows of `gauss_elim` divided by their Frobenius norms, which are
  positive because each kept row holds a nonzero pivot (`reduced_with`,
  which also records the row transformation `T` of the elimination). Both
  norm passes share the verified helper `packed_row_norm`.
- **Reduction residual bound** (`resid_spec.rs`): with the first-stage
  divisors `D` (1 for rows of norm zero), the second-stage divisors `S`, and
  `T = [T_K | T_D]` split at the kept rows, the input is recovered as
  `[A | b] = U [A_K | b_K] + [E | f]`, `U = D T_K S`, `[E | f] = D T_D H_D`
  with `H_D` the dropped echelon rows (`lemma_recovery_identity`). The
  dropped rows are small (zero in pivot columns, at most `2^rank * eps`
  elsewhere, right-hand side at most `TOL * 2^rank * beta`), so
  `||E||_F <= TOL * e_A` and `||f|| <= TOL * e_b` with
  `e_A = 2^rank sqrt(q n) alpha L`, `e_b = 2^rank sqrt(q) beta L`, `q` the
  number of dropped rows, `alpha`/`beta` the largest normalized coefficient /
  right-hand side, `L = ||D T_D||_F` (`lemma_err_bounds`). With
  `C = max(e_A / ||A||_F, e_b / ||b||)` (a zero denominator contributes 0)
  and `d(z) = ||A||_F ||z|| + ||b||`, every `z` satisfies
  `||A z - b|| <= C TOL d(z) + ||U||_F ||A_K z - b_K||` (`lemma_residual_bound`),
  divided by `d(z)` when `d(z) > 0` (`lemma_relative_residual_bound`);
  `d(z) = 0` forces `A z - b = 0` (`lemma_residual_zero_scale`) and `q = 0`
  gives `C = 0` (`lemma_red_const_no_drop`). `U` and `C` are spec functions
  of the data only (`recovery_r`, `red_const_r`);
  `lemma_reduction_residual_bound` states both forms and the zero case for
  `reduced_system` with the witnesses `red_pivots`, `red_transform`, and
  `lemma_reduced_from_residual_bound` (`sdpad_spec.rs`) for the dense rows of
  a setup with `check` (`reduced_from`, witness `red_echelon`). Nothing is
  claimed about the size of `C`.
- **Jacobi** (`jacobi.rs`): `jacobi_rot` gives `c^2 + s^2 = 1`,
  `A' = J^T A J`, `Q' = Q J`, `A'_jk = 0`. Both eigensolvers establish
  `eigensolver_ok`: `Q^T Q = Q Q^T = I`, `A = Q B Q^T`, `vals = diag(B)`,
  `||B - diag(B)||_F <= sqrt(dim (dim - 1)) * JACOBI_TOL * max(1, max_i |B_ii|)`
  (`eigensolver_ok(rv(JACOBI_TOL), ..)`).
  Each rotation preserves symmetry and eigenvalues
  (`lemma_jacobi_step_similar`), and so does the whole run: `A` and `B` have
  the same eigenvalues (`lemma_eigen_inv_eigenvalues`). The cached pivot has
  maximal off-diagonal magnitude; each cyclic sweep visits every pair `j < k`
  once in column-cyclic order.
- **Spectral parts**: `poseigendecomp` and `negeigendecomp` return the pairs
  `(B_kk, q_k)` with `B_kk > 0`, resp. `B_kk < -NEG_EIG_TOL * max(max_t |A_t|, 1)`
  (`spectral_part_ok`). Nothing relates them to the exact PSD projection.
- **Standard form** (`sdp.rs`): `sdp_to_standard` preserves the objective,
  the equality and LMI rows (`std_rows_ok`), the zero pattern (`zeros_ok`)
  and variable reconstruction (`Z_n = B_{s+2n} - B_{s+2n+1}`); the reduction
  is exported as `reduced_from`.
- **Solver** (`sdp_altdir.rs`):
  - `sdpad_setup` (`sdpad_setup_ok`): Frobenius weighting, Gram matrix, PSD
    partition, standard form.
  - `dual_solve`: `(G + s I) y_top = b_top` with `s = TOL * max(max_i |G_ii|, 1)`
    (0 with `check`), `y_bottom = 2 b_bottom`, and the residual test with
    `GRAM_RESID_TOL`.
  - `wgy_update` (`wgy_update_ok`): dual solve; `V = C_F - A*(y) - mu Z`;
    `S' = poseigendecomp` part of `V`; `Z' = (1 - rho) Z + (rho / mu)(S' - V)`.
  - `penalty_update`, `kkt_quantities` (`kkt_ok`: `pinf`, `dinf`, objective,
    gap, delta), `read_solution`.
  - `sdpad_loop` (`outcome_ok`): a ghost trajectory (`trajectory_ok`) starts
    at `Z = I, S = 0`; each state is the verified update of its predecessor
    and each penalty the verified update from the new infeasibilities. The
    returned solution, its objective value, the recorded deltas and the
    termination reason all refer to indices into this trajectory.
    `best_delta` starts at `0.0` and is set by the first iteration; after it,
    `lemma_outcome_best_delta` shows it is the least recorded delta.
- **Gap identity** (`sdpad_spec.rs`): for packed `Z`, `S` and multipliers `y`,
  `<C_F, Z>_F - b . y = <Z, S>_F + <Z, C_F - A*(y) - S>_F + y . (A(Z) - b)`
  (`lemma_gap_identity`); with `A(Z) = b` and `C_F - A*(y) = S` the gap is
  `<Z, S>_F` (`lemma_gap_complementarity`).
- **Feasibility correspondence** (`feas_spec.rs`): `user_feasible` states the
  user's problem with LMIs (`sum_n A_n(X_n) = b`, every slack `H_i(X) - M_i`
  symmetric PSD), `user_feasible_single` the problem without LMIs.
  - Soundness: a `Z` solving every row of the unreduced system, vanishing on
    the zero pattern and PSD is block diagonal, its LMI blocks are the PSD
    slacks, `vars_of(Z)` (what `read_solution` returns) is feasible and
    `<C_F, Z>_F` is its objective (`lemma_standard_feasible_sound`); without
    reduction this follows from `A(Z) = b` (`lemma_action_feasible_sound`,
    `lemma_action_feasible_sound_single`). With reduction, the dropped rows
    are controlled only by the reduction residual bound.
  - Converse, with or without reduction: feasible `X_n` lift to `lift_z`,
    block diagonal with blocks `H_i(X) - M_i`, `X_n + t_n I`, `t_n I`
    (`t_n = sum |X_n|`), which reads back the `X_n`, is PSD, solves the
    unreduced system and hence the setup's dense rows (a solution of the
    input solves the reduced system, `lemma_input_solution_reduced`), so
    `A(Z) = b`, and has the user's objective value
    (`lemma_user_feasible_lifts`, `lemma_user_feasible_lifts_single`). The
    unreduced system of a setup is the spec function `setup_aug`.
  - Approximate soundness, for any `Z`: the user-level residuals at
    `vars_of(Z)` are the residuals of the unreduced rows
    (`lemma_user_residuals`); without reduction these are entries of
    `A(Z) - b` (`lemma_approx_sound_unreduced`); a successful PSD check gives
    the LMI blocks `B_i >= -eps_i I` (`lemma_lmi_blocks_checked`), and
    `|res_l| <= pinf (1 + ||b||)`.
  - Approximate soundness with reduction (`lemma_approx_sound_reduced`): the
    unreduced residual norm `N = ||A_0 Z - b_0||` (`std_resid_vec_r`) satisfies
    `N <= C TOL phi(Z) + ||U||_F pinf (1 + ||b||)` with `phi`, `C`, `||U||_F`
    from the reduction residual bound (`setup_resid_scale`, `setup_red_const`,
    `setup_recovery_norm`), and `phi(Z) = 0` forces `N = 0`
    (`lemma_unreduced_residual_pinf`); every equality residual and every LMI
    linking residual `R_i = W_i - B_i` (`lmi_link_residual`, `W_i` the actual
    slack `lmi_slack_r`) satisfies `|res_e| <= N`, `||R_i||_F <= N`; a checked
    block `B_i >= -eps_i I` gives `x^T W_i x >= -(eps_i + ||R_i||_F) |x|^2` for
    every `x` (`lemma_lmi_slack_shifted`), a bound on the quadratic form since
    `W_i` need not be symmetric at an approximate point. Nothing bounds `C`
    and `||U||_F`.
  - Not certified with reduction (`check`): the feasibility of the user's
    problem at a returned point, small absolute LMI violations, and the
    equality of optimal values.
  - Supporting algebra (`mat_spec.rs`): a diagonal block of `A >= -eps I`
    satisfies the bound (`lemma_psd_shifted_block`), a block-diagonal matrix
    with such blocks does (`lemma_psd_shifted_two_blocks`, iterated in
    `lemma_block_diag_psd_shifted`), and `X >= -(sum |X_rc|) I`
    (`lemma_psd_shifted_abs_sum`).
- **Gram system** (`constr_spec.rs`, `sdpad_spec.rs`): `full_gram_view` is
  `A A*` for the full constraint operator (dense rows, then zero rows).
  `lemma_full_gram_block` shows it equals `(G, I/2)` when the dense rows
  vanish on a distinct zero pattern; `lemma_setup_gram_hypotheses` derives
  both conditions from the standard form. Hence `lemma_wgy_dual_gram`: every
  iteration's multipliers solve `(A A*) y = mu (b - A(Z)) + A(C - S)`, up to
  the regularization shift on the dense rows (`reg_rhs_r`).

## Specification architecture

- **Floats.** `rv: f64 -> real` is uninterpreted. Broadcast axioms
  (`crate::trusted::group_real_model`) give vstd's operator and comparison
  hooks exact-real meaning; std methods get `assume_specification` contracts,
  so call sites are unchanged. `sqrt_r` is the uninterpreted root of
  `trusted.rs`; `abs_r`, `max_r`, `min_r` live in `real_spec.rs`.
- **Views.** Vectors are `rvs(v@): Seq<real>`. Packed symmetric entry
  `(r, c)`, `r <= c`, sits at `pk_idx(r, c) = c (c + 1) / 2 + r`; `sym_entry`
  extends it by symmetry. Column-major uses `cm_idx`, packed lower-triangular
  `lt_idx`. Spec matrices are `spec_fn(int, int) -> real` with block equality
  `meq`, entered through `sym_view`, `cm_view`, `lt_view`, `diag_view`,
  `block_view`. `dim_fits(dim) := (dim + 2)^2 <= usize::MAX` covers all packed
  index arithmetic.
- **Sums.** `sum_fn_r` over closures, `sum2_fn_r` for square double sums,
  `tri_sum_r` for triangles. The packed Frobenius product `frob_r` and
  matrix-vector product `sym_mat_vec_r` (`mat_spec.rs`) are `frob_f` and
  `mat_vec` applied to `sym_view`; `frob_r` counts off-diagonal entries twice
  by proof (`lemma_frob_tri`), not by definition. Per-term spec functions are
  named so that closures match syntactically at call sites.
- **Matrix algebra** (`mat_spec.rs`): products, transposes, orthogonality,
  eigenvalues and similarity invariance, the order `psd_shifted(A, dim, eps)`
  (`x^T A x + eps |x|^2 >= 0`), congruence and diagonal lemmas shared by L07
  and L12.
- **Rectangular norms** (`norm_spec.rs`): `mat_vec_rc` and `frob_rc` for a
  `rows x cols` index-function matrix; Cauchy-Schwarz, the triangle
  inequality, `||A x|| <= ||A||_F ||x||`, `||A B||_F <= ||A||_F ||B||_F`, the
  entry bounds `||x|| <= sqrt(n) max |x_k|`,
  `||A||_F <= sqrt(rows cols) max |A_ik|`, the window bound
  `||x[off .. off + len]|| <= ||x||`, and `|x^T R x| <= ||R||_F ||x||^2`, which
  perturbs quadratic-form bounds: `x^T A x >= -eps |x|^2` for all `x` gives
  `x^T (A + R) x >= -(eps + ||R||_F) |x|^2` for all `x`, with no symmetry
  assumed (`lemma_psd_shifted_perturbed`).
- **Eigensolvers.** Both solvers maintain `eigen_decomp_inv` (`Q` orthogonal,
  `A = Q B Q^T`) through shared init/step lemmas and stop on `offdiag_below`;
  `lemma_converged_approx` yields `eigensolver_ok`. The spectral operations
  depend only on this interface.
- **Standard form and solver.** Contracts are stated on the user data and
  quantified over the variable `z`. The solver carries the standard form
  through `sdpad_setup_ok` (its objective witness is `setup_obj0`, exposed by
  `lemma_setup_obj0`); each iteration is specified by what the code
  computes (`wgy_update_ok`, `penalty_update_ok`); the loop by `outcome_ok`
  over the ghost trajectory, extended by `lemma_trajectory_step`. Nothing is
  claimed about convergence.

## Executable changes

The bugs in `EXPOSED_BUGS.md` were fixed. Other changes:

- `linear_remove_redundant_sym` computes both row-norm passes with the helper
  `packed_row_norm` and `sqrt(2)` once per call.
- `SDP` fields are `pub`. `sdp_to_standard` requires `sdp_single_input_ok` /
  `sdp_lmi_input_ok` (shapes, one map per objective entry, a nonzero
  constraint system when `check_constraints` is set without LMIs). `sdpad`
  informally requires `sdpad_input_ok`, which adds `dim_fits(total_rows(sdp))`.
- `sdpad` is split into the verified `sdpad_setup`, `dual_solve`,
  `wgy_update`, `penalty_update`, `kkt_quantities`, `read_solution`,
  `sdpad_loop`, and an unverified printing wrapper. `SdpadSetup`, `WgyState`,
  `SdpadOutcome` (with its ghost trajectory and histories), `Termination`,
  the tuning constants and `TOTAL_STEPS` are public.
- `dual_solve` always returns a full-length `dual_y` (the original left it
  empty when there was no reduction and no dense rows). The PSD block list is
  computed once before the loop; the best-solution copy reuses its buffer.
- Binary vector operations require equal lengths (the original `zip`
  truncated silently); `constraint_lin_comb` slices its argument accordingly.
- `ldlt_decomp` no longer rewrites the diagonal of `ident_mat_low_tri(dim)`,
  which is already 1 (a no-op loop of the original).
- `jacobi_eigen_cach`: the rotation limit uses `saturating_mul`; the eigenvector matrix is returned without a final copy;
  regression test `test_jacobi_cach_cache_update` in `src/sdp_tests.rs`.
- The tolerance literals `1e-9`, `1e-12`, `1e-8`, `1e-5`, `1e-4`, `1e-3` became
  the named constants `PSD_TOL`, `JACOBI_TOL`, `NEG_EIG_TOL`, `GRAM_RESID_TOL`,
  `STAG_TOL1..3` with the same values; the executable behavior is unchanged.

## Constructs rewritten for Verus compatibility

Every rewrite keeps the executable cost and was confirmed by a probe against
the installed Verus (0.2026.08.23).

| Construct | Replacement | Where |
|-----------|-------------|-------|
| Unary `-x` on `f64` | `crate::trusted::neg(x)` (T7) | `is_significantly_negative`, `negscal_vect` |
| `x += y`, `v[i] -= y`, `v[i] *= y` on `f64` (compiler crash) | `x = x + y`, etc. | all accumulation loops |
| `panic!`, `assert!`, `expect`, `unwrap_or_else(panic)` on data-dependent conditions | `crate::trusted::abort(..)` (T8); internal consistency checks became static `assert`s | solvers, eigensolvers, `linear_remove_redundant_sym`, `sdp_to_standard`, `dual_solve` |
| Iterator chains (`zip`, `map`, `fold`, `collect`, `all`, `enumerate`, `chunks_exact`, `scan`, `chain`, `sum`, `concat`), `for x in &v` | index loops over the same slices (`all` keeps its short circuit) | throughout |
| `continue` in `for` loops | `if` block, or `while` with explicit counter | `jacobi_rot`, `jacobi_eigen_cyc`, `gauss_elim`, `linear_remove_redundant_sym` |
| `return` inside `loop`, `&mut ident_mat(dim)` temporaries, final `to_vec()` | `loop .. break` with `ensures`; owned local `q0` returned after the loop | both eigensolvers |
| Destructuring assignment `(a, b, c) = f(..)` | `let (x, y, z) = f(..); a = x; ..` | both eigensolvers, `sdpad_loop` |
| Uninitialized `let (mut a, mut b);` before loops | initialized bindings inside the loop body | several |
| `slice.to_vec()`, `v[a..=b].to_vec()` | `vstd::slice::slice_to_vec(..)` | constraint operators, `mk_symmap`, `dual_solve` |
| `Vec::extend(vec)` | `Vec::append(&mut vec)` | `ident_mat`, `matrix_diagonal_mult_sym` |
| `Vec::swap` | swap through a temporary | `gauss_elim` |
| `clone_from` | `clear` + `extend_from_slice` | `sdpad_loop` |
| `&mut vect[..k]` on a `Vec` | `&mut vect.as_mut_slice()[..k]` | `pos_def_solver_upperleft` |
| `if let .. && ..` let-chains | nested `if` | `sdpad_loop` |
| Tuple patterns in parameters `(mat, list): &(..)` | `constr: &(..)` plus `let (mat, list) = constr;` | `constraint_action`, `constraint_lin_comb` |
| `std::f64::consts::SQRT_2` | `2.0_f64.sqrt()` | `linear_remove_redundant_sym` |
| `const`, `struct`, `enum` outside `verus!` | moved inside `verus!` | `matrix.rs`, `jacobi.rs`, `sdp.rs`, `sdp_altdir.rs` |
| `MAX_SWEEPS * dim * (dim - 1) / 2` | `(dim * (dim - 1) / 2).saturating_mul(MAX_SWEEPS)` | `jacobi_eigen_cach` |
| `println!` inside the solver loop | `sdpad_loop` returns `SdpadOutcome`; `sdpad` prints outside `verus!` | `sdp_altdir.rs` |
| Long bodies (`sdp_to_standard`, `sdpad`) | helpers following the proof structure | `sdp.rs`, `sdp_altdir.rs` |
| Imports of spec-only items | `#[cfg(verus_keep_ghost)]` | executable modules |

## Verus notes for this build

- Supported as written: `vec![x; n]`, slicing, `..=` ranges,
  `for j in it: (0..n).rev()` (with `it.index()`, `it.seq()`), `break` in
  `for`, `let .. else { abort(..) }`, `matches`, `Ghost<T>` struct fields,
  `&mut` struct parameters, `int as real`, tuples of `&mut` as return types
  (postconditions `final(p)@ == final(r)@`).
- Loop bodies are isolated: restate ghost facts, `v.len()` terms (for
  overflow checks) and quantified facts with their triggers as invariants.
- In `for x in a..b` the loop variable is unconstrained at entry when `a > b`
  is possible; prove `a <= b` first or use the named iterator.
- `nonlinear_arith` sees only its `requires`: prove one algebraic step at a
  time with identically bracketed monomials; use vstd `div_mod` lemmas for
  `/` and `%`. A `let`-bound name is an opaque variable inside the query:
  write its definition out or add it to the `requires`. The default query
  proves products with a literal constant, `x * 0 == 0`, and usually the
  commutativity, associativity and distribution of products of variables,
  but a bare `assert(p * q == q * p)` can fail when the surrounding context
  changes; `by (nonlinear_arith)` makes such steps robust. It is needed for
  facts that use hypotheses nonlinearly (sign reasoning, cancellation, bounds
  like `a <= b, c >= 0 ==> a * c <= b * c`, division).
- Closures match across the spec/proof boundary only when built from named
  spec functions and indexing.
- An `exists` postcondition needs a ground witness with the trigger's exact
  terms (bind tuple results with `let res = ..` first, or trigger on a
  result-free term).
- Ghost comparisons of `f64` values are uninterpreted (T4 covers executable
  comparisons only); compare `rv` values in ghost code.
- `Ghost<T>` fields may hold `int`, `real`, `Seq`s of executable structs and
  `ghost struct`s; such a `ghost struct` needs an unconditional `use`. Unused
  spec-only parameters need `#[allow(unused_variables)]` under `cargo`.
- A spec function whose body mentions `in_zero_region` (an `exists` whose
  trigger `sum_upto(bs, j)` re-triggers on `sum_upto(bs, j + 1)`) floods the
  prover wherever its value is unfolded; `lift_entry` is `#[verifier::opaque]`
  and revealed only in `lemma_lift_z_entry`, which exposes both cases.
- A two-level polynomial identity in four variables (the expansion of
  `(a v - c u)^2` in `lemma_cauchy_schwarz`) made `nonlinear_arith` hang,
  ignoring `--rlimit`, after unrelated lemmas were added to its module.
  Single-step product identities are proved once in `real_spec.rs`
  (`lemma_sq_diff`, `lemma_mul_rearrange`, `lemma_real_mul_assoc`, ..) and
  instantiated; the linear solver combines the instances.
- Large proofs can verify alone but fail in the full crate (`standard_lmi` at
  its call of `objective_sum`). Intermediate assertions, or naming a quantified
  precondition as a spec predicate (`objective_shapes_ok`) so that the call-site
  check is a ground fact, make them stable.

## Open issues

None.

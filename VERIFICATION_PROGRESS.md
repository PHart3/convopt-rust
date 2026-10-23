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

All 22 layers (L00-L21) are done; L21 holds the post-hoc proofs of `ADDITIONAL_PROOFS.md`.

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
| L13 Gaussian elimination | `src/lin_alg/gauss_spec.rs`; `gauss_elim` in `matrix.rs` (block 2) |
| L14 constraint reduction | `src/lin_alg/reduce_spec.rs`; `linear_remove_redundant_sym` |
| L15 symmetric maps and block embeddings | `sym_matrix.rs` (blocks 5-6); `src/sdp.rs` (block 2) |
| L16 standard-form conversion | `src/sdp_spec.rs`; `SDP`, `sdp_to_standard` and helpers in `sdp.rs` (block 1) |
| L17 solver setup | `src/sdpad_spec.rs`; `sdpad_setup`, `dual_solve`, `psd_block_list`, `initial_iterates` in `src/sdp_altdir.rs` |
| L18 Wen-Goldfarb-Yin iteration | `wgy_update`, `penalty_update`, `WgyState` |
| L19 residuals and KKT quantities | `sdpad_loop`, `kkt_quantities`, `read_solution`, `SdpadOutcome` |
| L20 final audit | this file, `TRUSTED_BASE.md`, `src/trusted.rs` |
| L21 post-hoc proofs (gap identity, feasibility correspondence, approximate soundness) | `src/feas_spec.rs`; `lemma_gap_identity` in `sdpad_spec.rs`; block PSD lemmas in `mat_spec.rs` |

## Verified-function count

Last full run: **867 verified, 0 errors** (about 27 s). `--time` reports no
slow proof, and no `rlimit` is raised. The `verus!` blocks contain 15,431
lines of code, not counting blank and comment lines.

Verus counts executable functions, lemmas, recursive spec functions, loop
bodies and `assert .. by (nonlinear_arith)` sub-queries; `external_body`
items are not counted. Earlier runs reported more (999, 916, then 891):
inline `nonlinear_arith` steps were removed or folded into shared lemmas (261
remain), and the scope reduction of October 2026 removed the reduction
residual bounds, the reduced-case lifting and the unused negative-part PSD
bound with their helpers (38 items). Such decreases are expected and are not
lost proofs. The update to the row-normalized, RHS-scaled elimination of
`main` (October 2026) added 14 items: row-scaling lemmas, the helper
`packed_row_norm`, and the loops of the new scales and tests.

| Module | Verified | Layers |
|--------|---------:|--------|
| `trusted` | 0 | trusted items only |
| `real_spec` | 22 | L00 |
| `lin_alg::vec_spec` | 20 | L01, L21 |
| `lin_alg::sym_spec` | 62 | L02, L03, L21 |
| `lin_alg::mat_spec` | 87 | L03, L07, L12, L21 |
| `lin_alg::constr_spec` | 23 | L04 |
| `lin_alg::solve_spec` | 11 | L05 |
| `lin_alg::ldlt_spec` | 9 | L06, L07 |
| `lin_alg::jacobi_spec` | 57 | L08 |
| `lin_alg::eigen_spec` | 19 | L09-L11 |
| `lin_alg::spectral_spec` | 8 | L12 |
| `lin_alg::gauss_spec` | 33 | L13, L21 |
| `lin_alg::reduce_spec` | 9 | L14, L21 |
| `lin_alg::matrix` | 48 | L01, L13 |
| `lin_alg::comparison` | 6 | L00 |
| `lin_alg::sym_matrix` | 152 | L02, L04-L07, L14, L15 |
| `lin_alg::jacobi` | 38 | L08, L10-L12 |
| `sdp_spec` | 58 | L16 |
| `sdpad_spec` | 35 | L17-L19, L21 |
| `feas_spec` | 37 | L21 |
| `sdp` | 98 | L15, L16 |
| `sdp_altdir` | 35 | L17-L19 |
| **total** | **853** | |

## Main contracts

- **Comparisons** (`comparison.rs`): the exact-real reading of each tolerance
  test, e.g. `scaled_near_zero(x, s, tol)` iff `|x| < tol * s`.
- **Dense matrices** (`matrix.rs`): `gauss_elim` preserves the solution set,
  keeps zero columns, and produces echelon form with nonzero pivots in the
  coefficient columns under the tolerance-based pivot/skip rules
  (`echelon_form`, `skipped_bounded`, tolerance `TOL * max |A_ij|`). A
  consistent result has right-hand sides zero or below
  `TOL * 2^rank * max |b_k|` in the rows without a pivot (`rhs_consistent`),
  and comes from a zero input if its rank is 0.
- **Constraint operator** (`sym_matrix.rs`): `constraint_action` is `A(X)`;
  `constraint_lin_comb` is `A*(y)` with the off-diagonal factors 1/2 and
  satisfies `<A(X), y> = <X, A*(y)>_F`; `diag_scale_gram` is `Q D Q^T`.
- **Solvers** (`sym_matrix.rs`): `forw_subst`, `back_subst`, `diag_solver`
  solve `L x = b`, `L^T x = b`, `D x = b`; `ldlt_decomp` gives `L D L^T = A`
  with pivots `>= TOL * max(max_i |A_ii|, 1)`; `pos_def_solver_upperleft`
  solves the block system `(A, I/2) x = b`.
- **PSD check**: success of `psd_block_check` gives `B >= -eps I` for each
  block, `eps = 1e-9 * max(max_i |B_ii|, 1)`.
- **Reduction**: `linear_remove_redundant_sym` divides the input rows by
  their Frobenius norms (`normalized_r`), then returns the first `rank`
  echelon rows of `gauss_elim` divided by their Frobenius norms, which are
  positive because each kept row holds a nonzero pivot (`reduced_with`).
  The dropped rows are only known to pass the consistency test. Both norm
  passes share the verified helper `packed_row_norm`.
- **Jacobi** (`jacobi.rs`): `jacobi_rot` gives `c^2 + s^2 = 1`,
  `A' = J^T A J`, `Q' = Q J`, `A'_jk = 0`. Both eigensolvers establish
  `eigensolver_ok`: `Q^T Q = Q Q^T = I`, `A = Q B Q^T`, `vals = diag(B)`,
  `||B - diag(B)||_F <= sqrt(dim (dim - 1)) * 1e-12 * max(1, max_i |B_ii|)`.
  Each rotation preserves symmetry and eigenvalues
  (`lemma_jacobi_step_similar`), and so does the whole run: `A` and `B` have
  the same eigenvalues (`lemma_eigen_inv_eigenvalues`). The cached pivot has
  maximal off-diagonal magnitude; each cyclic sweep visits every pair `j < k`
  once in column-cyclic order.
- **Spectral parts**: `poseigendecomp` and `negeigendecomp` return the pairs
  `(B_kk, q_k)` with `B_kk > 0`, resp. `B_kk < -1e-8 * max(max_t |A_t|, 1)`
  (`spectral_part_ok`). Nothing relates them to the exact PSD projection.
- **Standard form** (`sdp.rs`): `sdp_to_standard` preserves the objective,
  the equality and LMI rows (`std_rows_ok`), the zero pattern (`zeros_ok`)
  and variable reconstruction (`Z_n = B_{s+2n} - B_{s+2n+1}`); the reduction
  is exported as `reduced_from`.
- **Solver** (`sdp_altdir.rs`):
  - `sdpad_setup` (`sdpad_setup_ok`): Frobenius weighting, Gram matrix, PSD
    partition, standard form.
  - `dual_solve`: `(G + s I) y_top = b_top`, `y_bottom = 2 b_bottom`, with
    the residual contract.
  - `wgy_update` (`wgy_update_ok`): dual solve; `V = C_F - A*(y) - mu Z`;
    `S' = poseigendecomp` part of `V`; `Z' = (1 - rho) Z + (rho / mu)(S' - V)`.
  - `penalty_update`, `kkt_quantities` (`kkt_ok`: `pinf`, `dinf`, objective,
    gap, delta), `read_solution`.
  - `sdpad_loop` (`outcome_ok`): a ghost trajectory (`trajectory_ok`) starts
    at `Z = I, S = 0`; each state is the verified update of its predecessor
    and each penalty the verified update from the new infeasibilities. The
    returned solution, its objective value, the recorded deltas and the
    termination reason all refer to indices into this trajectory.
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
    `lemma_action_feasible_sound_single`). With reduction, nothing is
    claimed about the dropped rows.
  - Converse, without reduction: feasible `X_n` lift to `lift_z`, block
    diagonal with blocks `H_i(X) - M_i`, `X_n + t_n I`, `t_n I`
    (`t_n = sum |X_n|`), which reads back the `X_n`, is PSD, solves the
    unreduced system, so `A(Z) = b`, and has the user's objective value
    (`lemma_user_feasible_lifts`, `lemma_user_feasible_lifts_single`).
  - Approximate soundness, for any `Z`: the user-level residuals at
    `vars_of(Z)` are the residuals of the unreduced rows
    (`lemma_user_residuals`); without reduction these are entries of
    `A(Z) - b` (`lemma_approx_sound_unreduced`); a successful PSD check gives
    the LMI blocks `B_i >= -eps_i I` (`lemma_lmi_blocks_checked`), and
    `|res_l| <= pinf (1 + ||b||)`.
  - Not certified with reduction (`check`): the residuals of the dropped
    rows, the feasibility of the user's problem, and the equality of optimal
    values. `A(Z) - b` then measures the reduced system only.
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

The four bugs in `EXPOSED_BUGS.md` were fixed. Other changes:

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
- `jacobi_eigen_cach`: `biggest` starts at `0.0`; the rotation limit uses
  `saturating_mul`; the eigenvector matrix is returned without a final copy;
  regression test `test_jacobi_cach_cache_update` in `src/sdp_tests.rs`.

## Constructs rewritten for Verus compatibility

Every rewrite keeps the executable cost and was confirmed by a probe against
the installed Verus (0.2026.08.23).

| Construct | Replacement | Where |
|-----------|-------------|-------|
| Unary `-x` on `f64` | `crate::trusted::neg(x)` (T7) | `is_significantly_negative`, `negscal_vect` |
| `f64::INFINITY` | `pos_infinity()` (T7) | `sdpad_loop` sentinels |
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
  `/` and `%`. It is not needed for products with a literal constant or for
  `x * 0 == 0`. Commutativity, associativity and distribution of products of
  variables usually go through in the default query, but a bare
  `assert(p * q == q * p)` can fail when the surrounding context changes;
  `by (nonlinear_arith)` makes such steps robust. It is needed for facts that
  use hypotheses nonlinearly (sign reasoning, cancellation, bounds like
  `a <= b, c >= 0 ==> a * c <= b * c`, division).
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
- Large proofs can verify alone but fail in the full crate (`standard_lmi` at
  its call of `objective_sum`). Intermediate assertions, or naming a quantified
  precondition as a spec predicate (`objective_shapes_ok`) so that the call-site
  check is a ground fact, make them stable.

## Open issues

None.

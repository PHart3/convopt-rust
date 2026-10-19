# Verification Progress

Final state after the audit layer L20. Layer DAG: `DAG.md`. Trusted items:
`TRUSTED_BASE.md`. Bugs exposed by verification: `EXPOSED_BUGS.md`.

## Commands

Full verification (the Z3 shipped with this Verus build reports a version
string with a build hashcode, which fails Verus's exact version check, so the
extra flag is required):

    verus src/lib.rs --crate-type=lib --edition=2024 -V no-solver-version-check

Runtime checks: `cargo build`, `cargo test` (both without warnings).

## Layer status

All 21 layers (L00-L20) are done.

| Layer | Where |
|-------|-------|
| L00 scaffolding and trusted exact-real model | `src/trusted.rs`, `src/real_spec.rs`, `src/lin_alg/comparison.rs` |
| L01 vector views and basic operations | `src/lin_alg/vec_spec.rs`; `src/lin_alg/matrix.rs` (first `verus!` block) |
| L02 packed symmetric view and basic symmetric operations | `src/lin_alg/sym_spec.rs`; `src/lin_alg/sym_matrix.rs` (first block) |
| L03 spec-level matrix algebra | `src/lin_alg/mat_spec.rs` |
| L04 constraint operator and adjoint | `src/lin_alg/constr_spec.rs`; `sym_matrix.rs` (second block) |
| L05 triangular and diagonal solvers | `src/lin_alg/solve_spec.rs`; `sym_matrix.rs` (third block) |
| L06 LDLT and positive-definite solvers | `src/lin_alg/ldlt_spec.rs`; `sym_matrix.rs` (third block) |
| L07 PSD block check | `ldlt_spec.rs`, `mat_spec.rs`; `psd_block_check` in `sym_matrix.rs` |
| L08 shared Jacobi rotation core | `src/lin_alg/jacobi_spec.rs`; `jacobi_rot` in `src/lin_alg/jacobi.rs` |
| L09 common eigensolver specification | `src/lin_alg/eigen_spec.rs` |
| L10 cached Jacobi | `jacobi_eigen_cach`, `index_max_in_col`, `cached_pivot_col` in `jacobi.rs` |
| L11 column-cyclic Jacobi | `jacobi_eigen_cyc` in `jacobi.rs`; `cyc_order` in `eigen_spec.rs` |
| L12 PSD spectral operations | `src/lin_alg/spectral_spec.rs`; `poseigendecomp`, `negeigendecomp`, `select_eigenpairs` in `jacobi.rs` |
| L13 Gaussian elimination | `src/lin_alg/gauss_spec.rs`; `gauss_elim` in `matrix.rs` (second block) |
| L14 constraint reduction | `src/lin_alg/reduce_spec.rs`; `linear_remove_redundant_sym` in `sym_matrix.rs` |
| L15 user-facing symmetric maps and block embeddings | `sym_matrix.rs` (blocks 5-6); `src/sdp.rs` (second block) |
| L16 standard-form conversion | `src/sdp_spec.rs`; `SDP`, `sdp_to_standard` and helpers in `sdp.rs` (first block) |
| L17 solver linear-system setup | `src/sdpad_spec.rs`; `sdpad_setup`, `dual_solve`, `psd_block_list`, `initial_iterates` in `src/sdp_altdir.rs` |
| L18 Wen-Goldfarb-Yin iteration | `wgy_update`, `penalty_update`, `WgyState` in `sdp_altdir.rs` |
| L19 residuals and KKT quantities | `sdpad_loop`, `kkt_quantities`, `read_solution`, `SdpadOutcome` in `sdp_altdir.rs` |
| L20 final audit | this file, `TRUSTED_BASE.md`, `src/trusted.rs` |

## Verified-function count

Last full run: **866 verified, 0 errors** (about 27 s wall clock, 15 threads;
`--time` reports no slow proof; no `rlimit` is raised anywhere).

Per module (Verus counts executable functions, lemmas, recursive spec
functions and loop bodies; `external_body` items are not counted):

| Module | Verified | Layers |
|--------|---------:|--------|
| `trusted` | 0 | trusted items only (not counted) |
| `real_spec` | 21 | L00 |
| `lin_alg::vec_spec` | 23 | L01 |
| `lin_alg::sym_spec` | 55 | L02, L03 |
| `lin_alg::mat_spec` | 76 | L03, L07, L12 |
| `lin_alg::constr_spec` | 26 | L04 |
| `lin_alg::solve_spec` | 11 | L05 |
| `lin_alg::ldlt_spec` | 13 | L06, L07 |
| `lin_alg::jacobi_spec` | 61 | L08 |
| `lin_alg::eigen_spec` | 21 | L09-L11 |
| `lin_alg::spectral_spec` | 10 | L12 |
| `lin_alg::gauss_spec` | 25 | L13 |
| `lin_alg::reduce_spec` | 27 | L14 |
| `lin_alg::matrix` | 47 | L01, L13 |
| `lin_alg::comparison` | 6 | L00 |
| `lin_alg::sym_matrix` | 168 | L02, L04-L07, L14, L15 |
| `lin_alg::jacobi` | 41 | L08, L10-L12 |
| `sdp_spec` | 62 | L16 |
| `sdpad_spec` | 23 | L17-L19 |
| `sdp` | 113 | L15, L16 |
| `sdp_altdir` | 37 | L17-L19 |
| **total** | **866** | |

Executable functions verified (contract summary):

- **Vectors and dense matrices** (`matrix.rs`, 15): constructors, dot product,
  norms, linear combinations, equality, and `gauss_elim` (solution set preserved,
  echelon form with tolerance-based pivot/skip rules, `rank >= 1` for nonzero input).
- **Comparisons** (`comparison.rs`, 6): exact-real reading of each tolerance test
  (e.g. `scaled_near_zero(x, s, tol)` iff `|x| < tol * s`).
- **Packed symmetric matrices** (`sym_matrix.rs`, 36): basic operations;
  `constraint_action` (`A(X)`), `constraint_lin_comb` (adjoint `A*(y)` with the
  off-diagonal factors 1/2, adjoint identity `<A(X), y> = <X, A*(y)>_F`),
  `constraint_gram`, `diag_scale_gram` (`Q D Q^T`); `diag_solver`, `forw_subst`,
  `back_subst` (`L x = b`, `L^T x = b`), `ldlt_decomp` (`L D L^T = A`, pivots
  `>= TOL * max(max_i |A_ii|, 1)`), `pos_def_solver`, `pos_def_solver_upperleft`;
  `psd_block_check` (success gives `B >= -eps I`, `eps = 1e-9 * max(max_i |B_ii|, 1)`
  per block); `linear_remove_redundant_sym` (reduced system: first `rank` echelon
  rows divided by their Frobenius norms); block embeddings and symmetric-map helpers.
- **Jacobi eigensolvers** (`jacobi.rs`, 8): `jacobi_rot` (`exists c, s`:
  `c^2 + s^2 = 1`, `A' = J^T A J`, `Q' = Q J`, `A'_jk = 0`); `jacobi_eigen_cach`
  and `jacobi_eigen_cyc` share `eigensolver_ok`: `Q^T Q = Q Q^T = I`, `A = Q B Q^T`,
  `vals = diag(B)`, `||B - diag(B)||_F <= sqrt(dim (dim - 1)) * 1e-12 * max(1,
  max_i |B_ii|)`; the cached pivot is a position of maximal off-diagonal magnitude;
  each cyclic sweep visits every pair `j < k` exactly once in column-cyclic order;
  `poseigendecomp` / `negeigendecomp` return the pairs `(B_kk, q_k)` with `B_kk > 0`
  resp. `B_kk < -1e-8 * max(max_t |A_t|, 1)` (`spectral_part_ok`).
- **SDP representation** (`sdp.rs`, 14): `sdp_to_standard` preserves the objective
  (`<obj, z> = sum_n C_n(Z_n)`), the equality and LMI rows (`std_rows_ok`), the zero
  pattern (`zeros_ok`) and variable reconstruction (`Z_n = B_{s+2n} - B_{s+2n+1}`);
  the constraint reduction is exported as `reduced_from`.
- **Solver** (`sdp_altdir.rs`, 10): `sdpad_setup` (`sdpad_setup_ok`: Frobenius
  weighting, Gram matrix, PSD partition, standard form), `dual_solve`
  (`(G + s I) y = b` with the residual contract), `wgy_update` (`wgy_update_ok`:
  dual solve, `diff = C_F - A*(y)`, `V = diff - mu Z`, `S' = Pi_pos(V)` through
  `poseigendecomp`, `Z' = (1 - rho) Z + (rho / mu)(S' - V)`), `penalty_update`,
  `kkt_quantities` (`pinf`, `dinf`, objective, gap, delta as defined in `kkt_ok`),
  `read_solution`, `sdpad_loop` (`outcome_ok`: the returned solution is read from an
  iterate of the verified update, its value is the user's objective, and the
  termination reason's conditions hold).

## Specification architecture

- **Floats.** `rv: f64 -> real` is uninterpreted; vstd's operator and comparison
  hooks are given exact-real meaning by broadcast axioms (`broadcast use
  crate::trusted::group_real_model;`); std methods get `assume_specification`
  contracts, so call sites and performance are unchanged. Spec vocabulary
  `abs_r`, `max_r`, `min_r`, `sqrt_r` lives in `real_spec.rs`.
- **Views.** `rvs(v@): Seq<real>` for vectors; packed symmetric entry `(r, c)`,
  `r <= c`, at `pk_idx(r, c) = c (c + 1) / 2 + r` with `sym_entry` extended by
  symmetry; column-major `cm_idx`; packed lower-triangular `lt_idx`. Matrices in the
  algebra are index functions `spec_fn(int, int) -> real` with block equality `meq`;
  stored matrices enter through `sym_view`, `cm_view`, `lt_view`, `diag_view`,
  `block_view`. `dim_fits(dim) := (dim + 2)^2 <= usize::MAX` is the single
  precondition covering all packed index arithmetic.
- **Sums.** `sum_fn_r(f, n)` over closures, square double sums `sum2_fn_r`, triangle
  sums `tri_sum_r` (the bridge between packed loops, flat packed sums and the
  Frobenius product `frob_r`, which counts off-diagonal entries twice by proof, not
  by definition). Closures are matched syntactically: per-term spec functions are
  named so that lemma conclusions carry over to call sites.
- **Matrix algebra** (`mat_spec.rs`): products, transposes, orthogonality (both
  `Q^T Q = I` and `Q Q^T = I`), eigenvalues with similarity invariance, the
  semidefinite order `psd_shifted(A, dim, eps)` (`x^T A x + eps |x|^2 >= 0`),
  congruence and diagonal lemmas shared by L07 (LDLT) and L12 (spectral).
- **Eigensolvers.** One invariant `eigen_decomp_inv` (`Q` orthogonal, `A = Q B Q^T`)
  with init/step/eigenvalue lemmas consumed by both solvers; the stopping predicate
  `offdiag_below`; `lemma_converged_approx` gives the shared contract
  `eigensolver_ok`. The spectral operations are specified only through this
  interface (`spectral_part_ok`); no claim relates them to the exact PSD projection.
- **Standard form and solver.** Contracts are stated on the user data (`SDP` fields
  are `pub`) and quantified functionally over the variable `z`; the solver carries
  the standard form through `sdpad_setup_ok`, each iteration is specified by what
  the code computes (`wgy_update_ok`, `penalty_update_ok`), and the loop by
  `outcome_ok` over ghost histories of deltas and feasibility flags. Nothing is
  claimed about convergence.

## Executable changes

Bugs found by verification (details in `EXPOSED_BUGS.md`): the stale `biggest`
bound and the out-of-column cache read of `jacobi_eigen_cach` (L10); the packed
column start of the row-norm loop of `linear_remove_redundant_sym` (L14); the
empty PSD block list for single-variable problems (L17).

Other behaviour and interface changes:

- The off-diagonal row-norm factor of `linear_remove_redundant_sym` is `sqrt(2)`
  instead of `1/sqrt(2)` (author's decision, 2026-10-18): the scale is the
  Frobenius norm of the row's symmetric matrix.
- `SDP` fields are `pub`; `sdp_to_standard` requires `sdp_single_input_ok` /
  `sdp_lmi_input_ok` (shapes, one map per objective entry, a nonzero constraint
  system when `check_constraints` is set without LMIs); `sdpad` requires
  (informally) `sdpad_input_ok`, which adds `dim_fits(total_rows(sdp))`.
- `sdpad` is split into the verified `sdpad_setup`, `dual_solve`, `wgy_update`,
  `penalty_update`, `kkt_quantities`, `read_solution`, `sdpad_loop` and the
  unverified printing wrapper; `SdpadSetup`, `WgyState`, `SdpadOutcome`,
  `Termination`, the tuning constants and `TOTAL_STEPS` are public.
- `dual_solve` always returns a `dual_y` of full length (the original left it empty
  without reduction and with no dense rows); the PSD block list is computed once
  before the loop; the best-solution copy reuses its buffer.
- Binary vector operations require equal lengths (the original `zip` truncated
  silently); `constraint_lin_comb` slices its argument accordingly.
- `jacobi_eigen_cach`: `biggest` starts at `0.0`, the rotation limit uses
  `saturating_mul`, the eigenvector matrix is returned without the final copy.
- `jacobi_eigen_cach` has the regression test `test_jacobi_cach_cache_update`
  (`src/sdp_tests.rs`).

## Constructs rewritten for Verus compatibility

Every rewrite keeps the executable cost; each was confirmed by a probe against
the installed Verus (0.2026.08.23).

| Construct | Replacement | Where |
|-----------|-------------|-------|
| Unary `-x` on `f64` | `crate::trusted::neg(x)` (T7) | `is_significantly_negative`, `negscal_vect` |
| `f64::INFINITY` | `pos_infinity()` (T7) | `sdpad_loop` sentinels |
| `x += y`, `v[i] -= y`, `v[i] *= y` on `f64` (compiler crash) | `x = x + y`, etc. | all accumulation loops |
| `panic!`, `assert!`, `expect`, `unwrap_or_else(panic)` on data-dependent conditions | `crate::trusted::abort(..)` (T8); internal consistency checks became static `assert`s | solvers, eigensolvers, `linear_remove_redundant_sym`, `sdp_to_standard`, `dual_solve` |
| Iterator chains (`zip`, `map`, `fold`, `enumerate`, `chunks_exact`, `scan`, `chain`, `sum`, `concat`) and `for x in &v` / `for &x in v` loops | index loops over the same slices | throughout |
| `continue` in `for` loops | `if` block or `while` loop with explicit counter | `jacobi_rot`, `jacobi_eigen_cyc`, `gauss_elim` |
| `return` inside `loop`, `&mut ident_mat(dim)` temporaries, final `to_vec()` | `loop .. break` with `ensures`, owned local `q0` returned after the loop | both eigensolvers |
| Destructuring assignment `(a, b, c) = f(..)` | `let (x, y, z) = f(..); a = x; ..` | both eigensolvers, `sdpad_loop` |
| Uninitialized `let (mut a, mut b);` before loops | initialized bindings inside the loop body | several |
| `slice.to_vec()`, `v[a..=b].to_vec()` | `vstd::slice::slice_to_vec(..)` | constraint operators, `mk_symmap`, `dual_solve` |
| `Vec::extend(vec)` | `Vec::append(&mut vec)` | `ident_mat`, `matrix_diagonal_mult_sym` |
| `Vec::swap` | three-line swap through a temporary | `gauss_elim` |
| `clone_from` | `clear` + `extend_from_slice` | `sdpad_loop` |
| `&mut vect[..k]` on a `Vec` | `&mut vect.as_mut_slice()[..k]` | `pos_def_solver_upperleft` |
| `if let .. && ..` let-chains | nested `if` | `sdpad_loop` |
| Tuple patterns in parameters `(mat, list): &(..)` | `constr: &(..)` plus `let (mat, list) = constr;` | `constraint_action`, `constraint_lin_comb` |
| `std::f64::consts::FRAC_1_SQRT_2` | `2.0_f64.sqrt()` (also the factor correction) | `linear_remove_redundant_sym` |
| `const`s and `struct`/`enum` declarations outside `verus!` | moved inside `verus!` (`TOL`, `MAX_SWEEPS`, tuning constants, `SDP`, `JacobiVariant`) | `matrix.rs`, `jacobi.rs`, `sdp.rs`, `sdp_altdir.rs` |
| `MAX_SWEEPS * dim * (dim - 1) / 2` | `(dim * (dim - 1) / 2).saturating_mul(MAX_SWEEPS)` | `jacobi_eigen_cach` |
| `println!` inside the solver loop | the loop is `sdpad_loop` returning `SdpadOutcome`; `sdpad` prints outside `verus!` | `sdp_altdir.rs` |
| Long bodies (`sdp_to_standard`, `sdpad`) | split into helpers along the proof structure | `sdp.rs`, `sdp_altdir.rs` |
| Imports of spec-only items | gated with `#[cfg(verus_keep_ghost)]` | spec imports in executable modules |

## Verus notes for this build

- Supported as written: `vec![x; n]`, slicing, `..=` ranges, `for j in it: (0..n).rev()`
  (with `it.index()`, `it.seq()`), `break` in `for`, `let .. else { abort(..) }`,
  `matches` patterns, `Ghost<T>` struct fields, `&mut` struct parameters,
  `int as real`, tuple-of-`&mut` return types (postconditions `final(p)@ == final(r)@`).
- Loop bodies are isolated: restate ghost facts, `v.len()` terms (needed for overflow
  checks) and quantified facts with their triggers as invariants.
- In `for x in a..b` the loop variable is unconstrained at entry when `a > b` is
  possible; prove `a <= b` first or use the named iterator.
- `nonlinear_arith` sees only its `requires`; prove one algebraic step at a time and
  keep monomials bracketed identically; use vstd `div_mod` lemmas for `/` and `%`.
- Closures match syntactically across the spec/proof boundary only when built from
  named spec functions and indexing; name per-term functions.
- An `exists` postcondition needs a ground witness with the trigger's exact terms
  (bind tuple results to `let res = ..` first, or trigger on a result-free term).
- Ghost comparisons of `f64` values are not interpreted (T4 covers executable
  comparisons only); compare `rv` values in ghost code.
- `Ghost<T>` fields must hold `cargo`-buildable types (`f64`, `Seq<f64>`), not `int`
  or `real`; unused spec-only parameters need `#[allow(unused_variables)]` under `cargo`.

## Open issues

None. The final audit (L20) found no trusted construct outside
`src/trusted.rs`, no raised `rlimit`, no slow proof, and no warning from
`cargo build` or `cargo test`. It removed the unused literal axioms `1e3`,
`1e-6` and the unused wrapper `neg_infinity` from the trusted base, annotated
one low-confidence trigger in `part_concat`, and added the intermediate
assertion `0 * rr == 0` in `standard_lmi`: the following assertion
`cm_idx(rr, 0, 0) == 0` verifies in isolation but stopped verifying in the
full-crate context after the trusted axioms changed (proof instability, not
a nonlinear-arithmetic gap); the explicit step makes the proof robust.

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

All 21 layers (L00-L20) are done.

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

## Verified-function count

Last full run: **902 verified, 0 errors** (about 30 s). `--time` reports no
slow proof, and no `rlimit` is raised.

Verus counts executable functions, lemmas, recursive spec functions and loop
bodies; `external_body` items are not counted.

| Module | Verified | Layers |
|--------|---------:|--------|
| `trusted` | 0 | trusted items only |
| `real_spec` | 21 | L00 |
| `lin_alg::vec_spec` | 23 | L01 |
| `lin_alg::sym_spec` | 60 | L02, L03 |
| `lin_alg::mat_spec` | 76 | L03, L07, L12 |
| `lin_alg::constr_spec` | 30 | L04 |
| `lin_alg::solve_spec` | 11 | L05 |
| `lin_alg::ldlt_spec` | 13 | L06, L07 |
| `lin_alg::jacobi_spec` | 61 | L08 |
| `lin_alg::eigen_spec` | 21 | L09-L11 |
| `lin_alg::spectral_spec` | 10 | L12 |
| `lin_alg::gauss_spec` | 29 | L13 |
| `lin_alg::reduce_spec` | 26 | L14 |
| `lin_alg::matrix` | 47 | L01, L13 |
| `lin_alg::comparison` | 6 | L00 |
| `lin_alg::sym_matrix` | 168 | L02, L04-L07, L14, L15 |
| `lin_alg::jacobi` | 41 | L08, L10-L12 |
| `sdp_spec` | 70 | L16 |
| `sdpad_spec` | 37 | L17-L19 |
| `sdp` | 114 | L15, L16 |
| `sdp_altdir` | 38 | L17-L19 |
| **total** | **902** | |

## Main contracts

- **Comparisons** (`comparison.rs`): the exact-real reading of each tolerance
  test, e.g. `scaled_near_zero(x, s, tol)` iff `|x| < tol * s`.
- **Dense matrices** (`matrix.rs`): `gauss_elim` preserves the solution set,
  keeps zero columns, produces echelon form under the tolerance-based
  pivot/skip rules, and has `rank >= 1` for nonzero input.
- **Constraint operator** (`sym_matrix.rs`): `constraint_action` is `A(X)`;
  `constraint_lin_comb` is `A*(y)` with the off-diagonal factors 1/2 and
  satisfies `<A(X), y> = <X, A*(y)>_F`; `diag_scale_gram` is `Q D Q^T`.
- **Solvers** (`sym_matrix.rs`): `forw_subst`, `back_subst`, `diag_solver`
  solve `L x = b`, `L^T x = b`, `D x = b`; `ldlt_decomp` gives `L D L^T = A`
  with pivots `>= TOL * max(max_i |A_ii|, 1)`; `pos_def_solver_upperleft`
  solves the block system `(A, I/2) x = b`.
- **PSD check**: success of `psd_block_check` gives `B >= -eps I` for each
  block, `eps = 1e-9 * max(max_i |B_ii|, 1)`.
- **Reduction**: `linear_remove_redundant_sym` returns the first `rank`
  echelon rows divided by their Frobenius norms.
- **Jacobi** (`jacobi.rs`): `jacobi_rot` gives `c^2 + s^2 = 1`,
  `A' = J^T A J`, `Q' = Q J`, `A'_jk = 0`. Both eigensolvers establish
  `eigensolver_ok`: `Q^T Q = Q Q^T = I`, `A = Q B Q^T`, `vals = diag(B)`,
  `||B - diag(B)||_F <= sqrt(dim (dim - 1)) * 1e-12 * max(1, max_i |B_ii|)`.
  The cached pivot has maximal off-diagonal magnitude; each cyclic sweep
  visits every pair `j < k` once in column-cyclic order.
- **Spectral parts**: `poseigendecomp` and `negeigendecomp` return the pairs
  `(B_kk, q_k)` with `B_kk > 0`, resp. `B_kk < -1e-8 * max(max_t |A_t|, 1)`
  (`spectral_part_ok`). Nothing relates them to the exact PSD projection.
- **Standard form** (`sdp.rs`): `sdp_to_standard` preserves the objective,
  the equality and LMI rows (`std_rows_ok`), the zero pattern (`zeros_ok`)
  and variable reconstruction (`Z_n = B_{s+2n-1} - B_{s+2n}`); the reduction
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
  so call sites are unchanged. `abs_r`, `max_r`, `min_r`, `sqrt_r` live in
  `real_spec.rs`.
- **Views.** Vectors are `rvs(v@): Seq<real>`. Packed symmetric entry
  `(r, c)`, `r <= c`, sits at `pk_idx(r, c) = c (c + 1) / 2 + r`; `sym_entry`
  extends it by symmetry. Column-major uses `cm_idx`, packed lower-triangular
  `lt_idx`. Spec matrices are `spec_fn(int, int) -> real` with block equality
  `meq`, entered through `sym_view`, `cm_view`, `lt_view`, `diag_view`,
  `block_view`. `dim_fits(dim) := (dim + 2)^2 <= usize::MAX` covers all packed
  index arithmetic.
- **Sums.** `sum_fn_r` over closures, `sum2_fn_r` for square double sums,
  `tri_sum_r` for triangles. The Frobenius product `frob_r` counts
  off-diagonal entries twice by proof, not by definition. Per-term spec
  functions are named so that closures match syntactically at call sites.
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
  through `sdpad_setup_ok`; each iteration is specified by what the code
  computes (`wgy_update_ok`, `penalty_update_ok`); the loop by `outcome_ok`
  over the ghost trajectory, extended by `lemma_trajectory_step`. Nothing is
  claimed about convergence.

## Executable changes

The four bugs in `EXPOSED_BUGS.md` were fixed. Other changes:

- The off-diagonal row-norm factor of `linear_remove_redundant_sym` is
  `sqrt(2)` instead of `1/sqrt(2)` (author's decision), so the scale is the
  Frobenius norm of the row's symmetric matrix.
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
- `jacobi_eigen_cach`: `biggest` starts at `0.0`; the rotation limit uses
  `saturating_mul`; the
  eigenvector matrix is returned without a final copy; regression test
  `test_jacobi_cach_cache_update` in `src/sdp_tests.rs`.

## Constructs rewritten for Verus compatibility

Every rewrite keeps the executable cost and was confirmed by a probe against
the installed Verus (0.2026.08.23).

| Construct | Replacement | Where |
|-----------|-------------|-------|
| Unary `-x` on `f64` | `crate::trusted::neg(x)` (T7) | `is_significantly_negative`, `negscal_vect` |
| `f64::INFINITY` | `pos_infinity()` (T7) | `sdpad_loop` sentinels |
| `x += y`, `v[i] -= y`, `v[i] *= y` on `f64` (compiler crash) | `x = x + y`, etc. | all accumulation loops |
| `panic!`, `assert!`, `expect`, `unwrap_or_else(panic)` on data-dependent conditions | `crate::trusted::abort(..)` (T8); internal consistency checks became static `assert`s | solvers, eigensolvers, `linear_remove_redundant_sym`, `sdp_to_standard`, `dual_solve` |
| Iterator chains (`zip`, `map`, `fold`, `enumerate`, `chunks_exact`, `scan`, `chain`, `sum`, `concat`), `for x in &v` | index loops over the same slices | throughout |
| `continue` in `for` loops | `if` block, or `while` with explicit counter | `jacobi_rot`, `jacobi_eigen_cyc`, `gauss_elim` |
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
| `std::f64::consts::FRAC_1_SQRT_2` | `2.0_f64.sqrt()` (with the factor change above) | `linear_remove_redundant_sym` |
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
  `/` and `%`.
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
- Some assertions verify in isolation but not in the full crate. Intermediate
  assertions or a small wrapping lemma fix this (`standard_lmi`,
  `lemma_objective_shapes`).

## Open issues

None.

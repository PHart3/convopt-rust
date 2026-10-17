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
| L07 .. L20 | not started | |

## Verified-function count

Last full run: **335 verified, 0 errors** (Verus `--time`: 6.3 s SMT time; no slow
proofs; the whole run takes 14 s).

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

Known constructs that later layers must rewrite (found during L00 experiments):

- `x += y`, `x -= y`, `x *= y` on `f64` (including `v[i] *= y`) crash this Verus build
  (`get_range Float(64)`); write `x = x + y` (`v[i] = v[i] * y` verifies).
- `f64::INFINITY`, `f64::NEG_INFINITY`, `core::f64::consts::FRAC_1_SQRT_2`: use the
  wrappers in `src/trusted.rs`.
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

- `constraint_action` and `constraint_lin_comb` ignore the zero-pattern list when the dense
  constraint matrix is empty (early return in the original code). The verified
  postconditions state this behaviour; whether `sdpad` can reach it with a nonempty list
  (an SDP with LMI blocks but no linear constraints) must be settled in L17/L18.

- The full-verification command in CLAUDE.md needs `-V no-solver-version-check` on
  this machine (Z3 4.16.0 with build hashcode); without it Verus panics as soon as any
  function is verified.
- `cargo build` warns that `use vstd::prelude::*` in `src/lib.rs` is unused; the import
  is required by CLAUDE.md and is harmless.

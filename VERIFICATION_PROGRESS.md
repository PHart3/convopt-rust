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
| L03 .. L20 | not started | |

## Verified-function count

Last full run: **138 verified, 0 errors** (Verus `--time`: 2.6 s verify time; the
15-function `sym_matrix` module takes 1.5 s in total, no slow proofs).

- 15 executable symmetric-matrix functions (`src/lin_alg/sym_matrix.rs`), each with an
  exact-real postcondition (L02).
- 14 executable vector/matrix functions (`src/lin_alg/matrix.rs`) (L01).
- 6 executable comparison functions (`src/lin_alg/comparison.rs`) (L00).
- 33 proved lemmas in `src/lin_alg/sym_spec.rs`: packed index arithmetic (column starts,
  entry bounds, closed-form diagonal index, `usize` fit), sums/maxima over index
  functions (extensionality, additivity, single-term update, nonnegativity, scaling,
  agreement with `sum_r`), square double-sum expansion (general and symmetric),
  column form of the Frobenius product, `<A, A>_F >= 0`, and the sweep bookkeeping
  (init, off-diagonal step, diagonal step, next column, done).
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

## Constructs rewritten for Verus compatibility

| Construct | Replacement | Where |
|-----------|-------------|-------|
| `-(expr)` on `f64` (unsupported by Verus) | `crate::trusted::neg(expr)` (inline wrapper) | `is_significantly_negative`, `negscal_vect` |
| `iter().zip(..)` / `for v in vect` loops | index loops `for i in 0..n` (same cost) | `dot_prod`, `euclid_distance`, `vect_add`, `vect_subt`, `scal_vect`, `negscal_vect`, `vect_add_scaled`, `matrix_add_diff`, `vector_equality` |
| `Vec::extend(vec)` (no vstd spec) | `Vec::append(&mut vec)` (same memcpy) | `ident_mat` |
| imports of spec-only items / `vstd::std_specs` (erased under `cargo`) | gated with `#[cfg(verus_keep_ghost)]` | `trusted.rs`, `real_spec.rs`, `comparison.rs`, `matrix.rs`, `sym_matrix.rs` |
| `x += y`, `v[i] += y`, `v[i] *= y` on `f64` (compiler crash) | `x = x + y`, `v[i] = v[i] + y`, `v[i] = v[i] * y` | `frob_prod_sym`, `sym_mat_one_norm`, `sym_mat_act`, `scale_off_diag` |
| `iter().fold(0.0, max)` | index loop | `sym_mat_one_norm` |

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
- `vec![x; n]`, `&v[a..b]` slicing, `v.first()`, `Option::unwrap_or` and `for j in 0..=i`
  loops are supported by vstd (probed 2026-10-17); keep such code as written.
- Before rewriting a construct, probe it in a scratch file: every entry in the table above
  was re-confirmed by a probe on 2026-10-17 (unary `-x`, `+=` on `f64`, `INFINITY`,
  `FRAC_1_SQRT_2`, `Vec::extend`, `zip`, `fold` all fail in this build).

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

- The full-verification command in CLAUDE.md needs `-V no-solver-version-check` on
  this machine (Z3 4.16.0 with build hashcode); without it Verus panics as soon as any
  function is verified.
- `cargo build` warns that `use vstd::prelude::*` in `src/lib.rs` is unused; the import
  is required by CLAUDE.md and is harmless.

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
| L02 .. L20 | not started | |

## Verified-function count

Last full run: **59 verified, 0 errors** (Verus `--time`: 1.4 s verify time, no slow proofs).

- 14 executable vector/matrix functions (`src/lin_alg/matrix.rs`), each with an exact-real postcondition (L01).
- 6 executable comparison functions (`src/lin_alg/comparison.rs`), each with an exact-real postcondition (L00).
- 12 proved lemmas in `src/lin_alg/vec_spec.rs`: packed lower-triangular column offsets (zero, step, end, monotone, index bound), prefix sums (`take` step, `take` all), nonnegative sums, `<a, a> >= 0`.
- 3 proved lemmas about `sqrt_r` (`src/real_spec.rs`): uniqueness, `sqrt(x^2) = |x|`, monotonicity.
- 5 `external_body` wrappers in `src/trusted.rs` (counted as verified by Verus; they are trusted, see TRUSTED_BASE.md).
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

## Constructs rewritten for Verus compatibility

| Construct | Replacement | Where |
|-----------|-------------|-------|
| `-(expr)` on `f64` (unsupported by Verus) | `crate::trusted::neg(expr)` (inline wrapper) | `is_significantly_negative`, `negscal_vect` |
| `iter().zip(..)` / `for v in vect` loops | index loops `for i in 0..n` (same cost) | `dot_prod`, `euclid_distance`, `vect_add`, `vect_subt`, `scal_vect`, `negscal_vect`, `vect_add_scaled`, `matrix_add_diff`, `vector_equality` |
| `Vec::extend(vec)` (no vstd spec) | `Vec::append(&mut vec)` (same memcpy) | `ident_mat` |
| imports of spec-only items / `vstd::std_specs` (erased under `cargo`) | gated with `#[cfg(verus_keep_ghost)]` | `trusted.rs`, `real_spec.rs`, `comparison.rs` |

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
- `vec![x; n]` and `&v[a..b]` slicing are supported by vstd.

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

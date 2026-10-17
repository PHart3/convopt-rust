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
| L01 .. L20 | not started | |

## Verified-function count

Last full run: **14 verified, 0 errors** (Verus `--time`: 0.24 s total).

- 6 executable comparison functions (`src/lin_alg/comparison.rs`), each with an exact-real postcondition.
- 3 proved lemmas about `sqrt_r` (`src/real_spec.rs`): uniqueness, `sqrt(x^2) = |x|`, monotonicity.
- 5 `external_body` wrappers in `src/trusted.rs` (counted as verified by Verus; they are trusted, see TRUSTED_BASE.md).

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

## Constructs rewritten for Verus compatibility

| Construct | Replacement | Where |
|-----------|-------------|-------|
| `-(expr)` on `f64` (unsupported by Verus) | `crate::trusted::neg(expr)` (inline wrapper) | `is_significantly_negative` |
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

## Open issues

- The full-verification command in CLAUDE.md needs `-V no-solver-version-check` on
  this machine (Z3 4.16.0 with build hashcode); without it Verus panics as soon as any
  function is verified.
- `cargo build` warns that `use vstd::prelude::*` in `src/lib.rs` is unused; the import
  is required by CLAUDE.md and is harmless.

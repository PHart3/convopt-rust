// TRUSTED BASE: exact-real abstraction of f64 arithmetic.
//
// Every trusted item of the crate (axioms, assume_specification, external_body)
// lives in this file and is documented in TRUSTED_BASE.md.  No other file may
// introduce trusted items.
//
// Model: `rv(x)` is the exact real value denoted by the f64 `x`.  Primitive
// f64 operations are abstracted as the corresponding exact-real operations;
// rounding, overflow, NaN and infinities are not modeled.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use crate::real_spec::{abs_r, max_r, min_r};
#[cfg(verus_keep_ghost)]
use vstd::std_specs::ops::{AddSpec, SubSpec, MulSpec, DivSpec, add_ensures, sub_ensures, mul_ensures, div_ensures};
#[cfg(verus_keep_ghost)]
use vstd::std_specs::cmp::{eq_ensures, ne_ensures, lt_ensures, le_ensures, gt_ensures, ge_ensures};

verus! {

/* ---------- spec-level vocabulary ---------- */

/// Exact real value of an f64 (uninterpreted; constrained only by the axioms below).
pub uninterp spec fn rv(x: f64) -> real;

/// Nonnegative square root on reals (uninterpreted; axiomatized for x >= 0 only).
pub uninterp spec fn sqrt_r(x: real) -> real;

/// T1: sqrt_r is the nonnegative square root on nonnegative reals.
pub broadcast axiom fn axiom_sqrt_r(x: real)
    requires
        x >= 0.0real,
    ensures
        #[trigger] sqrt_r(x) >= 0.0real,
        sqrt_r(x) * sqrt_r(x) == x,
;

/* ---------- T2: primitive f64 operators are total ---------- */

pub broadcast axiom fn axiom_add_req(a: f64, b: f64)
    ensures
        #[trigger] a.add_req(b),
;

pub broadcast axiom fn axiom_sub_req(a: f64, b: f64)
    ensures
        #[trigger] a.sub_req(b),
;

pub broadcast axiom fn axiom_mul_req(a: f64, b: f64)
    ensures
        #[trigger] a.mul_req(b),
;

pub broadcast axiom fn axiom_div_req(a: f64, b: f64)
    ensures
        #[trigger] a.div_req(b),
;

/* ---------- T3: primitive f64 operators compute exact-real results ---------- */

pub broadcast axiom fn axiom_add(a: f64, b: f64, o: f64)
    requires
        #[trigger] add_ensures::<f64>(a, b, o),
    ensures
        rv(o) == rv(a) + rv(b),
;

pub broadcast axiom fn axiom_sub(a: f64, b: f64, o: f64)
    requires
        #[trigger] sub_ensures::<f64>(a, b, o),
    ensures
        rv(o) == rv(a) - rv(b),
;

pub broadcast axiom fn axiom_mul(a: f64, b: f64, o: f64)
    requires
        #[trigger] mul_ensures::<f64>(a, b, o),
    ensures
        rv(o) == rv(a) * rv(b),
;

/// Division by a nonzero value is exact; division by zero is left unspecified.
pub broadcast axiom fn axiom_div(a: f64, b: f64, o: f64)
    requires
        #[trigger] div_ensures::<f64>(a, b, o),
        rv(b) != 0.0real,
    ensures
        rv(o) == rv(a) / rv(b),
;

/* ---------- T4: f64 comparisons agree with exact-real comparisons ---------- */

pub broadcast axiom fn axiom_eq(a: f64, b: f64, o: bool)
    requires
        #[trigger] eq_ensures::<f64>(a, b, o),
    ensures
        o == (rv(a) == rv(b)),
;

pub broadcast axiom fn axiom_ne(a: f64, b: f64, o: bool)
    requires
        #[trigger] ne_ensures::<f64>(a, b, o),
    ensures
        o == (rv(a) != rv(b)),
;

pub broadcast axiom fn axiom_lt(a: f64, b: f64, o: bool)
    requires
        #[trigger] lt_ensures::<f64>(a, b, o),
    ensures
        o == (rv(a) < rv(b)),
;

pub broadcast axiom fn axiom_le(a: f64, b: f64, o: bool)
    requires
        #[trigger] le_ensures::<f64>(a, b, o),
    ensures
        o == (rv(a) <= rv(b)),
;

pub broadcast axiom fn axiom_gt(a: f64, b: f64, o: bool)
    requires
        #[trigger] gt_ensures::<f64>(a, b, o),
    ensures
        o == (rv(a) > rv(b)),
;

pub broadcast axiom fn axiom_ge(a: f64, b: f64, o: bool)
    requires
        #[trigger] ge_ensures::<f64>(a, b, o),
    ensures
        o == (rv(a) >= rv(b)),
;

/* ---------- T5: f64 literals denote their decimal values ---------- */

/// The f64 literals occurring in the verified code denote their decimal values exactly.
pub broadcast axiom fn axiom_literals()
    ensures
        #[trigger] rv(0.0f64) == 0.0real,
        #[trigger] rv(0.5f64) == 0.5real,
        #[trigger] rv(1.0f64) == 1.0real,
        #[trigger] rv(1.6f64) == 1.6real,
        #[trigger] rv(2.0f64) == 2.0real,
        #[trigger] rv(5.0f64) == 5.0real,
        #[trigger] rv(1e3f64) == 1000.0real,
        #[trigger] rv(1e4f64) == 10000.0real,
        #[trigger] rv(1e-3f64) == 0.001real,
        #[trigger] rv(1e-4f64) == 0.0001real,
        #[trigger] rv(1e-5f64) == 0.00001real,
        #[trigger] rv(1e-6f64) == 0.000001real,
        #[trigger] rv(1e-8f64) == 0.00000001real,
        #[trigger] rv(1e-9f64) == 0.000000001real,
        #[trigger] rv(1e-10f64) == 0.0000000001real,
        #[trigger] rv(1e-12f64) == 0.000000000001real,
;

pub broadcast group group_real_model {
    axiom_sqrt_r,
    axiom_add_req,
    axiom_sub_req,
    axiom_mul_req,
    axiom_div_req,
    axiom_add,
    axiom_sub,
    axiom_mul,
    axiom_div,
    axiom_eq,
    axiom_ne,
    axiom_lt,
    axiom_le,
    axiom_gt,
    axiom_ge,
    axiom_literals,
}

/* ---------- T6: exact-real contracts for f64 methods ---------- */

pub assume_specification[ f64::abs ](x: f64) -> (r: f64)
    ensures
        rv(r) == abs_r(rv(x)),
;

pub assume_specification[ f64::max ](x: f64, y: f64) -> (r: f64)
    ensures
        rv(r) == max_r(rv(x), rv(y)),
;

pub assume_specification[ f64::min ](x: f64, y: f64) -> (r: f64)
    ensures
        rv(r) == min_r(rv(x), rv(y)),
;

/// Square root of a nonnegative value is exact; of a negative value it is unspecified.
pub assume_specification[ f64::sqrt ](x: f64) -> (r: f64)
    ensures
        rv(x) >= 0.0real ==> rv(r) == sqrt_r(rv(x)),
;

/// Fused multiply-add: x * a + b.
pub assume_specification[ f64::mul_add ](x: f64, a: f64, b: f64) -> (r: f64)
    ensures
        rv(r) == rv(x) * rv(a) + rv(b),
;

/// hypot(x, y) = sqrt(x^2 + y^2).
pub assume_specification[ f64::hypot ](x: f64, y: f64) -> (r: f64)
    ensures
        rv(r) == sqrt_r(rv(x) * rv(x) + rv(y) * rv(y)),
;

/// Reciprocal of a nonzero value is exact; of zero it is unspecified.
pub assume_specification[ f64::recip ](x: f64) -> (r: f64)
    ensures
        rv(x) != 0.0real ==> rv(r) == 1.0real / rv(x),
;

/// signum is +1 or -1, agreeing with the sign of nonzero inputs (either sign for zero).
pub assume_specification[ f64::signum ](x: f64) -> (r: f64)
    ensures
        rv(r) == 1.0real || rv(r) == -1.0real,
        rv(x) > 0.0real ==> rv(r) == 1.0real,
        rv(x) < 0.0real ==> rv(r) == -1.0real,
;

/// Finiteness is not modeled: the result is unconstrained.
pub assume_specification[ f64::is_finite ](x: f64) -> bool;

/* ---------- T7: executable wrappers for unsupported primitives ---------- */

/// Unary negation (Verus does not support `-x` on f64).
#[verifier::external_body]
#[inline(always)]
pub fn neg(x: f64) -> (r: f64)
    ensures
        rv(r) == -rv(x),
{
    -x
}

/// f64::INFINITY; its real value is unconstrained.
#[verifier::external_body]
#[inline(always)]
pub fn pos_infinity() -> f64 {
    f64::INFINITY
}

/// f64::NEG_INFINITY; its real value is unconstrained.
#[verifier::external_body]
#[inline(always)]
pub fn neg_infinity() -> f64 {
    f64::NEG_INFINITY
}

/// 1 / sqrt(2), characterized as the positive root of r^2 = 1/2.
#[verifier::external_body]
#[inline(always)]
pub fn frac_1_sqrt_2() -> (r: f64)
    ensures
        rv(r) > 0.0real,
        rv(r) * rv(r) == 0.5real,
{
    core::f64::consts::FRAC_1_SQRT_2
}

/* ---------- T8: program abort ---------- */

/// Aborts the program with a message.  Nothing is claimed about reachability:
/// this is the verified counterpart of the data-dependent `panic!` calls in the solver.
#[verifier::external_body]
#[inline(never)]
pub fn abort(msg: &str) -> ! {
    panic!("{}", msg)
}

} // verus!

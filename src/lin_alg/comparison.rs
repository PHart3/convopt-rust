// A handful of numerical comparison functions that are used in the linear algebra routines.
//
// Each function is specified exactly in terms of the real values of its arguments
// (see src/trusted.rs for the exact-real abstraction of f64).

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use crate::trusted::rv;
#[cfg(verus_keep_ghost)]
use crate::real_spec::{abs_r, max_r};

verus! {

broadcast use crate::trusted::group_real_model;

/// a and b agree up to absolute or relative tolerance tol.
pub open spec fn approx_equal(a: real, b: real, tol: real) -> bool {
    abs_r(a - b) <= tol || abs_r(a - b) <= tol * max_r(abs_r(b), abs_r(a))
}

/// |value| < tol * scale.
pub open spec fn near_zero(value: real, scale: real, tol: real) -> bool {
    abs_r(value) < tol * scale
}

#[inline]
pub fn float_equality(a: f64, b: f64, tol: f64) -> (r: bool)
    ensures
        r == approx_equal(rv(a), rv(b), rv(tol)),
{
    let diff = (a - b).abs();
    diff <= tol || diff <= tol * b.abs().max(a.abs())
}

#[inline]
pub(crate) fn scaled_near_zero(value: f64, scale: f64, tol: f64) -> (r: bool)
    ensures
        r == near_zero(rv(value), rv(scale), rv(tol)),
{
    value.abs() < tol * scale
}

#[inline]
pub(crate) fn pivot_below_positive_threshold(pivot: f64, scale: f64, tol: f64) -> (r: bool)
    ensures
        r == (rv(pivot) < rv(tol) * rv(scale)),
{
    pivot < tol * scale
}

#[inline]
pub(crate) fn is_significantly_negative(value: f64, scale: f64, tol: f64) -> (r: bool)
    ensures
        r == (rv(value) < -(rv(tol) * rv(scale))),
{
    value < crate::trusted::neg(tol * scale)
}

#[inline]
pub(crate) fn residual_is_small(residual: f64, tol: f64) -> (r: bool)
    ensures
        r == (rv(residual) < rv(tol)),
{
    residual < tol
}

/// Relative residual check; unspecified when scale is zero.
#[inline]
pub(crate) fn relative_residual_is_small(residual: f64, scale: f64, tol: f64) -> (r: bool)
    ensures
        rv(scale) != 0.0real ==> r == (rv(residual) / rv(scale) < rv(tol)),
{
    residual_is_small(residual / scale, tol)
}

} // verus!

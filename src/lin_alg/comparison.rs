// A handful of numerical comparison functions that are used in the linear algebra routines.

#[inline]
pub fn float_equality(a: f64, b: f64, tol: f64) -> bool {
    let diff = (a - b).abs();
    diff <= tol || diff <= tol * b.abs().max(a.abs())
}

#[inline]
pub(crate) fn scaled_near_zero(value: f64, scale: f64, tol: f64) -> bool {
    value.abs() < tol * scale
}

#[inline]
pub(crate) fn pivot_below_positive_threshold(pivot: f64, scale: f64, tol: f64) -> bool {
    pivot < tol * scale
}

#[inline]
pub(crate) fn is_significantly_negative(value: f64, scale: f64, tol: f64) -> bool {
    value < -(tol * scale)
}

#[inline]
pub(crate) fn residual_is_small(residual: f64, tol: f64) -> bool {
    residual < tol
}

#[inline]
pub(crate) fn relative_residual_is_small(residual: f64, scale: f64, tol: f64) -> bool {
    residual_is_small(residual / scale, tol)
}

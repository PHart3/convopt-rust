// Spec-level real-number vocabulary and proved lemmas (not trusted).

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use crate::trusted::sqrt_r;

verus! {

broadcast use crate::trusted::group_real_model;

pub open spec fn abs_r(x: real) -> real {
    if x >= 0.0real { x } else { -x }
}

pub open spec fn max_r(x: real, y: real) -> real {
    if x >= y { x } else { y }
}

pub open spec fn min_r(x: real, y: real) -> real {
    if x <= y { x } else { y }
}

/// The nonnegative square root is unique.
pub proof fn lemma_sqrt_unique(x: real, y: real)
    requires
        x >= 0.0real,
        y >= 0.0real,
        y * y == x,
    ensures
        sqrt_r(x) == y,
{
    let s = sqrt_r(x);
    assert((s - y) * (s + y) == 0.0real) by (nonlinear_arith)
        requires s * s == x, y * y == x;
    if s + y == 0.0real {
        assert(s == 0.0real && y == 0.0real);
    } else {
        assert(s - y == 0.0real) by (nonlinear_arith)
            requires (s - y) * (s + y) == 0.0real, s + y != 0.0real;
    }
}

/// sqrt(x * x) = |x|.
pub proof fn lemma_sqrt_square(x: real)
    ensures
        sqrt_r(x * x) == abs_r(x),
{
    assert(x * x >= 0.0real) by (nonlinear_arith);
    assert(abs_r(x) * abs_r(x) == x * x) by (nonlinear_arith);
    lemma_sqrt_unique(x * x, abs_r(x));
}

/// sqrt is monotone on nonnegative reals.
pub proof fn lemma_sqrt_monotone(x: real, y: real)
    requires
        0.0real <= x <= y,
    ensures
        sqrt_r(x) <= sqrt_r(y),
{
    let (s, t) = (sqrt_r(x), sqrt_r(y));
    assert(s <= t) by (nonlinear_arith)
        requires s >= 0.0real, t >= 0.0real, s * s == x, t * t == y, x <= y;
}

} // verus!

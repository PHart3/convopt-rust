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

/// x^2 >= 0.
pub proof fn lemma_sq_nonneg(x: real)
    ensures
        x * x >= 0.0real,
{
    assert(x * x >= 0.0real) by (nonlinear_arith);
}

/// Single-step product identities, proved once so that call sites need no nonlinear query.
pub proof fn lemma_mul_comm(x: real, y: real)
    ensures
        x * y == y * x,
{
    assert(x * y == y * x) by (nonlinear_arith);
}

pub proof fn lemma_real_mul_assoc(x: real, y: real, z: real)
    ensures
        (x * y) * z == x * (y * z),
{
    assert((x * y) * z == x * (y * z)) by (nonlinear_arith);
}

/// (x y)(z w) = (x z)(y w).
pub proof fn lemma_mul_rearrange(x: real, y: real, z: real, w: real)
    ensures
        (x * y) * (z * w) == (x * z) * (y * w),
{
    assert((x * y) * (z * w) == (x * z) * (y * w)) by (nonlinear_arith);
}

/// x (y - z) = x y - x z.
pub proof fn lemma_mul_sub_right(x: real, y: real, z: real)
    ensures
        x * (y - z) == x * y - x * z,
{
    assert(x * (y - z) == x * y - x * z) by (nonlinear_arith);
}

/// (-2 x) y = -(2 x y).
pub proof fn lemma_neg_two_mul(x: real, y: real)
    ensures
        (-(2.0real * x)) * y == -(2.0real * (x * y)),
{
    assert((-(2.0real * x)) * y == -(2.0real * (x * y))) by (nonlinear_arith);
}

/// (p - q)^2 = p^2 - 2 p q + q^2.
pub proof fn lemma_sq_diff(p: real, q: real)
    ensures
        (p - q) * (p - q) == p * p - 2.0real * (p * q) + q * q,
{
    assert((p - q) * (p - q) == p * p - 2.0real * (p * q) + q * q) by (nonlinear_arith);
}

/// x > 0 and x t >= 0 give t >= 0.
pub proof fn lemma_nonneg_of_pos_factor(x: real, t: real)
    requires
        x > 0.0real,
        x * t >= 0.0real,
    ensures
        t >= 0.0real,
{
    assert(t >= 0.0real) by (nonlinear_arith)
        requires x > 0.0real, x * t >= 0.0real;
}

/// |x y| = |x| |y|.
pub proof fn lemma_abs_mul(x: real, y: real)
    ensures
        abs_r(x * y) == abs_r(x) * abs_r(y),
{
    if x >= 0.0real && y >= 0.0real {
        assert(x * y >= 0.0real) by (nonlinear_arith)
            requires x >= 0.0real, y >= 0.0real;
    } else if x >= 0.0real {
        assert(x * y <= 0.0real && x * (-y) == -(x * y)) by (nonlinear_arith)
            requires x >= 0.0real, y < 0.0real;
    } else if y >= 0.0real {
        assert(x * y <= 0.0real && (-x) * y == -(x * y)) by (nonlinear_arith)
            requires x < 0.0real, y >= 0.0real;
    } else {
        assert(x * y > 0.0real && (-x) * (-y) == x * y) by (nonlinear_arith)
            requires x < 0.0real, y < 0.0real;
    }
}

/// |a - b| <= |a| + |b|.
pub proof fn lemma_abs_sub_triangle(a: real, b: real)
    ensures
        abs_r(a - b) <= abs_r(a) + abs_r(b),
{
}

/// |num / den| <= 1 when |num| <= |den| and den != 0.
pub proof fn lemma_abs_quotient_le_one(num: real, den: real)
    requires
        den != 0.0real,
        abs_r(num) <= abs_r(den),
    ensures
        abs_r(num / den) <= 1.0real,
{
    let q = num / den;
    assert((num / den) * den == num) by (nonlinear_arith)
        requires den != 0.0real;
    lemma_abs_mul(q, den);
    assert(abs_r(q) <= 1.0real) by (nonlinear_arith)
        requires abs_r(q) * abs_r(den) <= abs_r(den), abs_r(den) > 0.0real;
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
    lemma_sq_nonneg(x);
    assert(abs_r(x) * abs_r(x) == x * x) by (nonlinear_arith);
    lemma_sqrt_unique(x * x, abs_r(x));
}

/// sqrt(x y) = sqrt(x) sqrt(y) for nonnegative x, y.
pub proof fn lemma_sqrt_mul(x: real, y: real)
    requires
        x >= 0.0real,
        y >= 0.0real,
    ensures
        sqrt_r(x * y) == sqrt_r(x) * sqrt_r(y),
{
    let (s, t) = (sqrt_r(x), sqrt_r(y));
    assert(x * y >= 0.0real) by (nonlinear_arith)
        requires x >= 0.0real, y >= 0.0real;
    assert(s * t >= 0.0real) by (nonlinear_arith)
        requires s >= 0.0real, t >= 0.0real;
    assert((s * t) * (s * t) == (s * s) * (t * t)) by (nonlinear_arith);
    lemma_sqrt_unique(x * y, s * t);
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

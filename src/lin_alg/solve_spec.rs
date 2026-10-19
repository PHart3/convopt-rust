// Spec-level support for the linear-system solvers (not trusted).
//
// The substitution solvers (`forw_subst`, `back_subst`) and the diagonal solver
// (`diag_solver`) are specified as `mat_vec(view, dim, solution) == rhs` over the
// views of mat_spec.rs.  Their loops accumulate row sums over a growing or
// shrinking index range, so this file provides partial row sums (`row_prefix_r`,
// `row_range_r`), the lemmas connecting them to `mat_vec` for lower-triangular,
// upper-triangular (transposed lower-triangular) and diagonal views, and the
// solvers' pivot scale `max(max_t |m_t|, 1)`.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::seq::*;
#[cfg(verus_keep_ghost)]
use crate::real_spec::{abs_r, max_r};
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::*;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- pivot scale ---------- */

/// max(0, |v_0|, ..., |v_{n-1}|), computed as a left fold.
pub open spec fn abs_max_prefix_r(v: Seq<real>, n: nat) -> real {
    max0_fn_r(|t: int| abs_r(v[t]), n)
}

/// max(0, |v_0|, ..., |v_{len-1}|).
pub open spec fn abs_max_r(v: Seq<real>) -> real {
    abs_max_prefix_r(v, v.len())
}

/// The solvers' pivot scale: max(abs_max_r(v), 1).
pub open spec fn pivot_scale_r(v: Seq<real>) -> real {
    max_r(abs_max_r(v), 1.0real)
}

/// One fold step of the prefix maximum.
pub proof fn lemma_abs_max_prefix_step(v: Seq<real>, n: nat)
    ensures
        abs_max_prefix_r(v, n + 1) == max_r(abs_max_prefix_r(v, n), abs_r(v[n as int])),
{
}

/// A positive prefix maximum is attained by some entry.
pub proof fn lemma_abs_max_prefix_attained(v: Seq<real>, n: nat)
    requires
        abs_max_prefix_r(v, n) > 0.0real,
    ensures
        exists|t: int| 0 <= t < n && abs_r(#[trigger] v[t]) == abs_max_prefix_r(v, n),
    decreases n,
{
    if n > 0 {
        lemma_abs_max_prefix_step(v, (n - 1) as nat);
        if abs_r(v[n - 1]) == abs_max_prefix_r(v, n) {
            assert(abs_r(v[n - 1]) == abs_max_prefix_r(v, n));
        } else {
            lemma_abs_max_prefix_attained(v, (n - 1) as nat);
        }
    }
}

/// A positive largest magnitude is attained by some entry.
pub proof fn lemma_abs_max_attained(v: Seq<real>)
    requires
        abs_max_r(v) > 0.0real,
    ensures
        exists|t: int| 0 <= t < v.len() && abs_r(#[trigger] v[t]) == abs_max_r(v),
{
    lemma_abs_max_prefix_attained(v, v.len());
}

/* ---------- partial row sums ---------- */

/// sum_{k < n} A_ik x_k: row i of A against the first n entries of x.
pub open spec fn row_prefix_r(a: MatF, x: Seq<real>, i: int, n: nat) -> real {
    sum_fn_r(|k: int| a(i, k) * x[k], n)
}

/// sum_{lo <= k < hi} A_ik x_k.
pub open spec fn row_range_r(a: MatF, x: Seq<real>, i: int, lo: nat, hi: nat) -> real {
    row_prefix_r(a, x, i, hi) - row_prefix_r(a, x, i, lo)
}

pub proof fn lemma_row_prefix_step(a: MatF, x: Seq<real>, i: int, n: nat)
    ensures
        row_prefix_r(a, x, i, n + 1) == row_prefix_r(a, x, i, n) + a(i, n as int) * x[n as int],
{
}

/// The prefix sum only reads the first n entries of x.
pub proof fn lemma_row_prefix_ext(a: MatF, x: Seq<real>, y: Seq<real>, i: int, n: nat)
    requires
        forall|k: int| 0 <= k < n ==> x[k] == y[k],
    ensures
        row_prefix_r(a, x, i, n) == row_prefix_r(a, y, i, n),
{
    let f = |k: int| a(i, k) * x[k];
    let g = |k: int| a(i, k) * y[k];
    assert forall|k: int| 0 <= k < n implies #[trigger] f(k) == g(k) by {}
    lemma_sum_fn_ext(f, g, n);
}

pub proof fn lemma_row_range_step(a: MatF, x: Seq<real>, i: int, lo: nat, hi: nat)
    requires
        lo < hi,
    ensures
        row_range_r(a, x, i, lo, hi) == row_range_r(a, x, i, lo + 1, hi) + a(i, lo as int) * x[lo as int],
{
    lemma_row_prefix_step(a, x, i, lo);
}

/// The range sum only reads the entries lo..hi of x.
pub proof fn lemma_row_range_ext(a: MatF, x: Seq<real>, y: Seq<real>, i: int, lo: nat, hi: nat)
    requires
        lo <= hi,
        forall|k: int| lo <= k < hi ==> x[k] == y[k],
    ensures
        row_range_r(a, x, i, lo, hi) == row_range_r(a, y, i, lo, hi),
    decreases hi - lo,
{
    if lo < hi {
        lemma_row_range_ext(a, x, y, i, lo + 1, hi);
        lemma_row_range_step(a, x, i, lo, hi);
        lemma_row_range_step(a, y, i, lo, hi);
    }
}

/* ---------- matrix-vector products of the solver views ---------- */

/// (A x)_i is the full row sum.
pub proof fn lemma_mat_vec_row_prefix(a: MatF, dim: int, x: Seq<real>, i: int)
    requires
        0 <= i < dim,
    ensures
        mat_vec(a, dim, x)[i] == row_prefix_r(a, x, i, dim as nat),
{
}

/// For lower-triangular L, (L x)_i = sum_{k <= i} L_ik x_k.
pub proof fn lemma_lt_mat_vec(m: Seq<real>, dim: int, x: Seq<real>, i: int)
    requires
        0 <= i < dim,
    ensures
        mat_vec(lt_view(m, dim), dim, x)[i] == row_prefix_r(lt_view(m, dim), x, i, (i + 1) as nat),
{
    let l = lt_view(m, dim);
    lemma_mat_vec_row_prefix(l, dim, x, i);
    let f = |k: int| l(i, k) * x[k];
    assert forall|k: int| i + 1 <= k < dim implies #[trigger] f(k) == 0.0real by {
        assert(l(i, k) == 0.0real);
    }
    lemma_sum_fn_tail_zero(f, (i + 1) as nat, dim as nat);
}

/// For upper-triangular U = L^T, (U x)_i = sum_{k >= i} L_ki x_k.
pub proof fn lemma_ut_mat_vec(m: Seq<real>, dim: int, x: Seq<real>, i: int)
    requires
        0 <= i < dim,
    ensures
        mat_vec(mtr(lt_view(m, dim)), dim, x)[i] == row_range_r(mtr(lt_view(m, dim)), x, i, i as nat, dim as nat),
{
    let u = mtr(lt_view(m, dim));
    lemma_mat_vec_row_prefix(u, dim, x, i);
    let f = |k: int| u(i, k) * x[k];
    let z = |k: int| 0.0real;
    assert forall|k: int| 0 <= k < i implies #[trigger] f(k) == z(k) by {
        assert(u(i, k) == 0.0real);
    }
    lemma_sum_fn_ext(f, z, i as nat);
    lemma_sum_fn_zero(i as nat);
}

/// For diagonal D, (D x)_i = d_i x_i.
pub proof fn lemma_diag_mat_vec(d: Seq<real>, dim: int, x: Seq<real>, i: int)
    requires
        0 <= i < dim,
    ensures
        mat_vec(diag_view(d), dim, x)[i] == d[i] * x[i],
{
    let f = |k: int| diag_view(d)(i, k) * x[k];
    assert forall|k: int| 0 <= k < dim && k != i implies #[trigger] f(k) == 0.0real by {
        assert(diag_view(d)(i, k) == 0.0real);
    }
    lemma_sum_fn_single(f, dim as nat, i);
}

} // verus!

// Spec-level Euclidean and Frobenius norms of rectangular matrices (not trusted).
//
// Products (`mat_vec_rc`) and Frobenius norms (`frob_rc`) of rows x cols index
// functions (`MatF`), with Cauchy-Schwarz, the triangle inequality,
// ||A x|| <= ||A||_F ||x||, ||A B||_F <= ||A||_F ||B||_F and the entry bounds
// ||x|| <= sqrt(n) max |x_k|, ||A||_F <= sqrt(rows cols) max |A_ik|.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::seq::*;
#[cfg(verus_keep_ghost)]
use crate::trusted::sqrt_r;
#[cfg(verus_keep_ghost)]
use crate::real_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::*;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- rectangular products and norms ---------- */

/// Row i of a against x: sum_{k < cols} a(i, k) x_k.
pub open spec fn row_dot_rc(a: MatF, cols: int, x: Seq<real>, i: int) -> real {
    sum_fn_r(|k: int| a(i, k) * x[k], cols as nat)
}

/// A x for a rows x cols matrix a and a vector x of length cols.
pub open spec fn mat_vec_rc(a: MatF, rows: int, cols: int, x: Seq<real>) -> Seq<real> {
    Seq::new(rows as nat, |i: int| row_dot_rc(a, cols, x, i))
}

/// Row i of a as a vector of length cols.
pub open spec fn row_vec_rc(a: MatF, cols: int, i: int) -> Seq<real> {
    Seq::new(cols as nat, |k: int| a(i, k))
}

/// Column c of a as a vector of length rows.
pub open spec fn col_vec_rc(a: MatF, rows: int, c: int) -> Seq<real> {
    Seq::new(rows as nat, |i: int| a(i, c))
}

/// Squared Frobenius norm sum_{i < rows} sum_{k < cols} a(i, k)^2.
pub open spec fn frob_sq_rc(a: MatF, rows: int, cols: int) -> real {
    sum_fn_r(|i: int| sum_fn_r(|k: int| a(i, k) * a(i, k), cols as nat), rows as nat)
}

/// Frobenius norm of a rows x cols matrix.
pub open spec fn frob_rc(a: MatF, rows: int, cols: int) -> real {
    sqrt_r(frob_sq_rc(a, rows, cols))
}

/* ---------- basic facts ---------- */

/// Row i of A x is the inner product of row i with x.
pub proof fn lemma_row_dot_rc(a: MatF, cols: int, x: Seq<real>, i: int)
    requires
        cols >= 0,
        x.len() == cols,
    ensures
        row_dot_rc(a, cols, x, i) == dot_r(row_vec_rc(a, cols, i), x),
{
    let r = row_vec_rc(a, cols, i);
    lemma_dot_sum_fn(r, x);
    let f = |k: int| a(i, k) * x[k];
    let g = |k: int| r[k] * x[k];
    assert forall|k: int| 0 <= k < cols implies #[trigger] f(k) == g(k) by {}
    lemma_sum_fn_ext(f, g, cols as nat);
}

/// The squared Frobenius norm is the sum of the squared row norms.
pub proof fn lemma_frob_sq_rc_rows(a: MatF, rows: int, cols: int)
    requires
        rows >= 0,
        cols >= 0,
    ensures
        frob_sq_rc(a, rows, cols) == sum_fn_r(|i: int| dot_r(row_vec_rc(a, cols, i), row_vec_rc(a, cols, i)), rows as nat),
{
    let f = |i: int| sum_fn_r(|k: int| a(i, k) * a(i, k), cols as nat);
    let g = |i: int| dot_r(row_vec_rc(a, cols, i), row_vec_rc(a, cols, i));
    assert forall|i: int| 0 <= i < rows implies #[trigger] f(i) == g(i) by {
        let r = row_vec_rc(a, cols, i);
        lemma_dot_sum_fn(r, r);
        let f2 = |k: int| a(i, k) * a(i, k);
        let g2 = |k: int| r[k] * r[k];
        assert forall|k: int| 0 <= k < cols implies #[trigger] f2(k) == g2(k) by {}
        lemma_sum_fn_ext(f2, g2, cols as nat);
    }
    lemma_sum_fn_ext(f, g, rows as nat);
}

/// ... and the sum of the squared column norms.
pub proof fn lemma_frob_sq_rc_cols(a: MatF, rows: int, cols: int)
    requires
        rows >= 0,
        cols >= 0,
    ensures
        frob_sq_rc(a, rows, cols) == sum_fn_r(|c: int| dot_r(col_vec_rc(a, rows, c), col_vec_rc(a, rows, c)), cols as nat),
{
    let t = |i: int, k: int| a(i, k) * a(i, k);
    let (m, n) = (rows as nat, cols as nat);
    let rows_a = |i: int| sum_fn_r(|k: int| a(i, k) * a(i, k), n);
    let rows_t = |i: int| sum_fn_r(|k: int| t(i, k), n);
    assert forall|i: int| 0 <= i < rows implies #[trigger] rows_a(i) == rows_t(i) by {
        lemma_sum_fn_ext(|k: int| a(i, k) * a(i, k), |k: int| t(i, k), n);
    }
    lemma_sum_fn_ext(rows_a, rows_t, m);
    lemma_sum_fn_swap(t, m, n);
    let cols_t = |c: int| sum_fn_r(|i: int| t(i, c), m);
    let cols_d = |c: int| dot_r(col_vec_rc(a, rows, c), col_vec_rc(a, rows, c));
    assert forall|c: int| 0 <= c < cols implies #[trigger] cols_t(c) == cols_d(c) by {
        let v = col_vec_rc(a, rows, c);
        lemma_dot_sum_fn(v, v);
        let f = |i: int| t(i, c);
        let g = |i: int| v[i] * v[i];
        assert forall|i: int| 0 <= i < rows implies #[trigger] f(i) == g(i) by {}
        lemma_sum_fn_ext(f, g, m);
    }
    lemma_sum_fn_ext(cols_t, cols_d, n);
}

/// ||A||_F^2 >= 0.
pub proof fn lemma_frob_sq_rc_nonneg(a: MatF, rows: int, cols: int)
    requires
        rows >= 0,
        cols >= 0,
    ensures
        frob_sq_rc(a, rows, cols) >= 0.0real,
        frob_rc(a, rows, cols) >= 0.0real,
{
    let inner = |i: int| sum_fn_r(|k: int| a(i, k) * a(i, k), cols as nat);
    assert forall|i: int| 0 <= i < rows implies #[trigger] inner(i) >= 0.0real by {
        let f = |k: int| a(i, k) * a(i, k);
        assert forall|k: int| 0 <= k < cols implies #[trigger] f(k) >= 0.0real by {
            lemma_sq_nonneg(a(i, k));
        }
        lemma_sum_fn_nonneg(f, cols as nat);
    }
    lemma_sum_fn_nonneg(inner, rows as nat);
}

/// A vector with <x, x> = 0 vanishes.
pub proof fn lemma_dot_self_zero(x: Seq<real>)
    requires
        dot_r(x, x) == 0.0real,
    ensures
        forall|k: int| 0 <= k < x.len() ==> #[trigger] x[k] == 0.0real,
{
    lemma_dot_sum_fn(x, x);
    let sq = |k: int| x[k] * x[k];
    assert forall|k: int| 0 <= k < x.len() implies #[trigger] sq(k) >= 0.0real by {
        lemma_sq_nonneg(x[k]);
    }
    assert forall|k: int| 0 <= k < x.len() implies #[trigger] x[k] == 0.0real by {
        lemma_sum_fn_ge_term(sq, x.len(), k);
        let u = x[k];
        assert(u == 0.0real) by (nonlinear_arith)
            requires u * u <= 0.0real;
    }
}

/// A vector of norm 0 vanishes.
pub proof fn lemma_norm_zero(x: Seq<real>)
    requires
        norm_r(x) == 0.0real,
    ensures
        forall|k: int| 0 <= k < x.len() ==> #[trigger] x[k] == 0.0real,
{
    lemma_dot_self_nonneg(x);
    assert(dot_r(x, x) == sqrt_r(dot_r(x, x)) * sqrt_r(dot_r(x, x)));
    lemma_dot_self_zero(x);
}

/// A matrix of Frobenius norm 0 vanishes.
pub proof fn lemma_frob_rc_zero(a: MatF, rows: int, cols: int)
    requires
        rows >= 0,
        cols >= 0,
        frob_rc(a, rows, cols) == 0.0real,
    ensures
        forall|i: int, k: int| 0 <= i < rows && 0 <= k < cols ==> #[trigger] a(i, k) == 0.0real,
{
    lemma_frob_sq_rc_nonneg(a, rows, cols);
    let fs = frob_sq_rc(a, rows, cols);
    assert(fs == sqrt_r(fs) * sqrt_r(fs));
    lemma_frob_sq_rc_rows(a, rows, cols);
    let rr = |i: int| dot_r(row_vec_rc(a, cols, i), row_vec_rc(a, cols, i));
    assert forall|i: int| 0 <= i < rows implies #[trigger] rr(i) >= 0.0real by {
        lemma_dot_self_nonneg(row_vec_rc(a, cols, i));
    }
    assert forall|i: int, k: int| 0 <= i < rows && 0 <= k < cols implies #[trigger] a(i, k) == 0.0real by {
        lemma_sum_fn_ge_term(rr, rows as nat, i);
        let r = row_vec_rc(a, cols, i);
        lemma_dot_self_zero(r);
        assert(r[k] == 0.0real);
    }
}

/* ---------- Cauchy-Schwarz and the triangle inequality ---------- */

/// Cauchy-Schwarz: <x, y>^2 <= <x, x> <y, y>.
pub proof fn lemma_cauchy_schwarz(x: Seq<real>, y: Seq<real>)
    requires
        x.len() == y.len(),
    ensures
        dot_r(x, y) * dot_r(x, y) <= dot_r(x, x) * dot_r(y, y),
{
    let n = x.len();
    let (a, b, c) = (dot_r(x, x), dot_r(y, y), dot_r(x, y));
    lemma_dot_self_nonneg(x);
    lemma_dot_self_nonneg(y);
    lemma_dot_sum_fn(x, x);
    lemma_dot_sum_fn(y, y);
    lemma_dot_sum_fn(x, y);
    let xx = |k: int| x[k] * x[k];
    let yy = |k: int| y[k] * y[k];
    let xy = |k: int| x[k] * y[k];
    if a == 0.0real {
        // every x_k vanishes, hence c = 0
        lemma_dot_self_zero(x);
        assert forall|k: int| 0 <= k < n implies #[trigger] xy(k) == 0.0real by {
            assert(x[k] == 0.0real);
        }
        lemma_sum_fn_all_zero(xy, n);
        assert(c == 0.0real);
        assert(c * c <= a * b) by (nonlinear_arith)
            requires c == 0.0real, a == 0.0real;
    } else {
        // 0 <= sum_k (a y_k - c x_k)^2 = a^2 b - 2 a c^2 + c^2 a = a (a b - c^2)
        let w = Seq::new(n, |k: int| a * y[k] - c * x[k]);
        lemma_dot_self_nonneg(w);
        lemma_dot_sum_fn(w, w);
        let ww = |k: int| w[k] * w[k];
        let (sa, sb, sc) = (a * a, -(2.0real * (a * c)), c * c);
        let s1 = |k: int| sa * yy(k);
        let s2 = |k: int| sb * xy(k);
        let s3 = |k: int| sc * xx(k);
        let s12 = |k: int| s1(k) + s2(k);
        let s123 = |k: int| s12(k) + s3(k);
        assert forall|k: int| 0 <= k < n implies #[trigger] ww(k) == s123(k) by {
            let (u, v) = (x[k], y[k]);
            assert(w[k] == a * v - c * u);
            assert((a * v - c * u) * (a * v - c * u) == (a * a) * (v * v) + (-(2.0real * (a * c))) * (u * v) + (c * c) * (u * u))
                by (nonlinear_arith);
        }
        lemma_sum_fn_ext(ww, s123, n);
        lemma_sum_fn_add(s12, s3, s123, n);
        lemma_sum_fn_add(s1, s2, s12, n);
        lemma_sum_fn_scale(yy, s1, sa, n);
        lemma_sum_fn_scale(xy, s2, sb, n);
        lemma_sum_fn_scale(xx, s3, sc, n);
        assert(dot_r(w, w) == sa * b + sb * c + sc * a);
        assert((a * a) * b + (-(2.0real * (a * c))) * c + (c * c) * a == a * (a * b - c * c)) by (nonlinear_arith);
        assert(a * b - c * c >= 0.0real) by (nonlinear_arith)
            requires a > 0.0real, a * (a * b - c * c) >= 0.0real;
    }
}

/// |<x, y>| <= ||x|| ||y||.
pub proof fn lemma_dot_le_norms(x: Seq<real>, y: Seq<real>)
    requires
        x.len() == y.len(),
    ensures
        abs_r(dot_r(x, y)) <= norm_r(x) * norm_r(y),
{
    lemma_cauchy_schwarz(x, y);
    let (a, b, c) = (dot_r(x, x), dot_r(y, y), dot_r(x, y));
    lemma_dot_self_nonneg(x);
    lemma_dot_self_nonneg(y);
    lemma_sq_nonneg(c);
    assert(a * b >= 0.0real) by (nonlinear_arith)
        requires a >= 0.0real, b >= 0.0real;
    lemma_sqrt_square(c);
    lemma_sqrt_monotone(c * c, a * b);
    lemma_sqrt_mul(a, b);
}

/// ||u + v|| <= ||u|| + ||v||.
pub proof fn lemma_norm_triangle(u: Seq<real>, v: Seq<real>)
    requires
        u.len() == v.len(),
    ensures
        norm_r(vadd_r(u, v)) <= norm_r(u) + norm_r(v),
{
    let w = vadd_r(u, v);
    let (nu, nv) = (norm_r(u), norm_r(v));
    lemma_dot_self_nonneg(u);
    lemma_dot_self_nonneg(v);
    lemma_dot_self_nonneg(w);
    // <w, w> = <u, u> + 2 <u, v> + <v, v>
    lemma_dot_add_right(w, u, v);
    lemma_dot_comm(w, u);
    lemma_dot_add_right(u, u, v);
    lemma_dot_comm(w, v);
    lemma_dot_add_right(v, u, v);
    lemma_dot_comm(v, u);
    assert(dot_r(w, w) == dot_r(u, u) + 2.0real * dot_r(u, v) + dot_r(v, v));
    lemma_dot_le_norms(u, v);
    assert(dot_r(u, u) == nu * nu);
    assert(dot_r(v, v) == nv * nv);
    assert(dot_r(w, w) <= (nu + nv) * (nu + nv)) by (nonlinear_arith)
        requires
            dot_r(w, w) == nu * nu + 2.0real * dot_r(u, v) + nv * nv,
            dot_r(u, v) <= nu * nv;
    lemma_sqrt_monotone(dot_r(w, w), (nu + nv) * (nu + nv));
    lemma_sqrt_square(nu + nv);
}

/// ||s x|| = |s| ||x||.
pub proof fn lemma_norm_scale(s: real, x: Seq<real>)
    ensures
        norm_r(vscale_r(s, x)) == abs_r(s) * norm_r(x),
{
    let y = vscale_r(s, x);
    let d = dot_r(x, x);
    lemma_dot_scale_right(y, s, x);
    lemma_dot_comm(y, x);
    lemma_dot_scale_right(x, s, x);
    assert(dot_r(y, y) == s * (s * d));
    assert(s * (s * d) == (s * s) * d) by (nonlinear_arith);
    lemma_dot_self_nonneg(x);
    lemma_sq_nonneg(s);
    lemma_sqrt_mul(s * s, d);
    lemma_sqrt_square(s);
}

/* ---------- products ---------- */

/// (A + B) x = A x + B x.
pub proof fn lemma_mat_vec_rc_add(a: MatF, b: MatF, rows: int, cols: int, x: Seq<real>)
    requires
        rows >= 0,
        cols >= 0,
    ensures
        mat_vec_rc(madd(a, b), rows, cols, x) == vadd_r(mat_vec_rc(a, rows, cols, x), mat_vec_rc(b, rows, cols, x)),
{
    let lhs = mat_vec_rc(madd(a, b), rows, cols, x);
    let rhs = vadd_r(mat_vec_rc(a, rows, cols, x), mat_vec_rc(b, rows, cols, x));
    assert forall|i: int| 0 <= i < rows implies #[trigger] lhs[i] == rhs[i] by {
        let f = |k: int| a(i, k) * x[k];
        let g = |k: int| b(i, k) * x[k];
        let h = |k: int| madd(a, b)(i, k) * x[k];
        assert forall|k: int| 0 <= k < cols implies #[trigger] h(k) == f(k) + g(k) by {
            let (u, v, w) = (a(i, k), b(i, k), x[k]);
            assert(h(k) == (u + v) * w);
            assert((u + v) * w == u * w + v * w) by (nonlinear_arith);
        }
        lemma_sum_fn_add(f, g, h, cols as nat);
    }
    assert(lhs =~= rhs);
}

/// A x only depends on the rows x cols block of A.
pub proof fn lemma_mat_vec_rc_meq(a: MatF, b: MatF, rows: int, cols: int, x: Seq<real>)
    requires
        rows >= 0,
        cols >= 0,
        meq(a, b, rows, cols),
    ensures
        mat_vec_rc(a, rows, cols, x) == mat_vec_rc(b, rows, cols, x),
{
    let lhs = mat_vec_rc(a, rows, cols, x);
    let rhs = mat_vec_rc(b, rows, cols, x);
    assert forall|i: int| 0 <= i < rows implies #[trigger] lhs[i] == rhs[i] by {
        let f = |k: int| a(i, k) * x[k];
        let g = |k: int| b(i, k) * x[k];
        assert forall|k: int| 0 <= k < cols implies #[trigger] f(k) == g(k) by {}
        lemma_sum_fn_ext(f, g, cols as nat);
    }
    assert(lhs =~= rhs);
}

/// (A B) x = A (B x) for a rows x inner, b inner x cols.
pub proof fn lemma_mat_vec_rc_mul(a: MatF, b: MatF, rows: int, inner: int, cols: int, x: Seq<real>)
    requires
        rows >= 0,
        inner >= 0,
        cols >= 0,
        x.len() == cols,
    ensures
        mat_vec_rc(mmul(a, b, inner), rows, cols, x) == mat_vec_rc(a, rows, inner, mat_vec_rc(b, inner, cols, x)),
{
    let ab = mmul(a, b, inner);
    let bx = mat_vec_rc(b, inner, cols, x);
    let lhs = mat_vec_rc(ab, rows, cols, x);
    let rhs = mat_vec_rc(a, rows, inner, bx);
    let (m, n) = (inner as nat, cols as nat);
    assert forall|i: int| 0 <= i < rows implies #[trigger] lhs[i] == rhs[i] by {
        let t = |c: int, k: int| (a(i, k) * b(k, c)) * x[c];
        // sum_c (ab)_ic x_c = sum_c sum_k t(c, k)
        let f = |c: int| ab(i, c) * x[c];
        let fc = |c: int| sum_fn_r(|k: int| t(c, k), m);
        assert forall|c: int| 0 <= c < cols implies #[trigger] f(c) == fc(c) by {
            let g = |k: int| a(i, k) * b(k, c);
            let h = |k: int| t(c, k);
            let xc = x[c];
            assert forall|k: int| 0 <= k < inner implies #[trigger] h(k) == xc * g(k) by {
                let v = g(k);
                assert(v * xc == xc * v) by (nonlinear_arith);
            }
            lemma_sum_fn_scale(g, h, xc, m);
            assert(ab(i, c) == sum_fn_r(g, m));
            let sg = sum_fn_r(g, m);
            assert(sg * xc == xc * sg) by (nonlinear_arith);
        }
        lemma_sum_fn_ext(f, fc, n);
        lemma_sum_fn_swap(t, n, m);
        // = sum_k a_ik (sum_c b_kc x_c)
        let g2 = |k: int| a(i, k) * bx[k];
        let gk = |k: int| sum_fn_r(|c: int| t(c, k), n);
        assert forall|k: int| 0 <= k < inner implies #[trigger] g2(k) == gk(k) by {
            let p = |c: int| b(k, c) * x[c];
            let q = |c: int| t(c, k);
            let aik = a(i, k);
            assert forall|c: int| 0 <= c < cols implies #[trigger] q(c) == aik * p(c) by {
                let (u, w) = (b(k, c), x[c]);
                assert((aik * u) * w == aik * (u * w)) by (nonlinear_arith);
            }
            lemma_sum_fn_scale(p, q, aik, n);
            assert(bx[k] == sum_fn_r(p, n));
        }
        lemma_sum_fn_ext(g2, gk, m);
    }
    assert(lhs =~= rhs);
}

/* ---------- norm bounds ---------- */

/// ||A x||^2 <= ||A||_F^2 ||x||^2 and ||A x|| <= ||A||_F ||x||.
pub proof fn lemma_mat_vec_rc_bound(a: MatF, rows: int, cols: int, x: Seq<real>)
    requires
        rows >= 0,
        cols >= 0,
        x.len() == cols,
    ensures
        dot_r(mat_vec_rc(a, rows, cols, x), mat_vec_rc(a, rows, cols, x)) <= frob_sq_rc(a, rows, cols) * dot_r(x, x),
        norm_r(mat_vec_rc(a, rows, cols, x)) <= frob_rc(a, rows, cols) * norm_r(x),
{
    let ax = mat_vec_rc(a, rows, cols, x);
    let xx = dot_r(x, x);
    let fs = frob_sq_rc(a, rows, cols);
    lemma_dot_sum_fn(ax, ax);
    lemma_frob_sq_rc_rows(a, rows, cols);
    let sq = |i: int| ax[i] * ax[i];
    let rr = |i: int| dot_r(row_vec_rc(a, cols, i), row_vec_rc(a, cols, i));
    let xrr = |i: int| xx * rr(i);
    assert forall|i: int| 0 <= i < rows implies #[trigger] sq(i) <= xrr(i) by {
        lemma_row_dot_rc(a, cols, x, i);
        lemma_cauchy_schwarz(row_vec_rc(a, cols, i), x);
        assert(rr(i) * xx == xx * rr(i)) by (nonlinear_arith);
    }
    lemma_sum_fn_le(sq, xrr, rows as nat);
    lemma_sum_fn_scale(rr, xrr, xx, rows as nat);
    assert(dot_r(ax, ax) <= xx * fs);
    assert(xx * fs == fs * xx) by (nonlinear_arith);
    lemma_frob_sq_rc_nonneg(a, rows, cols);
    lemma_dot_self_nonneg(x);
    lemma_dot_self_nonneg(ax);
    assert(fs * xx >= 0.0real) by (nonlinear_arith)
        requires fs >= 0.0real, xx >= 0.0real;
    lemma_sqrt_monotone(dot_r(ax, ax), fs * xx);
    lemma_sqrt_mul(fs, xx);
}

/// ||A B||_F^2 <= ||A||_F^2 ||B||_F^2 and ||A B||_F <= ||A||_F ||B||_F for a rows x inner, b inner x cols.
pub proof fn lemma_mmul_frob_rc(a: MatF, b: MatF, rows: int, inner: int, cols: int)
    requires
        rows >= 0,
        inner >= 0,
        cols >= 0,
    ensures
        frob_sq_rc(mmul(a, b, inner), rows, cols) <= frob_sq_rc(a, rows, inner) * frob_sq_rc(b, inner, cols),
        frob_rc(mmul(a, b, inner), rows, cols) <= frob_rc(a, rows, inner) * frob_rc(b, inner, cols),
{
    let ab = mmul(a, b, inner);
    let (fa, fb) = (frob_sq_rc(a, rows, inner), frob_sq_rc(b, inner, cols));
    lemma_frob_sq_rc_cols(b, inner, cols);
    let cc = |c: int| dot_r(col_vec_rc(b, inner, c), col_vec_rc(b, inner, c));
    // row i of A B: sum_c (ab)_ic^2 <= ||row_i(A)||^2 ||B||_F^2
    let inner_ab = |i: int| sum_fn_r(|c: int| ab(i, c) * ab(i, c), cols as nat);
    let rr = |i: int| dot_r(row_vec_rc(a, inner, i), row_vec_rc(a, inner, i));
    let bound = |i: int| fb * rr(i);
    assert forall|i: int| 0 <= i < rows implies #[trigger] inner_ab(i) <= bound(i) by {
        let sq = |c: int| ab(i, c) * ab(i, c);
        let rc = |c: int| rr(i) * cc(c);
        assert forall|c: int| 0 <= c < cols implies #[trigger] sq(c) <= rc(c) by {
            // (ab)_ic = <row_i(A), col_c(B)>
            let (ra, cb) = (row_vec_rc(a, inner, i), col_vec_rc(b, inner, c));
            lemma_dot_sum_fn(ra, cb);
            let f = |k: int| a(i, k) * b(k, c);
            let g = |k: int| ra[k] * cb[k];
            assert forall|k: int| 0 <= k < inner implies #[trigger] f(k) == g(k) by {}
            lemma_sum_fn_ext(f, g, inner as nat);
            assert(ab(i, c) == dot_r(ra, cb));
            lemma_cauchy_schwarz(ra, cb);
        }
        lemma_sum_fn_le(sq, rc, cols as nat);
        lemma_sum_fn_scale(cc, rc, rr(i), cols as nat);
        assert(rr(i) * fb == fb * rr(i)) by (nonlinear_arith);
    }
    lemma_sum_fn_le(inner_ab, bound, rows as nat);
    lemma_sum_fn_scale(rr, bound, fb, rows as nat);
    lemma_frob_sq_rc_rows(a, rows, inner);
    assert(frob_sq_rc(ab, rows, cols) <= fb * fa);
    assert(fb * fa == fa * fb) by (nonlinear_arith);
    lemma_frob_sq_rc_nonneg(a, rows, inner);
    lemma_frob_sq_rc_nonneg(b, inner, cols);
    lemma_frob_sq_rc_nonneg(ab, rows, cols);
    assert(fa * fb >= 0.0real) by (nonlinear_arith)
        requires fa >= 0.0real, fb >= 0.0real;
    lemma_sqrt_monotone(frob_sq_rc(ab, rows, cols), fa * fb);
    lemma_sqrt_mul(fa, fb);
}

/// Entries of magnitude at most m give ||x||^2 <= n m^2 and ||x|| <= sqrt(n) m.
pub proof fn lemma_norm_entry_bound(x: Seq<real>, m: real)
    requires
        m >= 0.0real,
        forall|k: int| 0 <= k < x.len() ==> abs_r(#[trigger] x[k]) <= m,
    ensures
        dot_r(x, x) <= (x.len() as real) * (m * m),
        norm_r(x) <= sqrt_r(x.len() as real) * m,
{
    let n = x.len();
    lemma_dot_sum_fn(x, x);
    let sq = |k: int| x[k] * x[k];
    let mm = |k: int| m * m;
    assert forall|k: int| 0 <= k < n implies #[trigger] sq(k) <= mm(k) by {
        lemma_abs_sq_le(x[k], m);
    }
    lemma_sum_fn_le(sq, mm, n);
    lemma_sum_fn_const(m * m, n);
    lemma_dot_self_nonneg(x);
    lemma_sq_nonneg(m);
    assert((n as real) * (m * m) >= 0.0real) by (nonlinear_arith)
        requires (n as real) >= 0.0real, m * m >= 0.0real;
    lemma_sqrt_monotone(dot_r(x, x), (n as real) * (m * m));
    lemma_sqrt_mul(n as real, m * m);
    lemma_sqrt_square(m);
}

/// |u| <= m gives u^2 <= m^2.
pub proof fn lemma_abs_sq_le(u: real, m: real)
    requires
        abs_r(u) <= m,
    ensures
        u * u <= m * m,
{
    lemma_abs_mul(u, u);
    lemma_sq_nonneg(u);
    assert(abs_r(u) * abs_r(u) <= m * m) by (nonlinear_arith)
        requires 0.0real <= abs_r(u) <= m;
}

/// Entries of magnitude at most m give ||A||_F^2 <= rows cols m^2 and ||A||_F <= sqrt(rows cols) m.
pub proof fn lemma_frob_rc_entry_bound(a: MatF, rows: int, cols: int, m: real)
    requires
        rows >= 0,
        cols >= 0,
        m >= 0.0real,
        forall|i: int, k: int| 0 <= i < rows && 0 <= k < cols ==> abs_r(#[trigger] a(i, k)) <= m,
    ensures
        frob_sq_rc(a, rows, cols) <= ((rows as real) * (cols as real)) * (m * m),
        frob_rc(a, rows, cols) <= sqrt_r((rows as real) * (cols as real)) * m,
{
    let inner = |i: int| sum_fn_r(|k: int| a(i, k) * a(i, k), cols as nat);
    let cb = |i: int| (cols as real) * (m * m);
    assert forall|i: int| 0 <= i < rows implies #[trigger] inner(i) <= cb(i) by {
        let sq = |k: int| a(i, k) * a(i, k);
        let mm = |k: int| m * m;
        assert forall|k: int| 0 <= k < cols implies #[trigger] sq(k) <= mm(k) by {
            lemma_abs_sq_le(a(i, k), m);
        }
        lemma_sum_fn_le(sq, mm, cols as nat);
        lemma_sum_fn_const(m * m, cols as nat);
    }
    lemma_sum_fn_le(inner, cb, rows as nat);
    lemma_sum_fn_const((cols as real) * (m * m), rows as nat);
    assert((rows as real) * ((cols as real) * (m * m)) == ((rows as real) * (cols as real)) * (m * m)) by (nonlinear_arith);
    lemma_frob_sq_rc_nonneg(a, rows, cols);
    lemma_sq_nonneg(m);
    assert((rows as real) * (cols as real) >= 0.0real) by (nonlinear_arith)
        requires (rows as real) >= 0.0real, (cols as real) >= 0.0real;
    let rc = (rows as real) * (cols as real);
    assert(rc * (m * m) >= 0.0real) by (nonlinear_arith)
        requires rc >= 0.0real, m * m >= 0.0real;
    lemma_sqrt_monotone(frob_sq_rc(a, rows, cols), rc * (m * m));
    lemma_sqrt_mul(rc, m * m);
    lemma_sqrt_square(m);
}

} // verus!

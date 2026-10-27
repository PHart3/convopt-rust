// Relative residual bound of the constraint reduction (not trusted).
//
// With the first-stage divisors D (1 for rows of norm zero), the row
// transformation T = [T_K | T_D] of the elimination (normalized input = T *
// echelon form, split at the kept rows), the second-stage divisors S and the
// dropped echelon rows H_D, the input of `reduced_with` is recovered as
//
//     [A | b] = U [A_K | b_K] + [E | f],   U = D T_K S,   [E | f] = D T_D H_D.
//
// The dropped rows are small (skip bounds, consistency test), so
// ||E||_F <= TOL e_A and ||f|| <= TOL e_b with e_A = 2^rank sqrt(q n) alpha L,
// e_b = 2^rank sqrt(q) beta L (alpha, beta the largest normalized coefficient
// and right-hand side, L = ||D T_D||_F, q dropped rows, n columns of A).  With
// C = max(e_A / ||A||_F, e_b / ||b||) (0 / 0 = 0) and d(z) = ||A||_F ||z|| + ||b||,
//
//     ||A z - b|| <= C TOL d(z) + ||U||_F ||A_K z - b_K||,
//
// divided by d(z) when d(z) > 0.  If d(z) = 0 the residual is 0; without
// dropped rows C = 0.  Nothing is claimed about the size of C.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::seq::*;
#[cfg(verus_keep_ghost)]
use vstd::arithmetic::div_mod::{lemma_fundamental_div_mod, lemma_fundamental_div_mod_converse, lemma_mod_bound};
#[cfg(verus_keep_ghost)]
use crate::trusted::{rv, sqrt_r};
#[cfg(verus_keep_ghost)]
use crate::real_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::norm_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::solve_spec::abs_max_prefix_r;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::gauss_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::reduce_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::matrix::TOL;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- residuals and norms of the augmented system ---------- */

/// The augmented test vector (z, -1): a row residual is the inner product of the row with it.
pub open spec fn aug_vec_r(z: Seq<real>) -> Seq<real> {
    z.push(-1.0real)
}

/// The residual vector A z - b of the augmented system m (row_dim rows).
pub open spec fn resid_vec_r(m: Seq<real>, row_dim: int, col_dim: int, z: Seq<real>) -> Seq<real> {
    Seq::new(row_dim as nat, |i: int| row_residual_r(m, row_dim, col_dim, i, z))
}

/// ||A||_F of the coefficient part of the augmented system m.
pub open spec fn coef_norm_r(m: Seq<real>, row_dim: int, col_dim: int) -> real {
    frob_rc(cm_view(m, row_dim), row_dim, col_dim - 1)
}

/// The right-hand side b of the augmented system m.
pub open spec fn rhs_vec_r(m: Seq<real>, row_dim: int, col_dim: int) -> Seq<real> {
    Seq::new(row_dim as nat, |i: int| rhs_r(m, row_dim, col_dim, i))
}

/// ||b||.
pub open spec fn rhs_norm_r(m: Seq<real>, row_dim: int, col_dim: int) -> real {
    norm_r(rhs_vec_r(m, row_dim, col_dim))
}

/// The residual scale d(z) = ||A||_F ||z|| + ||b||.
pub open spec fn resid_scale_r(m: Seq<real>, row_dim: int, col_dim: int, z: Seq<real>) -> real {
    coef_norm_r(m, row_dim, col_dim) * norm_r(z) + rhs_norm_r(m, row_dim, col_dim)
}

/* ---------- the recovery matrix and the error ---------- */

/// The divisor D_i of row i before the elimination: its scale, or 1 if the scale is 0.
pub open spec fn pre_div_r(a0: Seq<real>, row_dim: int, dim: int, i: int) -> real {
    let s = row_scale_r(a0, row_dim, dim, i);
    if s == 0.0real { 1.0real } else { s }
}

/// U = D T_K S (rd x rank): entry (i, k) is D_i t(i, k) S_k.
pub open spec fn recovery_r(a0: Seq<real>, e: Seq<real>, dim: int, t: MatF) -> MatF {
    let rd = red_row_dim(a0, dim);
    |i: int, k: int| (pre_div_r(a0, rd, dim, i) * t(i, k)) * row_scale_r(e, rd, dim, k)
}

/// D T_D (rd x q): the divisors times the columns of T at the dropped rows.
pub open spec fn drop_comb_r(a0: Seq<real>, dim: int, t: MatF, rank: int) -> MatF {
    let rd = red_row_dim(a0, dim);
    |i: int, l: int| pre_div_r(a0, rd, dim, i) * t(i, rank + l)
}

/// H_D (q x cd): the dropped echelon rows.
pub open spec fn drop_rows_r(e: Seq<real>, row_dim: int, rank: int) -> MatF {
    |l: int, c: int| cm_entry(e, row_dim, rank + l, c)
}

/// [E | f] = D T_D H_D (rd x cd).
pub open spec fn err_r(a0: Seq<real>, e: Seq<real>, dim: int, t: MatF, rank: int) -> MatF {
    let rd = red_row_dim(a0, dim);
    mmul(drop_comb_r(a0, dim, t, rank), drop_rows_r(e, rd, rank), rd - rank)
}

/// f: the right-hand side column of [E | f].
pub open spec fn err_rhs_r(a0: Seq<real>, e: Seq<real>, dim: int, t: MatF, rank: int) -> Seq<real> {
    col_vec_rc(err_r(a0, e, dim, t, rank), red_row_dim(a0, dim), red_col_dim(dim) - 1)
}

/* ---------- the constants ---------- */

/// alpha: the largest magnitude of the coefficient part of the normalized input (0 if empty).
pub open spec fn coef_max_r(a0: Seq<real>, dim: int) -> real {
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    abs_max_prefix_r(normalized_r(a0, dim), ((cd - 1) * rd) as nat)
}

/// beta: the largest magnitude of the right-hand side of the normalized input (0 if empty).
pub open spec fn rhs_max_r(a0: Seq<real>, dim: int) -> real {
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    rhs_abs_max_r(normalized_r(a0, dim), rd, cd)
}

/// L = ||D T_D||_F.
pub open spec fn drop_norm_r(a0: Seq<real>, dim: int, t: MatF, rank: int) -> real {
    let rd = red_row_dim(a0, dim);
    frob_rc(drop_comb_r(a0, dim, t, rank), rd, rd - rank)
}

/// e_A = 2^rank sqrt(q n) alpha L.
pub open spec fn err_coef_bound_r(a0: Seq<real>, dim: int, t: MatF, rank: int) -> real {
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    pow2_r(rank as nat) * sqrt_r(((rd - rank) as real) * ((cd - 1) as real)) * coef_max_r(a0, dim) * drop_norm_r(a0, dim, t, rank)
}

/// e_b = 2^rank sqrt(q) beta L.
pub open spec fn err_rhs_bound_r(a0: Seq<real>, dim: int, t: MatF, rank: int) -> real {
    let rd = red_row_dim(a0, dim);
    pow2_r(rank as nat) * sqrt_r((rd - rank) as real) * rhs_max_r(a0, dim) * drop_norm_r(a0, dim, t, rank)
}

/// x / y, read as 0 when y = 0.
pub open spec fn ratio0_r(x: real, y: real) -> real {
    if y == 0.0real { 0.0real } else { x / y }
}

/// C = max(e_A / ||A||_F, e_b / ||b||), a zero denominator contributing 0.
pub open spec fn red_const_r(a0: Seq<real>, dim: int, t: MatF, rank: int) -> real {
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    max_r(ratio0_r(err_coef_bound_r(a0, dim, t, rank), coef_norm_r(a0, rd, cd)),
          ratio0_r(err_rhs_bound_r(a0, dim, t, rank), rhs_norm_r(a0, rd, cd)))
}

/* ---------- shapes and the normalized input ---------- */

/// The row count of the augmented system: rd * cd <= a0.len().
pub proof fn lemma_red_dims(a0: Seq<real>, dim: int)
    requires
        dim >= 0,
    ensures
        red_col_dim(dim) >= 1,
        red_row_dim(a0, dim) >= 0,
        red_row_dim(a0, dim) * red_col_dim(dim) <= a0.len(),
{
    let cd = red_col_dim(dim);
    let len = a0.len() as int;
    lemma_pk_col_start_zero();
    lemma_pk_col_start_monotone(0, dim);
    lemma_fundamental_div_mod(len, cd);
    lemma_mod_bound(len, cd);
    let rd = len / cd;
    assert(rd >= 0 && rd * cd <= len) by (nonlinear_arith)
        requires cd >= 1, len >= 0, len == cd * rd + len % cd, 0 <= len % cd < cd;
}

/// Entry (i, c) of the normalized input is entry (i, c) of a0 divided by D_i.
pub proof fn lemma_normalized_entry(a0: Seq<real>, dim: int, i: int, c: int)
    requires
        dim >= 0,
        0 <= i < red_row_dim(a0, dim),
        0 <= c < red_col_dim(dim),
    ensures
        cm_entry(normalized_r(a0, dim), red_row_dim(a0, dim), i, c)
            == normalized_entry_r(a0, red_row_dim(a0, dim), dim, i, cm_idx(red_row_dim(a0, dim), i, c)),
        pre_div_r(a0, red_row_dim(a0, dim), dim, i) * cm_entry(normalized_r(a0, dim), red_row_dim(a0, dim), i, c)
            == cm_entry(a0, red_row_dim(a0, dim), i, c),
{
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    lemma_red_dims(a0, dim);
    lemma_cm_idx_bound(rd, i, c, cd);
    let idx = cm_idx(rd, i, c);
    assert(idx < cd * rd);
    assert(cd * rd == rd * cd) by (nonlinear_arith);
    lemma_fundamental_div_mod_converse(idx, rd, c, i);
    assert(idx % rd == i);
    let s = row_scale_r(a0, rd, dim, i);
    let x = a0[idx];
    if s != 0.0real {
        assert(s * (x / s) == x) by (nonlinear_arith)
            requires s != 0.0real;
    }
}

/* ---------- the dropped rows ---------- */

/// The dropped echelon rows have small entries: coefficients at most 2^rank TOL alpha,
/// right-hand sides at most 2^rank TOL beta.
pub proof fn lemma_dropped_rows_small(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, pivots: Seq<int>)
    requires
        dim >= 1,
        reduced_with(a0, e, res, dim, pivots),
    ensures
        coef_max_r(a0, dim) >= 0.0real,
        rhs_max_r(a0, dim) >= 0.0real,
        forall|l: int, c: int| 0 <= l < red_row_dim(a0, dim) - pivots.len() && 0 <= c < red_col_dim(dim) - 1
            ==> abs_r(#[trigger] drop_rows_r(e, red_row_dim(a0, dim), pivots.len() as int)(l, c))
                <= pow2_r(pivots.len()) * (rv(TOL) * coef_max_r(a0, dim)),
        forall|l: int| 0 <= l < red_row_dim(a0, dim) - pivots.len()
            ==> abs_r(#[trigger] drop_rows_r(e, red_row_dim(a0, dim), pivots.len() as int)(l, red_col_dim(dim) - 1))
                <= rv(TOL) * (pow2_r(pivots.len()) * rhs_max_r(a0, dim)),
{
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    let rank = pivots.len();
    let n0 = normalized_r(a0, dim);
    let eps = gauss_eps_r(n0, rd, cd);
    let bmax = rhs_abs_max_r(n0, rd, cd);
    let alpha = coef_max_r(a0, dim);
    lemma_max0_fn_bound(|u: int| abs_r(n0[u]), ((cd - 1) * rd) as nat);
    lemma_max0_fn_bound(|i: int| abs_r(rhs_r(n0, rd, cd, i)), rd as nat);
    assert(rv(TOL) == 0.0000000001real);
    assert(eps == rv(TOL) * alpha);
    assert(eps >= 0.0real) by (nonlinear_arith)
        requires rv(TOL) >= 0.0real, alpha >= 0.0real, eps == rv(TOL) * alpha;
    let hd = drop_rows_r(e, rd, rank as int);
    assert forall|l: int, c: int| 0 <= l < rd - rank && 0 <= c < cd - 1
        implies abs_r(#[trigger] hd(l, c)) <= pow2_r(rank) * (rv(TOL) * alpha) by {
        lemma_dropped_row_bounded(e, rd, cd, pivots, eps, bmax, rank + l);
        assert(hd(l, c) == cm_entry(e, rd, rank + l, c));
    }
    assert forall|l: int| 0 <= l < rd - rank
        implies abs_r(#[trigger] hd(l, cd - 1)) <= rv(TOL) * (pow2_r(rank) * bmax) by {
        lemma_dropped_row_bounded(e, rd, cd, pivots, eps, bmax, rank + l);
        assert(hd(l, cd - 1) == rhs_r(e, rd, cd, rank + l));
    }
}

/* ---------- recovering the input from the reduced system ---------- */

/// [A | b] = U [A_K | b_K] + [E | f].
pub proof fn lemma_recovery_identity(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, pivots: Seq<int>, t: MatF)
    requires
        dim >= 1,
        reduced_with(a0, e, res, dim, pivots),
        row_comb(normalized_r(a0, dim), e, red_row_dim(a0, dim), red_col_dim(dim), t),
    ensures
        meq(cm_view(a0, red_row_dim(a0, dim)),
            madd(mmul(recovery_r(a0, e, dim, t), cm_view(res, pivots.len() as int), pivots.len() as int),
                 err_r(a0, e, dim, t, pivots.len() as int)),
            red_row_dim(a0, dim), red_col_dim(dim)),
{
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    let rank = pivots.len();
    let rk = rank as int;
    let q = (rd - rank) as nat;
    let n0 = normalized_r(a0, dim);
    let u = recovery_r(a0, e, dim, t);
    let rv_ = cm_view(res, rk);
    let dt = drop_comb_r(a0, dim, t, rk);
    let hd = drop_rows_r(e, rd, rk);
    let er = err_r(a0, e, dim, t, rk);
    let lhs = cm_view(a0, rd);
    let rhs = madd(mmul(u, rv_, rk), er);
    assert forall|i: int, c: int| 0 <= i < rd && 0 <= c < cd implies #[trigger] lhs(i, c) == #[trigger] rhs(i, c) by {
        lemma_normalized_entry(a0, dim, i, c);
        let d = pre_div_r(a0, rd, dim, i);
        let f = |l: int| t(i, l) * cm_entry(e, rd, l, c);
        assert(cm_entry(n0, rd, i, c) == sum_fn_r(f, rd as nat));
        lemma_sum_fn_split(f, rank, q);
        // kept rows: t(i, l) e_lc = (t(i, l) S_l) res_lc
        let kept = |l: int| (t(i, l) * row_scale_r(e, rd, dim, l)) * cm_entry(res, rk, l, c);
        assert forall|l: int| 0 <= l < rank implies #[trigger] f(l) == kept(l) by {
            let s = row_scale_r(e, rd, dim, l);
            assert(s > 0.0real);
            assert(cm_entry(res, rk, l, c) == cm_entry(e, rd, l, c) / s);
            let (tl, el) = (t(i, l), cm_entry(e, rd, l, c));
            assert(tl * el == (tl * s) * (el / s)) by (nonlinear_arith)
                requires s != 0.0real;
        }
        lemma_sum_fn_ext(f, kept, rank);
        let ures = |l: int| u(i, l) * rv_(l, c);
        assert forall|l: int| 0 <= l < rank implies #[trigger] ures(l) == d * kept(l) by {
            let (tl, s, r) = (t(i, l), row_scale_r(e, rd, dim, l), cm_entry(res, rk, l, c));
            assert(ures(l) == ((d * tl) * s) * r);
            assert(((d * tl) * s) * r == d * ((tl * s) * r)) by (nonlinear_arith);
        }
        lemma_sum_fn_scale(kept, ures, d, rank);
        // dropped rows
        let tail = |l: int| f(rank + l);
        let derr = |l: int| dt(i, l) * hd(l, c);
        assert forall|l: int| 0 <= l < q implies #[trigger] derr(l) == d * tail(l) by {
            let (tl, el) = (t(i, rank + l), cm_entry(e, rd, rank + l, c));
            assert(derr(l) == (d * tl) * el);
            assert(tail(l) == tl * el);
            assert((d * tl) * el == d * (tl * el)) by (nonlinear_arith);
        }
        lemma_sum_fn_scale(tail, derr, d, q);
        // assemble
        let (sk, st) = (sum_fn_r(kept, rank), sum_fn_r(tail, q));
        assert(cm_entry(a0, rd, i, c) == d * (sk + st));
        assert(d * (sk + st) == d * sk + d * st) by (nonlinear_arith);
        assert(mmul(u, rv_, rk)(i, c) == sum_fn_r(ures, rank));
        assert(er(i, c) == sum_fn_r(derr, q));
    }
}

/// The residual vector as a matrix-vector product with the augmented vector (z, -1).
pub proof fn lemma_resid_vec_aug(m: Seq<real>, row_dim: int, col_dim: int, z: Seq<real>)
    requires
        row_dim >= 0,
        col_dim >= 1,
        z.len() == col_dim - 1,
    ensures
        resid_vec_r(m, row_dim, col_dim, z) == mat_vec_rc(cm_view(m, row_dim), row_dim, col_dim, aug_vec_r(z)),
{
    let zt = aug_vec_r(z);
    let n = (col_dim - 1) as nat;
    let lhs = resid_vec_r(m, row_dim, col_dim, z);
    let rhs = mat_vec_rc(cm_view(m, row_dim), row_dim, col_dim, zt);
    assert forall|i: int| 0 <= i < row_dim implies #[trigger] lhs[i] == rhs[i] by {
        let f = |c: int| cm_view(m, row_dim)(i, c) * zt[c];
        let g = |c: int| cm_entry(m, row_dim, i, c) * z[c];
        assert forall|c: int| 0 <= c < n implies #[trigger] f(c) == g(c) by {}
        lemma_sum_fn_ext(f, g, n);
        assert(sum_fn_r(f, n + 1) == sum_fn_r(f, n) + f(n as int));
        assert(f(n as int) == cm_entry(m, row_dim, i, col_dim - 1) * (-1.0real));
    }
    assert(lhs =~= rhs);
}

/// A (z, -1) = A' z - a_last for the coefficient columns A' and the last column a_last of A.
pub proof fn lemma_mat_vec_rc_aug(a: MatF, rows: int, cols: int, z: Seq<real>)
    requires
        rows >= 0,
        cols >= 1,
        z.len() == cols - 1,
    ensures
        mat_vec_rc(a, rows, cols, aug_vec_r(z))
            == vadd_r(mat_vec_rc(a, rows, cols - 1, z), vscale_r(-1.0real, col_vec_rc(a, rows, cols - 1))),
{
    let zt = aug_vec_r(z);
    let n = (cols - 1) as nat;
    let lhs = mat_vec_rc(a, rows, cols, zt);
    let rhs = vadd_r(mat_vec_rc(a, rows, cols - 1, z), vscale_r(-1.0real, col_vec_rc(a, rows, cols - 1)));
    assert forall|i: int| 0 <= i < rows implies #[trigger] lhs[i] == rhs[i] by {
        let f = |c: int| a(i, c) * zt[c];
        let g = |c: int| a(i, c) * z[c];
        assert forall|c: int| 0 <= c < n implies #[trigger] f(c) == g(c) by {}
        lemma_sum_fn_ext(f, g, n);
        assert(sum_fn_r(f, n + 1) == sum_fn_r(f, n) + f(n as int));
        assert(f(n as int) == a(i, cols - 1) * (-1.0real));
    }
    assert(lhs =~= rhs);
}

/// A z - b = U (A_K z - b_K) + (E z - f).
pub proof fn lemma_residual_decomp(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, pivots: Seq<int>, t: MatF, z: Seq<real>)
    requires
        dim >= 1,
        reduced_with(a0, e, res, dim, pivots),
        row_comb(normalized_r(a0, dim), e, red_row_dim(a0, dim), red_col_dim(dim), t),
        z.len() == red_col_dim(dim) - 1,
    ensures
        resid_vec_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z)
            == vadd_r(mat_vec_rc(recovery_r(a0, e, dim, t), red_row_dim(a0, dim), pivots.len() as int,
                                 resid_vec_r(res, pivots.len() as int, red_col_dim(dim), z)),
                      vadd_r(mat_vec_rc(err_r(a0, e, dim, t, pivots.len() as int), red_row_dim(a0, dim), red_col_dim(dim) - 1, z),
                             vscale_r(-1.0real, err_rhs_r(a0, e, dim, t, pivots.len() as int)))),
{
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    let rk = pivots.len() as int;
    lemma_red_dims(a0, dim);
    lemma_recovery_identity(a0, e, res, dim, pivots, t);
    let zt = aug_vec_r(z);
    let u = recovery_r(a0, e, dim, t);
    let rv_ = cm_view(res, rk);
    let er = err_r(a0, e, dim, t, rk);
    lemma_resid_vec_aug(a0, rd, cd, z);
    lemma_resid_vec_aug(res, rk, cd, z);
    lemma_mat_vec_rc_meq(cm_view(a0, rd), madd(mmul(u, rv_, rk), er), rd, cd, zt);
    lemma_mat_vec_rc_add(mmul(u, rv_, rk), er, rd, cd, zt);
    lemma_mat_vec_rc_mul(u, rv_, rd, rk, cd, zt);
    lemma_mat_vec_rc_aug(er, rd, cd, z);
}

/* ---------- bounding the error ---------- */

/// ||E||_F <= TOL e_A and ||f|| <= TOL e_b.
pub proof fn lemma_err_bounds(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, pivots: Seq<int>, t: MatF)
    requires
        dim >= 1,
        reduced_with(a0, e, res, dim, pivots),
    ensures
        frob_rc(err_r(a0, e, dim, t, pivots.len() as int), red_row_dim(a0, dim), red_col_dim(dim) - 1)
            <= rv(TOL) * err_coef_bound_r(a0, dim, t, pivots.len() as int),
        norm_r(err_rhs_r(a0, e, dim, t, pivots.len() as int))
            <= rv(TOL) * err_rhs_bound_r(a0, dim, t, pivots.len() as int),
{
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    let rank = pivots.len();
    let rk = rank as int;
    let (q, n) = (rd - rk, cd - 1);
    lemma_red_dims(a0, dim);
    lemma_dropped_rows_small(a0, e, res, dim, pivots);
    let (alpha, beta) = (coef_max_r(a0, dim), rhs_max_r(a0, dim));
    let dt = drop_comb_r(a0, dim, t, rk);
    let hd = drop_rows_r(e, rd, rk);
    let er = err_r(a0, e, dim, t, rk);
    let big_l = drop_norm_r(a0, dim, t, rk);
    let p2 = pow2_r(rank);
    lemma_pow2_pos(rank);
    assert(rv(TOL) == 0.0000000001real);
    lemma_frob_sq_rc_nonneg(dt, rd, q);
    // the coefficient part
    let ma = p2 * (rv(TOL) * alpha);
    assert(ma >= 0.0real) by (nonlinear_arith)
        requires p2 >= 1.0real, rv(TOL) >= 0.0real, alpha >= 0.0real, ma == p2 * (rv(TOL) * alpha);
    lemma_frob_rc_entry_bound(hd, q, n, ma);
    lemma_mmul_frob_rc(dt, hd, rd, q, n);
    let sq = sqrt_r((q as real) * (n as real));
    assert((q as real) * (n as real) >= 0.0real) by (nonlinear_arith)
        requires (q as real) >= 0.0real, (n as real) >= 0.0real;
    assert(frob_rc(er, rd, n) <= big_l * (sq * ma)) by (nonlinear_arith)
        requires
            big_l >= 0.0real,
            frob_rc(er, rd, n) <= big_l * frob_rc(hd, q, n),
            frob_rc(hd, q, n) <= sq * ma;
    assert(big_l * (sq * (p2 * (rv(TOL) * alpha))) == rv(TOL) * (p2 * sq * alpha * big_l)) by (nonlinear_arith);
    // the right-hand side
    let h = col_vec_rc(hd, q, n);
    let mb = rv(TOL) * (p2 * beta);
    assert(mb >= 0.0real) by (nonlinear_arith)
        requires p2 >= 1.0real, rv(TOL) >= 0.0real, beta >= 0.0real, mb == rv(TOL) * (p2 * beta);
    assert forall|l: int| 0 <= l < q implies abs_r(#[trigger] h[l]) <= mb by {
        assert(h[l] == hd(l, n));
    }
    lemma_norm_entry_bound(h, mb);
    let f = err_rhs_r(a0, e, dim, t, rk);
    let dth = mat_vec_rc(dt, rd, q, h);
    assert forall|i: int| 0 <= i < rd implies #[trigger] f[i] == dth[i] by {
        let p = |k: int| dt(i, k) * hd(k, n);
        let p2f = |k: int| dt(i, k) * h[k];
        assert forall|k: int| 0 <= k < q implies #[trigger] p(k) == p2f(k) by {}
        lemma_sum_fn_ext(p, p2f, q as nat);
        assert(f[i] == er(i, n));
        assert(er(i, n) == sum_fn_r(p, q as nat));
    }
    assert(f =~= dth);
    lemma_mat_vec_rc_bound(dt, rd, q, h);
    let sq1 = sqrt_r(q as real);
    assert(norm_r(f) <= big_l * (sq1 * mb)) by (nonlinear_arith)
        requires
            big_l >= 0.0real,
            norm_r(f) <= big_l * norm_r(h),
            norm_r(h) <= sq1 * mb;
    assert(big_l * (sq1 * (rv(TOL) * (p2 * beta))) == rv(TOL) * (p2 * sq1 * beta * big_l)) by (nonlinear_arith);
}

/* ---------- the constant ---------- */

/// A zero coefficient matrix has alpha = 0.
pub proof fn lemma_coef_norm_zero(a0: Seq<real>, dim: int)
    requires
        dim >= 1,
        coef_norm_r(a0, red_row_dim(a0, dim), red_col_dim(dim)) == 0.0real,
    ensures
        coef_max_r(a0, dim) == 0.0real,
{
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    let n = cd - 1;
    let n0 = normalized_r(a0, dim);
    lemma_red_dims(a0, dim);
    lemma_frob_rc_zero(cm_view(a0, rd), rd, n);
    let g = |u: int| abs_r(n0[u]);
    let total = (n * rd) as nat;
    assert forall|u: int| 0 <= u < total implies #[trigger] g(u) <= 0.0real by {
        lemma_fundamental_div_mod(u, rd);
        lemma_mod_bound(u, rd);
        let (i, c) = (u % rd, u / rd);
        assert(c < n) by (nonlinear_arith)
            requires u < n * rd, u == rd * c + i, i >= 0, rd > 0;
        assert(cm_idx(rd, i, c) == u) by (nonlinear_arith)
            requires u == rd * c + i;
        lemma_normalized_entry(a0, dim, i, c);
        assert(cm_view(a0, rd)(i, c) == 0.0real);
        assert(a0[u] == 0.0real);
        let s = row_scale_r(a0, rd, dim, i);
        if s != 0.0real {
            assert(0.0real / s == 0.0real) by (nonlinear_arith)
                requires s != 0.0real;
        }
        assert(n0[u] == 0.0real);
    }
    lemma_max0_fn_le(g, total, 0.0real);
    lemma_max0_fn_bound(g, total);
    assert(cd - 1 == n);
    assert(((cd - 1) * rd) as nat == total);
}

/// A zero right-hand side has beta = 0.
pub proof fn lemma_rhs_norm_zero(a0: Seq<real>, dim: int)
    requires
        dim >= 1,
        rhs_norm_r(a0, red_row_dim(a0, dim), red_col_dim(dim)) == 0.0real,
    ensures
        rhs_max_r(a0, dim) == 0.0real,
{
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    let n0 = normalized_r(a0, dim);
    lemma_red_dims(a0, dim);
    lemma_norm_zero(rhs_vec_r(a0, rd, cd));
    let g = |i: int| abs_r(rhs_r(n0, rd, cd, i));
    assert forall|i: int| 0 <= i < rd implies #[trigger] g(i) <= 0.0real by {
        assert(rhs_vec_r(a0, rd, cd)[i] == 0.0real);
        lemma_normalized_entry(a0, dim, i, cd - 1);
        let s = row_scale_r(a0, rd, dim, i);
        if s != 0.0real {
            assert(0.0real / s == 0.0real) by (nonlinear_arith)
                requires s != 0.0real;
        }
        assert(rhs_r(n0, rd, cd, i) == 0.0real);
    }
    lemma_max0_fn_le(g, rd as nat, 0.0real);
    lemma_max0_fn_bound(g, rd as nat);
}

/// e_A <= C ||A||_F and e_b <= C ||b||, with C >= 0.
pub proof fn lemma_red_const_dominates(a0: Seq<real>, dim: int, t: MatF, rank: int)
    requires
        dim >= 1,
        0 <= rank <= red_row_dim(a0, dim),
    ensures
        red_const_r(a0, dim, t, rank) >= 0.0real,
        err_coef_bound_r(a0, dim, t, rank) <= red_const_r(a0, dim, t, rank) * coef_norm_r(a0, red_row_dim(a0, dim), red_col_dim(dim)),
        err_rhs_bound_r(a0, dim, t, rank) <= red_const_r(a0, dim, t, rank) * rhs_norm_r(a0, red_row_dim(a0, dim), red_col_dim(dim)),
{
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    lemma_red_dims(a0, dim);
    let (ea, eb) = (err_coef_bound_r(a0, dim, t, rank), err_rhs_bound_r(a0, dim, t, rank));
    let (na, nb) = (coef_norm_r(a0, rd, cd), rhs_norm_r(a0, rd, cd));
    let c = red_const_r(a0, dim, t, rank);
    // the ingredients of e_A and e_b are nonnegative
    let (alpha, beta, big_l, p2) = (coef_max_r(a0, dim), rhs_max_r(a0, dim), drop_norm_r(a0, dim, t, rank), pow2_r(rank as nat));
    let n0 = normalized_r(a0, dim);
    lemma_max0_fn_bound(|u: int| abs_r(n0[u]), ((cd - 1) * rd) as nat);
    lemma_max0_fn_bound(|i: int| abs_r(rhs_r(n0, rd, cd, i)), rd as nat);
    lemma_frob_sq_rc_nonneg(drop_comb_r(a0, dim, t, rank), rd, rd - rank);
    lemma_pow2_pos(rank as nat);
    assert(((rd - rank) as real) * ((cd - 1) as real) >= 0.0real) by (nonlinear_arith)
        requires ((rd - rank) as real) >= 0.0real, ((cd - 1) as real) >= 0.0real;
    let (sqa, sqb) = (sqrt_r(((rd - rank) as real) * ((cd - 1) as real)), sqrt_r((rd - rank) as real));
    assert(ea >= 0.0real) by (nonlinear_arith)
        requires p2 >= 0.0real, sqa >= 0.0real, alpha >= 0.0real, big_l >= 0.0real, ea == p2 * sqa * alpha * big_l;
    assert(eb >= 0.0real) by (nonlinear_arith)
        requires p2 >= 0.0real, sqb >= 0.0real, beta >= 0.0real, big_l >= 0.0real, eb == p2 * sqb * beta * big_l;
    lemma_frob_sq_rc_nonneg(cm_view(a0, rd), rd, cd - 1);
    lemma_dot_self_nonneg(rhs_vec_r(a0, rd, cd));
    // the coefficient term
    let ra = ratio0_r(ea, na);
    if na == 0.0real {
        lemma_coef_norm_zero(a0, dim);
        assert(ea == 0.0real) by (nonlinear_arith)
            requires alpha == 0.0real, ea == p2 * sqa * alpha * big_l;
        assert(c * na == 0.0real) by (nonlinear_arith)
            requires na == 0.0real;
    } else {
        assert(ra == ea / na);
        assert(ra >= 0.0real) by (nonlinear_arith)
            requires ea >= 0.0real, na > 0.0real, ra == ea / na;
        assert(c >= ra);
        assert(c * na >= ea) by (nonlinear_arith)
            requires na > 0.0real, c >= ea / na;
    }
    // the right-hand side term
    let rb = ratio0_r(eb, nb);
    if nb == 0.0real {
        lemma_rhs_norm_zero(a0, dim);
        assert(eb == 0.0real) by (nonlinear_arith)
            requires beta == 0.0real, eb == p2 * sqb * beta * big_l;
        assert(c * nb == 0.0real) by (nonlinear_arith)
            requires nb == 0.0real;
    } else {
        assert(rb == eb / nb);
        assert(rb >= 0.0real) by (nonlinear_arith)
            requires eb >= 0.0real, nb > 0.0real, rb == eb / nb;
        assert(c >= rb);
        assert(c * nb >= eb) by (nonlinear_arith)
            requires nb > 0.0real, c >= eb / nb;
    }
}

/* ---------- the residual bound ---------- */

/// ||A z - b|| <= C TOL d(z) + ||U||_F ||A_K z - b_K||.
pub proof fn lemma_residual_bound(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, pivots: Seq<int>, t: MatF, z: Seq<real>)
    requires
        dim >= 1,
        reduced_with(a0, e, res, dim, pivots),
        row_comb(normalized_r(a0, dim), e, red_row_dim(a0, dim), red_col_dim(dim), t),
        z.len() == red_col_dim(dim) - 1,
    ensures
        norm_r(resid_vec_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z))
            <= red_const_r(a0, dim, t, pivots.len() as int) * rv(TOL) * resid_scale_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z)
               + frob_rc(recovery_r(a0, e, dim, t), red_row_dim(a0, dim), pivots.len() as int)
                 * norm_r(resid_vec_r(res, pivots.len() as int, red_col_dim(dim), z)),
{
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    let rk = pivots.len() as int;
    let n = cd - 1;
    lemma_red_dims(a0, dim);
    lemma_residual_decomp(a0, e, res, dim, pivots, t, z);
    let r = resid_vec_r(a0, rd, cd, z);
    let rk_vec = resid_vec_r(res, rk, cd, z);
    let u = recovery_r(a0, e, dim, t);
    let er = err_r(a0, e, dim, t, rk);
    let f = err_rhs_r(a0, e, dim, t, rk);
    let urk = mat_vec_rc(u, rd, rk, rk_vec);
    let ez = mat_vec_rc(er, rd, n, z);
    let mf = vscale_r(-1.0real, f);
    let (nu, nk, nz) = (frob_rc(u, rd, rk), norm_r(rk_vec), norm_r(z));
    let (na, nb) = (coef_norm_r(a0, rd, cd), rhs_norm_r(a0, rd, cd));
    let c = red_const_r(a0, dim, t, rk);
    let (ea, eb) = (err_coef_bound_r(a0, dim, t, rk), err_rhs_bound_r(a0, dim, t, rk));
    // triangle inequalities: ||r|| <= ||U r_K|| + ||E z|| + ||f||
    lemma_norm_triangle(urk, vadd_r(ez, mf));
    lemma_norm_triangle(ez, mf);
    lemma_norm_scale(-1.0real, f);
    assert(norm_r(mf) == norm_r(f));
    // matrix-vector bounds
    lemma_mat_vec_rc_bound(u, rd, rk, rk_vec);
    lemma_mat_vec_rc_bound(er, rd, n, z);
    lemma_err_bounds(a0, e, res, dim, pivots, t);
    lemma_red_const_dominates(a0, dim, t, rk);
    lemma_dot_self_nonneg(z);
    assert(rv(TOL) == 0.0000000001real);
    assert(norm_r(ez) <= rv(TOL) * (ea * nz)) by (nonlinear_arith)
        requires
            norm_r(ez) <= frob_rc(er, rd, n) * nz,
            frob_rc(er, rd, n) <= rv(TOL) * ea,
            nz >= 0.0real;
    assert(ea * nz <= (c * na) * nz) by (nonlinear_arith)
        requires ea <= c * na, nz >= 0.0real;
    assert(rv(TOL) * (ea * nz) <= rv(TOL) * ((c * na) * nz)) by (nonlinear_arith)
        requires ea * nz <= (c * na) * nz, rv(TOL) >= 0.0real;
    assert(rv(TOL) * eb <= rv(TOL) * (c * nb)) by (nonlinear_arith)
        requires eb <= c * nb, rv(TOL) >= 0.0real;
    assert(c * rv(TOL) * (na * nz + nb) == rv(TOL) * ((c * na) * nz) + rv(TOL) * (c * nb)) by (nonlinear_arith);
}

/// The relative residual bound: for d(z) > 0,
/// ||A z - b|| / d(z) <= C TOL + ||U||_F ||A_K z - b_K|| / d(z).
pub proof fn lemma_relative_residual_bound(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, pivots: Seq<int>, t: MatF, z: Seq<real>)
    requires
        dim >= 1,
        reduced_with(a0, e, res, dim, pivots),
        row_comb(normalized_r(a0, dim), e, red_row_dim(a0, dim), red_col_dim(dim), t),
        z.len() == red_col_dim(dim) - 1,
        resid_scale_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z) > 0.0real,
    ensures
        norm_r(resid_vec_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z)) / resid_scale_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z)
            <= red_const_r(a0, dim, t, pivots.len() as int) * rv(TOL)
               + frob_rc(recovery_r(a0, e, dim, t), red_row_dim(a0, dim), pivots.len() as int)
                 * norm_r(resid_vec_r(res, pivots.len() as int, red_col_dim(dim), z))
                 / resid_scale_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z),
{
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    let rk = pivots.len() as int;
    lemma_residual_bound(a0, e, res, dim, pivots, t, z);
    let nr = norm_r(resid_vec_r(a0, rd, cd, z));
    let d = resid_scale_r(a0, rd, cd, z);
    let ct = red_const_r(a0, dim, t, rk) * rv(TOL);
    let w = frob_rc(recovery_r(a0, e, dim, t), rd, rk) * norm_r(resid_vec_r(res, rk, cd, z));
    assert(nr / d <= ct + w / d) by (nonlinear_arith)
        requires d > 0.0real, nr <= ct * d + w;
}

/// A zero scale forces a zero residual: ||A||_F ||z|| + ||b|| = 0 means b = 0 and A z = 0.
pub proof fn lemma_residual_zero_scale(a0: Seq<real>, dim: int, z: Seq<real>)
    requires
        dim >= 1,
        z.len() == red_col_dim(dim) - 1,
        resid_scale_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z) == 0.0real,
    ensures
        norm_r(resid_vec_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z)) == 0.0real,
{
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    let n = (cd - 1) as nat;
    lemma_red_dims(a0, dim);
    let (na, nb, nz) = (coef_norm_r(a0, rd, cd), rhs_norm_r(a0, rd, cd), norm_r(z));
    lemma_frob_sq_rc_nonneg(cm_view(a0, rd), rd, cd - 1);
    lemma_dot_self_nonneg(rhs_vec_r(a0, rd, cd));
    lemma_dot_self_nonneg(z);
    assert(na * nz >= 0.0real) by (nonlinear_arith)
        requires na >= 0.0real, nz >= 0.0real;
    assert(nb == 0.0real);
    assert(na == 0.0real || nz == 0.0real) by (nonlinear_arith)
        requires na * nz == 0.0real;
    lemma_norm_zero(rhs_vec_r(a0, rd, cd));
    if na == 0.0real {
        lemma_frob_rc_zero(cm_view(a0, rd), rd, cd - 1);
    } else {
        lemma_norm_zero(z);
    }
    let r = resid_vec_r(a0, rd, cd, z);
    assert forall|i: int| 0 <= i < rd implies #[trigger] r[i] == 0.0real by {
        let f = |c: int| cm_entry(a0, rd, i, c) * z[c];
        assert forall|c: int| 0 <= c < n implies #[trigger] f(c) == 0.0real by {
            if na == 0.0real {
                assert(cm_view(a0, rd)(i, c) == 0.0real);
            } else {
                assert(z[c] == 0.0real);
            }
        }
        lemma_sum_fn_all_zero(f, n);
        assert(rhs_vec_r(a0, rd, cd)[i] == 0.0real);
    }
    lemma_dot_sum_fn(r, r);
    lemma_sum_fn_all_zero(|k: int| r[k] * r[k], r.len());
    lemma_sqrt_unique(0.0real, 0.0real);
}

/// Without dropped rows the error terms vanish: C = 0.
pub proof fn lemma_red_const_no_drop(a0: Seq<real>, dim: int, t: MatF, rank: int)
    requires
        dim >= 1,
        rank == red_row_dim(a0, dim),
    ensures
        red_const_r(a0, dim, t, rank) == 0.0real,
{
    let (rd, cd) = (red_row_dim(a0, dim), red_col_dim(dim));
    let dt = drop_comb_r(a0, dim, t, rank);
    let inner = |i: int| sum_fn_r(|k: int| dt(i, k) * dt(i, k), 0nat);
    assert forall|i: int| 0 <= i < rd implies #[trigger] inner(i) == 0.0real by {}
    lemma_sum_fn_all_zero(inner, rd as nat);
    assert(frob_sq_rc(dt, rd, 0) == 0.0real);
    lemma_sqrt_unique(0.0real, 0.0real);
    let big_l = drop_norm_r(a0, dim, t, rank);
    assert(big_l == 0.0real);
    let (ea, eb) = (err_coef_bound_r(a0, dim, t, rank), err_rhs_bound_r(a0, dim, t, rank));
    assert(ea == 0.0real) by (nonlinear_arith)
        requires big_l == 0.0real, ea == pow2_r(rank as nat) * sqrt_r(((rd - rank) as real) * ((cd - 1) as real)) * coef_max_r(a0, dim) * big_l;
    assert(eb == 0.0real) by (nonlinear_arith)
        requires big_l == 0.0real, eb == pow2_r(rank as nat) * sqrt_r((rd - rank) as real) * rhs_max_r(a0, dim) * big_l;
    let (na, nb) = (coef_norm_r(a0, rd, cd), rhs_norm_r(a0, rd, cd));
    if na != 0.0real {
        assert(0.0real / na == 0.0real) by (nonlinear_arith)
            requires na != 0.0real;
    }
    if nb != 0.0real {
        assert(0.0real / nb == 0.0real) by (nonlinear_arith)
            requires nb != 0.0real;
    }
}

/* ---------- the bound for a reduction ---------- */

/// The pivot sequence of a reduction (a witness of `reduced_system`).
pub open spec fn red_pivots(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int) -> Seq<int> {
    choose|pivots: Seq<int>| #![trigger pivots.len()] reduced_with(a0, e, res, dim, pivots)
}

/// The row transformation of a reduction: normalized input = T * echelon form.
pub open spec fn red_transform(a0: Seq<real>, e: Seq<real>, dim: int) -> MatF {
    choose|t: MatF| #[trigger] row_comb(normalized_r(a0, dim), e, red_row_dim(a0, dim), red_col_dim(dim), t)
}

/// The bound for `reduced_system`, with U and C built from the witnesses `red_pivots` and
/// `red_transform` (both independent of z): ||A z - b|| <= C TOL d(z) + ||U||_F ||A_K z - b_K||,
/// its relative form when d(z) > 0, and A z - b = 0 when d(z) = 0.
pub proof fn lemma_reduction_residual_bound(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, z: Seq<real>)
    requires
        dim >= 1,
        reduced_system(a0, e, res, dim),
        z.len() == red_col_dim(dim) - 1,
    ensures
        ({
            let t = red_transform(a0, e, dim);
            let rank = red_pivots(a0, e, res, dim).len() as int;
            let nr = norm_r(resid_vec_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z));
            let d = resid_scale_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z);
            let ct = red_const_r(a0, dim, t, rank) * rv(TOL);
            let w = frob_rc(recovery_r(a0, e, dim, t), red_row_dim(a0, dim), rank)
                * norm_r(resid_vec_r(res, rank, red_col_dim(dim), z));
            &&& nr <= ct * d + w
            &&& d > 0.0real ==> nr / d <= ct + w / d
            &&& d == 0.0real ==> nr == 0.0real
        }),
{
    let pivots = red_pivots(a0, e, res, dim);
    assert(reduced_with(a0, e, res, dim, pivots));
    let t = red_transform(a0, e, dim);
    assert(row_comb(normalized_r(a0, dim), e, red_row_dim(a0, dim), red_col_dim(dim), t));
    lemma_residual_bound(a0, e, res, dim, pivots, t, z);
    let d = resid_scale_r(a0, red_row_dim(a0, dim), red_col_dim(dim), z);
    if d > 0.0real {
        lemma_relative_residual_bound(a0, e, res, dim, pivots, t, z);
    } else if d == 0.0real {
        lemma_residual_zero_scale(a0, dim, z);
    }
}

} // verus!

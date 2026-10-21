// Spec-level support for the constraint reduction (not trusted).
//
// `linear_remove_redundant_sym` runs `gauss_elim` on the augmented constraint
// matrix [A | b] (row i is the packed symmetric matrix M_i followed by b_i),
// aborts if the elimination finds the system inconsistent, and returns the
// first `rank` rows of the echelon form, each divided by the Frobenius norm of
// the symmetric matrix M_i it represents,
//
//     s_i = ||M_i||_F = sqrt( sum_{r <= c} pk_wt(r, c) M_i[r, c]^2 )
//
// (off-diagonal packed entries count twice).  This module defines that scale, the
// result predicate `reduced_system`, and proves what the reduction guarantees:
// every solution of the input system solves the reduced system, and every
// solution of the reduced system satisfies the kept echelon rows exactly and
// each dropped echelon row up to the residual 2^rank * eps * (1 + ||x||_1),
// from the tolerance bounds `skipped_bounded` of gauss_spec.rs.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::seq::*;
#[cfg(verus_keep_ghost)]
use crate::trusted::{rv, sqrt_r};
#[cfg(verus_keep_ghost)]
use crate::real_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::solve_spec::abs_max_r;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::MatF;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::constr_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::gauss_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::matrix::TOL;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- shapes and tolerance ---------- */

/// Number of columns of the augmented constraint matrix for dim x dim symmetric matrices.
pub open spec fn red_col_dim(dim: int) -> int {
    pk_len(dim) + 1
}

/// The elimination tolerance of gauss_elim: TOL * max |entry of the input|.
pub open spec fn gauss_eps_r(a0: Seq<real>) -> real {
    rv(TOL) * abs_max_r(a0)
}

/* ---------- the Frobenius row norm ---------- */

/// pk_wt(r, c) * M_i[r, c]^2 for row i of the stored constraint matrix (weight 2 off the diagonal).
pub open spec fn row_sq_term_r(e: Seq<real>, row_dim: int, i: int, r: int, c: int) -> real {
    pk_wt(r, c) * (constr_entry(e, row_dim, i, pk_idx(r, c)) * constr_entry(e, row_dim, i, pk_idx(r, c)))
}

/// Partial sum of the terms of packed column c over rows r < m.
pub open spec fn col_sq_r(e: Seq<real>, row_dim: int, i: int, c: int, m: int) -> real {
    sum_fn_r(|r: int| row_sq_term_r(e, row_dim, i, r, c), m as nat)
}

/// The weighted sum of squares over the first k packed columns.
pub open spec fn row_sq_upto_r(e: Seq<real>, row_dim: int, i: int, k: int) -> real {
    tri_sum_r(|r: int, c: int| row_sq_term_r(e, row_dim, i, r, c), k as nat)
}

/// Squared Frobenius norm of the symmetric matrix of row i: sum_{r <= c < dim} pk_wt(r, c) M_i[r, c]^2.
pub open spec fn row_sq_r(e: Seq<real>, row_dim: int, dim: int, i: int) -> real {
    row_sq_upto_r(e, row_dim, i, dim)
}

/// The Frobenius norm ||M_i||_F of row i.
pub open spec fn row_scale_r(e: Seq<real>, row_dim: int, dim: int, i: int) -> real {
    sqrt_r(row_sq_r(e, row_dim, dim, i))
}

/// Row i of the stored constraint matrix as a packed symmetric matrix.
pub open spec fn packed_row_r(e: Seq<real>, row_dim: int, dim: int, i: int) -> Seq<real> {
    Seq::new(pk_len(dim) as nat, |n: int| constr_entry(e, row_dim, i, n))
}

/// The squared row norm is the Frobenius self-product <M_i, M_i>_F of sym_spec.rs.
pub proof fn lemma_row_sq_frob(e: Seq<real>, row_dim: int, dim: int, i: int)
    requires
        dim >= 0,
    ensures
        row_sq_r(e, row_dim, dim, i) == frob_r(packed_row_r(e, row_dim, dim, i), packed_row_r(e, row_dim, dim, i), dim),
{
    let m = packed_row_r(e, row_dim, dim, i);
    lemma_frob_tri(m, m, dim as nat);
    let f = |r: int, c: int| row_sq_term_r(e, row_dim, i, r, c);
    let g = |r: int, c: int| pk_wt(r, c) * (m[pk_idx(r, c)] * m[pk_idx(r, c)]);
    assert forall|r: int, c: int| 0 <= r <= c < dim implies #[trigger] f(r, c) == g(r, c) by {
        lemma_pk_idx_bound(r, c, dim);
    }
    lemma_tri_sum_ext(f, g, dim as nat);
}

pub proof fn lemma_col_sq_zero(e: Seq<real>, row_dim: int, i: int, c: int)
    ensures
        col_sq_r(e, row_dim, i, c, 0) == 0.0real,
{
}

/// One more row of a packed column.
pub proof fn lemma_col_sq_step(e: Seq<real>, row_dim: int, i: int, c: int, m: int)
    requires
        m >= 0,
    ensures
        col_sq_r(e, row_dim, i, c, m + 1) == col_sq_r(e, row_dim, i, c, m) + row_sq_term_r(e, row_dim, i, m, c),
{
}

pub proof fn lemma_row_sq_upto_zero(e: Seq<real>, row_dim: int, i: int)
    ensures
        row_sq_upto_r(e, row_dim, i, 0) == 0.0real,
{
}

/// One more packed column (rows 0..k of column k).
pub proof fn lemma_row_sq_upto_step(e: Seq<real>, row_dim: int, i: int, k: int)
    requires
        k >= 0,
    ensures
        row_sq_upto_r(e, row_dim, i, k + 1) == row_sq_upto_r(e, row_dim, i, k) + col_sq_r(e, row_dim, i, k, k + 1),
{
    let f = |r: int, c: int| row_sq_term_r(e, row_dim, i, r, c);
    let cols = |c: int| sum_fn_r(|r: int| f(r, c), (c + 1) as nat);
    assert(row_sq_upto_r(e, row_dim, i, k + 1) == sum_fn_r(cols, (k + 1) as nat));
    assert(row_sq_upto_r(e, row_dim, i, k) == sum_fn_r(cols, k as nat));
    assert(sum_fn_r(cols, (k + 1) as nat) == sum_fn_r(cols, k as nat) + cols(k));
    lemma_sum_fn_ext(|r: int| f(r, k), |r: int| row_sq_term_r(e, row_dim, i, r, k), (k + 1) as nat);
}

/* ---------- positivity of the row norm ---------- */

/// Triangle sums of nonnegative terms dominate each term.
pub proof fn lemma_tri_sum_ge_term(f: spec_fn(int, int) -> real, dim: nat, r0: int, c0: int)
    requires
        0 <= r0 <= c0 < dim,
        forall|r: int, c: int| 0 <= r <= c < dim ==> #[trigger] f(r, c) >= 0.0real,
    ensures
        tri_sum_r(f, dim) >= f(r0, c0),
{
    let cols = |c: int| sum_fn_r(|r: int| f(r, c), (c + 1) as nat);
    assert forall|c: int| 0 <= c < dim implies #[trigger] cols(c) >= 0.0real by {
        lemma_sum_fn_nonneg(|r: int| f(r, c), (c + 1) as nat);
    }
    lemma_sum_fn_ge_term(cols, dim, c0);
    lemma_sum_fn_ge_term(|r: int| f(r, c0), (c0 + 1) as nat, r0);
}


/// With eps > 0 every pivot row has a positive norm: its pivot entry has magnitude
/// at least eps and lies in the packed part (the pivot column is not the right-hand side).
pub proof fn lemma_row_scale_pos(e: Seq<real>, row_dim: int, dim: int, pivots: Seq<int>, eps: real, i: int)
    requires
        dim >= 0,
        eps > 0.0real,
        0 <= i < pivots.len(),
        echelon_form(e, row_dim, red_col_dim(dim), pivots, eps),
        pivots[i] < pk_len(dim),
    ensures
        row_sq_r(e, row_dim, dim, i) > 0.0real,
        row_scale_r(e, row_dim, dim, i) > 0.0real,
{
    let t = pivots[i];
    lemma_pk_idx_surjective(t, dim);
    let (r0, c0) = choose|r: int, c: int| 0 <= r <= c < dim && #[trigger] pk_idx(r, c) == t;
    let f = |r: int, c: int| row_sq_term_r(e, row_dim, i, r, c);
    assert forall|r: int, c: int| 0 <= r <= c < dim implies #[trigger] f(r, c) >= 0.0real by {
        let x = constr_entry(e, row_dim, i, pk_idx(r, c));
        assert(x * x >= 0.0real) by (nonlinear_arith);
    }
    lemma_tri_sum_ge_term(f, dim as nat, r0, c0);
    let x = constr_entry(e, row_dim, i, t);
    assert(abs_r(x) >= eps);
    assert(x * x >= eps * eps) by (nonlinear_arith)
        requires abs_r(x) >= eps, eps > 0.0real, abs_r(x) == x || abs_r(x) == -x;
    assert(eps * eps > 0.0real) by (nonlinear_arith)
        requires eps > 0.0real;
    assert(f(r0, c0) == pk_wt(r0, c0) * (x * x));
    let sq = row_sq_r(e, row_dim, dim, i);
    let s = sqrt_r(sq);
    assert(s > 0.0real) by (nonlinear_arith)
        requires s >= 0.0real, s * s == sq, sq > 0.0real;
}

/* ---------- residuals of dropped rows ---------- */

/// ||x||_1.
pub open spec fn one_norm_r(x: Seq<real>) -> real {
    sum_fn_r(|c: int| abs_r(x[c]), x.len())
}

pub proof fn lemma_pow2_monotone(k1: nat, k2: nat)
    requires
        k1 <= k2,
    ensures
        pow2_r(k1) <= pow2_r(k2),
    decreases k2 - k1,
{
    if k1 < k2 {
        lemma_pow2_monotone(k1, (k2 - 1) as nat);
        lemma_pow2_pos((k2 - 1) as nat);
    }
}

/// Every entry of a dropped row (i >= rank) of the echelon form has magnitude at most
/// 2^rank * eps: it is zero in pivot columns and below the skip bound elsewhere.
pub proof fn lemma_dropped_row_entry(e: Seq<real>, row_dim: int, col_dim: int, pivots: Seq<int>, eps: real, i: int, c: int)
    requires
        eps >= 0.0real,
        echelon_form(e, row_dim, col_dim, pivots, eps),
        skipped_bounded(e, row_dim, col_dim, pivots, eps, col_dim),
        pivots.len() <= i < row_dim,
        0 <= c < col_dim,
    ensures
        abs_r(cm_entry(e, row_dim, i, c)) <= pow2_r(pivots.len()) * eps,
{
    let rank = pivots.len();
    lemma_pow2_pos(rank);
    assert(pow2_r(rank) * eps >= 0.0real) by (nonlinear_arith)
        requires pow2_r(rank) > 0.0real, eps >= 0.0real;
    if is_pivot_col(pivots, c) {
        let l = choose|l: int| 0 <= l < pivots.len() && #[trigger] pivots[l] == c;
        assert(cm_entry(e, row_dim, i, pivots[l]) == 0.0real);
    } else {
        lemma_pivots_below_bounds(pivots, c);
        assert(abs_r(cm_entry(e, row_dim, i, c)) < skip_bound(pivots, c, eps));
        let k = (rank - pivots_below(pivots, c)) as nat;
        lemma_pow2_monotone(k, rank);
        assert(pow2_r(k) * eps <= pow2_r(rank) * eps) by (nonlinear_arith)
            requires pow2_r(k) <= pow2_r(rank), eps >= 0.0real;
    }
}

/// For any x, a dropped row of the echelon form has residual at most 2^rank * eps * (1 + ||x||_1).
pub proof fn lemma_dropped_row_residual(e: Seq<real>, row_dim: int, col_dim: int, pivots: Seq<int>, eps: real, i: int, x: Seq<real>)
    requires
        col_dim >= 1,
        eps >= 0.0real,
        echelon_form(e, row_dim, col_dim, pivots, eps),
        skipped_bounded(e, row_dim, col_dim, pivots, eps, col_dim),
        pivots.len() <= i < row_dim,
        x.len() == col_dim - 1,
    ensures
        abs_r(row_residual_r(e, row_dim, col_dim, i, x)) <= pow2_r(pivots.len()) * eps * (1.0real + one_norm_r(x)),
{
    let n = (col_dim - 1) as nat;
    let bnd = pow2_r(pivots.len()) * eps;
    let f = |c: int| cm_entry(e, row_dim, i, c) * x[c];
    let g = |c: int| abs_r(f(c));
    let h = |c: int| bnd * abs_r(x[c]);
    lemma_abs_sum_fn_le(f, n);
    assert forall|c: int| 0 <= c < n implies #[trigger] g(c) <= h(c) by {
        lemma_dropped_row_entry(e, row_dim, col_dim, pivots, eps, i, c);
        let (a, w) = (cm_entry(e, row_dim, i, c), x[c]);
        lemma_abs_mul(a, w);
        assert(abs_r(a) * abs_r(w) <= bnd * abs_r(w)) by (nonlinear_arith)
            requires abs_r(a) <= bnd, abs_r(w) >= 0.0real;
    }
    lemma_sum_fn_le(g, h, n);
    lemma_sum_fn_scale(|c: int| abs_r(x[c]), h, bnd, n);
    lemma_dropped_row_entry(e, row_dim, col_dim, pivots, eps, i, col_dim - 1);
    let (s, b) = (sum_fn_r(f, n), cm_entry(e, row_dim, i, col_dim - 1));
    lemma_abs_sub_triangle(s, b);
    assert(bnd * one_norm_r(x) + bnd == bnd * (1.0real + one_norm_r(x))) by (nonlinear_arith);
}

/* ---------- the result of the reduction ---------- */

/// The reduced system res (rank rows) obtained from the input a0 through the echelon form e
/// with pivot sequence pivots: res row i is echelon row i divided by its Frobenius norm, and the
/// zero columns of a0 are zero columns of e. The rank is positive (the input of the reduction
/// is nonzero).
pub open spec fn reduced_with(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, pivots: Seq<int>) -> bool {
    let cd = red_col_dim(dim);
    let rd = a0.len() as int / cd;
    let rank = pivots.len() as int;
    let eps = gauss_eps_r(a0);
    &&& e.len() == a0.len()
    &&& rank >= 1
    &&& res.len() == cd * rank
    &&& same_solutions(a0, e, rd, cd)
    &&& zero_cols_kept(a0, e, rd, cd)
    &&& echelon_form(e, rd, cd, pivots, eps)
    &&& skipped_bounded(e, rd, cd, pivots, eps, cd)
    &&& forall|l: int| 0 <= l < rank ==> #[trigger] pivots[l] < cd - 1
    &&& forall|i: int| 0 <= i < rank ==> #[trigger] row_scale_r(e, rd, dim, i) > 0.0real
    &&& forall|i: int, j: int| 0 <= i < rank && 0 <= j < cd
            ==> #[trigger] cm_entry(res, rank, i, j) == cm_entry(e, rd, i, j) / row_scale_r(e, rd, dim, i)
    &&& reduction_transform(a0, e, rd, cd, rank)
}

/// The input is recovered from the echelon form, a0 = T e, by a transformation with bounded
/// columns (gauss_elim). Opaque: its quantifier would be instantiated at every entry of the
/// large matrices the conversion handles; it is revealed where it is established and used.
#[verifier::opaque]
pub open spec fn reduction_transform(a0: Seq<real>, e: Seq<real>, rd: int, cd: int, rank: int) -> bool {
    exists|t: MatF| #[trigger] row_comb(a0, e, rd, cd, t) && cols_bounded(t, rd, rank)
}

/// The number of rows of an augmented system over dim x dim symmetric matrices.
pub open spec fn red_row_dim(a0: Seq<real>, dim: int) -> int {
    a0.len() as int / red_col_dim(dim)
}

/// The postcondition of `linear_remove_redundant_sym` (a0 the input, e its echelon form, res the result).
pub open spec fn reduced_system(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int) -> bool {
    exists|pivots: Seq<int>| #![trigger pivots.len()] reduced_with(a0, e, res, dim, pivots)
}

/// Every solution of the input system solves the reduced system.
pub proof fn lemma_reduced_of_input_solution(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, pivots: Seq<int>, x: Seq<real>)
    requires
        dim >= 0,
        reduced_with(a0, e, res, dim, pivots),
        x.len() == red_col_dim(dim) - 1,
        solves(a0, a0.len() as int / red_col_dim(dim), red_col_dim(dim), x),
    ensures
        solves(res, pivots.len() as int, red_col_dim(dim), x),
{
    let cd = red_col_dim(dim);
    let rd = a0.len() as int / cd;
    let rank = pivots.len() as int;
    let n = (cd - 1) as nat;
    assert(solves(e, rd, cd, x));
    assert forall|i: int| 0 <= i < rank implies #[trigger] aug_row_ok(res, rank, cd, i, x) by {
        assert(aug_row_ok(e, rd, cd, i, x));
        let s = row_scale_r(e, rd, dim, i);
        let f = |c: int| cm_entry(e, rd, i, c) * x[c];
        let g = |c: int| cm_entry(res, rank, i, c) * x[c];
        assert forall|c: int| 0 <= c < n implies #[trigger] g(c) == (1.0real / s) * f(c) by {
            let (u, w) = (cm_entry(e, rd, i, c), x[c]);
            assert(cm_entry(res, rank, i, c) == u / s);
            assert((u / s) * w == (1.0real / s) * (u * w)) by (nonlinear_arith)
                requires s != 0.0real;
        }
        lemma_sum_fn_scale(f, g, 1.0real / s, n);
        let b = cm_entry(e, rd, i, cd - 1);
        assert(cm_entry(res, rank, i, cd - 1) == b / s);
        assert((1.0real / s) * b == b / s) by (nonlinear_arith)
            requires s != 0.0real;
    }
}

/// Every solution of the reduced system satisfies the kept echelon rows exactly and each
/// dropped echelon row up to the residual 2^rank * eps * (1 + ||x||_1).
pub proof fn lemma_reduced_solution_echelon(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, pivots: Seq<int>, x: Seq<real>)
    requires
        dim >= 0,
        reduced_with(a0, e, res, dim, pivots),
        x.len() == red_col_dim(dim) - 1,
        solves(res, pivots.len() as int, red_col_dim(dim), x),
    ensures
        forall|i: int| 0 <= i < pivots.len() ==> #[trigger] aug_row_ok(e, a0.len() as int / red_col_dim(dim), red_col_dim(dim), i, x),
        forall|i: int| pivots.len() <= i < a0.len() as int / red_col_dim(dim)
            ==> abs_r(#[trigger] row_residual_r(e, a0.len() as int / red_col_dim(dim), red_col_dim(dim), i, x))
                <= pow2_r(pivots.len()) * gauss_eps_r(a0) * (1.0real + one_norm_r(x)),
{
    let cd = red_col_dim(dim);
    let rd = a0.len() as int / cd;
    let rank = pivots.len() as int;
    let n = (cd - 1) as nat;
    let eps = gauss_eps_r(a0);
    assert forall|i: int| 0 <= i < rank implies #[trigger] aug_row_ok(e, rd, cd, i, x) by {
        assert(aug_row_ok(res, rank, cd, i, x));
        let s = row_scale_r(e, rd, dim, i);
        let f = |c: int| cm_entry(e, rd, i, c) * x[c];
        let g = |c: int| cm_entry(res, rank, i, c) * x[c];
        assert forall|c: int| 0 <= c < n implies #[trigger] g(c) == (1.0real / s) * f(c) by {
            let (u, w) = (cm_entry(e, rd, i, c), x[c]);
            assert(cm_entry(res, rank, i, c) == u / s);
            assert((u / s) * w == (1.0real / s) * (u * w)) by (nonlinear_arith)
                requires s != 0.0real;
        }
        lemma_sum_fn_scale(f, g, 1.0real / s, n);
        let (sf, b) = (sum_fn_r(f, n), cm_entry(e, rd, i, cd - 1));
        assert(cm_entry(res, rank, i, cd - 1) == b / s);
        assert((1.0real / s) * sf == b / s);
        assert(sf == b) by (nonlinear_arith)
            requires s != 0.0real, (1.0real / s) * sf == b / s;
    }
    // eps >= 0
    lemma_max0_fn_bound(|t: int| abs_r(a0[t]), a0.len());
    assert(rv(TOL) == 0.0000000001real);
    assert forall|i: int| rank <= i < rd
        implies abs_r(#[trigger] row_residual_r(e, rd, cd, i, x)) <= pow2_r(pivots.len()) * eps * (1.0real + one_norm_r(x)) by {
        lemma_dropped_row_residual(e, rd, cd, pivots, eps, i, x);
    }
}

/* ---------- residuals of the original rows ---------- */

/// The residual of a kept echelon row is its Frobenius norm times the residual of the
/// corresponding reduced row.
pub proof fn lemma_kept_row_residual(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, pivots: Seq<int>, l: int, x: Seq<real>)
    requires
        dim >= 0,
        reduced_with(a0, e, res, dim, pivots),
        x.len() == red_col_dim(dim) - 1,
        0 <= l < pivots.len(),
    ensures
        row_residual_r(e, red_row_dim(a0, dim), red_col_dim(dim), l, x)
            == row_scale_r(e, red_row_dim(a0, dim), dim, l) * row_residual_r(res, pivots.len() as int, red_col_dim(dim), l, x),
{
    let cd = red_col_dim(dim);
    let rd = red_row_dim(a0, dim);
    let rank = pivots.len() as int;
    let n = (cd - 1) as nat;
    let s = row_scale_r(e, rd, dim, l);
    let f = |c: int| cm_entry(e, rd, l, c) * x[c];
    let g = |c: int| cm_entry(res, rank, l, c) * x[c];
    assert forall|c: int| 0 <= c < n implies #[trigger] g(c) == (1.0real / s) * f(c) by {
        let (u, w) = (cm_entry(e, rd, l, c), x[c]);
        assert(cm_entry(res, rank, l, c) == u / s);
        assert((u / s) * w == (1.0real / s) * (u * w)) by (nonlinear_arith)
            requires s != 0.0real;
    }
    lemma_sum_fn_scale(f, g, 1.0real / s, n);
    let (sf, b) = (sum_fn_r(f, n), cm_entry(e, rd, l, cd - 1));
    assert(cm_entry(res, rank, l, cd - 1) == b / s);
    assert(row_residual_r(res, rank, cd, l, x) == (1.0real / s) * sf - b / s);
    assert(s * ((1.0real / s) * sf - b / s) == sf - b) by (nonlinear_arith)
        requires s != 0.0real;
}

/// The residual bound of the reduction: for any x, every original row has residual at most
/// rd * sum_{l < rank} ||E_l||_F |res_l(x)| + (rd - rank) * 2^rank * eps * (1 + ||x||_1), where
/// res_l is the residual of reduced row l, ||E_l||_F the scale of echelon row l, and the
/// second term bounds the dropped echelon rows (lemma_dropped_row_residual).
pub proof fn lemma_reduced_residual_bound(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, pivots: Seq<int>, x: Seq<real>, i: int)
    requires
        dim >= 0,
        reduced_with(a0, e, res, dim, pivots),
        x.len() == red_col_dim(dim) - 1,
        0 <= i < red_row_dim(a0, dim),
    ensures
        abs_r(row_residual_r(a0, red_row_dim(a0, dim), red_col_dim(dim), i, x))
            <= (red_row_dim(a0, dim) as real)
                * sum_fn_r(|l: int| row_scale_r(e, red_row_dim(a0, dim), dim, l)
                    * abs_r(row_residual_r(res, pivots.len() as int, red_col_dim(dim), l, x)), pivots.len())
               + ((red_row_dim(a0, dim) - pivots.len()) as real)
                * (pow2_r(pivots.len()) * gauss_eps_r(a0) * (1.0real + one_norm_r(x))),
{
    let cd = red_col_dim(dim);
    let rd = red_row_dim(a0, dim);
    let rank = pivots.len() as int;
    let eps = gauss_eps_r(a0);
    lemma_pk_col_start_monotone(0, dim);
    reveal(reduction_transform);
    let t = choose|t: MatF| #[trigger] row_comb(a0, e, rd, cd, t) && cols_bounded(t, rd, rank);
    lemma_row_comb_residual_bound(a0, e, rd, cd, t, rank, i, x);
    // kept rows: scale times the reduced residual
    let ar = |l: int| abs_r(row_residual_r(e, rd, cd, l, x));
    let kept = |l: int| row_scale_r(e, rd, dim, l) * abs_r(row_residual_r(res, rank, cd, l, x));
    assert forall|l: int| 0 <= l < rank implies #[trigger] ar(l) == kept(l) by {
        lemma_kept_row_residual(a0, e, res, dim, pivots, l, x);
        let (s, v) = (row_scale_r(e, rd, dim, l), row_residual_r(res, rank, cd, l, x));
        lemma_abs_mul(s, v);
        assert(abs_r(s) == s);
    }
    lemma_sum_fn_ext(ar, kept, pivots.len());
    // dropped rows: the uniform bound
    lemma_max0_fn_bound(|k: int| abs_r(a0[k]), a0.len());
    assert(rv(TOL) == 0.0000000001real);
    let bnd = pow2_r(pivots.len()) * eps * (1.0real + one_norm_r(x));
    let ar2 = |l: int| abs_r(row_residual_r(e, rd, cd, rank + l, x));
    let cb = |l: int| bnd;
    assert forall|l: int| 0 <= l < rd - rank implies #[trigger] ar2(l) <= cb(l) by {
        lemma_dropped_row_residual(e, rd, cd, pivots, eps, rank + l, x);
    }
    lemma_sum_fn_le(ar2, cb, (rd - rank) as nat);
    lemma_sum_fn_const(bnd, (rd - rank) as nat);
}

} // verus!

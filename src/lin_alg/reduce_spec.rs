// Spec-level support for the constraint reduction (not trusted).
//
// `linear_remove_redundant_sym` runs `gauss_elim` on the augmented constraint
// matrix [A | b] (row i is the packed symmetric matrix M_i followed by b_i),
// aborts if the elimination finds the system inconsistent, and returns the
// first `rank` rows of the echelon form, each divided by the Frobenius norm
// s_i = ||M_i||_F of the symmetric matrix M_i it represents.  The loop
// accumulates this norm over the packed entries as
//
//     s_i^2 = sum_{r <= c} pk_wt(r, c) M_i[r, c]^2
//
// (off-diagonal packed entries count twice); `lemma_row_sq_frob` identifies the
// two.  This module defines the scale, proves that every kept row has a positive
// scale, and defines the result predicate `reduced_system`: the kept rows are the
// normalized echelon rows of an elimination satisfying the contract of `gauss_elim`
// (solution set, zero columns, echelon structure, tolerance-based pivot and skip
// rules).  Nothing is claimed about the rows the reduction drops.

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
use crate::lin_alg::mat_spec::{frob_r, frob_norm_r};
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

/// The weighted packed sum of squares the loop accumulates: sum_{r <= c < dim} pk_wt(r, c) M_i[r, c]^2.
pub open spec fn row_sq_r(e: Seq<real>, row_dim: int, dim: int, i: int) -> real {
    row_sq_upto_r(e, row_dim, i, dim)
}

/// Row i of the stored constraint matrix as a packed symmetric matrix M_i.
pub open spec fn packed_row_r(e: Seq<real>, row_dim: int, dim: int, i: int) -> Seq<real> {
    Seq::new(pk_len(dim) as nat, |n: int| constr_entry(e, row_dim, i, n))
}

/// The scale of row i: the Frobenius norm ||M_i||_F of its symmetric matrix.
pub open spec fn row_scale_r(e: Seq<real>, row_dim: int, dim: int, i: int) -> real {
    frob_norm_r(packed_row_r(e, row_dim, dim, i), dim)
}

/// The packed sum of squares is the Frobenius self-product <M_i, M_i>_F, so the scale is its square root.
pub proof fn lemma_row_sq_frob(e: Seq<real>, row_dim: int, dim: int, i: int)
    requires
        dim >= 0,
    ensures
        row_sq_r(e, row_dim, dim, i) == frob_r(packed_row_r(e, row_dim, dim, i), packed_row_r(e, row_dim, dim, i), dim),
        row_scale_r(e, row_dim, dim, i) == sqrt_r(row_sq_r(e, row_dim, dim, i)),
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
        lemma_sq_nonneg(x);
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
    lemma_row_sq_frob(e, row_dim, dim, i);
    let s = sqrt_r(sq);
    assert(s > 0.0real) by (nonlinear_arith)
        requires s >= 0.0real, s * s == sq, sq > 0.0real;
}

/* ---------- the result of the reduction ---------- */

/// The reduced system res (rank rows) obtained from the input a0 through the echelon form e
/// with pivot sequence pivots: res row i is echelon row i divided by its Frobenius norm, and the
/// zero columns of a0 are zero columns of e. The rank is positive (the input of the reduction
/// is nonzero).
pub open spec fn reduced_with(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int, pivots: Seq<int>) -> bool {
    let cd = red_col_dim(dim);
    let rd = red_row_dim(a0, dim);
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
}

/// The number of rows of an augmented system over dim x dim symmetric matrices.
pub open spec fn red_row_dim(a0: Seq<real>, dim: int) -> int {
    rows_of(a0.len() as int, red_col_dim(dim))
}

/// The postcondition of `linear_remove_redundant_sym` (a0 the input, e its echelon form, res the result).
pub open spec fn reduced_system(a0: Seq<real>, e: Seq<real>, res: Seq<real>, dim: int) -> bool {
    exists|pivots: Seq<int>| #![trigger pivots.len()] reduced_with(a0, e, res, dim, pivots)
}

} // verus!

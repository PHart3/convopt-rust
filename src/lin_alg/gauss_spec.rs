// Spec-level support for Gaussian elimination (not trusted).
//
// `gauss_elim` reduces the column-major augmented matrix [A | b] (row_dim rows,
// col_dim columns, the last column being b) by row swaps and row subtractions.
// This module defines the solution set of the augmented system, proves that
// both row operations preserve it, and defines the echelon structure of the
// result: strictly increasing pivot columns, pivots of magnitude at least the
// tolerance threshold, and exact zeros below every pivot.  The tolerance-based
// skip rule is recorded by `skipped_bounded`: a column without a pivot was
// found below the threshold in every remaining row when it was examined, and
// each later elimination step (whose multipliers have magnitude at most 1 by
// partial pivoting) at most doubles those entries.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::seq::*;
#[cfg(verus_keep_ghost)]
use crate::real_spec::{abs_r, lemma_abs_mul, lemma_abs_sub_triangle};
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- the augmented system and its solution set ---------- */

/// Row i of [A | b] is satisfied by x: sum_{c < col_dim - 1} A_ic x_c == b_i.
pub open spec fn aug_row_ok(m: Seq<real>, row_dim: int, col_dim: int, i: int, x: Seq<real>) -> bool {
    sum_fn_r(|c: int| cm_entry(m, row_dim, i, c) * x[c], (col_dim - 1) as nat) == cm_entry(m, row_dim, i, col_dim - 1)
}

/// x solves the augmented system.
pub open spec fn solves(m: Seq<real>, row_dim: int, col_dim: int, x: Seq<real>) -> bool {
    forall|i: int| 0 <= i < row_dim ==> #[trigger] aug_row_ok(m, row_dim, col_dim, i, x)
}

/// Two augmented systems have the same solution set.
pub open spec fn same_solutions(a: Seq<real>, b: Seq<real>, row_dim: int, col_dim: int) -> bool {
    forall|x: Seq<real>| #![trigger solves(a, row_dim, col_dim, x)] #![trigger solves(b, row_dim, col_dim, x)]
        x.len() == col_dim - 1 ==> (solves(a, row_dim, col_dim, x) <==> solves(b, row_dim, col_dim, x))
}

pub proof fn lemma_same_solutions_refl(a: Seq<real>, row_dim: int, col_dim: int)
    ensures
        same_solutions(a, a, row_dim, col_dim),
{
}

pub proof fn lemma_same_solutions_trans(a: Seq<real>, b: Seq<real>, c: Seq<real>, row_dim: int, col_dim: int)
    requires
        same_solutions(a, b, row_dim, col_dim),
        same_solutions(b, c, row_dim, col_dim),
    ensures
        same_solutions(a, c, row_dim, col_dim),
{
    assert forall|x: Seq<real>| #![trigger solves(a, row_dim, col_dim, x)] x.len() == col_dim - 1
        implies (solves(a, row_dim, col_dim, x) <==> solves(c, row_dim, col_dim, x)) by {
        assert(solves(a, row_dim, col_dim, x) <==> solves(b, row_dim, col_dim, x));
        assert(solves(b, row_dim, col_dim, x) <==> solves(c, row_dim, col_dim, x));
    }
}

/* ---------- row operations ---------- */

/// m2 is m with rows r1 and r2 exchanged.
pub open spec fn rows_swapped(m: Seq<real>, m2: Seq<real>, row_dim: int, col_dim: int, r1: int, r2: int) -> bool {
    &&& m2.len() == m.len()
    &&& forall|i: int, c: int| 0 <= i < row_dim && 0 <= c < col_dim
            ==> #[trigger] cm_entry(m2, row_dim, i, c)
                == cm_entry(m, row_dim, if i == r1 { r2 } else if i == r2 { r1 } else { i }, c)
}

/// m2 is m with row j replaced by row j - piv * row p.
pub open spec fn row_subtracted(m: Seq<real>, m2: Seq<real>, row_dim: int, col_dim: int, j: int, p: int, piv: real) -> bool {
    &&& m2.len() == m.len()
    &&& forall|i: int, c: int| 0 <= i < row_dim && 0 <= c < col_dim
            ==> #[trigger] cm_entry(m2, row_dim, i, c)
                == if i == j { cm_entry(m, row_dim, j, c) - piv * cm_entry(m, row_dim, p, c) } else { cm_entry(m, row_dim, i, c) }
}

/* ---------- zero columns ---------- */

/// Column c of m is zero.
pub open spec fn col_zero(m: Seq<real>, row_dim: int, c: int) -> bool {
    forall|i: int| 0 <= i < row_dim ==> #[trigger] cm_entry(m, row_dim, i, c) == 0.0real
}

/// Every zero column of a is a zero column of b (the row operations never fill a zero column).
pub open spec fn zero_cols_kept(a: Seq<real>, b: Seq<real>, row_dim: int, col_dim: int) -> bool {
    forall|c: int| 0 <= c < col_dim && col_zero(a, row_dim, c) ==> #[trigger] col_zero(b, row_dim, c)
}

pub proof fn lemma_zero_cols_refl(a: Seq<real>, row_dim: int, col_dim: int)
    ensures
        zero_cols_kept(a, a, row_dim, col_dim),
{
}

pub proof fn lemma_zero_cols_trans(a: Seq<real>, b: Seq<real>, c: Seq<real>, row_dim: int, col_dim: int)
    requires
        zero_cols_kept(a, b, row_dim, col_dim),
        zero_cols_kept(b, c, row_dim, col_dim),
    ensures
        zero_cols_kept(a, c, row_dim, col_dim),
{
    assert forall|k: int| 0 <= k < col_dim && col_zero(a, row_dim, k) implies #[trigger] col_zero(c, row_dim, k) by {
        assert(col_zero(b, row_dim, k));
    }
}

/// Exchanging two rows keeps zero columns.
pub proof fn lemma_swap_zero_cols(m: Seq<real>, m2: Seq<real>, row_dim: int, col_dim: int, r1: int, r2: int)
    requires
        0 <= r1 < row_dim,
        0 <= r2 < row_dim,
        rows_swapped(m, m2, row_dim, col_dim, r1, r2),
    ensures
        zero_cols_kept(m, m2, row_dim, col_dim),
{
    assert forall|c: int| 0 <= c < col_dim && col_zero(m, row_dim, c) implies #[trigger] col_zero(m2, row_dim, c) by {
        assert forall|i: int| 0 <= i < row_dim implies #[trigger] cm_entry(m2, row_dim, i, c) == 0.0real by {
            let src = if i == r1 { r2 } else if i == r2 { r1 } else { i };
            assert(cm_entry(m2, row_dim, i, c) == cm_entry(m, row_dim, src, c));
            assert(cm_entry(m, row_dim, src, c) == 0.0real);
        }
    }
}

/// Subtracting a multiple of a row keeps zero columns.
pub proof fn lemma_row_sub_zero_cols(m: Seq<real>, m2: Seq<real>, row_dim: int, col_dim: int, j: int, p: int, piv: real)
    requires
        0 <= j < row_dim,
        0 <= p < row_dim,
        row_subtracted(m, m2, row_dim, col_dim, j, p, piv),
    ensures
        zero_cols_kept(m, m2, row_dim, col_dim),
{
    assert forall|c: int| 0 <= c < col_dim && col_zero(m, row_dim, c) implies #[trigger] col_zero(m2, row_dim, c) by {
        assert forall|i: int| 0 <= i < row_dim implies #[trigger] cm_entry(m2, row_dim, i, c) == 0.0real by {
            assert(cm_entry(m, row_dim, i, c) == 0.0real);
            assert(cm_entry(m, row_dim, p, c) == 0.0real);
            if i == j {
                assert(cm_entry(m2, row_dim, i, c) == 0.0real - piv * 0.0real);
            }
        }
    }
}

/// Exchanging two rows preserves the solution set.
pub proof fn lemma_swap_same_solutions(m: Seq<real>, m2: Seq<real>, row_dim: int, col_dim: int, r1: int, r2: int)
    requires
        col_dim >= 1,
        0 <= r1 < row_dim,
        0 <= r2 < row_dim,
        rows_swapped(m, m2, row_dim, col_dim, r1, r2),
    ensures
        same_solutions(m, m2, row_dim, col_dim),
{
    let n = (col_dim - 1) as nat;
    // row i of m2 is row s(i) of m, so aug_row_ok(m2, i) == aug_row_ok(m, s(i))
    assert forall|i: int, x: Seq<real>| 0 <= i < row_dim && x.len() == col_dim - 1
        implies #[trigger] aug_row_ok(m2, row_dim, col_dim, i, x)
            == aug_row_ok(m, row_dim, col_dim, (if i == r1 { r2 } else if i == r2 { r1 } else { i }), x) by {
        let src = if i == r1 { r2 } else if i == r2 { r1 } else { i };
        let f = |c: int| cm_entry(m2, row_dim, i, c) * x[c];
        let g = |c: int| cm_entry(m, row_dim, src, c) * x[c];
        assert forall|c: int| 0 <= c < n implies #[trigger] f(c) == g(c) by {}
        lemma_sum_fn_ext(f, g, n);
    }
    assert forall|x: Seq<real>| #![trigger solves(m, row_dim, col_dim, x)] x.len() == col_dim - 1
        implies (solves(m, row_dim, col_dim, x) <==> solves(m2, row_dim, col_dim, x)) by {
        if solves(m, row_dim, col_dim, x) {
            assert forall|i: int| 0 <= i < row_dim implies #[trigger] aug_row_ok(m2, row_dim, col_dim, i, x) by {
                let src = if i == r1 { r2 } else if i == r2 { r1 } else { i };
                assert(aug_row_ok(m, row_dim, col_dim, src, x));
            }
        }
        if solves(m2, row_dim, col_dim, x) {
            assert forall|i: int| 0 <= i < row_dim implies #[trigger] aug_row_ok(m, row_dim, col_dim, i, x) by {
                let src = if i == r1 { r2 } else if i == r2 { r1 } else { i };
                assert(aug_row_ok(m2, row_dim, col_dim, src, x));
            }
        }
    }
}

/// Row j - piv * row p against x equals (row j against x) - piv * (row p against x).
proof fn lemma_row_sub_sum(m: Seq<real>, m2: Seq<real>, row_dim: int, col_dim: int, j: int, p: int, piv: real, x: Seq<real>)
    requires
        col_dim >= 1,
        0 <= j < row_dim,
        0 <= p < row_dim,
        x.len() == col_dim - 1,
        row_subtracted(m, m2, row_dim, col_dim, j, p, piv),
    ensures
        sum_fn_r(|c: int| cm_entry(m2, row_dim, j, c) * x[c], (col_dim - 1) as nat)
            == sum_fn_r(|c: int| cm_entry(m, row_dim, j, c) * x[c], (col_dim - 1) as nat)
               - piv * sum_fn_r(|c: int| cm_entry(m, row_dim, p, c) * x[c], (col_dim - 1) as nat),
{
    let n = (col_dim - 1) as nat;
    let h = |c: int| cm_entry(m2, row_dim, j, c) * x[c];
    let f = |c: int| cm_entry(m, row_dim, j, c) * x[c];
    let fp = |c: int| cm_entry(m, row_dim, p, c) * x[c];
    let g = |c: int| (-piv) * fp(c);
    let fg = |c: int| f(c) + g(c);
    assert forall|c: int| 0 <= c < n implies #[trigger] h(c) == fg(c) by {
        let (u, v, w) = (cm_entry(m, row_dim, j, c), cm_entry(m, row_dim, p, c), x[c]);
        assert(h(c) == (u - piv * v) * w);
        assert((u - piv * v) * w == u * w + (-piv) * (v * w)) by (nonlinear_arith);
    }
    lemma_sum_fn_ext(h, fg, n);
    lemma_sum_fn_scale(fp, g, -piv, n);
    lemma_sum_fn_add(f, g, fg, n);
    let sp = sum_fn_r(fp, n);
    assert((-piv) * sp == -(piv * sp)) by (nonlinear_arith);
}

/// Subtracting a multiple of another row preserves the solution set.
pub proof fn lemma_row_sub_same_solutions(m: Seq<real>, m2: Seq<real>, row_dim: int, col_dim: int, j: int, p: int, piv: real)
    requires
        col_dim >= 1,
        0 <= j < row_dim,
        0 <= p < row_dim,
        j != p,
        row_subtracted(m, m2, row_dim, col_dim, j, p, piv),
    ensures
        same_solutions(m, m2, row_dim, col_dim),
{
    assert forall|x: Seq<real>| #![trigger solves(m, row_dim, col_dim, x)] x.len() == col_dim - 1
        implies (solves(m, row_dim, col_dim, x) <==> solves(m2, row_dim, col_dim, x)) by {
        lemma_row_sub_sum(m, m2, row_dim, col_dim, j, p, piv, x);
        assert(cm_entry(m2, row_dim, j, col_dim - 1)
            == cm_entry(m, row_dim, j, col_dim - 1) - piv * cm_entry(m, row_dim, p, col_dim - 1));
        if solves(m, row_dim, col_dim, x) {
            assert forall|i: int| 0 <= i < row_dim implies #[trigger] aug_row_ok(m2, row_dim, col_dim, i, x) by {
                assert(aug_row_ok(m, row_dim, col_dim, i, x));
                assert(aug_row_ok(m, row_dim, col_dim, p, x));
                if i != j {
                    let f = |c: int| cm_entry(m2, row_dim, i, c) * x[c];
                    let g = |c: int| cm_entry(m, row_dim, i, c) * x[c];
                    assert forall|c: int| 0 <= c < col_dim - 1 implies #[trigger] f(c) == g(c) by {}
                    lemma_sum_fn_ext(f, g, (col_dim - 1) as nat);
                }
            }
        }
        if solves(m2, row_dim, col_dim, x) {
            assert forall|i: int| 0 <= i < row_dim implies #[trigger] aug_row_ok(m, row_dim, col_dim, i, x) by {
                assert(aug_row_ok(m2, row_dim, col_dim, i, x));
                assert(aug_row_ok(m2, row_dim, col_dim, p, x));
                if i != j {
                    let f = |c: int| cm_entry(m2, row_dim, i, c) * x[c];
                    let g = |c: int| cm_entry(m, row_dim, i, c) * x[c];
                    assert forall|c: int| 0 <= c < col_dim - 1 implies #[trigger] f(c) == g(c) by {}
                    lemma_sum_fn_ext(f, g, (col_dim - 1) as nat);
                } else {
                    // row p is unchanged, so its sum is the same in m and m2
                    let f = |c: int| cm_entry(m2, row_dim, p, c) * x[c];
                    let g = |c: int| cm_entry(m, row_dim, p, c) * x[c];
                    assert forall|c: int| 0 <= c < col_dim - 1 implies #[trigger] f(c) == g(c) by {}
                    lemma_sum_fn_ext(f, g, (col_dim - 1) as nat);
                }
            }
        }
    }
}

/* ---------- echelon structure ---------- */

/// The pivot columns are strictly increasing and lie in [0, col_dim).
pub open spec fn pivots_ok(pivots: Seq<int>, col_dim: int) -> bool {
    &&& forall|i: int| 0 <= i < pivots.len() ==> 0 <= #[trigger] pivots[i] < col_dim
    &&& forall|i: int, l: int| 0 <= i < l < pivots.len() ==> #[trigger] pivots[i] < #[trigger] pivots[l]
}

/// Echelon structure of m: row i has its pivot in column pivots[i], of magnitude at least
/// eps, and the pivot column is exactly zero in every row below i.
pub open spec fn echelon_form(m: Seq<real>, row_dim: int, col_dim: int, pivots: Seq<int>, eps: real) -> bool {
    &&& pivots.len() <= row_dim
    &&& pivots_ok(pivots, col_dim)
    &&& forall|i: int| 0 <= i < pivots.len() ==> abs_r(cm_entry(m, row_dim, i, #[trigger] pivots[i])) >= eps
    &&& forall|i: int, j: int| 0 <= i < pivots.len() && i < j < row_dim
            ==> #[trigger] cm_entry(m, row_dim, j, pivots[i]) == 0.0real
}

/// The rows of pivots are not touched by later operations on rows >= pivots.len():
/// echelon structure is preserved by a swap of two such rows.
pub proof fn lemma_echelon_swap(m: Seq<real>, m2: Seq<real>, row_dim: int, col_dim: int, pivots: Seq<int>, eps: real, r1: int, r2: int)
    requires
        pivots.len() <= r1 < row_dim,
        pivots.len() <= r2 < row_dim,
        rows_swapped(m, m2, row_dim, col_dim, r1, r2),
        echelon_form(m, row_dim, col_dim, pivots, eps),
    ensures
        echelon_form(m2, row_dim, col_dim, pivots, eps),
{
    assert forall|i: int| 0 <= i < pivots.len() implies abs_r(cm_entry(m2, row_dim, i, #[trigger] pivots[i])) >= eps by {
        assert(cm_entry(m2, row_dim, i, pivots[i]) == cm_entry(m, row_dim, i, pivots[i]));
    }
    assert forall|i: int, j: int| 0 <= i < pivots.len() && i < j < row_dim
        implies #[trigger] cm_entry(m2, row_dim, j, pivots[i]) == 0.0real by {
        let src = if j == r1 { r2 } else if j == r2 { r1 } else { j };
        assert(cm_entry(m, row_dim, src, pivots[i]) == 0.0real);
    }
}

/// ... and by subtracting a multiple of such a row from another.
pub proof fn lemma_echelon_row_sub(m: Seq<real>, m2: Seq<real>, row_dim: int, col_dim: int, pivots: Seq<int>, eps: real, j: int, p: int, piv: real)
    requires
        pivots.len() <= j < row_dim,
        pivots.len() <= p < row_dim,
        row_subtracted(m, m2, row_dim, col_dim, j, p, piv),
        echelon_form(m, row_dim, col_dim, pivots, eps),
    ensures
        echelon_form(m2, row_dim, col_dim, pivots, eps),
{
    assert forall|i: int| 0 <= i < pivots.len() implies abs_r(cm_entry(m2, row_dim, i, #[trigger] pivots[i])) >= eps by {
        assert(cm_entry(m2, row_dim, i, pivots[i]) == cm_entry(m, row_dim, i, pivots[i]));
    }
    assert forall|i: int, l: int| 0 <= i < pivots.len() && i < l < row_dim
        implies #[trigger] cm_entry(m2, row_dim, l, pivots[i]) == 0.0real by {
        assert(cm_entry(m, row_dim, l, pivots[i]) == 0.0real);
        assert(cm_entry(m, row_dim, p, pivots[i]) == 0.0real);
        assert(piv * 0.0real == 0.0real);
    }
}

/// Appending a new pivot: column n (greater than every pivot column) has magnitude at
/// least eps in row pivots.len() and is zero below it.
pub proof fn lemma_echelon_push(m: Seq<real>, row_dim: int, col_dim: int, pivots: Seq<int>, eps: real, n: int)
    requires
        pivots.len() < row_dim,
        0 <= n < col_dim,
        forall|i: int| 0 <= i < pivots.len() ==> #[trigger] pivots[i] < n,
        echelon_form(m, row_dim, col_dim, pivots, eps),
        abs_r(cm_entry(m, row_dim, pivots.len() as int, n)) >= eps,
        forall|j: int| pivots.len() < j < row_dim ==> #[trigger] cm_entry(m, row_dim, j, n) == 0.0real,
    ensures
        echelon_form(m, row_dim, col_dim, pivots.push(n), eps),
{
    let p2 = pivots.push(n);
    assert forall|i: int| 0 <= i < p2.len() implies 0 <= #[trigger] p2[i] < col_dim by {
        if i < pivots.len() { assert(p2[i] == pivots[i]); }
    }
    assert forall|i: int, l: int| 0 <= i < l < p2.len() implies #[trigger] p2[i] < #[trigger] p2[l] by {
        assert(p2[i] == pivots[i]);
        if l < pivots.len() { assert(p2[l] == pivots[l]); }
    }
    assert forall|i: int| 0 <= i < p2.len() implies abs_r(cm_entry(m, row_dim, i, #[trigger] p2[i])) >= eps by {
        if i < pivots.len() { assert(p2[i] == pivots[i]); }
    }
    assert forall|i: int, j: int| 0 <= i < p2.len() && i < j < row_dim
        implies #[trigger] cm_entry(m, row_dim, j, p2[i]) == 0.0real by {
        if i < pivots.len() { assert(p2[i] == pivots[i]); }
    }
}

/* ---------- the tolerance-based skip rule ---------- */

/// 2^k.
pub open spec fn pow2_r(k: nat) -> real
    decreases k,
{
    if k == 0 { 1.0real } else { 2.0real * pow2_r((k - 1) as nat) }
}

pub proof fn lemma_pow2_pos(k: nat)
    ensures
        pow2_r(k) > 0.0real,
    decreases k,
{
    if k > 0 {
        lemma_pow2_pos((k - 1) as nat);
    }
}

pub proof fn lemma_pow2_step(k: nat, e: real)
    ensures
        pow2_r(k + 1) * e == pow2_r(k) * e + pow2_r(k) * e,
{
    let p = pow2_r(k);
    assert(pow2_r(k + 1) == 2.0real * p);
    assert((2.0real * p) * e == p * e + p * e) by (nonlinear_arith);
}

/// Number of pivot columns strictly before column c: the row index at which column c was
/// examined by the elimination.
pub open spec fn pivots_below(pivots: Seq<int>, c: int) -> int
    decreases pivots.len(),
{
    if pivots.len() == 0 { 0 } else { pivots_below(pivots.drop_last(), c) + (if pivots.last() < c { 1int } else { 0int }) }
}

pub proof fn lemma_pivots_below_bounds(pivots: Seq<int>, c: int)
    ensures
        0 <= pivots_below(pivots, c) <= pivots.len(),
    decreases pivots.len(),
{
    if pivots.len() > 0 {
        lemma_pivots_below_bounds(pivots.drop_last(), c);
    }
}

pub proof fn lemma_pivots_below_push(pivots: Seq<int>, c: int, n: int)
    ensures
        pivots_below(pivots.push(n), c) == pivots_below(pivots, c) + (if n < c { 1int } else { 0int }),
{
    assert(pivots.push(n).drop_last() =~= pivots);
}

/// When every pivot column is before c, column c was examined at row pivots.len().
pub proof fn lemma_pivots_below_all(pivots: Seq<int>, c: int)
    requires
        forall|i: int| 0 <= i < pivots.len() ==> #[trigger] pivots[i] < c,
    ensures
        pivots_below(pivots, c) == pivots.len(),
    decreases pivots.len(),
{
    if pivots.len() > 0 {
        lemma_pivots_below_all(pivots.drop_last(), c);
    }
}

/// Column c holds a pivot.
pub open spec fn is_pivot_col(pivots: Seq<int>, c: int) -> bool {
    exists|i: int| 0 <= i < pivots.len() && #[trigger] pivots[i] == c
}

pub proof fn lemma_is_pivot_col_push(pivots: Seq<int>, c: int, n: int)
    requires
        c != n,
    ensures
        is_pivot_col(pivots.push(n), c) == is_pivot_col(pivots, c),
{
    let p2 = pivots.push(n);
    if is_pivot_col(pivots, c) {
        let i = choose|i: int| 0 <= i < pivots.len() && #[trigger] pivots[i] == c;
        assert(p2[i] == c);
    }
    if is_pivot_col(p2, c) {
        let i = choose|i: int| 0 <= i < p2.len() && #[trigger] p2[i] == c;
        assert(pivots[i] == c);
    }
}

/// The bound on the entries of a skipped column c after all later elimination steps:
/// 2^(number of pivots found after column c was skipped) * eps.
pub open spec fn skip_bound(pivots: Seq<int>, c: int, eps: real) -> real {
    pow2_r((pivots.len() - pivots_below(pivots, c)) as nat) * eps
}

/// The tolerance-based skip rule for the columns before ncols: a column without a pivot was
/// below eps in every remaining row when it was examined (row pivots_below(c) onwards), and
/// each later elimination step, whose multipliers have magnitude at most 1 by partial
/// pivoting, at most doubled those entries.
pub open spec fn skipped_bounded(m: Seq<real>, row_dim: int, col_dim: int, pivots: Seq<int>, eps: real, ncols: int) -> bool {
    forall|c: int, j: int| 0 <= c < ncols && !is_pivot_col(pivots, c) && pivots_below(pivots, c) <= j < row_dim
        ==> abs_r(#[trigger] cm_entry(m, row_dim, j, c)) < skip_bound(pivots, c, eps)
}

/// Swapping two rows at or below pivots.len() preserves the skip bounds.
pub proof fn lemma_skipped_swap(m: Seq<real>, m2: Seq<real>, row_dim: int, col_dim: int, pivots: Seq<int>, eps: real, ncols: int, r1: int, r2: int)
    requires
        pivots.len() <= r1 < row_dim,
        pivots.len() <= r2 < row_dim,
        rows_swapped(m, m2, row_dim, col_dim, r1, r2),
        ncols <= col_dim,
        skipped_bounded(m, row_dim, col_dim, pivots, eps, ncols),
    ensures
        skipped_bounded(m2, row_dim, col_dim, pivots, eps, ncols),
{
    assert forall|c: int, j: int| 0 <= c < ncols && !is_pivot_col(pivots, c) && pivots_below(pivots, c) <= j < row_dim
        implies abs_r(#[trigger] cm_entry(m2, row_dim, j, c)) < skip_bound(pivots, c, eps) by {
        lemma_pivots_below_bounds(pivots, c);
        let src = if j == r1 { r2 } else if j == r2 { r1 } else { j };
        assert(cm_entry(m2, row_dim, j, c) == cm_entry(m, row_dim, src, c));
        assert(abs_r(cm_entry(m, row_dim, src, c)) < skip_bound(pivots, c, eps));
    }
}

/// Skipping column n (all pivots are before n, every remaining entry of column n is below eps).
pub proof fn lemma_skipped_skip(m: Seq<real>, row_dim: int, col_dim: int, pivots: Seq<int>, eps: real, n: int)
    requires
        forall|i: int| 0 <= i < pivots.len() ==> #[trigger] pivots[i] < n,
        forall|j: int| pivots.len() <= j < row_dim ==> abs_r(#[trigger] cm_entry(m, row_dim, j, n)) < eps,
        skipped_bounded(m, row_dim, col_dim, pivots, eps, n),
    ensures
        skipped_bounded(m, row_dim, col_dim, pivots, eps, n + 1),
{
    lemma_pivots_below_all(pivots, n);
    assert(!is_pivot_col(pivots, n));
    assert(skip_bound(pivots, n, eps) == eps) by {
        assert(pow2_r(0) == 1.0real);
    }
    assert forall|c: int, j: int| 0 <= c < n + 1 && !is_pivot_col(pivots, c) && pivots_below(pivots, c) <= j < row_dim
        implies abs_r(#[trigger] cm_entry(m, row_dim, j, c)) < skip_bound(pivots, c, eps) by {
        if c < n {
            assert(abs_r(cm_entry(m, row_dim, j, c)) < skip_bound(pivots, c, eps));
        }
    }
}

/// One row of a skipped column after subtracting a multiple of magnitude at most 1 of
/// another bounded row: the bound doubles.
pub proof fn lemma_abs_sub_double(a: real, b: real, piv: real, bound: real)
    requires
        abs_r(a) < bound,
        abs_r(b) < bound,
        abs_r(piv) <= 1.0real,
    ensures
        abs_r(a - piv * b) < bound + bound,
{
    lemma_abs_sub_triangle(a, piv * b);
    lemma_abs_mul(piv, b);
    assert(abs_r(piv) * abs_r(b) <= abs_r(b)) by (nonlinear_arith)
        requires abs_r(piv) <= 1.0real, abs_r(b) >= 0.0real;
}

/// After the elimination step of pivot column n (every pivot before n), with every skipped
/// entry below twice its bound, the bounds with the new pivot hold for the columns up to n.
pub proof fn lemma_skipped_after_pivot(m: Seq<real>, row_dim: int, col_dim: int, pivots: Seq<int>, eps: real, n: int)
    requires
        forall|i: int| 0 <= i < pivots.len() ==> #[trigger] pivots[i] < n,
        forall|c: int, j: int| 0 <= c < n && !is_pivot_col(pivots, c) && pivots_below(pivots, c) <= j < row_dim
            ==> abs_r(#[trigger] cm_entry(m, row_dim, j, c)) < skip_bound(pivots, c, eps) + skip_bound(pivots, c, eps),
    ensures
        skipped_bounded(m, row_dim, col_dim, pivots.push(n), eps, n + 1),
{
    let p2 = pivots.push(n);
    assert(is_pivot_col(p2, n)) by {
        assert(p2[pivots.len() as int] == n);
    }
    assert forall|c: int, j: int| 0 <= c < n + 1 && !is_pivot_col(p2, c) && pivots_below(p2, c) <= j < row_dim
        implies abs_r(#[trigger] cm_entry(m, row_dim, j, c)) < skip_bound(p2, c, eps) by {
        assert(c != n);
        lemma_is_pivot_col_push(pivots, c, n);
        lemma_pivots_below_push(pivots, c, n);
        lemma_pivots_below_bounds(pivots, c);
        let k = (pivots.len() - pivots_below(pivots, c)) as nat;
        assert(p2.len() - pivots_below(p2, c) == k + 1);
        lemma_pow2_step(k, eps);
        assert(abs_r(cm_entry(m, row_dim, j, c)) < skip_bound(pivots, c, eps) + skip_bound(pivots, c, eps));
    }
}

/// Columns never examined (the rows ran out: pivots.len() == row_dim) satisfy the bounds vacuously.
pub proof fn lemma_skipped_extend(m: Seq<real>, row_dim: int, col_dim: int, pivots: Seq<int>, eps: real, n: int)
    requires
        pivots.len() == row_dim,
        forall|i: int| 0 <= i < pivots.len() ==> #[trigger] pivots[i] < n,
        skipped_bounded(m, row_dim, col_dim, pivots, eps, n),
    ensures
        skipped_bounded(m, row_dim, col_dim, pivots, eps, col_dim),
{
    assert forall|c: int, j: int| 0 <= c < col_dim && !is_pivot_col(pivots, c) && pivots_below(pivots, c) <= j < row_dim
        implies abs_r(#[trigger] cm_entry(m, row_dim, j, c)) < skip_bound(pivots, c, eps) by {
        if c >= n {
            lemma_pivots_below_all(pivots, c);
        } else {
            assert(abs_r(cm_entry(m, row_dim, j, c)) < skip_bound(pivots, c, eps));
        }
    }
}

} // verus!

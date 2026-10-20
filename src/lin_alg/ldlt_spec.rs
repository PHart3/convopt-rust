// Spec-level support for the LDLT factorization and the positive-definite solvers
// (not trusted).
//
// `ldlt_decomp` computes a unit lower-triangular L (packed, vec_spec.rs) and a
// pivot vector d with L D L^T = A by the recurrences
//
//     L_rc = (A_rc - sum_{k < c} d_k L_rk L_ck) / d_c        (c < r),
//     d_r  =  A_rr - sum_{k < r} d_k L_rk L_rk.
//
// Both are instances of one entry predicate, `ldlt_entry_ok`, stated with the
// partial weighted row product `ldlt_sum_r`.  The lemmas below give the sum's
// step and locality (which entries of L and d it reads), the preservation of
// the predicate under the writes performed by the algorithm, and the final
// identity L D L^T = A over the views of mat_spec.rs.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::seq::*;
#[cfg(verus_keep_ghost)]
use crate::real_spec::{abs_r, max_r};
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::*;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- the LDLT recurrence ---------- */

/// sum_{k < n} d_k L_ik L_jk: the partial product of rows i and j of L weighted by d.
pub open spec fn ldlt_sum_r(l: Seq<real>, d: Seq<real>, dim: int, i: int, j: int, n: nat) -> real {
    sum_fn_r(|k: int| d[k] * lt_entry(l, dim, i, k) * lt_entry(l, dim, j, k), n)
}

/// The recurrence for entry (r, c), c <= r: d_c L_rc + sum_{k < c} d_k L_rk L_ck = A_rc.
/// With L_cc = 1 this is entry (r, c) of L D L^T (lemma_ldlt_product_entry).
pub open spec fn ldlt_entry_ok(a: Seq<real>, l: Seq<real>, d: Seq<real>, dim: int, r: int, c: int) -> bool {
    d[c] * lt_entry(l, dim, r, c) + ldlt_sum_r(l, d, dim, r, c, c as nat) == sym_entry(a, r, c)
}

/// L is unit lower-triangular and every entry of the triangle satisfies the recurrence.
pub open spec fn ldlt_ok(a: Seq<real>, l: Seq<real>, d: Seq<real>, dim: int) -> bool {
    &&& forall|r: int| 0 <= r < dim ==> #[trigger] lt_entry(l, dim, r, r) == 1.0real
    &&& forall|r: int, c: int| 0 <= c <= r < dim ==> #[trigger] ldlt_entry_ok(a, l, d, dim, r, c)
}

pub proof fn lemma_ldlt_sum_step(l: Seq<real>, d: Seq<real>, dim: int, i: int, j: int, n: nat)
    ensures
        ldlt_sum_r(l, d, dim, i, j, n + 1)
            == ldlt_sum_r(l, d, dim, i, j, n) + d[n as int] * lt_entry(l, dim, i, n as int) * lt_entry(l, dim, j, n as int),
{
}

/// The sum only reads d_k and the entries (i, k), (j, k) of L for k < n.
pub proof fn lemma_ldlt_sum_ext(l: Seq<real>, l2: Seq<real>, d: Seq<real>, d2: Seq<real>, dim: int, i: int, j: int, n: nat)
    requires
        forall|k: int| 0 <= k < n ==> d[k] == d2[k],
        forall|k: int| 0 <= k < n ==> #[trigger] lt_entry(l, dim, i, k) == lt_entry(l2, dim, i, k),
        forall|k: int| 0 <= k < n ==> #[trigger] lt_entry(l, dim, j, k) == lt_entry(l2, dim, j, k),
    ensures
        ldlt_sum_r(l, d, dim, i, j, n) == ldlt_sum_r(l2, d2, dim, i, j, n),
{
    let f = |k: int| d[k] * lt_entry(l, dim, i, k) * lt_entry(l, dim, j, k);
    let g = |k: int| d2[k] * lt_entry(l2, dim, i, k) * lt_entry(l2, dim, j, k);
    assert forall|k: int| 0 <= k < n implies #[trigger] f(k) == g(k) by {}
    lemma_sum_fn_ext(f, g, n);
}

/// The sum is symmetric in i and j.
pub proof fn lemma_ldlt_sum_comm(l: Seq<real>, d: Seq<real>, dim: int, i: int, j: int, n: nat)
    ensures
        ldlt_sum_r(l, d, dim, i, j, n) == ldlt_sum_r(l, d, dim, j, i, n),
{
    let f = |k: int| d[k] * lt_entry(l, dim, i, k) * lt_entry(l, dim, j, k);
    let g = |k: int| d[k] * lt_entry(l, dim, j, k) * lt_entry(l, dim, i, k);
    assert forall|k: int| 0 <= k < n implies #[trigger] f(k) == g(k) by {
        let (u, v, w) = (d[k], lt_entry(l, dim, i, k), lt_entry(l, dim, j, k));
        assert(u * v * w == u * w * v) by (nonlinear_arith);
    }
    lemma_sum_fn_ext(f, g, n);
}

/* ---------- preservation under the algorithm's writes ---------- */

/// Writing position (i, j) of L leaves every other entry (r, c) of the triangle unchanged.
pub proof fn lemma_lt_entry_write_other(l: Seq<real>, l2: Seq<real>, dim: int, i: int, j: int, r: int, c: int)
    requires
        0 <= j <= i < dim,
        0 <= c <= r < dim,
        !(r == i && c == j),
        l2.len() == l.len(),
        l.len() >= lt_col_start(dim, dim),
        forall|t: int| 0 <= t < l.len() && t != lt_idx(dim, i, j) ==> l2[t] == l[t],
    ensures
        lt_entry(l2, dim, r, c) == lt_entry(l, dim, r, c),
{
    lemma_lt_idx_bound(dim, r, c, dim);
    if lt_idx(dim, r, c) == lt_idx(dim, i, j) {
        lemma_lt_idx_injective(dim, r, c, i, j);
    }
}

/// Writing position (i, j) of L (j < i) preserves the recurrence of an entry (r, c) that does
/// not read position (i, j): either neither row r nor row c is row i, or the sum stops before
/// column j.
pub proof fn lemma_ldlt_entry_ok_write(a: Seq<real>, l: Seq<real>, l2: Seq<real>, d: Seq<real>, dim: int, i: int, j: int, r: int, c: int)
    requires
        0 <= j < i < dim,
        0 <= c <= r < dim,
        !(r == i && c == j),
        (r != i && c != i) || c <= j,
        l2.len() == l.len(),
        l.len() >= lt_col_start(dim, dim),
        forall|t: int| 0 <= t < l.len() && t != lt_idx(dim, i, j) ==> l2[t] == l[t],
        ldlt_entry_ok(a, l, d, dim, r, c),
    ensures
        ldlt_entry_ok(a, l2, d, dim, r, c),
{
    lemma_lt_entry_write_other(l, l2, dim, i, j, r, c);
    assert forall|k: int| 0 <= k < c implies #[trigger] lt_entry(l, dim, r, k) == lt_entry(l2, dim, r, k) by {
        lemma_lt_entry_write_other(l, l2, dim, i, j, r, k);
    }
    assert forall|k: int| 0 <= k < c implies #[trigger] lt_entry(l, dim, c, k) == lt_entry(l2, dim, c, k) by {
        lemma_lt_entry_write_other(l, l2, dim, i, j, c, k);
    }
    lemma_ldlt_sum_ext(l, l2, d, d, dim, r, c, c as nat);
}

/// Appending pivots preserves the recurrence of an entry whose pivots already exist.
pub proof fn lemma_ldlt_entry_ok_push(a: Seq<real>, l: Seq<real>, d: Seq<real>, d2: Seq<real>, dim: int, r: int, c: int)
    requires
        0 <= c <= r,
        c < d.len(),
        forall|k: int| 0 <= k < d.len() ==> d2[k] == d[k],
        ldlt_entry_ok(a, l, d, dim, r, c),
    ensures
        ldlt_entry_ok(a, l, d2, dim, r, c),
{
    lemma_ldlt_sum_ext(l, l, d, d2, dim, r, c, c as nat);
}

/* ---------- L D L^T = A ---------- */

/// Entry (r, c), c <= r, of L D L^T for L with L_cc = 1: the recurrence's left-hand side.
pub proof fn lemma_ldlt_product_entry(l: Seq<real>, d: Seq<real>, dim: int, r: int, c: int)
    requires
        0 <= c <= r < dim,
        lt_entry(l, dim, c, c) == 1.0real,
    ensures
        mmul(mmul(lt_view(l, dim), diag_view(d), dim), mtr(lt_view(l, dim)), dim)(r, c)
            == d[c] * lt_entry(l, dim, r, c) + ldlt_sum_r(l, d, dim, r, c, c as nat),
{
    let q = lt_view(l, dim);
    lemma_mul_diag_transpose(q, d, dim, r, c);
    let f = |k: int| d[k] * q(r, k) * q(c, k);
    let g = |k: int| d[k] * lt_entry(l, dim, r, k) * lt_entry(l, dim, c, k);
    // the terms k > c vanish since L_ck = 0 there
    assert forall|k: int| c + 1 <= k < dim implies #[trigger] f(k) == 0.0real by {
        assert(q(c, k) == 0.0real);
        let (u, v) = (d[k], q(r, k));
        assert(u * v * 0.0real == 0.0real) by (nonlinear_arith);
    }
    lemma_sum_fn_tail_zero(f, (c + 1) as nat, dim as nat);
    // the terms k < c are those of the recurrence sum, and the term k = c is d_c L_rc
    assert forall|k: int| 0 <= k < c implies #[trigger] f(k) == g(k) by {
        assert(q(r, k) == lt_entry(l, dim, r, k));
        assert(q(c, k) == lt_entry(l, dim, c, k));
    }
    lemma_sum_fn_ext(f, g, c as nat);
    assert(q(r, c) == lt_entry(l, dim, r, c));
    assert(q(c, c) == 1.0real);
    let (u, v) = (d[c], lt_entry(l, dim, r, c));
    assert(u * v * 1.0real == u * v) by (nonlinear_arith);
}

/// L D L^T = A on the dim x dim block when L is unit lower-triangular and every entry of the
/// triangle satisfies the recurrence.
pub proof fn lemma_ldlt_product(a: Seq<real>, l: Seq<real>, d: Seq<real>, dim: int)
    requires
        ldlt_ok(a, l, d, dim),
    ensures
        meq(mmul(mmul(lt_view(l, dim), diag_view(d), dim), mtr(lt_view(l, dim)), dim), sym_view(a), dim, dim),
{
    let p = mmul(mmul(lt_view(l, dim), diag_view(d), dim), mtr(lt_view(l, dim)), dim);
    assert forall|r: int, c: int| 0 <= r < dim && 0 <= c < dim implies #[trigger] p(r, c) == sym_view(a)(r, c) by {
        if c <= r {
            lemma_ldlt_product_entry(l, d, dim, r, c);
            assert(ldlt_entry_ok(a, l, d, dim, r, c));
        } else {
            lemma_mul_diag_transpose_symmetric(lt_view(l, dim), d, dim, r, c);
            lemma_ldlt_product_entry(l, d, dim, c, r);
            assert(ldlt_entry_ok(a, l, d, dim, c, r));
            assert(sym_entry(a, r, c) == sym_entry(a, c, r));
        }
    }
}

/* ---------- the block-diagonal solver (A, I/2) ---------- */

/// If A x_head = b_head and the tail of x is twice the tail of b, then (A, I/2) x = b.
pub proof fn lemma_block_half_solve(a: MatF, n: int, total: int, x: Seq<real>, b: Seq<real>)
    requires
        0 <= n <= total,
        x.len() == total,
        b.len() == total,
        mat_vec(a, n, x.subrange(0, n)) == b.subrange(0, n),
        forall|t: int| n <= t < total ==> #[trigger] x[t] == 2.0real * b[t],
    ensures
        mat_vec(block_half_view(a, n), total, x) == b,
{
    let m = block_half_view(a, n);
    assert forall|r: int| 0 <= r < total implies #[trigger] mat_vec(m, total, x)[r] == b[r] by {
        let f = |k: int| m(r, k) * x[k];
        if r < n {
            let g = |k: int| a(r, k) * x.subrange(0, n)[k];
            assert forall|k: int| n <= k < total implies #[trigger] f(k) == 0.0real by {
                assert(m(r, k) == 0.0real);
            }
            lemma_sum_fn_tail_zero(f, n as nat, total as nat);
            assert forall|k: int| 0 <= k < n implies #[trigger] f(k) == g(k) by {
                assert(m(r, k) == a(r, k));
                assert(x.subrange(0, n)[k] == x[k]);
            }
            lemma_sum_fn_ext(f, g, n as nat);
            assert(mat_vec(a, n, x.subrange(0, n))[r] == sum_fn_r(g, n as nat));
            assert(b.subrange(0, n)[r] == b[r]);
        } else {
            assert forall|k: int| 0 <= k < total && k != r implies #[trigger] f(k) == 0.0real by {
                assert(m(r, k) == 0.0real);
            }
            lemma_sum_fn_single(f, total as nat, r);
            assert(m(r, r) == 0.5real);
        }
    }
    assert(mat_vec(m, total, x) =~= b);
}

/* ---------- the PSD block check ---------- */

/// |A_ii| of the diagonal block starting at `start` (per-term form, see block_max_abs_diag_r).
pub open spec fn block_diag_abs_r(a: Seq<real>, start: int, i: int) -> real {
    abs_r(sym_entry(a, start + i, start + i))
}

/// max(max_{i < dim} |A_{start+i, start+i}|, 0): the diagonal scale of the block.
pub open spec fn block_max_abs_diag_r(a: Seq<real>, start: int, dim: int) -> real {
    max0_fn_r(|i: int| block_diag_abs_r(a, start, i), dim as nat)
}

/// The shift eps = 1e-9 * max(max_i |B_ii|, 1) used by psd_block_check on the block B.
pub open spec fn psd_check_eps_r(a: Seq<real>, start: int, dim: int) -> real {
    0.000000001real * max_r(block_max_abs_diag_r(a, start, dim), 1.0real)
}

/// The guarantee of a successful check on one block: B >= -eps I.
pub open spec fn block_psd_ok(a: Seq<real>, start: int, dim: int) -> bool {
    psd_shifted(block_view(a, start), dim, psd_check_eps_r(a, start, dim))
}

/// Every listed block (start, dim) lies inside a packed matrix of length len whose
/// index arithmetic fits in usize.
pub open spec fn blocks_fit(blocks: Seq<(usize, usize)>, len: nat) -> bool {
    forall|b: int| 0 <= b < blocks.len() ==> {
        &&& dim_fits((#[trigger] blocks[b]).0 + blocks[b].1)
        &&& len >= pk_len(blocks[b].0 + blocks[b].1)
    }
}

/// The diagonal scale of an extracted block equals that of the block inside the full matrix.
pub proof fn lemma_block_max_abs_diag(b: Seq<real>, a: Seq<real>, start: int, dim: int)
    requires
        dim >= 0,
        forall|i: int| 0 <= i < dim ==> #[trigger] sym_diag(b, i) == sym_entry(a, start + i, start + i),
    ensures
        max_abs_diag_r(b, dim) == block_max_abs_diag_r(a, start, dim),
{
    let f = |i: int| abs_r(sym_diag(b, i));
    let g = |i: int| block_diag_abs_r(a, start, i);
    assert forall|i: int| 0 <= i < dim implies #[trigger] f(i) == g(i) by {}
    lemma_max0_fn_ext(f, g, dim as nat);
}

} // verus!

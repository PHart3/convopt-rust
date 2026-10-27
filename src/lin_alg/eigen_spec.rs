// Common eigensolver specification (not trusted).
//
// Both Jacobi eigensolvers (jacobi.rs) iterate the verified rotation
// `jacobi_rot` on a packed symmetric matrix B (initially A) while
// accumulating the rotations into a column-major matrix Q (initially I),
// and stop once every off-diagonal entry of B is below the threshold
//
//     thr = tol * max(1, max_i |B_ii|)
//
// in magnitude, for a nonnegative relative tolerance tol (JACOBI_TOL in jacobi.rs). This module defines the invariant they maintain,
//
//     A = Q B Q^T with Q orthogonal            (`eigen_decomp_inv`),
//
// proves that Q = I, B = A establishes it, that every `jacobi_step` preserves
// it together with the eigenvalues, and that the stopping criterion yields the
// approximate eigendecomposition
//
//     ||B - diag(B)||_F <= sqrt(dim (dim - 1)) * thr    (`eigen_decomp_approx`),
//
// which is the whole contract of either eigensolver (`eigensolver_ok`).
// Nothing here claims that B is exactly diagonal.

use vstd::prelude::*;
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
#[cfg(verus_keep_ghost)]
use crate::lin_alg::jacobi_spec::*;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- the invariant of the rotation sequence ---------- */

/// A = Q B Q^T on the dim x dim block with Q orthogonal, for the stored
/// representations (A, B packed symmetric; Q column-major).
pub open spec fn eigen_decomp_inv(a: Seq<real>, q: Seq<real>, b: Seq<real>, dim: int) -> bool {
    &&& is_orthogonal(cm_view(q, dim), dim)
    &&& meq(sym_view(a), mmul(mmul(cm_view(q, dim), sym_view(b), dim), mtr(cm_view(q, dim)), dim), dim, dim)
}

/// Q = I and B = A establish the invariant.
pub proof fn lemma_eigen_inv_init(a: Seq<real>, q: Seq<real>, dim: int)
    requires
        dim >= 0,
        is_ident_r(q, dim),
    ensures
        eigen_decomp_inv(a, q, a, dim),
{
    let (qv, av) = (cm_view(q, dim), sym_view(a));
    lemma_ident_views(q, dim);
    lemma_orthogonal_ident(dim);
    lemma_meq_sym(qv, mident(), dim, dim);
    lemma_orthogonal_meq(mident(), qv, dim);
    // Q A Q^T = I A I = A
    lemma_transpose_meq(qv, mident(), dim, dim);
    lemma_transpose_ident_meq(dim, dim);
    lemma_meq_trans(mtr(qv), mtr(mident()), mident(), dim, dim);
    lemma_mul_meq(qv, mident(), av, av, dim, dim, dim);
    lemma_mul_ident_left_meq(av, dim, dim);
    lemma_meq_trans(mmul(qv, av, dim), mmul(mident(), av, dim), av, dim, dim);
    lemma_mul_meq(mmul(qv, av, dim), av, mtr(qv), mident(), dim, dim, dim);
    lemma_mul_ident_right_meq(av, dim, dim);
    lemma_meq_trans(mmul(mmul(qv, av, dim), mtr(qv), dim), mmul(av, mident(), dim), av, dim, dim);
    lemma_meq_sym(mmul(mmul(qv, av, dim), mtr(qv), dim), av, dim, dim);
}

/// One call of `jacobi_rot` at pivot (j, k), which produces (B', Q') with
/// `jacobi_step(B, Q, B', Q')` for some c, s, preserves the invariant.
pub proof fn lemma_eigen_inv_step(a: Seq<real>, q: Seq<real>, b: Seq<real>, q2: Seq<real>, b2: Seq<real>, dim: int, j: int, k: int)
    requires
        0 <= j < k < dim,
        eigen_decomp_inv(a, q, b, dim),
        exists|c: real, s: real| jacobi_step(b, q, b2, q2, dim, j, k, c, s),
    ensures
        eigen_decomp_inv(a, q2, b2, dim),
{
    let (c, s) = choose|c: real, s: real| jacobi_step(b, q, b2, q2, dim, j, k, c, s);
    lemma_jacobi_step_preserves(sym_view(a), b, q, b2, q2, dim, j, k, c, s);
}

/// Under the invariant, A and B have the same eigenvalues (B = Q^T A Q).
pub proof fn lemma_eigen_inv_eigenvalues(a: Seq<real>, q: Seq<real>, b: Seq<real>, dim: int, lambda: real)
    requires
        dim >= 0,
        eigen_decomp_inv(a, q, b, dim),
    ensures
        is_eigenvalue(sym_view(a), dim, lambda) <==> is_eigenvalue(sym_view(b), dim, lambda),
{
    let (qv, av, bv) = (cm_view(q, dim), sym_view(a), sym_view(b));
    let qt = mtr(qv);
    // the similarity lemma for the orthogonal matrix Q^T: eig(B) <==> eig(Q^T^T B Q^T)
    lemma_orthogonal_transpose(qv, dim);
    lemma_orthogonal_similarity_eigenvalues(bv, qt, dim, lambda);
    let sim = mmul(mmul(mtr(qt), bv, dim), qt, dim);
    lemma_transpose_twice_meq(qv, dim, dim);
    lemma_mul_meq(mtr(qt), qv, bv, bv, dim, dim, dim);
    lemma_mul_meq(mmul(mtr(qt), bv, dim), mmul(qv, bv, dim), qt, qt, dim, dim, dim);
    lemma_meq_sym(av, mmul(mmul(qv, bv, dim), qt, dim), dim, dim);
    lemma_meq_trans(sim, mmul(mmul(qv, bv, dim), qt, dim), av, dim, dim);
    if is_eigenvalue(sim, dim, lambda) {
        lemma_eigenvalue_meq(sim, av, dim, lambda);
    }
    if is_eigenvalue(av, dim, lambda) {
        lemma_meq_sym(sim, av, dim, dim);
        lemma_eigenvalue_meq(av, sim, dim, lambda);
    }
}

/* ---------- the stopping criterion ---------- */

/// Every off-diagonal entry of B is below thr in magnitude.
pub open spec fn offdiag_below(b: Seq<real>, dim: int, thr: real) -> bool {
    forall|r: int, c: int| 0 <= r < dim && 0 <= c < dim && r != c
        ==> #[trigger] abs_r(sym_entry(b, r, c)) < thr
}

/// The stopping threshold of both eigensolvers for the relative tolerance tol:
/// tol * max(1, max_i |B_ii|) (`scaled_near_zero(.., 1.0.max(max_diag), JACOBI_TOL)` in jacobi.rs).
pub open spec fn jacobi_thr_r(tol: real, b: Seq<real>, dim: int) -> real {
    tol * max_r(1.0real, max_abs_diag_r(b, dim))
}

/// The stopping criterion: the largest off-diagonal magnitude of B is below the threshold.
pub open spec fn jacobi_converged(tol: real, b: Seq<real>, dim: int) -> bool {
    offdiag_below(b, dim, jacobi_thr_r(tol, b, dim))
}

/// For a nonnegative tolerance the threshold is nonnegative.
pub proof fn lemma_jacobi_thr_nonneg(tol: real, b: Seq<real>, dim: int)
    requires
        tol >= 0.0real,
    ensures
        jacobi_thr_r(tol, b, dim) >= tol,
{
    lemma_tol_scale_ge(tol, max_r(1.0real, max_abs_diag_r(b, dim)));
}

/* ---------- the off-diagonal part and its Frobenius norm ---------- */

/// diag(A): the diagonal part of a matrix.
pub open spec fn mdiag(a: MatF) -> MatF {
    |r: int, c: int| if r == c { a(r, c) } else { 0.0real }
}

/// A - diag(A).
pub open spec fn moffdiag(a: MatF) -> MatF {
    msub(a, mdiag(a))
}

/// ||B - diag(B)||_F for a packed symmetric B.
pub open spec fn offdiag_norm_r(b: Seq<real>, dim: int) -> real {
    sqrt_r(frob_f(moffdiag(sym_view(b)), moffdiag(sym_view(b)), dim))
}

/// The number dim (dim - 1) of off-diagonal positions, as a real.
pub open spec fn offdiag_count_r(dim: int) -> real {
    (dim as real) * ((dim as real) - 1.0real)
}

/// If every off-diagonal entry is below thr then
/// ||B - diag(B)||_F^2 <= dim (dim - 1) thr^2.
pub proof fn lemma_offdiag_frob_bound(b: Seq<real>, dim: int, thr: real)
    requires
        dim >= 0,
        thr >= 0.0real,
        offdiag_below(b, dim, thr),
    ensures
        frob_f(moffdiag(sym_view(b)), moffdiag(sym_view(b)), dim) <= offdiag_count_r(dim) * (thr * thr),
{
    let n = dim as nat;
    let (off, t2) = (moffdiag(sym_view(b)), thr * thr);
    let (dr, dr1) = (dim as real, (dim as real) - 1.0real);
    let f = |r: int, c: int| off(r, c) * off(r, c);
    let rows = |r: int| sum_fn_r(|c: int| f(r, c), n);
    let row_bound = |r: int| dr1 * t2;
    assert(frob_f(off, off, dim) == sum_fn_r(rows, n));
    // each row: the diagonal term is 0 and every other term is at most thr^2
    assert forall|r: int| 0 <= r < dim implies #[trigger] rows(r) <= row_bound(r) by {
        let g = |c: int| if c == r { 0.0real } else { t2 };
        assert forall|c: int| 0 <= c < dim implies #[trigger] f(r, c) <= g(c) by {
            if c == r {
                assert(off(r, c) == 0.0real);
                assert(f(r, c) == 0.0real * 0.0real);
            } else {
                let x = sym_entry(b, r, c);
                assert(off(r, c) == x);
                let ax = abs_r(x);
                assert(ax * ax == x * x) by (nonlinear_arith)
                    requires ax == x || ax == -x;
                assert(x * x <= t2) by (nonlinear_arith)
                    requires ax * ax == x * x, 0.0real <= ax, ax <= thr, t2 == thr * thr;
            }
        }
        lemma_sum_fn_le(|c: int| f(r, c), g, n);
        lemma_sum_fn_const(t2, n);
        lemma_sum_fn_update(|c: int| t2, g, n, r, -t2);
        assert(dr * t2 - t2 == dr1 * t2) by (nonlinear_arith)
            requires dr1 == dr - 1.0real;
    }
    lemma_sum_fn_le(rows, row_bound, n);
    lemma_sum_fn_const(dr1 * t2, n);
    assert(dr * (dr1 * t2) == (dr * dr1) * t2) by (nonlinear_arith);
}

/// The stopping criterion with threshold thr gives ||B - diag(B)||_F <= sqrt(dim (dim - 1)) * thr.
pub proof fn lemma_offdiag_norm_bound(b: Seq<real>, dim: int, thr: real)
    requires
        dim >= 0,
        thr >= 0.0real,
        offdiag_below(b, dim, thr),
    ensures
        offdiag_norm_r(b, dim) <= sqrt_r(offdiag_count_r(dim)) * thr,
{
    let off = moffdiag(sym_view(b));
    let (s, cnt) = (frob_f(off, off, dim), offdiag_count_r(dim));
    lemma_offdiag_frob_bound(b, dim, thr);
    lemma_frob_f_self_nonneg(off, dim);
    assert(cnt >= 0.0real) by (nonlinear_arith)
        requires cnt == (dim as real) * ((dim as real) - 1.0real), dim as real >= 1.0real || dim as real == 0.0real;
    lemma_sq_nonneg(thr);
    assert(cnt * (thr * thr) >= 0.0real);
    lemma_sqrt_monotone(s, cnt * (thr * thr));
    lemma_sqrt_mul(cnt, thr * thr);
    lemma_sqrt_square(thr);
}

/* ---------- the maximal off-diagonal position ---------- */

/// (j, k) with j < k is a position of maximal off-diagonal magnitude of B.
pub open spec fn offdiag_max_at(b: Seq<real>, dim: int, j: int, k: int) -> bool {
    &&& 0 <= j < k < dim
    &&& forall|r: int, c: int| 0 <= r < c < dim
            ==> abs_r(#[trigger] sym_entry(b, r, c)) <= abs_r(sym_entry(b, j, k))
}

/// If the maximal off-diagonal magnitude is below thr, so is every off-diagonal entry.
pub proof fn lemma_offdiag_max_below(b: Seq<real>, dim: int, j: int, k: int, thr: real)
    requires
        offdiag_max_at(b, dim, j, k),
        abs_r(sym_entry(b, j, k)) < thr,
    ensures
        offdiag_below(b, dim, thr),
{
    assert forall|r: int, c: int| 0 <= r < dim && 0 <= c < dim && r != c
        implies #[trigger] abs_r(sym_entry(b, r, c)) < thr by {
        if r > c {
            assert(sym_entry(b, r, c) == sym_entry(b, c, r));
        }
    }
}

/* ---------- the column-cyclic pivot order ---------- */

/// The pivot positions (j, k) of one column-cyclic sweep: column by column k = 1, .., dim - 1,
/// and within column k the rows j = 0, .., k - 1.
pub open spec fn cyc_order(dim: int) -> Seq<(int, int)>
    decreases dim,
{
    if dim <= 1 {
        Seq::empty()
    } else {
        cyc_order(dim - 1) + Seq::new((dim - 1) as nat, |j: int| (j, dim - 1))
    }
}

/// The position of (j, k) in the sweep: the k (k - 1) / 2 pairs of the columns before k, then j.
pub open spec fn cyc_pos(j: int, k: int) -> int {
    k * (k - 1) / 2 + j
}

/// A sweep of order dim lists dim (dim - 1) / 2 positions.
pub proof fn lemma_cyc_order_len(dim: int)
    requires
        dim >= 1,
    ensures
        cyc_order(dim).len() == cyc_pos(0, dim),
    decreases dim,
{
    if dim > 1 {
        lemma_cyc_order_len(dim - 1);
        assert((dim - 1) * (dim - 2) / 2 + (dim - 1) == dim * (dim - 1) / 2) by (nonlinear_arith);
    }
}

/// Every listed position is a pair j < k < dim, and it is listed at its own position cyc_pos(j, k).
pub proof fn lemma_cyc_order_entries(dim: int)
    requires
        dim >= 1,
    ensures
        forall|t: int| 0 <= t < cyc_order(dim).len()
            ==> 0 <= (#[trigger] cyc_order(dim)[t]).0 < cyc_order(dim)[t].1 < dim
                && t == cyc_pos(cyc_order(dim)[t].0, cyc_order(dim)[t].1),
    decreases dim,
{
    if dim > 1 {
        lemma_cyc_order_len(dim - 1);
        lemma_cyc_order_entries(dim - 1);
        let prev = cyc_order(dim - 1);
        let col = Seq::new((dim - 1) as nat, |j: int| (j, dim - 1));
        assert forall|t: int| 0 <= t < cyc_order(dim).len()
            implies 0 <= (#[trigger] cyc_order(dim)[t]).0 < cyc_order(dim)[t].1 < dim
                && t == cyc_pos(cyc_order(dim)[t].0, cyc_order(dim)[t].1) by {
            if t < prev.len() {
                assert(cyc_order(dim)[t] == prev[t]);
            } else {
                assert(cyc_order(dim)[t] == col[t - prev.len()]);
            }
        }
    }
}

/// Every pair j < k < dim is listed, at position cyc_pos(j, k).
pub proof fn lemma_cyc_order_at(dim: int, j: int, k: int)
    requires
        0 <= j < k < dim,
    ensures
        0 <= cyc_pos(j, k) < cyc_order(dim).len(),
        cyc_order(dim)[cyc_pos(j, k)] == (j, k),
    decreases dim,
{
    lemma_cyc_order_len(dim);
    lemma_cyc_order_len(dim - 1);
    if k == dim - 1 {
        let col = Seq::new((dim - 1) as nat, |j: int| (j, dim - 1));
        assert(cyc_order(dim)[cyc_pos(j, k)] == col[j]);
    } else {
        lemma_cyc_order_at(dim - 1, j, k);
        assert(cyc_order(dim)[cyc_pos(j, k)] == cyc_order(dim - 1)[cyc_pos(j, k)]);
    }
}

/// A sweep in column-cyclic order visits every pair j < k < dim exactly once.
pub open spec fn visits_each_pair_once(visited: Seq<(int, int)>, dim: int) -> bool {
    &&& forall|j: int, k: int| 0 <= j < k < dim
            ==> 0 <= #[trigger] cyc_pos(j, k) < visited.len() && visited[cyc_pos(j, k)] == (j, k)
    &&& forall|t: int| 0 <= t < visited.len()
            ==> 0 <= (#[trigger] visited[t]).0 < visited[t].1 < dim && t == cyc_pos(visited[t].0, visited[t].1)
}

/// The column-cyclic order visits every pair j < k exactly once, at position cyc_pos(j, k).
pub proof fn lemma_cyc_order_visits_each_pair_once(dim: int)
    requires
        dim >= 1,
    ensures
        visits_each_pair_once(cyc_order(dim), dim),
{
    lemma_cyc_order_entries(dim);
    assert forall|j: int, k: int| 0 <= j < k < dim
        implies 0 <= #[trigger] cyc_pos(j, k) < cyc_order(dim).len() && cyc_order(dim)[cyc_pos(j, k)] == (j, k) by {
        lemma_cyc_order_at(dim, j, k);
    }
}

/* ---------- the common eigensolver contract ---------- */

/// Approximate eigendecomposition of A: A = Q B Q^T with Q orthogonal and B symmetric,
/// where B is within eps of its diagonal in Frobenius norm.
pub open spec fn eigen_decomp_approx(a: Seq<real>, q: Seq<real>, b: Seq<real>, dim: int, eps: real) -> bool {
    &&& eigen_decomp_inv(a, q, b, dim)
    &&& is_symmetric(sym_view(b), dim)
    &&& offdiag_norm_r(b, dim) <= eps
}

/// The bound established by the stopping criterion with tolerance tol: sqrt(dim (dim - 1)) * thr.
pub open spec fn jacobi_eps_r(tol: real, b: Seq<real>, dim: int) -> real {
    sqrt_r(offdiag_count_r(dim)) * jacobi_thr_r(tol, b, dim)
}

/// The invariant plus the stopping criterion with a nonnegative tolerance give the
/// approximate eigendecomposition with the eigensolvers' bound.
pub proof fn lemma_converged_approx(tol: real, a: Seq<real>, q: Seq<real>, b: Seq<real>, dim: int)
    requires
        tol >= 0.0real,
        dim >= 0,
        eigen_decomp_inv(a, q, b, dim),
        jacobi_converged(tol, b, dim),
    ensures
        eigen_decomp_approx(a, q, b, dim, jacobi_eps_r(tol, b, dim)),
{
    lemma_sym_view_symmetric(b, dim);
    lemma_jacobi_thr_nonneg(tol, b, dim);
    lemma_offdiag_norm_bound(b, dim, jacobi_thr_r(tol, b, dim));
}

/// The contract shared by both eigensolvers with stopping tolerance tol, for the input A,
/// the final transformed matrix B, the accumulated Q and the returned eigenvalue vector: the
/// returned values are the diagonal of B, and (Q, B) is an approximate eigendecomposition of A
/// within the stopping bound `jacobi_eps_r`.
pub open spec fn eigensolver_ok(tol: real, a: Seq<real>, b: Seq<real>, q: Seq<real>, vals: Seq<real>, dim: int) -> bool {
    &&& q.len() == dim * dim
    &&& vals == sym_diag_r(b, dim)
    &&& eigen_decomp_approx(a, q, b, dim, jacobi_eps_r(tol, b, dim))
}

/// A 1 x 1 matrix is its own eigendecomposition: Q = (1) and B = A have no off-diagonal entries.
pub proof fn lemma_eigensolver_dim1(tol: real, a: Seq<real>, q: Seq<real>)
    requires
        tol >= 0.0real,
        a.len() >= 1,
        q.len() == 1,
        q[0] == 1.0real,
    ensures
        eigensolver_ok(tol, a, a, q, sym_diag_r(a, 1), 1),
{
    assert forall|r: int, c: int| 0 <= r < 1 && 0 <= c < 1
        implies #[trigger] cm_entry(q, 1, r, c) == (if r == c { 1.0real } else { 0.0real }) by {
        assert(cm_idx(1, r, c) == 0);
    }
    lemma_eigen_inv_init(a, q, 1);
    lemma_converged_approx(tol, a, q, a, 1);
}

} // verus!

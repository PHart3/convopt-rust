// Spec-level support for the Jacobi rotation core (not trusted).
//
// `jacobi_rot` applies one Jacobi rotation J in the (j, k) plane (j < k) to a
// packed symmetric matrix A and accumulates it into the column-major
// eigenvector matrix Q:
//
//     A' = J^T A J,    Q' = Q J,    J = rot_mat(j, k, c, s),    c^2 + s^2 = 1.
//
// This module defines J (`rot_mat`), the explicit entries of J^T A J
// (`rot_target`) and Q J (`rot_cols`), proves that J is orthogonal and that
// these explicit forms are the products, gives the scalar identities behind
// the executable choice of c, s and the annihilation of A'_jk, and proves the
// step invariant shared by both eigensolvers: a decomposition A = Q B Q^T
// with Q orthogonal is preserved, as are symmetry and eigenvalues of B.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::seq::*;
#[cfg(verus_keep_ghost)]
use crate::real_spec::abs_r;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::lemma_pk_idx_injective;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- the rotation and its explicit products ---------- */

/// The Jacobi rotation in the (j, k) plane: the identity except
/// J_jj = J_kk = c, J_jk = s, J_kj = -s.
pub open spec fn rot_mat(j: int, k: int, c: real, s: real) -> MatF {
    |r: int, l: int|
        if r == l { if r == j || r == k { c } else { 1.0real } }
        else if r == j && l == k { s }
        else if r == k && l == j { -s }
        else { 0.0real }
}

/// Entry (r, l) of J^T A J, written out as the products (J^T (A J))(r, l) compute them:
/// rows and columns j, k are rotated, the rest of A is unchanged.
pub open spec fn rot_entry(a: MatF, j: int, k: int, c: real, s: real, r: int, l: int) -> real {
    if r == j && l == j { c * (c * a(j, j) - s * a(k, j)) - s * (c * a(j, k) - s * a(k, k)) }
    else if r == k && l == k { s * (s * a(j, j) + c * a(k, j)) + c * (s * a(j, k) + c * a(k, k)) }
    else if r == j && l == k { s * (c * a(j, j) - s * a(k, j)) + c * (c * a(j, k) - s * a(k, k)) }
    else if r == k && l == j { c * (s * a(j, j) + c * a(k, j)) - s * (s * a(j, k) + c * a(k, k)) }
    else if l == j { c * a(r, j) - s * a(r, k) }
    else if l == k { s * a(r, j) + c * a(r, k) }
    else if r == j { c * a(j, l) - s * a(k, l) }
    else if r == k { s * a(j, l) + c * a(k, l) }
    else { a(r, l) }
}

/// J^T A J as an index function (for symmetric A).
pub open spec fn rot_target(a: MatF, j: int, k: int, c: real, s: real) -> MatF {
    |r: int, l: int| rot_entry(a, j, k, c, s, r, l)
}

/// Entry (r, l) of Q J: columns j and k are rotated, the others unchanged.
pub open spec fn rot_col_entry(q: MatF, j: int, k: int, c: real, s: real, r: int, l: int) -> real {
    if l == j { c * q(r, j) - s * q(r, k) }
    else if l == k { s * q(r, j) + c * q(r, k) }
    else { q(r, l) }
}

/// Q J as an index function.
pub open spec fn rot_cols(q: MatF, j: int, k: int, c: real, s: real) -> MatF {
    |r: int, l: int| rot_col_entry(q, j, k, c, s, r, l)
}

/// The executable rotation step on the stored representations: for J = rot_mat(j, k, c, s)
/// with c^2 + s^2 = 1, the packed matrix becomes J^T B J and the column-major matrix Q J.
pub open spec fn jacobi_step(b: Seq<real>, q: Seq<real>, b2: Seq<real>, q2: Seq<real>, dim: int, j: int, k: int, c: real, s: real) -> bool {
    &&& c * c + s * s == 1.0real
    &&& meq(sym_view(b2), mmul(mmul(mtr(rot_mat(j, k, c, s)), sym_view(b), dim), rot_mat(j, k, c, s), dim), dim, dim)
    &&& meq(cm_view(q2, dim), mmul(cm_view(q, dim), rot_mat(j, k, c, s), dim), dim, dim)
}

/* ---------- sums with two nonzero terms ---------- */

/// A sum whose only nonzero summands are at p and q.
pub proof fn lemma_sum_fn_two(f: spec_fn(int) -> real, n: nat, p: int, q: int)
    requires
        0 <= p < n,
        0 <= q < n,
        p != q,
        forall|i: int| 0 <= i < n && i != p && i != q ==> #[trigger] f(i) == 0.0real,
    ensures
        sum_fn_r(f, n) == f(p) + f(q),
{
    let g = |i: int| if i == q { 0.0real } else { f(i) };
    assert forall|i: int| 0 <= i < n && i != p implies #[trigger] g(i) == 0.0real by {}
    lemma_sum_fn_single(g, n, p);
    lemma_sum_fn_update(g, f, n, q, f(q));
}

/* ---------- products with the rotation ---------- */

/// (A J)(r, l) = rot_col_entry(A)(r, l).
pub proof fn lemma_mul_rot_right(a: MatF, dim: int, j: int, k: int, c: real, s: real, r: int, l: int)
    requires
        0 <= j < k < dim,
        0 <= l < dim,
    ensures
        mmul(a, rot_mat(j, k, c, s), dim)(r, l) == rot_col_entry(a, j, k, c, s, r, l),
{
    let jm = rot_mat(j, k, c, s);
    let f = |i: int| a(r, i) * jm(i, l);
    let (x, y) = (a(r, j), a(r, k));
    if l == j {
        assert forall|i: int| 0 <= i < dim && i != j && i != k implies #[trigger] f(i) == 0.0real by {
            assert(jm(i, l) == 0.0real);
        }
        lemma_sum_fn_two(f, dim as nat, j, k);
        assert(f(j) == x * c);
        assert(f(k) == y * (-s));
        assert(x * c + y * (-s) == c * x - s * y) by (nonlinear_arith);
    } else if l == k {
        assert forall|i: int| 0 <= i < dim && i != j && i != k implies #[trigger] f(i) == 0.0real by {
            assert(jm(i, l) == 0.0real);
        }
        lemma_sum_fn_two(f, dim as nat, j, k);
        assert(f(j) == x * s);
        assert(f(k) == y * c);
        assert(x * s + y * c == s * x + c * y) by (nonlinear_arith);
    } else {
        assert forall|i: int| 0 <= i < dim && i != l implies #[trigger] f(i) == 0.0real by {
            assert(jm(i, l) == 0.0real);
        }
        lemma_sum_fn_single(f, dim as nat, l);
        assert(f(l) == a(r, l) * 1.0real);
    }
}

/// (J^T B)(r, l): rows j and k are rotated, the others unchanged.
pub proof fn lemma_mul_rot_left(b: MatF, dim: int, j: int, k: int, c: real, s: real, r: int, l: int)
    requires
        0 <= j < k < dim,
        0 <= r < dim,
    ensures
        mmul(mtr(rot_mat(j, k, c, s)), b, dim)(r, l)
            == if r == j { c * b(j, l) - s * b(k, l) }
               else if r == k { s * b(j, l) + c * b(k, l) }
               else { b(r, l) },
{
    let jm = rot_mat(j, k, c, s);
    let f = |i: int| mtr(jm)(r, i) * b(i, l);
    let (x, y) = (b(j, l), b(k, l));
    if r == j {
        assert forall|i: int| 0 <= i < dim && i != j && i != k implies #[trigger] f(i) == 0.0real by {
            assert(mtr(jm)(r, i) == 0.0real);
        }
        lemma_sum_fn_two(f, dim as nat, j, k);
        assert(f(j) == c * x);
        assert(f(k) == (-s) * y);
        assert((-s) * y == -(s * y)) by (nonlinear_arith);
    } else if r == k {
        assert forall|i: int| 0 <= i < dim && i != j && i != k implies #[trigger] f(i) == 0.0real by {
            assert(mtr(jm)(r, i) == 0.0real);
        }
        lemma_sum_fn_two(f, dim as nat, j, k);
        assert(f(j) == s * x);
        assert(f(k) == c * y);
    } else {
        assert forall|i: int| 0 <= i < dim && i != r implies #[trigger] f(i) == 0.0real by {
            assert(mtr(jm)(r, i) == 0.0real);
        }
        lemma_sum_fn_single(f, dim as nat, r);
        assert(f(r) == 1.0real * b(r, l));
    }
}

/// J^T A J = rot_target(A) on the block.
pub proof fn lemma_rot_congruence(a: MatF, dim: int, j: int, k: int, c: real, s: real)
    requires
        0 <= j < k < dim,
    ensures
        meq(mmul(mmul(mtr(rot_mat(j, k, c, s)), a, dim), rot_mat(j, k, c, s), dim), rot_target(a, j, k, c, s), dim, dim),
{
    let jm = rot_mat(j, k, c, s);
    let ja = mmul(mtr(jm), a, dim);
    let jaj = mmul(ja, jm, dim);
    assert forall|r: int, l: int| 0 <= r < dim && 0 <= l < dim
        implies #[trigger] jaj(r, l) == rot_target(a, j, k, c, s)(r, l) by {
        lemma_mul_rot_right(ja, dim, j, k, c, s, r, l);
        lemma_mul_rot_left(a, dim, j, k, c, s, r, j);
        lemma_mul_rot_left(a, dim, j, k, c, s, r, k);
        lemma_mul_rot_left(a, dim, j, k, c, s, r, l);
    }
}

/// Q J = rot_cols(Q) on the block.
pub proof fn lemma_rot_cols(q: MatF, dim: int, j: int, k: int, c: real, s: real)
    requires
        0 <= j < k < dim,
    ensures
        meq(mmul(q, rot_mat(j, k, c, s), dim), rot_cols(q, j, k, c, s), dim, dim),
{
    assert forall|r: int, l: int| 0 <= r < dim && 0 <= l < dim
        implies #[trigger] mmul(q, rot_mat(j, k, c, s), dim)(r, l) == rot_cols(q, j, k, c, s)(r, l) by {
        lemma_mul_rot_right(q, dim, j, k, c, s, r, l);
    }
}

/* ---------- orthogonality ---------- */

/// J^T J = I when c^2 + s^2 = 1.
pub proof fn lemma_rot_tt_ident(dim: int, j: int, k: int, c: real, s: real)
    requires
        0 <= j < k < dim,
        c * c + s * s == 1.0real,
    ensures
        meq(mmul(mtr(rot_mat(j, k, c, s)), rot_mat(j, k, c, s), dim), mident(), dim, dim),
{
    let jm = rot_mat(j, k, c, s);
    assert forall|r: int, l: int| 0 <= r < dim && 0 <= l < dim
        implies #[trigger] mmul(mtr(jm), jm, dim)(r, l) == mident()(r, l) by {
        lemma_mul_rot_left(jm, dim, j, k, c, s, r, l);
        if r == j {
            if l == j {
                assert(c * c - s * (-s) == 1.0real) by (nonlinear_arith)
                    requires c * c + s * s == 1.0real;
            } else if l == k {
                assert(c * s - s * c == 0.0real) by (nonlinear_arith);
            } else {
                assert(jm(j, l) == 0.0real);
                assert(jm(k, l) == 0.0real);
            }
        } else if r == k {
            if l == j {
                assert(s * c + c * (-s) == 0.0real) by (nonlinear_arith);
            } else if l == k {
                assert(s * s + c * c == 1.0real);
            } else {
                assert(jm(j, l) == 0.0real);
                assert(jm(k, l) == 0.0real);
            }
        }
    }
}

/// The rotation is orthogonal: J^T J = I and J J^T = I.
pub proof fn lemma_rot_orthogonal(dim: int, j: int, k: int, c: real, s: real)
    requires
        0 <= j < k < dim,
        c * c + s * s == 1.0real,
    ensures
        is_orthogonal(rot_mat(j, k, c, s), dim),
{
    let jm = rot_mat(j, k, c, s);
    let jm2 = rot_mat(j, k, c, -s);
    lemma_rot_tt_ident(dim, j, k, c, s);
    // J^T = rot(c, -s) and J = rot(c, -s)^T on the block, so J J^T = rot(c, -s)^T rot(c, -s) = I
    assert((-s) * (-s) == s * s) by (nonlinear_arith);
    lemma_rot_tt_ident(dim, j, k, c, -s);
    assert forall|r: int, l: int| 0 <= r < dim && 0 <= l < dim implies #[trigger] jm(r, l) == mtr(jm2)(r, l) by {}
    assert forall|r: int, l: int| 0 <= r < dim && 0 <= l < dim implies #[trigger] mtr(jm)(r, l) == jm2(r, l) by {}
    lemma_mul_meq(jm, mtr(jm2), mtr(jm), jm2, dim, dim, dim);
    lemma_meq_trans(mmul(jm, mtr(jm), dim), mmul(mtr(jm2), jm2, dim), mident(), dim, dim);
}

/* ---------- scalar identities of the executable rotation ---------- */

/// t = sigma / (|tau| + sqrt(1 + tau^2)) with sigma = sign(tau) solves t^2 + 2 tau t - 1 = 0.
pub proof fn lemma_jacobi_t(tau: real, sigma: real, sq: real, t: real)
    requires
        sigma * sigma == 1.0real,
        sigma * tau == abs_r(tau),
        sq >= 0.0real,
        sq * sq == 1.0real + tau * tau,
        t * (abs_r(tau) + sq) == sigma,
    ensures
        t * t + 2.0real * tau * t - 1.0real == 0.0real,
{
    let (ab, den, u) = (abs_r(tau), abs_r(tau) + sq, sq - abs_r(tau));
    assert(ab == tau || ab == -tau);
    assert((-tau) * (-tau) == tau * tau) by (nonlinear_arith);
    assert(ab * ab == tau * tau);
    // (|tau| + S)(S - |tau|) = S^2 - tau^2 = 1, hence t = sigma (S - |tau|)
    assert((ab + sq) * (sq - ab) == sq * sq - ab * ab) by (nonlinear_arith);
    assert(den * u == 1.0real);
    assert(t * (den * u) == (t * den) * u) by (nonlinear_arith);
    assert(t == sigma * u);
    // t^2 + 2 tau t - 1 = u^2 + 2 |tau| u - 1 = (S - |tau|)^2 + 2 |tau| (S - |tau|) - 1 = S^2 - tau^2 - 1 = 0
    assert((sigma * u) * (sigma * u) == (sigma * sigma) * (u * u)) by (nonlinear_arith);
    assert(2.0real * tau * (sigma * u) == 2.0real * (sigma * tau) * u) by (nonlinear_arith);
    assert(t * t + 2.0real * tau * t - 1.0real == u * u + 2.0real * ab * u - 1.0real);
    assert((sq - ab) * (sq - ab) + 2.0real * ab * (sq - ab) == sq * sq - ab * ab) by (nonlinear_arith);
}

/// c = 1 / sqrt(1 + t^2) and s = c t give c^2 (1 + t^2) = 1 and c^2 + s^2 = 1.
pub proof fn lemma_jacobi_cs(t: real, sq: real, c: real, s: real)
    requires
        sq * sq == 1.0real + t * t,
        c * sq == 1.0real,
        s == c * t,
    ensures
        c * c * (1.0real + t * t) == 1.0real,
        c * c + s * s == 1.0real,
{
    assert(c * c * (sq * sq) == (c * sq) * (c * sq)) by (nonlinear_arith);
    assert(c * c + (c * t) * (c * t) == c * c * (1.0real + t * t)) by (nonlinear_arith);
}

/// With s = c t, c^2 (1 + t^2) = 1 and the pivot equation a_jk t^2 + (a_kk - a_jj) t - a_jk = 0,
/// the rotated 2 x 2 block (rot_entry with a_kj = a_jk) is diag(a_jj - t a_jk, a_kk + t a_jk):
/// the executable diagonal updates and the annihilation of the pivot entry.
/// (Each nonlinear step is a single distributive or associative law; the linear solver
/// combines them, treating monomials as atoms.)
pub proof fn lemma_jacobi_scalars(ajj: real, akk: real, ajk: real, t: real, c: real, s: real)
    requires
        s == c * t,
        c * c * (1.0real + t * t) == 1.0real,
        ajk * t * t + (akk - ajj) * t - ajk == 0.0real,
    ensures
        c * (c * ajj - s * ajk) - s * (c * ajk - s * akk) == ajj - t * ajk,
        s * (s * ajj + c * ajk) + c * (s * ajk + c * akk) == akk + t * ajk,
        s * (c * ajj - s * ajk) + c * (c * ajk - s * akk) == 0.0real,
        c * (s * ajj + c * ajk) - s * (s * ajk + c * akk) == 0.0real,
{
    let p = ajk * t * t + (akk - ajj) * t - ajk;
    // monomials in s rewritten with s = c t and c^2 factored out
    assert(c * (c * t) * ajk == c * c * (t * ajk)) by (nonlinear_arith);
    assert((c * t) * c * ajk == c * c * (t * ajk)) by (nonlinear_arith);
    assert((c * t) * (c * t) * akk == c * c * (t * t * akk)) by (nonlinear_arith);
    assert((c * t) * (c * t) * ajj == c * c * (t * t * ajj)) by (nonlinear_arith);
    assert((c * t) * (c * t) * ajk == c * c * (t * t * ajk)) by (nonlinear_arith);
    assert((c * t) * c * ajj == c * c * (t * ajj)) by (nonlinear_arith);
    assert(c * (c * t) * ajj == c * c * (t * ajj)) by (nonlinear_arith);
    assert(c * (c * t) * akk == c * c * (t * akk)) by (nonlinear_arith);
    assert((c * t) * c * akk == c * c * (t * akk)) by (nonlinear_arith);
    // the pivot equation and its multiple by t, expanded
    assert(ajk * t * t + (akk - ajj) * t - ajk == t * t * ajk + t * akk - t * ajj - ajk) by (nonlinear_arith);
    assert(t * (ajk * t * t + (akk - ajj) * t - ajk) == t * t * (t * ajk) + t * t * akk - t * t * ajj - t * ajk) by (nonlinear_arith);
    assert(t * p == 0.0real);
    // entry (j, j): c^2 (a_jj - 2 t a_jk + t^2 a_kk) = c^2 (1 + t^2)(a_jj - t a_jk) + c^2 t p
    let d1 = ajj - t * ajk;
    assert(c * (c * ajj - s * ajk) == c * c * ajj - c * s * ajk) by (nonlinear_arith);
    assert(s * (c * ajk - s * akk) == s * c * ajk - s * s * akk) by (nonlinear_arith);
    assert(c * c * ajj - c * c * (t * ajk) - c * c * (t * ajk) + c * c * (t * t * akk) == c * c * (ajj - t * ajk - t * ajk + t * t * akk)) by (nonlinear_arith);
    assert((c * c * (1.0real + t * t)) * d1 == c * c * ((1.0real + t * t) * d1)) by (nonlinear_arith);
    assert((1.0real + t * t) * d1 == d1 + t * t * d1) by (nonlinear_arith);
    assert(t * t * (ajj - t * ajk) == t * t * ajj - t * t * (t * ajk)) by (nonlinear_arith);
    assert(ajj - t * ajk - t * ajk + t * t * akk == (1.0real + t * t) * d1);
    assert(d1 == 1.0real * d1);
    // entry (k, k): c^2 (t^2 a_jj + 2 t a_jk + a_kk) = c^2 (1 + t^2)(a_kk + t a_jk) - c^2 t p
    let d2 = akk + t * ajk;
    assert(s * (s * ajj + c * ajk) == s * s * ajj + s * c * ajk) by (nonlinear_arith);
    assert(c * (s * ajk + c * akk) == c * s * ajk + c * c * akk) by (nonlinear_arith);
    assert(c * c * (t * t * ajj) + c * c * (t * ajk) + c * c * (t * ajk) + c * c * akk == c * c * (t * t * ajj + t * ajk + t * ajk + akk)) by (nonlinear_arith);
    assert((c * c * (1.0real + t * t)) * d2 == c * c * ((1.0real + t * t) * d2)) by (nonlinear_arith);
    assert((1.0real + t * t) * d2 == d2 + t * t * d2) by (nonlinear_arith);
    assert(t * t * (akk + t * ajk) == t * t * akk + t * t * (t * ajk)) by (nonlinear_arith);
    assert(t * t * ajj + t * ajk + t * ajk + akk == (1.0real + t * t) * d2);
    assert(d2 == 1.0real * d2);
    // entries (j, k) and (k, j): c^2 (t a_jj - t^2 a_jk + a_jk - t a_kk) = -c^2 p = 0
    assert(s * (c * ajj - s * ajk) == s * c * ajj - s * s * ajk) by (nonlinear_arith);
    assert(c * (c * ajk - s * akk) == c * c * ajk - c * s * akk) by (nonlinear_arith);
    assert(c * c * (t * ajj) - c * c * (t * t * ajk) + c * c * ajk - c * c * (t * akk) == c * c * (t * ajj - t * t * ajk + ajk - t * akk)) by (nonlinear_arith);
    assert(t * ajj - t * t * ajk + ajk - t * akk == -p);
    assert(c * c * (-p) == 0.0real);
    assert(c * (s * ajj + c * ajk) == c * s * ajj + c * c * ajk) by (nonlinear_arith);
    assert(s * (s * ajk + c * akk) == s * s * ajk + s * c * akk) by (nonlinear_arith);
    assert(c * c * (t * ajj) + c * c * ajk - c * c * (t * t * ajk) - c * c * (t * akk) == c * c * (t * ajj + ajk - t * t * ajk - t * akk)) by (nonlinear_arith);
    assert(t * ajj + ajk - t * t * ajk - t * akk == -p);
}

/* ---------- locality of single-entry writes ---------- */

/// Writing packed entry (r0, c0) leaves every other symmetric entry unchanged.
pub proof fn lemma_sym_entry_write_other(prev: Seq<real>, cur: Seq<real>, dim: int, r0: int, c0: int, r: int, l: int)
    requires
        0 <= r0 <= c0,
        0 <= r < dim,
        0 <= l < dim,
        prev.len() >= pk_len(dim),
        cur.len() == prev.len(),
        forall|t: int| 0 <= t < prev.len() && t != pk_idx(r0, c0) ==> cur[t] == prev[t],
        !((r == r0 && l == c0) || (r == c0 && l == r0)),
    ensures
        sym_entry(cur, r, l) == sym_entry(prev, r, l),
{
    let (lo, hi) = if r <= l { (r, l) } else { (l, r) };
    lemma_pk_idx_injective(lo, hi, r0, c0);
    lemma_pk_idx_bound(lo, hi, dim);
    assert(cur[pk_idx(lo, hi)] == prev[pk_idx(lo, hi)]);
}

/* ---------- the step invariant shared by both eigensolvers ---------- */

/// Orthogonality only depends on the entries on the block.
pub proof fn lemma_orthogonal_meq(p: MatF, p2: MatF, dim: int)
    requires
        dim >= 0,
        meq(p, p2, dim, dim),
        is_orthogonal(p, dim),
    ensures
        is_orthogonal(p2, dim),
{
    lemma_transpose_meq(p, p2, dim, dim);
    lemma_mul_meq(mtr(p), mtr(p2), p, p2, dim, dim, dim);
    lemma_meq_sym(mmul(mtr(p), p, dim), mmul(mtr(p2), p2, dim), dim, dim);
    lemma_meq_trans(mmul(mtr(p2), p2, dim), mmul(mtr(p), p, dim), mident(), dim, dim);
    lemma_mul_meq(p, p2, mtr(p), mtr(p2), dim, dim, dim);
    lemma_meq_sym(mmul(p, mtr(p), dim), mmul(p2, mtr(p2), dim), dim, dim);
    lemma_meq_trans(mmul(p2, mtr(p2), dim), mmul(p, mtr(p), dim), mident(), dim, dim);
}

/// Symmetry only depends on the entries on the block.
pub proof fn lemma_symmetric_meq(a: MatF, b: MatF, dim: int)
    requires
        meq(a, b, dim, dim),
        is_symmetric(a, dim),
    ensures
        is_symmetric(b, dim),
{
    assert forall|i: int, j: int| 0 <= i < dim && 0 <= j < dim implies #[trigger] b(i, j) == b(j, i) by {
        assert(a(i, j) == a(j, i));
    }
}

/// One orthogonal similarity step preserves the decomposition A = Q B Q^T with Q orthogonal:
/// if B' = J^T B J and Q' = Q J for an orthogonal J, then Q' is orthogonal and A = Q' B' Q'^T.
pub proof fn lemma_rotation_step(a: MatF, q: MatF, b: MatF, q2: MatF, b2: MatF, jm: MatF, dim: int)
    requires
        dim >= 0,
        is_orthogonal(jm, dim),
        is_orthogonal(q, dim),
        meq(a, mmul(mmul(q, b, dim), mtr(q), dim), dim, dim),
        meq(b2, mmul(mmul(mtr(jm), b, dim), jm, dim), dim, dim),
        meq(q2, mmul(q, jm, dim), dim, dim),
    ensures
        is_orthogonal(q2, dim),
        meq(a, mmul(mmul(q2, b2, dim), mtr(q2), dim), dim, dim),
{
    let n = dim;
    let (qj, jt, qt) = (mmul(q, jm, n), mtr(jm), mtr(q));
    lemma_orthogonal_mul(q, jm, n);
    lemma_meq_sym(q2, qj, n, n);
    lemma_orthogonal_meq(qj, q2, n);
    // Q' B' Q'^T = (Q J) (J^T B J) (J^T Q^T)
    let jtbj = mmul(mmul(jt, b, n), jm, n);
    lemma_transpose_meq(q2, qj, n, n);
    lemma_transpose_mul_meq(q, jm, n, n, n);
    lemma_meq_trans(mtr(q2), mtr(qj), mmul(jt, qt, n), n, n);
    lemma_mul_meq(q2, qj, b2, jtbj, n, n, n);
    lemma_mul_meq(mmul(q2, b2, n), mmul(qj, jtbj, n), mtr(q2), mmul(jt, qt, n), n, n, n);
    // (Q J) (J^T B J) = Q (J (J^T (B J))) = Q (B J) = (Q B) J
    let bj = mmul(b, jm, n);
    lemma_mul_assoc_meq(jt, b, jm, n, n, n, n);
    lemma_mul_meq(qj, qj, jtbj, mmul(jt, bj, n), n, n, n);
    lemma_mul_assoc_meq(q, jm, mmul(jt, bj, n), n, n, n, n);
    lemma_cancel_inverse(jm, jt, bj, n);
    lemma_mul_meq(q, q, mmul(jm, mmul(jt, bj, n), n), bj, n, n, n);
    lemma_mul_assoc_meq(q, b, jm, n, n, n, n);
    lemma_meq_sym(mmul(mmul(q, b, n), jm, n), mmul(q, bj, n), n, n);
    let qb = mmul(q, b, n);
    lemma_meq_trans(mmul(qj, jtbj, n), mmul(qj, mmul(jt, bj, n), n), mmul(q, mmul(jm, mmul(jt, bj, n), n), n), n, n);
    lemma_meq_trans(mmul(qj, jtbj, n), mmul(q, mmul(jm, mmul(jt, bj, n), n), n), mmul(q, bj, n), n, n);
    lemma_meq_trans(mmul(qj, jtbj, n), mmul(q, bj, n), mmul(qb, jm, n), n, n);
    // ((Q B) J) (J^T Q^T) = (Q B) (J (J^T Q^T)) = (Q B) Q^T
    lemma_mul_meq(mmul(qj, jtbj, n), mmul(qb, jm, n), mmul(jt, qt, n), mmul(jt, qt, n), n, n, n);
    lemma_mul_assoc_meq(qb, jm, mmul(jt, qt, n), n, n, n, n);
    lemma_cancel_inverse(jm, jt, qt, n);
    lemma_mul_meq(qb, qb, mmul(jm, mmul(jt, qt, n), n), qt, n, n, n);
    let x = mmul(mmul(q2, b2, n), mtr(q2), n);
    lemma_meq_trans(x, mmul(mmul(qj, jtbj, n), mmul(jt, qt, n), n), mmul(mmul(qb, jm, n), mmul(jt, qt, n), n), n, n);
    lemma_meq_trans(x, mmul(mmul(qb, jm, n), mmul(jt, qt, n), n), mmul(qb, mmul(jm, mmul(jt, qt, n), n), n), n, n);
    lemma_meq_trans(x, mmul(qb, mmul(jm, mmul(jt, qt, n), n), n), mmul(qb, qt, n), n, n);
    lemma_meq_sym(x, mmul(qb, qt, n), n, n);
    lemma_meq_trans(a, mmul(qb, qt, n), x, n, n);
}

/// The executable step preserves the eigensolver invariant on the stored representations:
/// Q' = Q J stays orthogonal and A = Q' B' Q'^T, where J is orthogonal by c^2 + s^2 = 1.
pub proof fn lemma_jacobi_step_preserves(a: MatF, b: Seq<real>, q: Seq<real>, b2: Seq<real>, q2: Seq<real>, dim: int, j: int, k: int, c: real, s: real)
    requires
        0 <= j < k < dim,
        jacobi_step(b, q, b2, q2, dim, j, k, c, s),
        is_orthogonal(cm_view(q, dim), dim),
        meq(a, mmul(mmul(cm_view(q, dim), sym_view(b), dim), mtr(cm_view(q, dim)), dim), dim, dim),
    ensures
        is_orthogonal(cm_view(q2, dim), dim),
        meq(a, mmul(mmul(cm_view(q2, dim), sym_view(b2), dim), mtr(cm_view(q2, dim)), dim), dim, dim),
{
    lemma_rot_orthogonal(dim, j, k, c, s);
    lemma_rotation_step(a, cm_view(q, dim), sym_view(b), cm_view(q2, dim), sym_view(b2), rot_mat(j, k, c, s), dim);
}

/// The rotated matrix J^T B J is symmetric and has the eigenvalues of B.
pub proof fn lemma_jacobi_step_similar(b: Seq<real>, b2: Seq<real>, dim: int, j: int, k: int, c: real, s: real, lambda: real)
    requires
        0 <= j < k < dim,
        c * c + s * s == 1.0real,
        meq(sym_view(b2), mmul(mmul(mtr(rot_mat(j, k, c, s)), sym_view(b), dim), rot_mat(j, k, c, s), dim), dim, dim),
    ensures
        is_symmetric(sym_view(b2), dim),
        is_eigenvalue(sym_view(b), dim, lambda) <==> is_eigenvalue(sym_view(b2), dim, lambda),
{
    let jm = rot_mat(j, k, c, s);
    let rotated = mmul(mmul(mtr(jm), sym_view(b), dim), jm, dim);
    lemma_sym_view_symmetric(b, dim);
    lemma_congruence_symmetric(sym_view(b), jm, dim);
    lemma_meq_sym(sym_view(b2), rotated, dim, dim);
    lemma_symmetric_meq(rotated, sym_view(b2), dim);
    lemma_rot_orthogonal(dim, j, k, c, s);
    lemma_orthogonal_similarity_eigenvalues(sym_view(b), jm, dim, lambda);
    if is_eigenvalue(rotated, dim, lambda) {
        lemma_eigenvalue_meq(rotated, sym_view(b2), dim, lambda);
    }
    if is_eigenvalue(sym_view(b2), dim, lambda) {
        lemma_eigenvalue_meq(sym_view(b2), rotated, dim, lambda);
    }
}

/* ---------- what a rotation leaves unchanged ---------- */

/// The entries outside rows and columns j, k of cur agree with those of prev.
pub open spec fn same_outside(prev: Seq<real>, cur: Seq<real>, dim: int, j: int, k: int) -> bool {
    forall|r: int, l: int| 0 <= r < dim && 0 <= l < dim && r != j && r != k && l != j && l != k
        ==> #[trigger] sym_entry(cur, r, l) == sym_entry(prev, r, l)
}

/// A rotation at (j, k) (given by the existential postcondition of `jacobi_rot`)
/// changes only rows and columns j and k.
pub proof fn lemma_jacobi_step_same_outside(b: Seq<real>, q: Seq<real>, b2: Seq<real>, q2: Seq<real>, dim: int, j: int, k: int)
    requires
        0 <= j < k < dim,
        exists|c: real, s: real| jacobi_step(b, q, b2, q2, dim, j, k, c, s),
    ensures
        same_outside(b, b2, dim, j, k),
{
    let (c, s) = choose|c: real, s: real| jacobi_step(b, q, b2, q2, dim, j, k, c, s);
    let jm = rot_mat(j, k, c, s);
    let rotated = mmul(mmul(mtr(jm), sym_view(b), dim), jm, dim);
    lemma_rot_congruence(sym_view(b), dim, j, k, c, s);
    assert forall|r: int, l: int| 0 <= r < dim && 0 <= l < dim && r != j && r != k && l != j && l != k
        implies #[trigger] sym_entry(b2, r, l) == sym_entry(b, r, l) by {
        assert(sym_view(b2)(r, l) == rotated(r, l));
        assert(rotated(r, l) == rot_target(sym_view(b), j, k, c, s)(r, l));
        assert(rot_entry(sym_view(b), j, k, c, s, r, l) == sym_view(b)(r, l));
    }
}

} // verus!

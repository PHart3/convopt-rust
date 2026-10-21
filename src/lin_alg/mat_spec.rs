// Spec-level matrix algebra (not trusted).
//
// A matrix is viewed as an index function `MatF = spec_fn(int, int) -> real`
// whose dimensions are supplied separately by every operation and predicate.
// Stored matrices enter this algebra through views: `sym_view` (packed
// symmetric, see sym_spec.rs), `cm_view` (column-major), `lt_view` (packed
// lower-triangular) and `diag_view` (a diagonal given by its vector).  The
// algebra provides products, transposes, the identity, orthogonality,
// symmetry, eigenvalues (with similarity invariance), quadratic forms and
// the semidefinite order `A >= -eps I`, and the Frobenius product.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::seq::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::real_spec::abs_r;

verus! {

broadcast use crate::trusted::group_real_model;

pub type MatF = spec_fn(int, int) -> real;

/* ---------- views of stored matrices ---------- */

/// A packed symmetric matrix (sym_spec.rs) as an index function.
pub open spec fn sym_view(m: Seq<real>) -> MatF {
    |r: int, c: int| sym_entry(m, r, c)
}

/// A column-major matrix with row_dim rows as an index function.
pub open spec fn cm_view(m: Seq<real>, row_dim: int) -> MatF {
    |r: int, c: int| cm_entry(m, row_dim, r, c)
}

/// A packed lower-triangular matrix (vec_spec.rs) as an index function (zero above the diagonal).
pub open spec fn lt_view(m: Seq<real>, dim: int) -> MatF {
    |r: int, c: int| if c <= r { lt_entry(m, dim, r, c) } else { 0.0real }
}

/// The diagonal matrix with diagonal d.
pub open spec fn diag_view(d: Seq<real>) -> MatF {
    |r: int, c: int| if r == c { d[r] } else { 0.0real }
}

/// The block-diagonal matrix (A, I/2) whose leading n x n block is a.
pub open spec fn block_half_view(a: MatF, n: int) -> MatF {
    |r: int, c: int| if r < n && c < n { a(r, c) } else if r == c { 0.5real } else { 0.0real }
}

/// The diagonal block of a packed symmetric matrix starting at row/column `start`.
pub open spec fn block_view(m: Seq<real>, start: int) -> MatF {
    |r: int, c: int| sym_entry(m, start + r, start + c)
}

/// The diagonal block of an index-function matrix starting at row/column k.
pub open spec fn shift_view(a: MatF, k: int) -> MatF {
    |r: int, c: int| a(k + r, k + c)
}

/// x placed at offset start in a zero vector of length dim.
pub open spec fn pad_r(x: Seq<real>, dim: int, start: int) -> Seq<real> {
    Seq::new(dim as nat, |i: int| if start <= i < start + x.len() { x[i - start] } else { 0.0real })
}

/// The sum of the magnitudes of all entries of a packed matrix of order dim.
pub open spec fn abs_sum_r(m: Seq<real>, dim: int) -> real {
    sum2_fn_r(|r: int, c: int| abs_r(sym_entry(m, r, c)), dim as nat)
}

/* ---------- matrix operations ---------- */

/// Matrix product with inner dimension `inner`: (A B)_ij = sum_{k < inner} A_ik B_kj.
pub open spec fn mmul(a: MatF, b: MatF, inner: int) -> MatF {
    |i: int, j: int| sum_fn_r(|k: int| a(i, k) * b(k, j), inner as nat)
}

/// Transpose.
pub open spec fn mtr(a: MatF) -> MatF {
    |i: int, j: int| a(j, i)
}

/// Identity (of every dimension).
pub open spec fn mident() -> MatF {
    |i: int, j: int| if i == j { 1.0real } else { 0.0real }
}

pub open spec fn madd(a: MatF, b: MatF) -> MatF {
    |i: int, j: int| a(i, j) + b(i, j)
}

pub open spec fn msub(a: MatF, b: MatF) -> MatF {
    |i: int, j: int| a(i, j) - b(i, j)
}

pub open spec fn mscale(s: real, a: MatF) -> MatF {
    |i: int, j: int| s * a(i, j)
}

/// Matrix-vector product A x of a dim x dim matrix with a vector of length dim.
pub open spec fn mat_vec(a: MatF, dim: int, x: Seq<real>) -> Seq<real> {
    Seq::new(dim as nat, |i: int| sum_fn_r(|k: int| a(i, k) * x[k], dim as nat))
}

/* ---------- predicates ---------- */

/// a and b agree on the rows x cols block.
pub open spec fn meq(a: MatF, b: MatF, rows: int, cols: int) -> bool {
    forall|i: int, j: int| #![trigger a(i, j)] #![trigger b(i, j)]
        0 <= i < rows && 0 <= j < cols ==> a(i, j) == b(i, j)
}

pub open spec fn is_symmetric(a: MatF, dim: int) -> bool {
    forall|i: int, j: int| 0 <= i < dim && 0 <= j < dim ==> #[trigger] a(i, j) == a(j, i)
}

/// Q^T Q = I and Q Q^T = I on the dim x dim block.
pub open spec fn is_orthogonal(q: MatF, dim: int) -> bool {
    &&& meq(mmul(mtr(q), q, dim), mident(), dim, dim)
    &&& meq(mmul(q, mtr(q), dim), mident(), dim, dim)
}

/// P and Q are mutually inverse on the dim x dim block.
pub open spec fn are_inverse(p: MatF, q: MatF, dim: int) -> bool {
    &&& meq(mmul(p, q, dim), mident(), dim, dim)
    &&& meq(mmul(q, p, dim), mident(), dim, dim)
}

pub open spec fn is_zero_vec(x: Seq<real>) -> bool {
    forall|i: int| 0 <= i < x.len() ==> #[trigger] x[i] == 0.0real
}

/// lambda is an eigenvalue of the dim x dim matrix a: A x = lambda x for some nonzero x.
pub open spec fn is_eigenvalue(a: MatF, dim: int, lambda: real) -> bool {
    exists|x: Seq<real>| #![trigger mat_vec(a, dim, x)]
        x.len() == dim && !is_zero_vec(x) && mat_vec(a, dim, x) == vscale_r(lambda, x)
}

/// Quadratic form x^T A x.
pub open spec fn quad_r(a: MatF, dim: int, x: Seq<real>) -> real {
    dot_r(x, mat_vec(a, dim, x))
}

/// A >= -eps I in the semidefinite order: x^T A x + eps |x|^2 >= 0 for all x.
pub open spec fn psd_shifted(a: MatF, dim: int, eps: real) -> bool {
    forall|x: Seq<real>| x.len() == dim ==> #[trigger] quad_r(a, dim, x) + eps * dot_r(x, x) >= 0.0real
}

/// A is positive semidefinite.
pub open spec fn is_psd(a: MatF, dim: int) -> bool {
    psd_shifted(a, dim, 0.0real)
}

/// Frobenius inner product sum_{r, c < dim} A_rc B_rc.
pub open spec fn frob_f(a: MatF, b: MatF, dim: int) -> real {
    sum2_fn_r(|r: int, c: int| a(r, c) * b(r, c), dim as nat)
}

/* ---------- product algebra (entrywise) ---------- */

/// (A B) C = A (B C) entrywise, for inner dimensions n (A B) and p (B C).
pub proof fn lemma_mul_assoc(a: MatF, b: MatF, c: MatF, n: int, p: int, i: int, j: int)
    requires
        n >= 0,
        p >= 0,
    ensures
        mmul(mmul(a, b, n), c, p)(i, j) == mmul(a, mmul(b, c, p), n)(i, j),
{
    let t = |k: int, l: int| a(i, k) * b(k, l) * c(l, j);
    // left side: sum_l (sum_k a_ik b_kl) c_lj = sum_l sum_k t(k, l)
    let lhs = |l: int| mmul(a, b, n)(i, l) * c(l, j);
    let lhs_t = |l: int| sum_fn_r(|k: int| t(k, l), n as nat);
    assert forall|l: int| 0 <= l < p implies #[trigger] lhs(l) == lhs_t(l) by {
        let f = |k: int| a(i, k) * b(k, l);
        let g = |k: int| t(k, l);
        assert forall|k: int| 0 <= k < n implies #[trigger] g(k) == c(l, j) * f(k) by {
            let (x, y, z) = (a(i, k), b(k, l), c(l, j));
            assert(x * y * z == z * (x * y)) by (nonlinear_arith);
        }
        lemma_sum_fn_scale(f, g, c(l, j), n as nat);
        assert(lhs(l) == sum_fn_r(f, n as nat) * c(l, j));
    }
    lemma_sum_fn_ext(lhs, lhs_t, p as nat);
    // exchange the order of summation
    lemma_sum_fn_swap(t, n as nat, p as nat);
    // right side: sum_k a_ik (sum_l b_kl c_lj) = sum_k sum_l t(k, l)
    let rhs = |k: int| a(i, k) * mmul(b, c, p)(k, j);
    let rhs_t = |k: int| sum_fn_r(|l: int| t(k, l), p as nat);
    assert forall|k: int| 0 <= k < n implies #[trigger] rhs(k) == rhs_t(k) by {
        let f = |l: int| b(k, l) * c(l, j);
        let g = |l: int| t(k, l);
        assert forall|l: int| 0 <= l < p implies #[trigger] g(l) == a(i, k) * f(l) by {
            let (x, y, z) = (a(i, k), b(k, l), c(l, j));
            assert(x * y * z == x * (y * z)) by (nonlinear_arith);
        }
        lemma_sum_fn_scale(f, g, a(i, k), p as nat);
        assert(rhs(k) == a(i, k) * sum_fn_r(f, p as nat));
    }
    lemma_sum_fn_ext(rhs, rhs_t, n as nat);
}

/// (A B)^T = B^T A^T entrywise.
pub proof fn lemma_transpose_mul(a: MatF, b: MatF, n: int, i: int, j: int)
    requires
        n >= 0,
    ensures
        mtr(mmul(a, b, n))(i, j) == mmul(mtr(b), mtr(a), n)(i, j),
{
    let f = |k: int| a(j, k) * b(k, i);
    let g = |k: int| mtr(b)(i, k) * mtr(a)(k, j);
    assert forall|k: int| 0 <= k < n implies #[trigger] f(k) == g(k) by {
        assert(g(k) == b(k, i) * a(j, k));
    }
    lemma_sum_fn_ext(f, g, n as nat);
}

/// I A = A on rows < n.
pub proof fn lemma_mul_ident_left(a: MatF, n: int, i: int, j: int)
    requires
        0 <= i < n,
    ensures
        mmul(mident(), a, n)(i, j) == a(i, j),
{
    let f = |k: int| mident()(i, k) * a(k, j);
    assert forall|k: int| 0 <= k < n && k != i implies #[trigger] f(k) == 0.0real by {
        assert(mident()(i, k) == 0.0real);
    }
    lemma_sum_fn_single(f, n as nat, i);
    assert(f(i) == 1.0real * a(i, j));
}

/// A I = A on columns < n.
pub proof fn lemma_mul_ident_right(a: MatF, n: int, i: int, j: int)
    requires
        0 <= j < n,
    ensures
        mmul(a, mident(), n)(i, j) == a(i, j),
{
    let f = |k: int| a(i, k) * mident()(k, j);
    assert forall|k: int| 0 <= k < n && k != j implies #[trigger] f(k) == 0.0real by {
        assert(mident()(k, j) == 0.0real);
    }
    lemma_sum_fn_single(f, n as nat, j);
    assert(f(j) == a(i, j) * 1.0real);
}

/* ---------- product algebra (block equalities) ---------- */
//
// `meq` compares matrices on a finite block only, since index functions are
// unconstrained outside it.  The lemmas below restate the entrywise facts as
// block equalities so that proofs can chain them with `lemma_meq_trans`.

/// Block equality is transitive.
pub proof fn lemma_meq_trans(a: MatF, b: MatF, c: MatF, rows: int, cols: int)
    requires
        meq(a, b, rows, cols),
        meq(b, c, rows, cols),
    ensures
        meq(a, c, rows, cols),
{
}

/// Block equality is symmetric.
pub proof fn lemma_meq_sym(a: MatF, b: MatF, rows: int, cols: int)
    requires
        meq(a, b, rows, cols),
    ensures
        meq(b, a, rows, cols),
{
}

/// Products only depend on the factors' entries on the blocks involved:
/// if A = A' on rows x n and B = B' on n x cols then A B = A' B' on rows x cols.
pub proof fn lemma_mul_meq(a: MatF, a2: MatF, b: MatF, b2: MatF, rows: int, n: int, cols: int)
    requires
        n >= 0,
        meq(a, a2, rows, n),
        meq(b, b2, n, cols),
    ensures
        meq(mmul(a, b, n), mmul(a2, b2, n), rows, cols),
{
    assert forall|i: int, j: int| 0 <= i < rows && 0 <= j < cols
        implies #[trigger] mmul(a, b, n)(i, j) == mmul(a2, b2, n)(i, j) by {
        let f = |k: int| a(i, k) * b(k, j);
        let g = |k: int| a2(i, k) * b2(k, j);
        assert forall|k: int| 0 <= k < n implies #[trigger] f(k) == g(k) by {}
        lemma_sum_fn_ext(f, g, n as nat);
    }
}

/// The transpose respects block equality.
pub proof fn lemma_transpose_meq(a: MatF, b: MatF, rows: int, cols: int)
    requires
        meq(a, b, rows, cols),
    ensures
        meq(mtr(a), mtr(b), cols, rows),
{
    assert forall|i: int, j: int| 0 <= i < cols && 0 <= j < rows
        implies #[trigger] mtr(a)(i, j) == mtr(b)(i, j) by {
        assert(a(j, i) == b(j, i));
    }
}

/// (A B) C = A (B C) on any block.
pub proof fn lemma_mul_assoc_meq(a: MatF, b: MatF, c: MatF, n: int, p: int, rows: int, cols: int)
    requires
        n >= 0,
        p >= 0,
    ensures
        meq(mmul(mmul(a, b, n), c, p), mmul(a, mmul(b, c, p), n), rows, cols),
{
    assert forall|i: int, j: int| 0 <= i < rows && 0 <= j < cols
        implies #[trigger] mmul(mmul(a, b, n), c, p)(i, j) == mmul(a, mmul(b, c, p), n)(i, j) by {
        lemma_mul_assoc(a, b, c, n, p, i, j);
    }
}

/// (A B)^T = B^T A^T on any block.
pub proof fn lemma_transpose_mul_meq(a: MatF, b: MatF, n: int, rows: int, cols: int)
    requires
        n >= 0,
    ensures
        meq(mtr(mmul(a, b, n)), mmul(mtr(b), mtr(a), n), rows, cols),
{
    assert forall|i: int, j: int| 0 <= i < rows && 0 <= j < cols
        implies #[trigger] mtr(mmul(a, b, n))(i, j) == mmul(mtr(b), mtr(a), n)(i, j) by {
        lemma_transpose_mul(a, b, n, i, j);
    }
}

/// I A = A on the n x cols block.
pub proof fn lemma_mul_ident_left_meq(a: MatF, n: int, cols: int)
    ensures
        meq(mmul(mident(), a, n), a, n, cols),
{
    assert forall|i: int, j: int| 0 <= i < n && 0 <= j < cols
        implies #[trigger] mmul(mident(), a, n)(i, j) == a(i, j) by {
        lemma_mul_ident_left(a, n, i, j);
    }
}

/// A I = A on the rows x n block.
pub proof fn lemma_mul_ident_right_meq(a: MatF, n: int, rows: int)
    ensures
        meq(mmul(a, mident(), n), a, rows, n),
{
    assert forall|i: int, j: int| 0 <= i < rows && 0 <= j < n
        implies #[trigger] mmul(a, mident(), n)(i, j) == a(i, j) by {
        lemma_mul_ident_right(a, n, i, j);
    }
}

/// (A^T)^T = A on any block.
pub proof fn lemma_transpose_twice_meq(a: MatF, rows: int, cols: int)
    ensures
        meq(mtr(mtr(a)), a, rows, cols),
{
    assert forall|i: int, j: int| 0 <= i < rows && 0 <= j < cols
        implies #[trigger] mtr(mtr(a))(i, j) == a(i, j) by {}
}

/// I^T = I on any block.
pub proof fn lemma_transpose_ident_meq(rows: int, cols: int)
    ensures
        meq(mtr(mident()), mident(), rows, cols),
{
    assert forall|i: int, j: int| 0 <= i < rows && 0 <= j < cols
        implies #[trigger] mtr(mident())(i, j) == mident()(i, j) by {}
}

/* ---------- orthogonality ---------- */

pub proof fn lemma_orthogonal_ident(dim: int)
    requires
        dim >= 0,
    ensures
        is_orthogonal(mident(), dim),
{
    lemma_transpose_ident_meq(dim, dim);
    lemma_mul_meq(mtr(mident()), mident(), mident(), mident(), dim, dim, dim);
    lemma_mul_meq(mident(), mident(), mtr(mident()), mident(), dim, dim, dim);
    lemma_mul_ident_left_meq(mident(), dim, dim);
    lemma_meq_trans(mmul(mtr(mident()), mident(), dim), mmul(mident(), mident(), dim), mident(), dim, dim);
    lemma_meq_trans(mmul(mident(), mtr(mident()), dim), mmul(mident(), mident(), dim), mident(), dim, dim);
}

/// The transpose of an orthogonal matrix is orthogonal.
pub proof fn lemma_orthogonal_transpose(q: MatF, dim: int)
    requires
        dim >= 0,
        is_orthogonal(q, dim),
    ensures
        is_orthogonal(mtr(q), dim),
{
    lemma_transpose_twice_meq(q, dim, dim);
    lemma_mul_meq(mtr(mtr(q)), q, mtr(q), mtr(q), dim, dim, dim);
    lemma_meq_trans(mmul(mtr(mtr(q)), mtr(q), dim), mmul(q, mtr(q), dim), mident(), dim, dim);
    lemma_mul_meq(mtr(q), mtr(q), mtr(mtr(q)), q, dim, dim, dim);
    lemma_meq_trans(mmul(mtr(q), mtr(mtr(q)), dim), mmul(mtr(q), q, dim), mident(), dim, dim);
}

/// If A B = I on the block then (A B) C = C and C (A B) = C on the block (a cancellation step).
pub proof fn lemma_cancel_inverse(a: MatF, b: MatF, c: MatF, dim: int)
    requires
        dim >= 0,
        meq(mmul(a, b, dim), mident(), dim, dim),
    ensures
        meq(mmul(a, mmul(b, c, dim), dim), c, dim, dim),
        meq(mmul(mmul(c, a, dim), b, dim), c, dim, dim),
{
    lemma_mul_assoc_meq(a, b, c, dim, dim, dim, dim);
    lemma_mul_meq(mmul(a, b, dim), mident(), c, c, dim, dim, dim);
    lemma_mul_ident_left_meq(c, dim, dim);
    lemma_meq_sym(mmul(mmul(a, b, dim), c, dim), mmul(a, mmul(b, c, dim), dim), dim, dim);
    lemma_meq_trans(mmul(a, mmul(b, c, dim), dim), mmul(mmul(a, b, dim), c, dim), mmul(mident(), c, dim), dim, dim);
    lemma_meq_trans(mmul(a, mmul(b, c, dim), dim), mmul(mident(), c, dim), c, dim, dim);
    lemma_mul_assoc_meq(c, a, b, dim, dim, dim, dim);
    lemma_mul_meq(c, c, mmul(a, b, dim), mident(), dim, dim, dim);
    lemma_mul_ident_right_meq(c, dim, dim);
    lemma_meq_trans(mmul(mmul(c, a, dim), b, dim), mmul(c, mmul(a, b, dim), dim), mmul(c, mident(), dim), dim, dim);
    lemma_meq_trans(mmul(mmul(c, a, dim), b, dim), mmul(c, mident(), dim), c, dim, dim);
}

/// Products of orthogonal matrices are orthogonal.
pub proof fn lemma_orthogonal_mul(p: MatF, q: MatF, dim: int)
    requires
        dim >= 0,
        is_orthogonal(p, dim),
        is_orthogonal(q, dim),
    ensures
        is_orthogonal(mmul(p, q, dim), dim),
{
    let pq = mmul(p, q, dim);
    let qtpt = mmul(mtr(q), mtr(p), dim);
    lemma_transpose_mul_meq(p, q, dim, dim, dim);
    // (P Q)^T (P Q) = Q^T (P^T (P Q)) = Q^T Q = I
    lemma_mul_meq(mtr(pq), qtpt, pq, pq, dim, dim, dim);
    lemma_mul_assoc_meq(mtr(q), mtr(p), pq, dim, dim, dim, dim);
    lemma_cancel_inverse(mtr(p), p, q, dim);
    lemma_mul_meq(mtr(q), mtr(q), mmul(mtr(p), pq, dim), q, dim, dim, dim);
    lemma_meq_trans(mmul(mtr(pq), pq, dim), mmul(qtpt, pq, dim), mmul(mtr(q), mmul(mtr(p), pq, dim), dim), dim, dim);
    lemma_meq_trans(mmul(mtr(pq), pq, dim), mmul(mtr(q), mmul(mtr(p), pq, dim), dim), mmul(mtr(q), q, dim), dim, dim);
    lemma_meq_trans(mmul(mtr(pq), pq, dim), mmul(mtr(q), q, dim), mident(), dim, dim);
    // (P Q) (P Q)^T = P (Q (Q^T P^T)) = P P^T = I
    lemma_mul_meq(pq, pq, mtr(pq), qtpt, dim, dim, dim);
    lemma_mul_assoc_meq(p, q, qtpt, dim, dim, dim, dim);
    lemma_cancel_inverse(q, mtr(q), mtr(p), dim);
    lemma_mul_meq(p, p, mmul(q, qtpt, dim), mtr(p), dim, dim, dim);
    lemma_meq_trans(mmul(pq, mtr(pq), dim), mmul(pq, qtpt, dim), mmul(p, mmul(q, qtpt, dim), dim), dim, dim);
    lemma_meq_trans(mmul(pq, mtr(pq), dim), mmul(p, mmul(q, qtpt, dim), dim), mmul(p, mtr(p), dim), dim, dim);
    lemma_meq_trans(mmul(pq, mtr(pq), dim), mmul(p, mtr(p), dim), mident(), dim, dim);
}

/// An orthogonal matrix and its transpose are mutually inverse.
pub proof fn lemma_orthogonal_inverse(q: MatF, dim: int)
    requires
        is_orthogonal(q, dim),
    ensures
        are_inverse(q, mtr(q), dim),
        are_inverse(mtr(q), q, dim),
{
}

/* ---------- symmetry ---------- */

/// A symmetric matrix equals its transpose on the block.
pub proof fn lemma_symmetric_transpose_meq(a: MatF, dim: int)
    requires
        is_symmetric(a, dim),
    ensures
        meq(mtr(a), a, dim, dim),
{
    assert forall|i: int, j: int| 0 <= i < dim && 0 <= j < dim implies #[trigger] mtr(a)(i, j) == a(i, j) by {
        assert(a(j, i) == a(i, j));
    }
}

/// The congruence P^T A P of a symmetric A is symmetric.
pub proof fn lemma_congruence_symmetric(a: MatF, p: MatF, dim: int)
    requires
        dim >= 0,
        is_symmetric(a, dim),
    ensures
        is_symmetric(mmul(mmul(mtr(p), a, dim), p, dim), dim),
{
    let pta = mmul(mtr(p), a, dim);
    let b = mmul(pta, p, dim);
    // B^T = P^T (P^T A)^T = P^T (A^T P) = P^T (A P) = B
    lemma_transpose_mul_meq(pta, p, dim, dim, dim);
    lemma_transpose_mul_meq(mtr(p), a, dim, dim, dim);
    lemma_symmetric_transpose_meq(a, dim);
    lemma_transpose_twice_meq(p, dim, dim);
    lemma_mul_meq(mtr(a), a, mtr(mtr(p)), p, dim, dim, dim);
    lemma_meq_trans(mtr(pta), mmul(mtr(a), mtr(mtr(p)), dim), mmul(a, p, dim), dim, dim);
    lemma_mul_meq(mtr(p), mtr(p), mtr(pta), mmul(a, p, dim), dim, dim, dim);
    lemma_mul_assoc_meq(mtr(p), a, p, dim, dim, dim, dim);
    lemma_meq_sym(b, mmul(mtr(p), mmul(a, p, dim), dim), dim, dim);
    lemma_meq_trans(mtr(b), mmul(mtr(p), mtr(pta), dim), mmul(mtr(p), mmul(a, p, dim), dim), dim, dim);
    lemma_meq_trans(mtr(b), mmul(mtr(p), mmul(a, p, dim), dim), b, dim, dim);
    assert forall|i: int, j: int| 0 <= i < dim && 0 <= j < dim implies #[trigger] b(i, j) == b(j, i) by {
        assert(mtr(b)(j, i) == b(j, i));
    }
}

/* ---------- matrix-vector products ---------- */

/// (A B) x = A (B x).
pub proof fn lemma_mat_vec_mul(a: MatF, b: MatF, dim: int, x: Seq<real>)
    requires
        dim >= 0,
        x.len() == dim,
    ensures
        mat_vec(mmul(a, b, dim), dim, x) == mat_vec(a, dim, mat_vec(b, dim, x)),
{
    let cx = |k: int, l: int| x[k];  // x as a column matrix
    let bx = mat_vec(b, dim, x);
    let ab = mmul(a, b, dim);
    assert forall|i: int| 0 <= i < dim implies #[trigger] mat_vec(ab, dim, x)[i] == mat_vec(a, dim, bx)[i] by {
        let f = |k: int| ab(i, k) * x[k];
        let g = |k: int| ab(i, k) * cx(k, 0);
        assert forall|k: int| 0 <= k < dim implies #[trigger] f(k) == g(k) by {}
        lemma_sum_fn_ext(f, g, dim as nat);
        lemma_mul_assoc(a, b, cx, dim, dim, i, 0);
        let bcx = mmul(b, cx, dim);
        assert forall|k: int| 0 <= k < dim implies #[trigger] bcx(k, 0) == bx[k] by {
            let f2 = |l: int| b(k, l) * cx(l, 0);
            let g2 = |l: int| b(k, l) * x[l];
            assert forall|l: int| 0 <= l < dim implies #[trigger] f2(l) == g2(l) by {}
            lemma_sum_fn_ext(f2, g2, dim as nat);
        }
        let f3 = |k: int| a(i, k) * bcx(k, 0);
        let g3 = |k: int| a(i, k) * bx[k];
        assert forall|k: int| 0 <= k < dim implies #[trigger] f3(k) == g3(k) by {}
        lemma_sum_fn_ext(f3, g3, dim as nat);
    }
    assert(mat_vec(ab, dim, x) =~= mat_vec(a, dim, bx));
}

/// A (s x) = s (A x).
pub proof fn lemma_mat_vec_scale(a: MatF, dim: int, s: real, x: Seq<real>)
    requires
        dim >= 0,
        x.len() == dim,
    ensures
        mat_vec(a, dim, vscale_r(s, x)) == vscale_r(s, mat_vec(a, dim, x)),
{
    let sx = vscale_r(s, x);
    assert forall|i: int| 0 <= i < dim implies #[trigger] mat_vec(a, dim, sx)[i] == vscale_r(s, mat_vec(a, dim, x))[i] by {
        let f = |k: int| a(i, k) * x[k];
        let g = |k: int| a(i, k) * sx[k];
        assert forall|k: int| 0 <= k < dim implies #[trigger] g(k) == s * f(k) by {
            let (u, v) = (a(i, k), x[k]);
            assert(u * (s * v) == s * (u * v)) by (nonlinear_arith);
        }
        lemma_sum_fn_scale(f, g, s, dim as nat);
    }
    assert(mat_vec(a, dim, sx) =~= vscale_r(s, mat_vec(a, dim, x)));
}

/// I x = x.
pub proof fn lemma_mat_vec_ident(dim: int, x: Seq<real>)
    requires
        x.len() == dim,
    ensures
        mat_vec(mident(), dim, x) == x,
{
    assert forall|i: int| 0 <= i < dim implies #[trigger] mat_vec(mident(), dim, x)[i] == x[i] by {
        let f = |k: int| mident()(i, k) * x[k];
        assert forall|k: int| 0 <= k < dim && k != i implies #[trigger] f(k) == 0.0real by {
            assert(mident()(i, k) == 0.0real);
        }
        lemma_sum_fn_single(f, dim as nat, i);
        assert(f(i) == 1.0real * x[i]);
    }
    assert(mat_vec(mident(), dim, x) =~= x);
}

/// A 0 = 0.
pub proof fn lemma_mat_vec_zero(a: MatF, dim: int, x: Seq<real>)
    requires
        x.len() == dim,
        is_zero_vec(x),
    ensures
        is_zero_vec(mat_vec(a, dim, x)),
{
    assert forall|i: int| 0 <= i < dim implies #[trigger] mat_vec(a, dim, x)[i] == 0.0real by {
        let f = |k: int| a(i, k) * x[k];
        let z = |k: int| 0.0real;
        assert forall|k: int| 0 <= k < dim implies #[trigger] f(k) == z(k) by {}
        lemma_sum_fn_ext(f, z, dim as nat);
        lemma_sum_fn_zero(dim as nat);
    }
}

/// A = B on the block implies A x = B x.
pub proof fn lemma_mat_vec_meq(a: MatF, b: MatF, dim: int, x: Seq<real>)
    requires
        dim >= 0,
        x.len() == dim,
        meq(a, b, dim, dim),
    ensures
        mat_vec(a, dim, x) == mat_vec(b, dim, x),
{
    assert forall|i: int| 0 <= i < dim implies #[trigger] mat_vec(a, dim, x)[i] == mat_vec(b, dim, x)[i] by {
        let f = |k: int| a(i, k) * x[k];
        let g = |k: int| b(i, k) * x[k];
        assert forall|k: int| 0 <= k < dim implies #[trigger] f(k) == g(k) by {}
        lemma_sum_fn_ext(f, g, dim as nat);
    }
    assert(mat_vec(a, dim, x) =~= mat_vec(b, dim, x));
}

/// (A + s I) x = A x + s x.
pub proof fn lemma_mat_vec_shift(a: MatF, dim: int, s: real, x: Seq<real>)
    requires
        dim >= 0,
        x.len() == dim,
    ensures
        mat_vec(madd(a, mscale(s, mident())), dim, x) == vadd_r(mat_vec(a, dim, x), vscale_r(s, x)),
{
    let sh = madd(a, mscale(s, mident()));
    assert forall|i: int| 0 <= i < dim
        implies #[trigger] mat_vec(sh, dim, x)[i] == vadd_r(mat_vec(a, dim, x), vscale_r(s, x))[i] by {
        let f = |k: int| a(i, k) * x[k];
        let g = |k: int| mscale(s, mident())(i, k) * x[k];
        let h = |k: int| sh(i, k) * x[k];
        assert forall|k: int| 0 <= k < dim implies #[trigger] h(k) == f(k) + g(k) by {
            let (u, v, w) = (a(i, k), mscale(s, mident())(i, k), x[k]);
            assert(sh(i, k) == u + v);
            assert((u + v) * w == u * w + v * w) by (nonlinear_arith);
        }
        lemma_sum_fn_add(f, g, h, dim as nat);
        assert forall|k: int| 0 <= k < dim && k != i implies #[trigger] g(k) == 0.0real by {
            assert(mident()(i, k) == 0.0real);
        }
        lemma_sum_fn_single(g, dim as nat, i);
        assert(g(i) == (s * 1.0real) * x[i]);
    }
    assert(mat_vec(sh, dim, x) =~= vadd_r(mat_vec(a, dim, x), vscale_r(s, x)));
}

/* ---------- eigenvalues ---------- */

/// Eigenvalues only depend on the entries on the block.
pub proof fn lemma_eigenvalue_meq(a: MatF, b: MatF, dim: int, lambda: real)
    requires
        dim >= 0,
        meq(a, b, dim, dim),
        is_eigenvalue(a, dim, lambda),
    ensures
        is_eigenvalue(b, dim, lambda),
{
    let x = choose|x: Seq<real>| #![trigger mat_vec(a, dim, x)]
        x.len() == dim && !is_zero_vec(x) && mat_vec(a, dim, x) == vscale_r(lambda, x);
    lemma_mat_vec_meq(a, b, dim, x);
}

/// Similarity preserves eigenvalues: if P Q = Q P = I on the block and lambda is an
/// eigenvalue of A, then lambda is an eigenvalue of Q A P.
pub proof fn lemma_similarity_preserves_eigenvalues(a: MatF, p: MatF, q: MatF, dim: int, lambda: real)
    requires
        dim >= 0,
        are_inverse(p, q, dim),
        is_eigenvalue(a, dim, lambda),
    ensures
        is_eigenvalue(mmul(mmul(q, a, dim), p, dim), dim, lambda),
{
    let x = choose|x: Seq<real>| #![trigger mat_vec(a, dim, x)]
        x.len() == dim && !is_zero_vec(x) && mat_vec(a, dim, x) == vscale_r(lambda, x);
    let y = mat_vec(q, dim, x);
    let b = mmul(mmul(q, a, dim), p, dim);
    // P y = (P Q) x = x
    lemma_mat_vec_mul(p, q, dim, x);
    lemma_mat_vec_meq(mmul(p, q, dim), mident(), dim, x);
    lemma_mat_vec_ident(dim, x);
    assert(mat_vec(p, dim, y) == x);
    // B y = (Q A) (P y) = Q (A x) = Q (lambda x) = lambda y
    lemma_mat_vec_mul(mmul(q, a, dim), p, dim, y);
    lemma_mat_vec_mul(q, a, dim, x);
    lemma_mat_vec_scale(q, dim, lambda, x);
    assert(mat_vec(b, dim, y) == vscale_r(lambda, y));
    // y is nonzero since P y = x is nonzero
    if is_zero_vec(y) {
        lemma_mat_vec_zero(p, dim, y);
        assert(false);
    }
}

/// For orthogonal Q, the matrices A and Q^T A Q have the same eigenvalues.
pub proof fn lemma_orthogonal_similarity_eigenvalues(a: MatF, q: MatF, dim: int, lambda: real)
    requires
        dim >= 0,
        is_orthogonal(q, dim),
    ensures
        is_eigenvalue(a, dim, lambda) <==> is_eigenvalue(mmul(mmul(mtr(q), a, dim), q, dim), dim, lambda),
{
    let b = mmul(mmul(mtr(q), a, dim), q, dim);
    if is_eigenvalue(a, dim, lambda) {
        lemma_similarity_preserves_eigenvalues(a, q, mtr(q), dim, lambda);
    }
    if is_eigenvalue(b, dim, lambda) {
        // Q B Q^T = (Q Q^T) A (Q Q^T) = A on the block
        lemma_similarity_preserves_eigenvalues(b, mtr(q), q, dim, lambda);
        let qb = mmul(q, b, dim);
        lemma_cancel_inverse(q, mtr(q), mmul(a, q, dim), dim);
        lemma_mul_assoc_meq(mtr(q), a, q, dim, dim, dim, dim);
        lemma_mul_meq(q, q, mmul(mmul(mtr(q), a, dim), q, dim), mmul(mtr(q), mmul(a, q, dim), dim), dim, dim, dim);
        lemma_meq_trans(qb, mmul(q, mmul(mtr(q), mmul(a, q, dim), dim), dim), mmul(a, q, dim), dim, dim);
        lemma_mul_meq(qb, mmul(a, q, dim), mtr(q), mtr(q), dim, dim, dim);
        lemma_cancel_inverse(q, mtr(q), a, dim);
        lemma_meq_trans(mmul(qb, mtr(q), dim), mmul(mmul(a, q, dim), mtr(q), dim), a, dim, dim);
        lemma_eigenvalue_meq(mmul(qb, mtr(q), dim), a, dim, lambda);
    }
}

/* ---------- quadratic forms and the semidefinite order ---------- */

/// The inner product as an index sum.
pub proof fn lemma_dot_sum_fn(x: Seq<real>, y: Seq<real>)
    requires
        x.len() == y.len(),
    ensures
        dot_r(x, y) == sum_fn_r(|i: int| x[i] * y[i], x.len()),
{
    lemma_sum_fn_seq(|i: int| x[i] * y[i], x.len());
    assert(vmul_r(x, y) =~= Seq::new(x.len(), |i: int| x[i] * y[i]));
}

/// <x, y + z> = <x, y> + <x, z>.
pub proof fn lemma_dot_add_right(x: Seq<real>, y: Seq<real>, z: Seq<real>)
    requires
        x.len() == y.len(),
        x.len() == z.len(),
    ensures
        dot_r(x, vadd_r(y, z)) == dot_r(x, y) + dot_r(x, z),
{
    let n = x.len();
    lemma_dot_sum_fn(x, vadd_r(y, z));
    lemma_dot_sum_fn(x, y);
    lemma_dot_sum_fn(x, z);
    let f = |i: int| x[i] * y[i];
    let g = |i: int| x[i] * z[i];
    let h = |i: int| x[i] * vadd_r(y, z)[i];
    assert forall|i: int| 0 <= i < n implies #[trigger] h(i) == f(i) + g(i) by {
        let (u, v, w) = (x[i], y[i], z[i]);
        assert(u * (v + w) == u * v + u * w) by (nonlinear_arith);
    }
    lemma_sum_fn_add(f, g, h, n);
}

/// <x, s y> = s <x, y>.
pub proof fn lemma_dot_scale_right(x: Seq<real>, s: real, y: Seq<real>)
    requires
        x.len() == y.len(),
    ensures
        dot_r(x, vscale_r(s, y)) == s * dot_r(x, y),
{
    let n = x.len();
    lemma_dot_sum_fn(x, vscale_r(s, y));
    lemma_dot_sum_fn(x, y);
    let f = |i: int| x[i] * y[i];
    let g = |i: int| x[i] * vscale_r(s, y)[i];
    assert forall|i: int| 0 <= i < n implies #[trigger] g(i) == s * f(i) by {
        let (u, v) = (x[i], y[i]);
        assert(u * (s * v) == s * (u * v)) by (nonlinear_arith);
    }
    lemma_sum_fn_scale(f, g, s, n);
}

/// <x, A y> = <A^T x, y>.
pub proof fn lemma_dot_mat_vec_transpose(a: MatF, dim: int, x: Seq<real>, y: Seq<real>)
    requires
        dim >= 0,
        x.len() == dim,
        y.len() == dim,
    ensures
        dot_r(x, mat_vec(a, dim, y)) == dot_r(mat_vec(mtr(a), dim, x), y),
{
    let n = dim as nat;
    let ay = mat_vec(a, dim, y);
    let atx = mat_vec(mtr(a), dim, x);
    lemma_dot_sum_fn(x, ay);
    lemma_dot_sum_fn(atx, y);
    let t = |i: int, k: int| x[i] * (a(i, k) * y[k]);
    // <x, A y> = sum_i x_i (sum_k a_ik y_k) = sum_i sum_k t(i, k)
    let lhs = |i: int| x[i] * ay[i];
    let lhs_t = |i: int| sum_fn_r(|k: int| t(i, k), n);
    assert forall|i: int| 0 <= i < dim implies #[trigger] lhs(i) == lhs_t(i) by {
        let f = |k: int| a(i, k) * y[k];
        let g = |k: int| t(i, k);
        assert forall|k: int| 0 <= k < dim implies #[trigger] g(k) == x[i] * f(k) by {}
        lemma_sum_fn_scale(f, g, x[i], n);
    }
    lemma_sum_fn_ext(lhs, lhs_t, n);
    lemma_sum_fn_swap(t, n, n);
    // <A^T x, y> = sum_k (sum_i a_ik x_i) y_k = sum_k sum_i t(i, k)
    let rhs = |k: int| atx[k] * y[k];
    let rhs_t = |k: int| sum_fn_r(|i: int| t(i, k), n);
    assert forall|k: int| 0 <= k < dim implies #[trigger] rhs(k) == rhs_t(k) by {
        let f = |i: int| mtr(a)(k, i) * x[i];
        let g = |i: int| t(i, k);
        assert forall|i: int| 0 <= i < dim implies #[trigger] g(i) == y[k] * f(i) by {
            let (u, v, w) = (x[i], a(i, k), y[k]);
            assert(f(i) == v * u);
            assert(u * (v * w) == w * (v * u)) by (nonlinear_arith);
        }
        lemma_sum_fn_scale(f, g, y[k], n);
        assert(atx[k] == sum_fn_r(f, n));
        assert(rhs_t(k) == sum_fn_r(g, n));
        assert(rhs(k) == sum_fn_r(f, n) * y[k]);
    }
    lemma_sum_fn_ext(rhs, rhs_t, n);
}

/// x^T (P^T A P) x = (P x)^T A (P x).
pub proof fn lemma_quad_congruence(a: MatF, p: MatF, dim: int, x: Seq<real>)
    requires
        dim >= 0,
        x.len() == dim,
    ensures
        quad_r(mmul(mmul(mtr(p), a, dim), p, dim), dim, x) == quad_r(a, dim, mat_vec(p, dim, x)),
{
    let px = mat_vec(p, dim, x);
    let apx = mat_vec(a, dim, px);
    lemma_mat_vec_mul(mmul(mtr(p), a, dim), p, dim, x);
    lemma_mat_vec_mul(mtr(p), a, dim, px);
    lemma_dot_mat_vec_transpose(mtr(p), dim, x, apx);
    lemma_transpose_twice_meq(p, dim, dim);
    lemma_mat_vec_meq(mtr(mtr(p)), p, dim, x);
}

/// Quadratic forms only depend on the entries on the block.
pub proof fn lemma_quad_meq(a: MatF, b: MatF, dim: int, x: Seq<real>)
    requires
        dim >= 0,
        x.len() == dim,
        meq(a, b, dim, dim),
    ensures
        quad_r(a, dim, x) == quad_r(b, dim, x),
{
    lemma_mat_vec_meq(a, b, dim, x);
}

/// x^T (A + s I) x = x^T A x + s |x|^2.
pub proof fn lemma_quad_shift(a: MatF, dim: int, s: real, x: Seq<real>)
    requires
        dim >= 0,
        x.len() == dim,
    ensures
        quad_r(madd(a, mscale(s, mident())), dim, x) == quad_r(a, dim, x) + s * dot_r(x, x),
{
    lemma_mat_vec_shift(a, dim, s, x);
    lemma_dot_add_right(x, mat_vec(a, dim, x), vscale_r(s, x));
    lemma_dot_scale_right(x, s, x);
}

/// x^T D x = sum_k d_k x_k^2 for the diagonal matrix D = diag(d).
pub proof fn lemma_quad_diag(d: Seq<real>, dim: int, x: Seq<real>)
    requires
        x.len() == dim,
    ensures
        quad_r(diag_view(d), dim, x) == sum_fn_r(|k: int| d[k] * x[k] * x[k], dim as nat),
{
    let dx = mat_vec(diag_view(d), dim, x);
    assert forall|i: int| 0 <= i < dim implies #[trigger] dx[i] == d[i] * x[i] by {
        let f = |k: int| diag_view(d)(i, k) * x[k];
        assert forall|k: int| 0 <= k < dim && k != i implies #[trigger] f(k) == 0.0real by {
            assert(diag_view(d)(i, k) == 0.0real);
        }
        lemma_sum_fn_single(f, dim as nat, i);
    }
    lemma_dot_sum_fn(x, dx);
    let f = |i: int| x[i] * dx[i];
    let g = |k: int| d[k] * x[k] * x[k];
    assert forall|i: int| 0 <= i < dim implies #[trigger] f(i) == g(i) by {
        let (u, v) = (x[i], d[i]);
        assert(u * (v * u) == v * u * u) by (nonlinear_arith);
    }
    lemma_sum_fn_ext(f, g, dim as nat);
}

/// If every diagonal entry is at least m then x^T D x >= m |x|^2.
pub proof fn lemma_quad_diag_lower_bound(d: Seq<real>, dim: int, x: Seq<real>, m: real)
    requires
        x.len() == dim,
        forall|k: int| 0 <= k < dim ==> #[trigger] d[k] >= m,
    ensures
        quad_r(diag_view(d), dim, x) >= m * dot_r(x, x),
{
    lemma_quad_diag(d, dim, x);
    lemma_dot_sum_fn(x, x);
    let sq = |k: int| x[k] * x[k];
    let msq = |k: int| m * (x[k] * x[k]);
    let g = |k: int| d[k] * x[k] * x[k];
    assert forall|k: int| 0 <= k < dim implies #[trigger] msq(k) == m * sq(k) by {}
    lemma_sum_fn_scale(sq, msq, m, dim as nat);
    assert forall|k: int| 0 <= k < dim implies #[trigger] msq(k) <= g(k) by {
        let (u, v) = (x[k], d[k]);
        assert(m * (u * u) <= v * u * u) by (nonlinear_arith)
            requires v >= m;
    }
    lemma_sum_fn_le(msq, g, dim as nat);
}

/// diag(d) with all d_k >= 0 is positive semidefinite.
pub proof fn lemma_diag_psd(d: Seq<real>, dim: int)
    requires
        forall|k: int| 0 <= k < dim ==> #[trigger] d[k] >= 0.0real,
    ensures
        is_psd(diag_view(d), dim),
{
    assert forall|x: Seq<real>| x.len() == dim
        implies #[trigger] quad_r(diag_view(d), dim, x) + 0.0real * dot_r(x, x) >= 0.0real by {
        lemma_quad_diag_lower_bound(d, dim, x, 0.0real);
        assert(0.0real * dot_r(x, x) == 0.0real);
    }
}

/// Positive semidefiniteness is preserved by congruence: A >= 0 implies P^T A P >= 0.
pub proof fn lemma_psd_congruence(a: MatF, p: MatF, dim: int)
    requires
        dim >= 0,
        is_psd(a, dim),
    ensures
        is_psd(mmul(mmul(mtr(p), a, dim), p, dim), dim),
{
    let b = mmul(mmul(mtr(p), a, dim), p, dim);
    assert forall|x: Seq<real>| x.len() == dim
        implies #[trigger] quad_r(b, dim, x) + 0.0real * dot_r(x, x) >= 0.0real by {
        lemma_quad_congruence(a, p, dim, x);
        let px = mat_vec(p, dim, x);
        assert(quad_r(a, dim, px) + 0.0real * dot_r(px, px) >= 0.0real);
        assert(0.0real * dot_r(x, x) == 0.0real);
        assert(0.0real * dot_r(px, px) == 0.0real);
    }
}

/// The semidefinite order only depends on the entries on the block.
pub proof fn lemma_psd_shifted_meq(a: MatF, b: MatF, dim: int, eps: real)
    requires
        dim >= 0,
        meq(a, b, dim, dim),
        psd_shifted(a, dim, eps),
    ensures
        psd_shifted(b, dim, eps),
{
    assert forall|x: Seq<real>| x.len() == dim
        implies #[trigger] quad_r(b, dim, x) + eps * dot_r(x, x) >= 0.0real by {
        lemma_quad_meq(a, b, dim, x);
        assert(quad_r(a, dim, x) + eps * dot_r(x, x) >= 0.0real);
    }
}

/// Every matrix of dimension 0 satisfies A >= -eps I (there is only the empty vector).
pub proof fn lemma_psd_shifted_zero_dim(a: MatF, eps: real)
    ensures
        psd_shifted(a, 0, eps),
{
    assert forall|x: Seq<real>| x.len() == 0
        implies #[trigger] quad_r(a, 0, x) + eps * dot_r(x, x) >= 0.0real by {
        assert(vmul_r(x, mat_vec(a, 0, x)).len() == 0);
        assert(vmul_r(x, x).len() == 0);
        assert(quad_r(a, 0, x) == 0.0real);
        assert(dot_r(x, x) == 0.0real);
    }
}

/// Q D Q^T with all d_k >= 0 is positive semidefinite (for any Q).
pub proof fn lemma_diag_congruence_psd(q: MatF, d: Seq<real>, dim: int)
    requires
        dim >= 0,
        forall|k: int| 0 <= k < dim ==> #[trigger] d[k] >= 0.0real,
    ensures
        is_psd(mmul(mmul(q, diag_view(d), dim), mtr(q), dim), dim),
{
    let dv = diag_view(d);
    lemma_diag_psd(d, dim);
    // (Q^T)^T D Q^T >= 0, and (Q^T)^T = Q on the block
    lemma_psd_congruence(dv, mtr(q), dim);
    lemma_transpose_twice_meq(q, dim, dim);
    lemma_mul_meq(mtr(mtr(q)), q, dv, dv, dim, dim, dim);
    lemma_mul_meq(mmul(mtr(mtr(q)), dv, dim), mmul(q, dv, dim), mtr(q), mtr(q), dim, dim, dim);
    lemma_psd_shifted_meq(mmul(mmul(mtr(mtr(q)), dv, dim), mtr(q), dim), mmul(mmul(q, dv, dim), mtr(q), dim), dim, 0.0real);
}

/// |Q^T x|^2 = |x|^2 for orthogonal Q.
pub proof fn lemma_orthogonal_norm(q: MatF, dim: int, x: Seq<real>)
    requires
        dim >= 0,
        x.len() == dim,
        is_orthogonal(q, dim),
    ensures
        dot_r(mat_vec(mtr(q), dim, x), mat_vec(mtr(q), dim, x)) == dot_r(x, x),
{
    let qtx = mat_vec(mtr(q), dim, x);
    // <x, Q (Q^T x)> = <Q^T x, Q^T x>
    lemma_dot_mat_vec_transpose(q, dim, x, qtx);
    // Q (Q^T x) = (Q Q^T) x = I x = x
    lemma_mat_vec_mul(q, mtr(q), dim, x);
    lemma_mat_vec_meq(mmul(q, mtr(q), dim), mident(), dim, x);
    lemma_mat_vec_ident(dim, x);
}

/// Q D Q^T >= -eps I for orthogonal Q and all d_k >= -eps:
/// x^T Q D Q^T x = y^T D y >= -eps |y|^2 = -eps |x|^2 with y = Q^T x.
pub proof fn lemma_diag_congruence_psd_shifted(q: MatF, d: Seq<real>, dim: int, eps: real)
    requires
        dim >= 0,
        is_orthogonal(q, dim),
        forall|k: int| 0 <= k < dim ==> #[trigger] d[k] >= -eps,
    ensures
        psd_shifted(mmul(mmul(q, diag_view(d), dim), mtr(q), dim), dim, eps),
{
    let dv = diag_view(d);
    let a = mmul(mmul(q, dv, dim), mtr(q), dim);
    // (Q^T)^T D Q^T = Q D Q^T on the block
    let a2 = mmul(mmul(mtr(mtr(q)), dv, dim), mtr(q), dim);
    lemma_transpose_twice_meq(q, dim, dim);
    lemma_mul_meq(mtr(mtr(q)), q, dv, dv, dim, dim, dim);
    lemma_mul_meq(mmul(mtr(mtr(q)), dv, dim), mmul(q, dv, dim), mtr(q), mtr(q), dim, dim, dim);
    assert forall|x: Seq<real>| x.len() == dim
        implies #[trigger] quad_r(a, dim, x) + eps * dot_r(x, x) >= 0.0real by {
        lemma_quad_meq(a2, a, dim, x);
        lemma_quad_congruence(dv, mtr(q), dim, x);
        let y = mat_vec(mtr(q), dim, x);
        lemma_quad_diag_lower_bound(d, dim, y, -eps);
        lemma_orthogonal_norm(q, dim, x);
        assert(-eps * dot_r(x, x) == -(eps * dot_r(x, x))) by (nonlinear_arith);
    }
}

/// A factorization Q D Q^T = A + eps I with all d_k >= 0 establishes A >= -eps I.
pub proof fn lemma_shifted_factorization_psd(a: MatF, q: MatF, d: Seq<real>, dim: int, eps: real)
    requires
        dim >= 0,
        forall|k: int| 0 <= k < dim ==> #[trigger] d[k] >= 0.0real,
        meq(mmul(mmul(q, diag_view(d), dim), mtr(q), dim), madd(a, mscale(eps, mident())), dim, dim),
    ensures
        psd_shifted(a, dim, eps),
{
    lemma_diag_congruence_psd(q, d, dim);
    lemma_psd_shifted_meq(mmul(mmul(q, diag_view(d), dim), mtr(q), dim), madd(a, mscale(eps, mident())), dim, 0.0real);
    lemma_psd_shifted_iff(a, dim, eps);
}

/// A >= -eps I if and only if A + eps I >= 0.
pub proof fn lemma_psd_shifted_iff(a: MatF, dim: int, eps: real)
    requires
        dim >= 0,
    ensures
        psd_shifted(a, dim, eps) <==> is_psd(madd(a, mscale(eps, mident())), dim),
{
    let sh = madd(a, mscale(eps, mident()));
    if psd_shifted(a, dim, eps) {
        assert forall|x: Seq<real>| x.len() == dim
            implies #[trigger] quad_r(sh, dim, x) + 0.0real * dot_r(x, x) >= 0.0real by {
            lemma_quad_shift(a, dim, eps, x);
            assert(quad_r(a, dim, x) + eps * dot_r(x, x) >= 0.0real);
            assert(0.0real * dot_r(x, x) == 0.0real);
        }
    }
    if is_psd(sh, dim) {
        assert forall|x: Seq<real>| x.len() == dim
            implies #[trigger] quad_r(a, dim, x) + eps * dot_r(x, x) >= 0.0real by {
            lemma_quad_shift(a, dim, eps, x);
            assert(quad_r(sh, dim, x) + 0.0real * dot_r(x, x) >= 0.0real);
            assert(0.0real * dot_r(x, x) == 0.0real);
        }
    }
}

/* ---------- block structure of the semidefinite order ---------- */

/// x^T A x = sum_{r, c} x_r A_rc x_c.
pub proof fn lemma_quad_sum2(a: MatF, dim: int, x: Seq<real>)
    requires
        dim >= 0,
        x.len() == dim,
    ensures
        quad_r(a, dim, x) == sum2_fn_r(|r: int, c: int| x[r] * (a(r, c) * x[c]), dim as nat),
{
    let n = dim as nat;
    let ax = mat_vec(a, dim, x);
    lemma_dot_sum_fn(x, ax);
    let t = |r: int, c: int| x[r] * (a(r, c) * x[c]);
    let lhs = |r: int| x[r] * ax[r];
    let rows = |r: int| sum_fn_r(|c: int| t(r, c), n);
    assert forall|r: int| 0 <= r < dim implies #[trigger] lhs(r) == rows(r) by {
        let f = |c: int| a(r, c) * x[c];
        let g = |c: int| t(r, c);
        assert forall|c: int| 0 <= c < dim implies #[trigger] g(c) == x[r] * f(c) by {}
        lemma_sum_fn_scale(f, g, x[r], n);
    }
    lemma_sum_fn_ext(lhs, rows, n);
}

/// <x, y> = <x[..k], y[..k]> + <x[k..], y[k..]>.
pub proof fn lemma_dot_split(x: Seq<real>, y: Seq<real>, k: int)
    requires
        x.len() == y.len(),
        0 <= k <= x.len(),
    ensures
        dot_r(x, y) == dot_r(x.subrange(0, k), y.subrange(0, k))
            + dot_r(x.subrange(k, x.len() as int), y.subrange(k, x.len() as int)),
{
    let n = x.len() as int;
    let (x1, y1) = (x.subrange(0, k), y.subrange(0, k));
    let (x2, y2) = (x.subrange(k, n), y.subrange(k, n));
    lemma_dot_sum_fn(x, y);
    lemma_dot_sum_fn(x1, y1);
    lemma_dot_sum_fn(x2, y2);
    let f = |i: int| x[i] * y[i];
    let f1 = |i: int| x1[i] * y1[i];
    let f2 = |i: int| x2[i] * y2[i];
    let tail = |i: int| f(k + i);
    lemma_sum_fn_split(f, k as nat, (n - k) as nat);
    assert forall|i: int| 0 <= i < k implies #[trigger] f(i) == f1(i) by {}
    lemma_sum_fn_ext(f, f1, k as nat);
    assert forall|i: int| 0 <= i < n - k implies #[trigger] tail(i) == f2(i) by {}
    lemma_sum_fn_ext(tail, f2, (n - k) as nat);
}

/// For a matrix with no coupling between the leading k rows/columns and the trailing m, the
/// quadratic form splits: x^T A x = x1^T A x1 + x2^T (shift_view(A, k)) x2 with x = (x1, x2).
pub proof fn lemma_quad_split(a: MatF, k: int, m: int, x: Seq<real>)
    requires
        k >= 0,
        m >= 0,
        x.len() == k + m,
        forall|r: int, c: int| #![trigger a(r, c)] #![trigger a(c, r)]
            0 <= r < k && k <= c < k + m ==> a(r, c) == 0.0real && a(c, r) == 0.0real,
    ensures
        quad_r(a, k + m, x) == quad_r(a, k, x.subrange(0, k)) + quad_r(shift_view(a, k), m, x.subrange(k, k + m)),
{
    let n = k + m;
    let x1 = x.subrange(0, k);
    let x2 = x.subrange(k, n);
    let a2 = shift_view(a, k);
    let ax = mat_vec(a, n, x);
    let ax1 = mat_vec(a, k, x1);
    let ax2 = mat_vec(a2, m, x2);
    // (A x)_r = (A x1)_r for r < k: the trailing coefficients vanish
    assert forall|r: int| 0 <= r < k implies #[trigger] ax[r] == ax1[r] by {
        let f = |c: int| a(r, c) * x[c];
        let g = |c: int| a(r, c) * x1[c];
        assert forall|c: int| k <= c < n implies #[trigger] f(c) == 0.0real by {
            assert(a(r, c) == 0.0real);
        }
        lemma_sum_fn_tail_zero(f, k as nat, n as nat);
        assert forall|c: int| 0 <= c < k implies #[trigger] f(c) == g(c) by {}
        lemma_sum_fn_ext(f, g, k as nat);
    }
    // (A x)_{k + i} = (A22 x2)_i for i < m: the leading coefficients vanish
    assert forall|i: int| 0 <= i < m implies #[trigger] ax[k + i] == ax2[i] by {
        let f = |c: int| a(k + i, c) * x[c];
        let g = |j: int| a2(i, j) * x2[j];
        let tail = |j: int| f(k + j);
        lemma_sum_fn_split(f, k as nat, m as nat);
        assert forall|c: int| 0 <= c < k implies #[trigger] f(c) == 0.0real by {
            assert(a(k + i, c) == 0.0real);
        }
        lemma_sum_fn_ext(f, |c: int| 0.0real, k as nat);
        lemma_sum_fn_zero(k as nat);
        assert forall|j: int| 0 <= j < m implies #[trigger] tail(j) == g(j) by {}
        lemma_sum_fn_ext(tail, g, m as nat);
    }
    lemma_dot_split(x, ax, k);
    assert(ax.subrange(0, k) =~= ax1);
    assert(ax.subrange(k, n) =~= ax2);
}

/// A block-diagonal matrix with blocks A11 (order k) and A22 = shift_view(A, k) (order m)
/// satisfies A >= -eps I when both blocks do: the quadratic form and the norm split by block.
pub proof fn lemma_psd_shifted_two_blocks(a: MatF, k: int, m: int, eps: real)
    requires
        k >= 0,
        m >= 0,
        forall|r: int, c: int| #![trigger a(r, c)] #![trigger a(c, r)]
            0 <= r < k && k <= c < k + m ==> a(r, c) == 0.0real && a(c, r) == 0.0real,
        psd_shifted(a, k, eps),
        psd_shifted(shift_view(a, k), m, eps),
    ensures
        psd_shifted(a, k + m, eps),
{
    let n = k + m;
    assert forall|x: Seq<real>| x.len() == n
        implies #[trigger] quad_r(a, n, x) + eps * dot_r(x, x) >= 0.0real by {
        let x1 = x.subrange(0, k);
        let x2 = x.subrange(k, n);
        lemma_quad_split(a, k, m, x);
        lemma_dot_split(x, x, k);
        assert(quad_r(a, k, x1) + eps * dot_r(x1, x1) >= 0.0real);
        assert(quad_r(shift_view(a, k), m, x2) + eps * dot_r(x2, x2) >= 0.0real);
        assert(eps * (dot_r(x1, x1) + dot_r(x2, x2)) == eps * dot_r(x1, x1) + eps * dot_r(x2, x2)) by (nonlinear_arith);
    }
}

/// The padded vector has the quadratic form of the block, x_pad^T A x_pad = x^T A_blk x with
/// A_blk = shift_view(A, start), and the norm of x.
pub proof fn lemma_quad_pad(a: MatF, dim: int, start: int, k: int, x: Seq<real>)
    requires
        0 <= start,
        0 <= k,
        start + k <= dim,
        x.len() == k,
    ensures
        quad_r(a, dim, pad_r(x, dim, start)) == quad_r(shift_view(a, start), k, x),
        dot_r(pad_r(x, dim, start), pad_r(x, dim, start)) == dot_r(x, x),
{
    let xp = pad_r(x, dim, start);
    let axp = mat_vec(a, dim, xp);
    let ab = shift_view(a, start);
    let abx = mat_vec(ab, k, x);
    // (A x_pad)_{start + i} = (A_blk x)_i
    assert forall|i: int| 0 <= i < k implies #[trigger] axp[start + i] == abx[i] by {
        let f = |c: int| a(start + i, c) * xp[c];
        let g = |j: int| ab(i, j) * x[j];
        let win = |j: int| f(start + j);
        assert forall|c: int| 0 <= c < dim && !(start <= c < start + k) implies #[trigger] f(c) == 0.0real by {
            assert(xp[c] == 0.0real);
            let u = a(start + i, c);
            assert(u * 0.0real == 0.0real) by (nonlinear_arith);
        }
        lemma_sum_fn_window(f, dim as nat, start, k as nat);
        assert forall|j: int| 0 <= j < k implies #[trigger] win(j) == g(j) by {
            assert(xp[start + j] == x[j]);
        }
        lemma_sum_fn_ext(win, g, k as nat);
    }
    // <x_pad, A x_pad> is the sum over the window
    lemma_dot_sum_fn(xp, axp);
    lemma_dot_sum_fn(x, abx);
    let f = |i: int| xp[i] * axp[i];
    let g = |i: int| x[i] * abx[i];
    let win = |j: int| f(start + j);
    assert forall|i: int| 0 <= i < dim && !(start <= i < start + k) implies #[trigger] f(i) == 0.0real by {
        assert(xp[i] == 0.0real);
        let u = axp[i];
        assert(0.0real * u == 0.0real) by (nonlinear_arith);
    }
    lemma_sum_fn_window(f, dim as nat, start, k as nat);
    assert forall|j: int| 0 <= j < k implies #[trigger] win(j) == g(j) by {
        assert(xp[start + j] == x[j]);
    }
    lemma_sum_fn_ext(win, g, k as nat);
    // |x_pad|^2 = |x|^2
    lemma_dot_sum_fn(xp, xp);
    lemma_dot_sum_fn(x, x);
    let f2 = |i: int| xp[i] * xp[i];
    let g2 = |i: int| x[i] * x[i];
    let win2 = |j: int| f2(start + j);
    assert forall|i: int| 0 <= i < dim && !(start <= i < start + k) implies #[trigger] f2(i) == 0.0real by {
        assert(xp[i] == 0.0real);
    }
    lemma_sum_fn_window(f2, dim as nat, start, k as nat);
    assert forall|j: int| 0 <= j < k implies #[trigger] win2(j) == g2(j) by {
        assert(xp[start + j] == x[j]);
    }
    lemma_sum_fn_ext(win2, g2, k as nat);
}

/// A diagonal block of a matrix with A >= -eps I satisfies the same bound: a test vector of
/// the block is padded with zeros to a test vector of A.
pub proof fn lemma_psd_shifted_block(m: Seq<real>, dim: int, start: int, k: int, eps: real)
    requires
        0 <= start,
        0 <= k,
        start + k <= dim,
        psd_shifted(sym_view(m), dim, eps),
    ensures
        psd_shifted(block_view(m, start), k, eps),
{
    let a = sym_view(m);
    let b = block_view(m, start);
    assert forall|r: int, c: int| 0 <= r < k && 0 <= c < k
        implies #[trigger] shift_view(a, start)(r, c) == #[trigger] b(r, c) by {}
    assert forall|x: Seq<real>| x.len() == k
        implies #[trigger] quad_r(b, k, x) + eps * dot_r(x, x) >= 0.0real by {
        let xp = pad_r(x, dim, start);
        lemma_quad_pad(a, dim, start, k, x);
        assert(quad_r(a, dim, xp) + eps * dot_r(xp, xp) >= 0.0real);
        lemma_quad_meq(shift_view(a, start), b, k, x);
    }
}

/// The sum of the magnitudes of the entries is nonnegative.
pub proof fn lemma_abs_sum_nonneg(m: Seq<real>, dim: int)
    requires
        dim >= 0,
    ensures
        abs_sum_r(m, dim) >= 0.0real,
{
    let f = |r: int, c: int| abs_r(sym_entry(m, r, c));
    let rows = |r: int| sum_fn_r(|c: int| f(r, c), dim as nat);
    assert forall|r: int| 0 <= r < dim implies #[trigger] rows(r) >= 0.0real by {
        assert forall|c: int| 0 <= c < dim implies #[trigger] f(r, c) >= 0.0real by {}
        lemma_sum_fn_nonneg(|c: int| f(r, c), dim as nat);
    }
    lemma_sum_fn_nonneg(rows, dim as nat);
}

/// X >= -t I for t = sum_{r, c} |X_rc|: each term X_rc x_r x_c of the quadratic form is at
/// least -|X_rc| |x|^2, because |x_r x_c| <= (x_r^2 + x_c^2) / 2 <= |x|^2.
pub proof fn lemma_psd_shifted_abs_sum(m: Seq<real>, dim: int)
    requires
        dim >= 0,
    ensures
        psd_shifted(sym_view(m), dim, abs_sum_r(m, dim)),
{
    let a = sym_view(m);
    let t = abs_sum_r(m, dim);
    let n = dim as nat;
    assert forall|x: Seq<real>| x.len() == dim
        implies #[trigger] quad_r(a, dim, x) + t * dot_r(x, x) >= 0.0real by {
        lemma_quad_sum2(a, dim, x);
        let xx = dot_r(x, x);
        let f = |r: int, c: int| x[r] * (a(r, c) * x[c]);
        let g = |r: int, c: int| abs_r(sym_entry(m, r, c));
        let h = |r: int, c: int| (-xx) * g(r, c);
        lemma_dot_sum_fn(x, x);
        let sq = |i: int| x[i] * x[i];
        assert forall|i: int| 0 <= i < dim implies #[trigger] sq(i) >= 0.0real by {
            assert(x[i] * x[i] >= 0.0real) by (nonlinear_arith);
        }
        assert forall|r: int, c: int| 0 <= r < dim && 0 <= c < dim implies #[trigger] h(r, c) <= f(r, c) by {
            lemma_sum_fn_ge_term(sq, n, r);
            lemma_sum_fn_ge_term(sq, n, c);
            let (u, v, w) = (x[r], sym_entry(m, r, c), x[c]);
            let av = abs_r(v);
            let p = u * w;
            assert(f(r, c) == u * (v * w));
            assert(u * (v * w) == v * (u * w)) by (nonlinear_arith);
            // |u w| <= |x|^2 from 2 |u w| <= u^2 + w^2
            assert(p <= xx && -p <= xx) by (nonlinear_arith)
                requires p == u * w, u * u <= xx, w * w <= xx;
            if v >= 0.0real {
                assert(v * p >= -(v * xx)) by (nonlinear_arith)
                    requires v >= 0.0real, -p <= xx;
                assert(h(r, c) == (-xx) * v);
                assert((-xx) * v == -(v * xx)) by (nonlinear_arith);
            } else {
                assert(v * p >= v * xx) by (nonlinear_arith)
                    requires v < 0.0real, p <= xx;
                assert(h(r, c) == (-xx) * (-v));
                assert((-xx) * (-v) == v * xx) by (nonlinear_arith);
            }
        }
        lemma_sum2_fn_le(h, f, n);
        lemma_sum2_fn_scale(g, h, -xx, n);
        assert((-xx) * t == -(t * xx)) by (nonlinear_arith);
    }
}

/* ---------- products with a diagonal matrix ---------- */

/// (Q D)_rk = Q_rk d_k for D = diag(d), k < n.
pub proof fn lemma_mul_diag_right(q: MatF, d: Seq<real>, n: int, r: int, k: int)
    requires
        0 <= k < n,
    ensures
        mmul(q, diag_view(d), n)(r, k) == q(r, k) * d[k],
{
    let f = |l: int| q(r, l) * diag_view(d)(l, k);
    assert forall|l: int| 0 <= l < n && l != k implies #[trigger] f(l) == 0.0real by {
        assert(diag_view(d)(l, k) == 0.0real);
    }
    lemma_sum_fn_single(f, n as nat, k);
}

/// (D Q)_kc = d_k Q_kc for D = diag(d), k < n.
pub proof fn lemma_mul_diag_left(d: Seq<real>, q: MatF, n: int, k: int, c: int)
    requires
        0 <= k < n,
    ensures
        mmul(diag_view(d), q, n)(k, c) == d[k] * q(k, c),
{
    let f = |l: int| diag_view(d)(k, l) * q(l, c);
    assert forall|l: int| 0 <= l < n && l != k implies #[trigger] f(l) == 0.0real by {
        assert(diag_view(d)(k, l) == 0.0real);
    }
    lemma_sum_fn_single(f, n as nat, k);
}

/// (Q D Q^T)_rc = sum_{k < n} d_k Q_rk Q_ck: the spectral reconstruction as a sum of
/// scaled outer products of the columns of Q.
pub proof fn lemma_mul_diag_transpose(q: MatF, d: Seq<real>, n: int, r: int, c: int)
    requires
        n >= 0,
    ensures
        mmul(mmul(q, diag_view(d), n), mtr(q), n)(r, c) == sum_fn_r(|k: int| d[k] * q(r, k) * q(c, k), n as nat),
{
    let qd = mmul(q, diag_view(d), n);
    let f = |k: int| qd(r, k) * mtr(q)(k, c);
    let g = |k: int| d[k] * q(r, k) * q(c, k);
    assert forall|k: int| 0 <= k < n implies #[trigger] f(k) == g(k) by {
        lemma_mul_diag_right(q, d, n, r, k);
        let (u, v, w) = (q(r, k), d[k], q(c, k));
        assert(mtr(q)(k, c) == w);
        assert(f(k) == (u * v) * w);
        assert((u * v) * w == v * u * w) by (nonlinear_arith);
    }
    lemma_sum_fn_ext(f, g, n as nat);
}

/// Q D Q^T is symmetric (entrywise, for any r and c).
pub proof fn lemma_mul_diag_transpose_symmetric(q: MatF, d: Seq<real>, n: int, r: int, c: int)
    requires
        n >= 0,
    ensures
        mmul(mmul(q, diag_view(d), n), mtr(q), n)(r, c) == mmul(mmul(q, diag_view(d), n), mtr(q), n)(c, r),
{
    lemma_mul_diag_transpose(q, d, n, r, c);
    lemma_mul_diag_transpose(q, d, n, c, r);
    let f = |k: int| d[k] * q(r, k) * q(c, k);
    let g = |k: int| d[k] * q(c, k) * q(r, k);
    assert forall|k: int| 0 <= k < n implies #[trigger] f(k) == g(k) by {
        let (u, v, w) = (d[k], q(r, k), q(c, k));
        assert(u * v * w == u * w * v) by (nonlinear_arith);
    }
    lemma_sum_fn_ext(f, g, n as nat);
}

/* ---------- Frobenius product ---------- */

/// The packed Frobenius product (sym_spec.rs) is the Frobenius product of the views.
pub proof fn lemma_frob_sym_view(a: Seq<real>, b: Seq<real>, dim: int)
    requires
        dim >= 0,
    ensures
        frob_r(a, b, dim) == frob_f(sym_view(a), sym_view(b), dim),
{
    let f = |r: int, c: int| sym_entry(a, r, c) * sym_entry(b, r, c);
    let g = |r: int, c: int| sym_view(a)(r, c) * sym_view(b)(r, c);
    assert forall|r: int, c: int| 0 <= r < dim && 0 <= c < dim implies #[trigger] f(r, c) == g(r, c) by {}
    lemma_sum2_fn_ext(f, g, dim as nat);
}

/// <A, A>_F >= 0.
pub proof fn lemma_frob_f_self_nonneg(a: MatF, dim: int)
    requires
        dim >= 0,
    ensures
        frob_f(a, a, dim) >= 0.0real,
{
    let f = |r: int, c: int| a(r, c) * a(r, c);
    let rows = |r: int| sum_fn_r(|c: int| f(r, c), dim as nat);
    assert forall|r: int| 0 <= r < dim implies #[trigger] rows(r) >= 0.0real by {
        assert forall|c: int| 0 <= c < dim implies #[trigger] f(r, c) >= 0.0real by {
            assert(a(r, c) * a(r, c) >= 0.0real) by (nonlinear_arith);
        }
        lemma_sum_fn_nonneg(|c: int| f(r, c), dim as nat);
    }
    lemma_sum_fn_nonneg(rows, dim as nat);
}

/// <A, B>_F = <B, A>_F.
pub proof fn lemma_frob_f_comm(a: MatF, b: MatF, dim: int)
    requires
        dim >= 0,
    ensures
        frob_f(a, b, dim) == frob_f(b, a, dim),
{
    let f = |r: int, c: int| a(r, c) * b(r, c);
    let g = |r: int, c: int| b(r, c) * a(r, c);
    assert forall|r: int, c: int| 0 <= r < dim && 0 <= c < dim implies #[trigger] f(r, c) == g(r, c) by {}
    lemma_sum2_fn_ext(f, g, dim as nat);
}

/// The Frobenius product only depends on the entries on the block.
pub proof fn lemma_frob_f_meq(a: MatF, a2: MatF, b: MatF, b2: MatF, dim: int)
    requires
        dim >= 0,
        meq(a, a2, dim, dim),
        meq(b, b2, dim, dim),
    ensures
        frob_f(a, b, dim) == frob_f(a2, b2, dim),
{
    let f = |r: int, c: int| a(r, c) * b(r, c);
    let g = |r: int, c: int| a2(r, c) * b2(r, c);
    assert forall|r: int, c: int| 0 <= r < dim && 0 <= c < dim implies #[trigger] f(r, c) == g(r, c) by {}
    lemma_sum2_fn_ext(f, g, dim as nat);
}

/// <A, B + C>_F = <A, B>_F + <A, C>_F.
pub proof fn lemma_frob_f_add(a: MatF, b: MatF, c: MatF, dim: int)
    requires
        dim >= 0,
    ensures
        frob_f(a, madd(b, c), dim) == frob_f(a, b, dim) + frob_f(a, c, dim),
{
    let f = |r: int, k: int| a(r, k) * b(r, k);
    let g = |r: int, k: int| a(r, k) * c(r, k);
    let h = |r: int, k: int| a(r, k) * madd(b, c)(r, k);
    assert forall|r: int, k: int| 0 <= r < dim && 0 <= k < dim implies #[trigger] h(r, k) == f(r, k) + g(r, k) by {
        let (u, v, w) = (a(r, k), b(r, k), c(r, k));
        assert(h(r, k) == u * (v + w));
        assert(u * (v + w) == u * v + u * w) by (nonlinear_arith);
    }
    lemma_sum2_fn_add(f, g, h, dim as nat);
}

/// <A, s B>_F = s <A, B>_F.
pub proof fn lemma_frob_f_scale(a: MatF, s: real, b: MatF, dim: int)
    requires
        dim >= 0,
    ensures
        frob_f(a, mscale(s, b), dim) == s * frob_f(a, b, dim),
{
    let f = |r: int, k: int| a(r, k) * b(r, k);
    let g = |r: int, k: int| a(r, k) * mscale(s, b)(r, k);
    assert forall|r: int, k: int| 0 <= r < dim && 0 <= k < dim implies #[trigger] g(r, k) == s * f(r, k) by {
        let (u, v) = (a(r, k), b(r, k));
        assert(g(r, k) == u * (s * v));
        assert(u * (s * v) == s * (u * v)) by (nonlinear_arith);
    }
    lemma_sum2_fn_scale(f, g, s, dim as nat);
}

/* ---------- connecting the stored-matrix specifications to the algebra ---------- */

/// The packed symmetric view is symmetric.
pub proof fn lemma_sym_view_symmetric(m: Seq<real>, dim: int)
    ensures
        is_symmetric(sym_view(m), dim),
{
    assert forall|i: int, j: int| 0 <= i < dim && 0 <= j < dim implies #[trigger] sym_view(m)(i, j) == sym_view(m)(j, i) by {}
}

/// The packed matrix-vector product (sym_spec.rs) is the product of the view.
pub proof fn lemma_sym_mat_vec_view(a: Seq<real>, dim: int, x: Seq<real>)
    requires
        dim >= 0,
    ensures
        sym_mat_vec_r(a, dim, x) == mat_vec(sym_view(a), dim, x),
{
    assert forall|i: int| 0 <= i < dim implies #[trigger] sym_mat_vec_r(a, dim, x)[i] == mat_vec(sym_view(a), dim, x)[i] by {
        let f = |j: int| sym_entry(a, i, j) * x[j];
        let g = |k: int| sym_view(a)(i, k) * x[k];
        assert forall|j: int| 0 <= j < dim implies #[trigger] f(j) == g(j) by {}
        lemma_sum_fn_ext(f, g, dim as nat);
        assert(sym_mat_vec_r(a, dim, x)[i] == sum_fn_r(f, dim as nat));
        assert(mat_vec(sym_view(a), dim, x)[i] == sum_fn_r(g, dim as nat));
    }
    assert(sym_mat_vec_r(a, dim, x) =~= mat_vec(sym_view(a), dim, x));
}

/// The identity predicates of the stored representations give the identity view.
pub proof fn lemma_ident_views(m: Seq<real>, dim: int)
    ensures
        is_ident_r(m, dim) ==> meq(cm_view(m, dim), mident(), dim, dim),
        is_sym_ident_r(m, dim) ==> meq(sym_view(m), mident(), dim, dim),
        is_lt_ident_r(m, dim) ==> meq(lt_view(m, dim), mident(), dim, dim),
{
    if is_ident_r(m, dim) {
        assert forall|i: int, j: int| 0 <= i < dim && 0 <= j < dim implies #[trigger] cm_view(m, dim)(i, j) == mident()(i, j) by {
            assert(cm_entry(m, dim, i, j) == (if i == j { 1.0real } else { 0.0real }));
        }
    }
    if is_sym_ident_r(m, dim) {
        assert forall|i: int, j: int| 0 <= i < dim && 0 <= j < dim implies #[trigger] sym_view(m)(i, j) == mident()(i, j) by {
            if i <= j {
                assert(sym_entry(m, i, j) == (if i == j { 1.0real } else { 0.0real }));
            } else {
                assert(sym_entry(m, j, i) == 0.0real);
            }
        }
    }
    if is_lt_ident_r(m, dim) {
        assert forall|i: int, j: int| 0 <= i < dim && 0 <= j < dim implies #[trigger] lt_view(m, dim)(i, j) == mident()(i, j) by {
            if j <= i {
                assert(lt_entry(m, dim, i, j) == (if i == j { 1.0real } else { 0.0real }));
            }
        }
    }
}

} // verus!

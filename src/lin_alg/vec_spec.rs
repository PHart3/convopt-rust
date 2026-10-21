// Spec-level views of vectors and matrices (not trusted).
//
// A float vector `v: Vec<f64>` or `&[f64]` is viewed as the real sequence
// `rvs(v@)`.  Matrices stored as flat vectors are viewed through index maps:
// column-major (`cm_*`) and packed lower-triangular column-major (`lt_*`).
// All definitions are over Verus's native `real` type.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::seq::*;
#[cfg(verus_keep_ghost)]
use vstd::seq_lib::*;
#[cfg(verus_keep_ghost)]
use crate::trusted::{rv, sqrt_r};
#[cfg(verus_keep_ghost)]
use crate::real_spec::lemma_sq_nonneg;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- vector views ---------- */

/// Exact-real view of a sequence of floats.
pub open spec fn rvs(s: Seq<f64>) -> Seq<real> {
    Seq::new(s.len(), |i: int| rv(s[i]))
}

/// Sum of a sequence of reals (empty sum is 0).
pub open spec fn sum_r(s: Seq<real>) -> real
    decreases s.len(),
{
    if s.len() == 0 { 0.0real } else { sum_r(s.drop_last()) + s.last() }
}

/// Constant sequence.
pub open spec fn const_r(n: nat, x: real) -> Seq<real> {
    Seq::new(n, |i: int| x)
}

/// Standard basis vector e_i of dimension dim (the zero vector if i >= dim).
pub open spec fn std_basis_r(i: int, dim: nat) -> Seq<real> {
    Seq::new(dim, |j: int| if j == i { 1.0real } else { 0.0real })
}

/// Pointwise sum (length taken from a; recommends equal lengths).
pub open spec fn vadd_r(a: Seq<real>, b: Seq<real>) -> Seq<real>
    recommends a.len() == b.len(),
{
    Seq::new(a.len(), |i: int| a[i] + b[i])
}

/// Pointwise difference.
pub open spec fn vsub_r(a: Seq<real>, b: Seq<real>) -> Seq<real>
    recommends a.len() == b.len(),
{
    Seq::new(a.len(), |i: int| a[i] - b[i])
}

/// Pointwise (Hadamard) product.
pub open spec fn vmul_r(a: Seq<real>, b: Seq<real>) -> Seq<real>
    recommends a.len() == b.len(),
{
    Seq::new(a.len(), |i: int| a[i] * b[i])
}

/// Scalar multiple.
pub open spec fn vscale_r(s: real, a: Seq<real>) -> Seq<real> {
    Seq::new(a.len(), |i: int| s * a[i])
}

/// Standard inner product sum_i a[i] * b[i].
pub open spec fn dot_r(a: Seq<real>, b: Seq<real>) -> real
    recommends a.len() == b.len(),
{
    sum_r(vmul_r(a, b))
}

/// Euclidean norm.
pub open spec fn norm_r(a: Seq<real>) -> real {
    sqrt_r(dot_r(a, a))
}

/// Euclidean distance.
pub open spec fn dist_r(a: Seq<real>, b: Seq<real>) -> real
    recommends a.len() == b.len(),
{
    norm_r(vsub_r(a, b))
}

/* ---------- column-major matrix view ---------- */

/// Flat index of entry (r, c) of a column-major matrix with row_dim rows.
pub open spec fn cm_idx(row_dim: int, r: int, c: int) -> int {
    c * row_dim + r
}

/// Entry (r, c) of a column-major matrix with row_dim rows.
pub open spec fn cm_entry(m: Seq<real>, row_dim: int, r: int, c: int) -> real {
    m[cm_idx(row_dim, r, c)]
}

/// m is the dim x dim identity matrix in column-major form.
pub open spec fn is_ident_r(m: Seq<real>, dim: int) -> bool {
    &&& m.len() == dim * dim
    &&& forall|r: int, c: int| 0 <= r < dim && 0 <= c < dim
            ==> #[trigger] cm_entry(m, dim, r, c) == (if r == c { 1.0real } else { 0.0real })
}

/// Column-major indices are injective on in-range row indices.
pub proof fn lemma_cm_idx_injective(row_dim: int, r1: int, c1: int, r2: int, c2: int)
    requires
        0 <= r1 < row_dim,
        0 <= r2 < row_dim,
    ensures
        cm_idx(row_dim, r1, c1) == cm_idx(row_dim, r2, c2) <==> (r1 == r2 && c1 == c2),
{
    if c1 != c2 {
        assert(c1 * row_dim + r1 != c2 * row_dim + r2) by (nonlinear_arith)
            requires 0 <= r1 < row_dim, 0 <= r2 < row_dim, c1 != c2;
    }
}

/// Column-major index of entry (r, c) lies before the end of column c (and of any later column).
pub proof fn lemma_cm_idx_bound(row_dim: int, r: int, c: int, cols: int)
    requires
        0 <= r < row_dim,
        0 <= c < cols,
    ensures
        0 <= cm_idx(row_dim, r, c) < (c + 1) * row_dim <= cols * row_dim,
{
    assert((c + 1) * row_dim == c * row_dim + row_dim) by (nonlinear_arith);
    assert((c + 1) * row_dim <= cols * row_dim) by (nonlinear_arith)
        requires c + 1 <= cols, row_dim >= 0;
    assert(c * row_dim >= 0);
}

/// Writing column-major entry (r0, c0) leaves every other entry unchanged.
pub proof fn lemma_cm_entry_write_other(prev: Seq<real>, cur: Seq<real>, row_dim: int, col_dim: int, r0: int, c0: int, r: int, l: int)
    requires
        0 <= r0 < row_dim,
        0 <= r < row_dim,
        0 <= l < col_dim,
        prev.len() >= row_dim * col_dim,
        cur.len() == prev.len(),
        forall|t: int| 0 <= t < prev.len() && t != cm_idx(row_dim, r0, c0) ==> cur[t] == prev[t],
        !(r == r0 && l == c0),
    ensures
        cm_entry(cur, row_dim, r, l) == cm_entry(prev, row_dim, r, l),
{
    lemma_cm_idx_injective(row_dim, r, l, r0, c0);
    assert(l * row_dim + r < row_dim * col_dim) by (nonlinear_arith)
        requires 0 <= r < row_dim, 0 <= l < col_dim;
    assert(cur[cm_idx(row_dim, r, l)] == prev[cm_idx(row_dim, r, l)]);
}

/* ---------- packed lower-triangular column-major view ---------- */
//
// Column c of a dim x dim lower-triangular matrix stores entries (c, c), ..., (dim-1, c)
// contiguously, columns in increasing order; column c starts at c * (2 * dim - c + 1) / 2.

/// Start offset of column c.
pub open spec fn lt_col_start(dim: int, c: int) -> int {
    c * (2 * dim - c + 1) / 2
}

/// Flat index of entry (r, c), c <= r < dim.
pub open spec fn lt_idx(dim: int, r: int, c: int) -> int {
    lt_col_start(dim, c) + (r - c)
}

/// Entry (r, c), c <= r < dim.
pub open spec fn lt_entry(m: Seq<real>, dim: int, r: int, c: int) -> real {
    m[lt_idx(dim, r, c)]
}

/// m is the dim x dim identity matrix in packed lower-triangular form.
pub open spec fn is_lt_ident_r(m: Seq<real>, dim: int) -> bool {
    &&& m.len() == dim * (dim + 1) / 2
    &&& forall|r: int, c: int| 0 <= c <= r < dim
            ==> #[trigger] lt_entry(m, dim, r, c) == (if r == c { 1.0real } else { 0.0real })
}

/// Column c + 1 starts dim - c entries after column c.
pub proof fn lemma_lt_col_start_step(dim: int, c: int)
    requires
        0 <= c < dim,
    ensures
        lt_col_start(dim, c + 1) == lt_col_start(dim, c) + (dim - c),
{
    assert((c + 1) * (2 * dim - (c + 1) + 1) == c * (2 * dim - c + 1) + 2 * (dim - c))
        by (nonlinear_arith);
}

/// The packed form of a dim x dim lower-triangular matrix has dim * (dim + 1) / 2 entries.
pub proof fn lemma_lt_col_start_end(dim: int)
    requires
        dim >= 0,
    ensures
        lt_col_start(dim, dim) == dim * (dim + 1) / 2,
{
    assert(dim * (2 * dim - dim + 1) == dim * (dim + 1));
}

/// Column 0 starts at offset 0.
pub proof fn lemma_lt_col_start_zero(dim: int)
    ensures
        lt_col_start(dim, 0) == 0,
{
    assert(0 * (2 * dim - 0 + 1) == 0);
}

/// Column starts are nondecreasing in the column index.
pub proof fn lemma_lt_col_start_monotone(dim: int, c1: int, c2: int)
    requires
        0 <= c1 <= c2 <= dim,
    ensures
        lt_col_start(dim, c1) <= lt_col_start(dim, c2),
    decreases c2 - c1,
{
    if c1 < c2 {
        lemma_lt_col_start_step(dim, c1);
        lemma_lt_col_start_monotone(dim, c1 + 1, c2);
    }
}

/// Entry (r, c) lies in column c, which lies before every column c' >= c + 1.
pub proof fn lemma_lt_idx_bound(dim: int, r: int, c: int, c2: int)
    requires
        0 <= c <= r < dim,
        c < c2 <= dim,
    ensures
        0 <= lt_col_start(dim, c) <= lt_idx(dim, r, c) < lt_col_start(dim, c + 1) <= lt_col_start(dim, c2),
{
    lemma_lt_col_start_zero(dim);
    lemma_lt_col_start_monotone(dim, 0, c);
    lemma_lt_col_start_step(dim, c);
    lemma_lt_col_start_monotone(dim, c + 1, c2);
}

/// Distinct entries of the triangle have distinct flat indices.
pub proof fn lemma_lt_idx_injective(dim: int, r1: int, c1: int, r2: int, c2: int)
    requires
        0 <= c1 <= r1 < dim,
        0 <= c2 <= r2 < dim,
        lt_idx(dim, r1, c1) == lt_idx(dim, r2, c2),
    ensures
        r1 == r2 && c1 == c2,
{
    if c1 < c2 {
        lemma_lt_idx_bound(dim, r1, c1, c2);
        lemma_lt_idx_bound(dim, r2, c2, dim);
    } else if c2 < c1 {
        lemma_lt_idx_bound(dim, r2, c2, c1);
        lemma_lt_idx_bound(dim, r1, c1, dim);
    }
}

/* ---------- lemmas about sums ---------- */

/// sum over a prefix extended by one element.
pub proof fn lemma_sum_r_take_succ(s: Seq<real>, i: int)
    requires
        0 <= i < s.len(),
    ensures
        sum_r(s.take(i + 1)) == sum_r(s.take(i)) + s[i],
{
    assert(s.take(i + 1).drop_last() =~= s.take(i));
    assert(s.take(i + 1).last() == s[i]);
}

/// Taking the whole sequence changes nothing.
pub proof fn lemma_take_all<A>(s: Seq<A>)
    ensures
        s.take(s.len() as int) =~= s,
{
}

/// A sum of nonnegative reals is nonnegative.
pub proof fn lemma_sum_r_nonneg(s: Seq<real>)
    requires
        forall|i: int| 0 <= i < s.len() ==> #[trigger] s[i] >= 0.0real,
    ensures
        sum_r(s) >= 0.0real,
    decreases s.len(),
{
    if s.len() > 0 {
        lemma_sum_r_nonneg(s.drop_last());
    }
}

/// <a, a> >= 0.
pub proof fn lemma_dot_self_nonneg(a: Seq<real>)
    ensures
        dot_r(a, a) >= 0.0real,
{
    assert forall|i: int| 0 <= i < vmul_r(a, a).len() implies #[trigger] vmul_r(a, a)[i] >= 0.0real by {
        lemma_sq_nonneg(a[i]);
    }
    lemma_sum_r_nonneg(vmul_r(a, a));
}

/// <a, b> = <b, a>.
pub proof fn lemma_dot_comm(a: Seq<real>, b: Seq<real>)
    requires
        a.len() == b.len(),
    ensures
        dot_r(a, b) == dot_r(b, a),
{
    assert forall|i: int| 0 <= i < a.len() implies #[trigger] vmul_r(a, b)[i] == vmul_r(b, a)[i] by {
        let (u, v) = (a[i], b[i]);
        assert(u * v == v * u);
    }
    assert(vmul_r(a, b) =~= vmul_r(b, a));
}

} // verus!

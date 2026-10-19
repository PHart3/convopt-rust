// Spec-level view of packed symmetric matrices (not trusted).
//
// A dim x dim symmetric matrix is stored as a flat vector holding its upper
// triangle in column-major order: column c stores entries (0, c), ..., (c, c)
// contiguously, columns in increasing order, so entry (r, c) with r <= c sits
// at index c * (c + 1) / 2 + r.  The view `sym_entry` extends this to all
// (r, c) by symmetry.  This file also provides the generic sums over index
// functions and the "sweep" bookkeeping used to verify loops that traverse
// the packed triangle once while updating both rows r and c of each entry.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::seq::*;
#[cfg(verus_keep_ghost)]
use crate::trusted::sqrt_r;
#[cfg(verus_keep_ghost)]
use crate::real_spec::{abs_r, max_r};
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- packed upper-triangular column-major index map ---------- */

/// Start offset of column c (also the number of packed entries of a c x c matrix).
pub open spec fn pk_col_start(c: int) -> int {
    c * (c + 1) / 2
}

/// Number of packed entries of a dim x dim symmetric matrix.
pub open spec fn pk_len(dim: int) -> int {
    pk_col_start(dim)
}

/// Flat index of entry (r, c) with 0 <= r <= c.
pub open spec fn pk_idx(r: int, c: int) -> int {
    pk_col_start(c) + r
}

/// Entry (r, c) of the symmetric matrix with packed view m (either order of indices).
pub open spec fn sym_entry(m: Seq<real>, r: int, c: int) -> real {
    if r <= c { m[pk_idx(r, c)] } else { m[pk_idx(c, r)] }
}

/// Diagonal entry (i, i).
pub open spec fn sym_diag(m: Seq<real>, i: int) -> real {
    m[pk_idx(i, i)]
}

/// The index arithmetic of a packed dim x dim matrix fits in usize
/// (covers c * (c + 1), i * i + 3 * i and k * (2 * dim - k + 1) for all indices <= dim).
pub open spec fn dim_fits(dim: int) -> bool {
    (dim + 2) * (dim + 2) <= usize::MAX
}

/// Column c + 1 starts c + 1 entries after column c.
pub proof fn lemma_pk_col_start_step(c: int)
    requires
        c >= 0,
    ensures
        pk_col_start(c + 1) == pk_col_start(c) + c + 1,
{
    assert((c + 1) * (c + 2) == c * (c + 1) + 2 * (c + 1)) by (nonlinear_arith);
}

pub proof fn lemma_pk_col_start_zero()
    ensures
        pk_col_start(0) == 0,
{
}

/// Column starts are nondecreasing.
pub proof fn lemma_pk_col_start_monotone(c1: int, c2: int)
    requires
        0 <= c1 <= c2,
    ensures
        pk_col_start(c1) <= pk_col_start(c2),
    decreases c2 - c1,
{
    if c1 < c2 {
        lemma_pk_col_start_step(c1);
        lemma_pk_col_start_monotone(c1 + 1, c2);
    }
}

/// Entry (r, c) lies in column c, strictly before every column c2 > c.
pub proof fn lemma_pk_idx_bound(r: int, c: int, c2: int)
    requires
        0 <= r <= c < c2,
    ensures
        0 <= pk_col_start(c) <= pk_idx(r, c) < pk_col_start(c + 1) <= pk_col_start(c2),
{
    lemma_pk_col_start_monotone(0, c);
    lemma_pk_col_start_step(c);
    lemma_pk_col_start_monotone(c + 1, c2);
}

/// The closed-form diagonal index used by the implementation.
pub proof fn lemma_pk_diag_idx(i: int)
    requires
        i >= 0,
    ensures
        pk_idx(i, i) == (i * i + 3 * i) / 2,
{
    assert(i * (i + 1) + 2 * i == i * i + 3 * i) by (nonlinear_arith);
}

/// The index arithmetic for i < dim fits in usize when dim_fits(dim).
pub proof fn lemma_dim_fits_idx(dim: int, i: int)
    requires
        dim_fits(dim),
        0 <= i <= dim,
    ensures
        i * (i + 1) <= usize::MAX,
        i * i + 3 * i <= usize::MAX,
        (i + 1) * (i + 2) <= usize::MAX,
        pk_col_start(i + 1) <= usize::MAX,
        dim * (dim + 1) <= usize::MAX,
{
    assert(i * (i + 1) <= (dim + 2) * (dim + 2)) by (nonlinear_arith)
        requires 0 <= i <= dim;
    assert((i + 1) * (i + 2) <= (dim + 2) * (dim + 2)) by (nonlinear_arith)
        requires 0 <= i <= dim;
    assert(i * i + 3 * i <= (dim + 2) * (dim + 2)) by (nonlinear_arith)
        requires 0 <= i <= dim;
    assert(dim * (dim + 1) <= (dim + 2) * (dim + 2)) by (nonlinear_arith)
        requires dim >= 0;
}

/// The packed lower-triangular column offset k * (2 * dim - k + 1) / 2 fits in usize when dim_fits(dim).
pub proof fn lemma_lt_idx_fits(dim: int, k: int)
    requires
        dim_fits(dim),
        0 <= k <= dim,
    ensures
        2 * dim + 1 <= usize::MAX,
        k * (2 * dim - k + 1) <= usize::MAX,
{
    assert(2 * dim + 1 <= (dim + 2) * (dim + 2)) by (nonlinear_arith)
        requires dim >= 0;
    assert(k * (2 * dim - k + 1) <= (dim + 2) * (dim + 2)) by (nonlinear_arith)
        requires 0 <= k <= dim;
}

/// dim_fits is monotone.
pub proof fn lemma_dim_fits_monotone(d1: int, d2: int)
    requires
        0 <= d1 <= d2,
        dim_fits(d2),
    ensures
        dim_fits(d1),
{
    assert((d1 + 2) * (d1 + 2) <= (d2 + 2) * (d2 + 2)) by (nonlinear_arith)
        requires 0 <= d1 <= d2;
}

/// Packed indices order pairs (r, c), r <= c, lexicographically by (c, r).
pub proof fn lemma_pk_idx_order(r1: int, c1: int, r2: int, c2: int)
    requires
        0 <= r1 <= c1,
        0 <= r2 <= c2,
    ensures
        pk_idx(r1, c1) < pk_idx(r2, c2) <==> (c1 < c2 || (c1 == c2 && r1 < r2)),
{
    if c1 < c2 {
        lemma_pk_idx_bound(r1, c1, c2);
    } else if c2 < c1 {
        lemma_pk_idx_bound(r2, c2, c1);
    }
}

/* ---------- the diagonal block of a larger packed matrix ---------- */

/// t is the packed index, in a larger matrix, of an entry of the dim x dim diagonal block at `start`.
pub open spec fn is_block_idx(t: int, start: int, dim: int) -> bool {
    exists|a: int, b: int| 0 <= a <= b < dim && t == #[trigger] pk_idx(start + a, start + b)
}

/// The packed index just after the block entry preceding (a, b) in packed order
/// (0 for the first entry (0, 0)).
pub open spec fn block_prev_end(start: int, a: int, b: int) -> int {
    if a > 0 { pk_idx(start + a - 1, start + b) + 1 }
    else if b > 0 { pk_idx(start + b - 1, start + b - 1) + 1 }
    else { 0 }
}

/// No block entry lies strictly between consecutive block entries: every index in
/// [block_prev_end(a, b), pk_idx(start + a, start + b)) is outside the block.
pub proof fn lemma_block_idx_gap(start: int, dim: int, a: int, b: int, t: int)
    requires
        start >= 0,
        0 <= a <= b < dim,
        block_prev_end(start, a, b) <= t < pk_idx(start + a, start + b),
    ensures
        !is_block_idx(t, start, dim),
{
    if is_block_idx(t, start, dim) {
        let (a2, b2) = choose|a2: int, b2: int| 0 <= a2 <= b2 < dim && t == #[trigger] pk_idx(start + a2, start + b2);
        // (a2, b2) precedes (a, b), hence is at most its predecessor
        lemma_pk_idx_order(start + a2, start + b2, start + a, start + b);
        if a > 0 {
            lemma_pk_idx_order(start + a2, start + b2, start + a - 1, start + b);
            if pk_idx(start + a2, start + b2) < pk_idx(start + a - 1, start + b) {
            }
        } else {
            lemma_pk_idx_order(start + a2, start + b2, start + b - 1, start + b - 1);
        }
    }
}

/// A block entry (a, b) preceding (ga, gb) in packed order lies before block_prev_end(ga, gb).
pub proof fn lemma_block_prev_end_bound(start: int, a: int, b: int, ga: int, gb: int)
    requires
        start >= 0,
        0 <= a <= b,
        0 <= ga <= gb,
        pk_idx(a, b) < pk_idx(ga, gb),
    ensures
        pk_idx(start + a, start + b) < block_prev_end(start, ga, gb),
{
    lemma_pk_idx_order(a, b, ga, gb);
    if ga > 0 {
        lemma_pk_idx_order(start + a, start + b, start + ga - 1, start + gb);
        lemma_pk_idx_order(start + ga - 1, start + gb, start + a, start + b);
    } else {
        lemma_pk_idx_order(start + a, start + b, start + gb - 1, start + gb - 1);
        lemma_pk_idx_order(start + gb - 1, start + gb - 1, start + a, start + b);
    }
}

/// The predecessor's end is at most the entry itself.
pub proof fn lemma_block_prev_end_le(start: int, a: int, b: int)
    requires
        start >= 0,
        0 <= a <= b,
    ensures
        0 <= block_prev_end(start, a, b) <= pk_idx(start + a, start + b),
{
    lemma_pk_col_start_monotone(0, start + b);
    lemma_pk_col_start_zero();
    if a == 0 && b > 0 {
        lemma_pk_col_start_step(start + b - 1);
    }
}

/// Every block entry has packed index at most that of the last diagonal entry.
pub proof fn lemma_block_idx_max(start: int, dim: int, t: int)
    requires
        start >= 0,
        dim >= 1,
        is_block_idx(t, start, dim),
    ensures
        t <= pk_idx(start + dim - 1, start + dim - 1),
{
    let (a, b) = choose|a: int, b: int| 0 <= a <= b < dim && t == #[trigger] pk_idx(start + a, start + b);
    lemma_pk_idx_order(start + a, start + b, start + dim - 1, start + dim - 1);
    lemma_pk_idx_order(start + dim - 1, start + dim - 1, start + a, start + b);
}

/* ---------- partitioned symmetric matrices (Vec<Vector>) ---------- */

/// A dim x dim symmetric matrix partitioned by the columns of its upper triangle:
/// v[c] holds the entries (0, c), ..., (c, c), so entry (r, c) with r <= c is v[c][r].
pub open spec fn part_shape(v: Seq<Vec<f64>>, dim: int) -> bool {
    &&& v.len() == dim
    &&& forall|c: int| 0 <= c < dim ==> (#[trigger] v[c])@.len() == c + 1
}

/// A list of partitioned dim x dim symmetric matrices (the rows of a MatrixSym).
pub open spec fn symmap_shape(m: Seq<Vec<Vec<f64>>>, dim: int) -> bool {
    forall|k: int| 0 <= k < m.len() ==> part_shape((#[trigger] m[k])@, dim)
}

/* ---------- symmetric-matrix predicates ---------- */

/// m is the packed dim x dim identity matrix.
pub open spec fn is_sym_ident_r(m: Seq<real>, dim: int) -> bool {
    &&& m.len() == pk_len(dim)
    &&& forall|r: int, c: int| 0 <= r <= c < dim
            ==> #[trigger] sym_entry(m, r, c) == (if r == c { 1.0real } else { 0.0real })
}

/// The diagonal of m as a vector.
pub open spec fn sym_diag_r(m: Seq<real>, dim: int) -> Seq<real> {
    Seq::new(dim as nat, |i: int| sym_diag(m, i))
}

/// Largest absolute diagonal entry (folded from 0).
pub open spec fn max_abs_diag_r(m: Seq<real>, dim: int) -> real {
    max0_fn_r(|i: int| abs_r(sym_diag(m, i)), dim as nat)
}

/// Frobenius inner product sum_{r, c < dim} A_rc B_rc over the full matrices.
pub open spec fn frob_r(a: Seq<real>, b: Seq<real>, dim: int) -> real {
    sum2_fn_r(|r: int, c: int| sym_entry(a, r, c) * sym_entry(b, r, c), dim as nat)
}

/// Frobenius norm.
pub open spec fn frob_norm_r(a: Seq<real>, dim: int) -> real {
    sqrt_r(frob_r(a, a, dim))
}

/// Matrix-vector product A x as a vector.
pub open spec fn sym_mat_vec_r(a: Seq<real>, dim: int, x: Seq<real>) -> Seq<real> {
    Seq::new(dim as nat, |i: int| sum_fn_r(|j: int| sym_entry(a, i, j) * x[j], dim as nat))
}

/// Induced 1-norm max_i sum_j |A_ij| (folded from 0).
pub open spec fn sym_one_norm_r(a: Seq<real>, dim: int) -> real {
    max0_fn_r(|i: int| sum_fn_r(|j: int| abs_r(sym_entry(a, i, j)), dim as nat), dim as nat)
}

/* ---------- sums and maxima over index functions ---------- */

/// f(0) + ... + f(n - 1).
pub open spec fn sum_fn_r(f: spec_fn(int) -> real, n: nat) -> real
    decreases n,
{
    if n == 0 { 0.0real } else { sum_fn_r(f, (n - 1) as nat) + f(n - 1) }
}

/// Square double sum sum_{r < n} sum_{c < n} f(r, c).
pub open spec fn sum2_fn_r(f: spec_fn(int, int) -> real, n: nat) -> real {
    sum_fn_r(|r: int| sum_fn_r(|c: int| f(r, c), n), n)
}

/// max(0, f(0), ..., f(n - 1)) computed by the left fold `if f(i) > acc { f(i) } else { acc }`.
pub open spec fn max0_fn_r(f: spec_fn(int) -> real, n: nat) -> real
    decreases n,
{
    if n == 0 { 0.0real } else { max_r(max0_fn_r(f, (n - 1) as nat), f(n - 1)) }
}

/// Sums only depend on the values of the summand on 0..n.
pub proof fn lemma_sum_fn_ext(f: spec_fn(int) -> real, g: spec_fn(int) -> real, n: nat)
    requires
        forall|i: int| 0 <= i < n ==> #[trigger] f(i) == g(i),
    ensures
        sum_fn_r(f, n) == sum_fn_r(g, n),
    decreases n,
{
    if n > 0 {
        lemma_sum_fn_ext(f, g, (n - 1) as nat);
    }
}

/// Sum of a pointwise sum.
pub proof fn lemma_sum_fn_add(f: spec_fn(int) -> real, g: spec_fn(int) -> real, h: spec_fn(int) -> real, n: nat)
    requires
        forall|i: int| 0 <= i < n ==> #[trigger] h(i) == f(i) + g(i),
    ensures
        sum_fn_r(h, n) == sum_fn_r(f, n) + sum_fn_r(g, n),
    decreases n,
{
    if n > 0 {
        lemma_sum_fn_add(f, g, h, (n - 1) as nat);
    }
}

/// Changing one summand by delta changes the sum by delta.
pub proof fn lemma_sum_fn_update(f: spec_fn(int) -> real, g: spec_fn(int) -> real, n: nat, k: int, delta: real)
    requires
        0 <= k < n,
        g(k) == f(k) + delta,
        forall|i: int| 0 <= i < n && i != k ==> #[trigger] g(i) == f(i),
    ensures
        sum_fn_r(g, n) == sum_fn_r(f, n) + delta,
    decreases n,
{
    if k == n - 1 {
        lemma_sum_fn_ext(f, g, (n - 1) as nat);
    } else {
        lemma_sum_fn_update(f, g, (n - 1) as nat, k, delta);
    }
}

/// Trailing zero summands do not change the sum.
pub proof fn lemma_sum_fn_tail_zero(f: spec_fn(int) -> real, n: nat, m: nat)
    requires
        n <= m,
        forall|k: int| n <= k < m ==> #[trigger] f(k) == 0.0real,
    ensures
        sum_fn_r(f, m) == sum_fn_r(f, n),
    decreases m - n,
{
    if n < m {
        lemma_sum_fn_tail_zero(f, n, (m - 1) as nat);
    }
}

/// A sum of nonnegative terms is nonnegative.
pub proof fn lemma_sum_fn_nonneg(f: spec_fn(int) -> real, n: nat)
    requires
        forall|i: int| 0 <= i < n ==> #[trigger] f(i) >= 0.0real,
    ensures
        sum_fn_r(f, n) >= 0.0real,
    decreases n,
{
    if n > 0 {
        lemma_sum_fn_nonneg(f, (n - 1) as nat);
    }
}

/// Scaling every summand scales the sum.
pub proof fn lemma_sum_fn_scale(f: spec_fn(int) -> real, g: spec_fn(int) -> real, s: real, n: nat)
    requires
        forall|i: int| 0 <= i < n ==> #[trigger] g(i) == s * f(i),
    ensures
        sum_fn_r(g, n) == s * sum_fn_r(f, n),
    decreases n,
{
    if n > 0 {
        lemma_sum_fn_scale(f, g, s, (n - 1) as nat);
        assert(s * sum_fn_r(f, (n - 1) as nat) + s * f(n - 1) == s * (sum_fn_r(f, (n - 1) as nat) + f(n - 1)))
            by (nonlinear_arith);
    }
}

/// sum_fn_r agrees with sum_r on the sequence of values.
pub proof fn lemma_sum_fn_seq(f: spec_fn(int) -> real, n: nat)
    ensures
        sum_fn_r(f, n) == sum_r(Seq::new(n, f)),
    decreases n,
{
    if n > 0 {
        lemma_sum_fn_seq(f, (n - 1) as nat);
        assert(Seq::new(n, f).drop_last() =~= Seq::new((n - 1) as nat, f));
    }
}

/// The zero summand sums to 0.
pub proof fn lemma_sum_fn_zero(n: nat)
    ensures
        sum_fn_r(|i: int| 0.0real, n) == 0.0real,
    decreases n,
{
    if n > 0 {
        lemma_sum_fn_zero((n - 1) as nat);
    }
}

/// A constant summand sums to n times the constant.
pub proof fn lemma_sum_fn_const(x: real, n: nat)
    ensures
        sum_fn_r(|i: int| x, n) == (n as real) * x,
    decreases n,
{
    if n > 0 {
        let m = (n - 1) as nat;
        lemma_sum_fn_const(x, m);
        assert((m as real) == (n as real) - 1.0real);
        assert((m as real) * x + x == (n as real) * x) by (nonlinear_arith)
            requires (m as real) == (n as real) - 1.0real;
    } else {
        assert((n as real) * x == 0.0real) by (nonlinear_arith)
            requires n as real == 0.0real;
    }
}

/// A sum with a single nonzero term k equals that term.
pub proof fn lemma_sum_fn_single(f: spec_fn(int) -> real, n: nat, k: int)
    requires
        0 <= k < n,
        forall|i: int| 0 <= i < n && i != k ==> #[trigger] f(i) == 0.0real,
    ensures
        sum_fn_r(f, n) == f(k),
{
    lemma_sum_fn_zero(n);
    lemma_sum_fn_update(|i: int| 0.0real, f, n, k, f(k));
}

/// Sums are monotone in the summand.
pub proof fn lemma_sum_fn_le(f: spec_fn(int) -> real, g: spec_fn(int) -> real, n: nat)
    requires
        forall|i: int| 0 <= i < n ==> #[trigger] f(i) <= g(i),
    ensures
        sum_fn_r(f, n) <= sum_fn_r(g, n),
    decreases n,
{
    if n > 0 {
        lemma_sum_fn_le(f, g, (n - 1) as nat);
    }
}

/// Exchange of the order of summation in a finite double sum.
pub proof fn lemma_sum_fn_swap(f: spec_fn(int, int) -> real, m: nat, n: nat)
    ensures
        sum_fn_r(|i: int| sum_fn_r(|j: int| f(i, j), n), m)
            == sum_fn_r(|j: int| sum_fn_r(|i: int| f(i, j), m), n),
    decreases n,
{
    if n == 0 {
        let rows = |i: int| sum_fn_r(|j: int| f(i, j), 0);
        let zero = |i: int| 0.0real;
        assert forall|i: int| 0 <= i < m implies #[trigger] rows(i) == zero(i) by {}
        lemma_sum_fn_ext(rows, zero, m);
        lemma_sum_fn_zero(m);
    } else {
        let n1 = (n - 1) as nat;
        lemma_sum_fn_swap(f, m, n1);
        let rows_n = |i: int| sum_fn_r(|j: int| f(i, j), n);
        let rows_n1 = |i: int| sum_fn_r(|j: int| f(i, j), n1);
        let last_col = |i: int| f(i, n1 as int);
        assert forall|i: int| 0 <= i < m implies #[trigger] rows_n(i) == rows_n1(i) + last_col(i) by {}
        lemma_sum_fn_add(rows_n1, last_col, rows_n, m);
    }
}

/// Double sums only depend on the values of the summand on the square.
pub proof fn lemma_sum2_fn_ext(f: spec_fn(int, int) -> real, g: spec_fn(int, int) -> real, n: nat)
    requires
        forall|r: int, c: int| 0 <= r < n && 0 <= c < n ==> #[trigger] f(r, c) == g(r, c),
    ensures
        sum2_fn_r(f, n) == sum2_fn_r(g, n),
{
    let rows_f = |r: int| sum_fn_r(|c: int| f(r, c), n);
    let rows_g = |r: int| sum_fn_r(|c: int| g(r, c), n);
    assert forall|r: int| 0 <= r < n implies #[trigger] rows_f(r) == rows_g(r) by {
        lemma_sum_fn_ext(|c: int| f(r, c), |c: int| g(r, c), n);
    }
    lemma_sum_fn_ext(rows_f, rows_g, n);
}

/// The fold maximum dominates every term and 0.
pub proof fn lemma_max0_fn_bound(f: spec_fn(int) -> real, n: nat)
    ensures
        max0_fn_r(f, n) >= 0.0real,
        forall|i: int| 0 <= i < n ==> #[trigger] f(i) <= max0_fn_r(f, n),
    decreases n,
{
    if n > 0 {
        lemma_max0_fn_bound(f, (n - 1) as nat);
    }
}

/// The 1-norm is nonnegative.
pub proof fn lemma_sym_one_norm_nonneg(a: Seq<real>, dim: int)
    ensures
        sym_one_norm_r(a, dim) >= 0.0real,
{
    lemma_max0_fn_bound(|i: int| sum_fn_r(|j: int| abs_r(sym_entry(a, i, j)), dim as nat), dim as nat);
}

/// Maxima only depend on the values of the function on 0..n.
pub proof fn lemma_max0_fn_ext(f: spec_fn(int) -> real, g: spec_fn(int) -> real, n: nat)
    requires
        forall|i: int| 0 <= i < n ==> #[trigger] f(i) == g(i),
    ensures
        max0_fn_r(f, n) == max0_fn_r(g, n),
    decreases n,
{
    if n > 0 {
        lemma_max0_fn_ext(f, g, (n - 1) as nat);
    }
}

/// A square double sum grows by one row and one column:
/// sum2(n + 1) = sum2(n) + sum_{r < n} f(r, n) + sum_{c < n} f(n, c) + f(n, n).
pub proof fn lemma_sum2_fn_step(f: spec_fn(int, int) -> real, n: nat)
    ensures
        sum2_fn_r(f, n + 1) == sum2_fn_r(f, n)
            + sum_fn_r(|r: int| f(r, n as int), n)
            + sum_fn_r(|c: int| f(n as int, c), n)
            + f(n as int, n as int),
{
    let row_n = |r: int| sum_fn_r(|c: int| f(r, c), n);
    let row_n1 = |r: int| sum_fn_r(|c: int| f(r, c), n + 1);
    let last_col = |r: int| f(r, n as int);
    assert forall|r: int| 0 <= r < n implies #[trigger] row_n1(r) == row_n(r) + last_col(r) by {}
    lemma_sum_fn_add(row_n, last_col, row_n1, n);
}

/// For a symmetric summand the two border sums coincide, so
/// sum2(n + 1) = sum2(n) + 2 * sum_{r < n} f(r, n) + f(n, n).
pub proof fn lemma_sum2_fn_step_sym(f: spec_fn(int, int) -> real, n: nat)
    requires
        forall|r: int, c: int| 0 <= r < n && 0 <= c < n + 1 ==> #[trigger] f(r, c) == f(c, r),
    ensures
        sum2_fn_r(f, n + 1) == sum2_fn_r(f, n) + 2.0real * sum_fn_r(|r: int| f(r, n as int), n) + f(n as int, n as int),
{
    lemma_sum2_fn_step(f, n);
    lemma_sum_fn_ext(|c: int| f(n as int, c), |r: int| f(r, n as int), n);
}

/// Double sum of a pointwise sum.
pub proof fn lemma_sum2_fn_add(f: spec_fn(int, int) -> real, g: spec_fn(int, int) -> real, h: spec_fn(int, int) -> real, n: nat)
    requires
        forall|r: int, c: int| 0 <= r < n && 0 <= c < n ==> #[trigger] h(r, c) == f(r, c) + g(r, c),
    ensures
        sum2_fn_r(h, n) == sum2_fn_r(f, n) + sum2_fn_r(g, n),
{
    let rows_f = |r: int| sum_fn_r(|c: int| f(r, c), n);
    let rows_g = |r: int| sum_fn_r(|c: int| g(r, c), n);
    let rows_h = |r: int| sum_fn_r(|c: int| h(r, c), n);
    assert forall|r: int| 0 <= r < n implies #[trigger] rows_h(r) == rows_f(r) + rows_g(r) by {
        lemma_sum_fn_add(|c: int| f(r, c), |c: int| g(r, c), |c: int| h(r, c), n);
    }
    lemma_sum_fn_add(rows_f, rows_g, rows_h, n);
}

/// Scaling every term of a double sum scales the double sum.
pub proof fn lemma_sum2_fn_scale(f: spec_fn(int, int) -> real, g: spec_fn(int, int) -> real, s: real, n: nat)
    requires
        forall|r: int, c: int| 0 <= r < n && 0 <= c < n ==> #[trigger] g(r, c) == s * f(r, c),
    ensures
        sum2_fn_r(g, n) == s * sum2_fn_r(f, n),
{
    let rows_f = |r: int| sum_fn_r(|c: int| f(r, c), n);
    let rows_g = |r: int| sum_fn_r(|c: int| g(r, c), n);
    assert forall|r: int| 0 <= r < n implies #[trigger] rows_g(r) == s * rows_f(r) by {
        lemma_sum_fn_scale(|c: int| f(r, c), |c: int| g(r, c), s, n);
    }
    lemma_sum_fn_scale(rows_f, rows_g, s, n);
}

/* ---------- Frobenius product facts ---------- */

/// Column-by-column form of the Frobenius product (what the packed loop computes).
pub proof fn lemma_frob_step(a: Seq<real>, b: Seq<real>, n: int)
    requires
        n >= 0,
    ensures
        frob_r(a, b, n + 1) == frob_r(a, b, n)
            + 2.0real * sum_fn_r(|r: int| a[pk_idx(r, n)] * b[pk_idx(r, n)], n as nat)
            + a[pk_idx(n, n)] * b[pk_idx(n, n)],
{
    let f = |r: int, c: int| sym_entry(a, r, c) * sym_entry(b, r, c);
    lemma_sum2_fn_step_sym(f, n as nat);
    lemma_sum_fn_ext(|r: int| f(r, n), |r: int| a[pk_idx(r, n)] * b[pk_idx(r, n)], n as nat);
}

/// <A, A>_F >= 0.
pub proof fn lemma_frob_self_nonneg(a: Seq<real>, dim: int)
    requires
        dim >= 0,
    ensures
        frob_r(a, a, dim) >= 0.0real,
{
    let f = |r: int, c: int| sym_entry(a, r, c) * sym_entry(a, r, c);
    let rows = |r: int| sum_fn_r(|c: int| f(r, c), dim as nat);
    assert forall|r: int| 0 <= r < dim implies #[trigger] rows(r) >= 0.0real by {
        assert forall|c: int| 0 <= c < dim implies #[trigger] f(r, c) >= 0.0real by {
            assert(sym_entry(a, r, c) * sym_entry(a, r, c) >= 0.0real) by (nonlinear_arith);
        }
        lemma_sum_fn_nonneg(|c: int| f(r, c), dim as nat);
    }
    lemma_sum_fn_nonneg(rows, dim as nat);
}

/* ---------- sweep bookkeeping ---------- */
//
// Several routines traverse the packed triangle column by column (column i,
// rows j = 0..i) and, at entry (j, i), contribute to both row j and row i.
// `swept(k, l, i, j)` says whether the unordered pair {k, l} has been visited
// when the traversal is about to process entry (j, i); `partial_row` is the
// corresponding partial row sum of a term function t.

/// The pair {k, l} has been visited before position (column i, row j).
pub open spec fn swept(k: int, l: int, i: int, j: int) -> bool {
    let (lo, hi) = if k <= l { (k, l) } else { (l, k) };
    hi < i || (hi == i && lo < j)
}

/// Partial row sum of t(k, .) over the visited pairs.
pub open spec fn partial_row(t: spec_fn(int, int) -> real, dim: int, k: int, i: int, j: int) -> real {
    sum_fn_r(|l: int| if swept(k, l, i, j) { t(k, l) } else { 0.0real }, dim as nat)
}

/// Nothing has been visited at the start.
pub proof fn lemma_partial_row_init(t: spec_fn(int, int) -> real, dim: int, k: int)
    requires
        dim >= 0,
        k >= 0,
    ensures
        partial_row(t, dim, k, 0, 0) == 0.0real,
{
    let f = |l: int| if swept(k, l, 0, 0) { t(k, l) } else { 0.0real };
    let z = |l: int| 0.0real;
    assert forall|l: int| 0 <= l < dim implies #[trigger] f(l) == z(l) by {}
    lemma_sum_fn_ext(f, z, dim as nat);
    lemma_sum_fn_scale(z, z, 0.0real, dim as nat);
}

/// Finishing column i (position (i, i + 1)) is the same as starting column i + 1.
pub proof fn lemma_partial_row_next_col(t: spec_fn(int, int) -> real, dim: int, k: int, i: int)
    requires
        dim >= 0,
        k >= 0,
    ensures
        partial_row(t, dim, k, i, i + 1) == partial_row(t, dim, k, i + 1, 0),
{
    let f = |l: int| if swept(k, l, i, i + 1) { t(k, l) } else { 0.0real };
    let g = |l: int| if swept(k, l, i + 1, 0) { t(k, l) } else { 0.0real };
    assert forall|l: int| 0 <= l < dim implies #[trigger] f(l) == g(l) by {}
    lemma_sum_fn_ext(f, g, dim as nat);
}

/// After the last column every pair has been visited.
pub proof fn lemma_partial_row_done(t: spec_fn(int, int) -> real, dim: int, k: int)
    requires
        0 <= k < dim,
    ensures
        partial_row(t, dim, k, dim, 0) == sum_fn_r(|l: int| t(k, l), dim as nat),
{
    let f = |l: int| if swept(k, l, dim, 0) { t(k, l) } else { 0.0real };
    let g = |l: int| t(k, l);
    assert forall|l: int| 0 <= l < dim implies #[trigger] f(l) == g(l) by {}
    lemma_sum_fn_ext(f, g, dim as nat);
}

/// Visiting an off-diagonal entry (j, i), j < i, adds t(j, i) to row j and t(i, j) to row i
/// and leaves every other row unchanged.
pub proof fn lemma_partial_row_offdiag(t: spec_fn(int, int) -> real, dim: int, i: int, j: int)
    requires
        0 <= j < i < dim,
    ensures
        partial_row(t, dim, j, i, j + 1) == partial_row(t, dim, j, i, j) + t(j, i),
        partial_row(t, dim, i, i, j + 1) == partial_row(t, dim, i, i, j) + t(i, j),
        forall|k: int| 0 <= k < dim && k != i && k != j
            ==> #[trigger] partial_row(t, dim, k, i, j + 1) == partial_row(t, dim, k, i, j),
{
    lemma_sum_fn_update(
        |l: int| if swept(j, l, i, j) { t(j, l) } else { 0.0real },
        |l: int| if swept(j, l, i, j + 1) { t(j, l) } else { 0.0real },
        dim as nat, i, t(j, i),
    );
    lemma_sum_fn_update(
        |l: int| if swept(i, l, i, j) { t(i, l) } else { 0.0real },
        |l: int| if swept(i, l, i, j + 1) { t(i, l) } else { 0.0real },
        dim as nat, j, t(i, j),
    );
    assert forall|k: int| 0 <= k < dim && k != i && k != j
        implies #[trigger] partial_row(t, dim, k, i, j + 1) == partial_row(t, dim, k, i, j) by {
        lemma_sum_fn_ext(
            |l: int| if swept(k, l, i, j) { t(k, l) } else { 0.0real },
            |l: int| if swept(k, l, i, j + 1) { t(k, l) } else { 0.0real },
            dim as nat,
        );
    }
}

/// Visiting the diagonal entry (i, i) adds t(i, i) to row i and leaves every other row unchanged.
pub proof fn lemma_partial_row_diag(t: spec_fn(int, int) -> real, dim: int, i: int)
    requires
        0 <= i < dim,
    ensures
        partial_row(t, dim, i, i, i + 1) == partial_row(t, dim, i, i, i) + t(i, i),
        forall|k: int| 0 <= k < dim && k != i
            ==> #[trigger] partial_row(t, dim, k, i, i + 1) == partial_row(t, dim, k, i, i),
{
    lemma_sum_fn_update(
        |l: int| if swept(i, l, i, i) { t(i, l) } else { 0.0real },
        |l: int| if swept(i, l, i, i + 1) { t(i, l) } else { 0.0real },
        dim as nat, i, t(i, i),
    );
    assert forall|k: int| 0 <= k < dim && k != i
        implies #[trigger] partial_row(t, dim, k, i, i + 1) == partial_row(t, dim, k, i, i) by {
        lemma_sum_fn_ext(
            |l: int| if swept(k, l, i, i) { t(k, l) } else { 0.0real },
            |l: int| if swept(k, l, i, i + 1) { t(k, l) } else { 0.0real },
            dim as nat,
        );
    }
}

} // verus!

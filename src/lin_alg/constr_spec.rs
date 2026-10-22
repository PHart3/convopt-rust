// Spec-level view of the constraint operator of the standard-form SDP (not trusted).
//
// The dense linear constraints are stored as a column-major matrix with
// row_dim rows and pk_len(dim) columns: row i is the packed symmetric matrix
// M_i, so entry (i, n) sits at n * row_dim + i.  A list of index pairs (r, c)
// with r < c adds the zero-pattern constraints X_rc = 0.  The operator is
//
//     A(X) = ( <M_i, X>_packed )_i  ++  ( X_rc )_(r, c),
//
// where <., .>_packed is the plain dot product of packed vectors.  Its adjoint
// under the Frobenius product therefore carries a factor 1/2 on every
// off-diagonal packed entry (the Frobenius product counts those twice).  This
// file defines A, A*, the Gram matrix A A* of the dense rows, the sums over
// the packed triangle needed to relate them, and proves the adjoint identity
//
//     <A(X), y> = <X, A*(y)>_F.

use vstd::prelude::*;
#[cfg(verus_keep_ghost)]
use vstd::seq::*;
#[cfg(verus_keep_ghost)]
use vstd::arithmetic::div_mod::lemma_fundamental_div_mod;
#[cfg(verus_keep_ghost)]
use crate::trusted::rv;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::*;

verus! {

broadcast use crate::trusted::group_real_model;

/* ---------- sums over the packed triangle ---------- */

/// sum_{c < dim} sum_{r <= c} f(r, c): the sum over the stored (upper) triangle,
/// column by column, in packed order.
pub open spec fn tri_sum_r(f: spec_fn(int, int) -> real, dim: nat) -> real {
    sum_fn_r(|c: int| sum_fn_r(|r: int| f(r, c), (c + 1) as nat), dim)
}

/// Frobenius weight of a packed entry: 1 on the diagonal, 2 off it.
pub open spec fn pk_wt(r: int, c: int) -> real {
    if r == c { 1.0real } else { 2.0real }
}

/// Adjoint weight of a packed entry: 1 on the diagonal, 1/2 off it (pk_wt * adj_wt == 1).
pub open spec fn adj_wt(r: int, c: int) -> real {
    if r == c { 1.0real } else { 0.5real }
}

/// Triangle sums only depend on the summand on the triangle.
pub proof fn lemma_tri_sum_ext(f: spec_fn(int, int) -> real, g: spec_fn(int, int) -> real, dim: nat)
    requires
        forall|r: int, c: int| 0 <= r <= c < dim ==> #[trigger] f(r, c) == g(r, c),
    ensures
        tri_sum_r(f, dim) == tri_sum_r(g, dim),
{
    let cols_f = |c: int| sum_fn_r(|r: int| f(r, c), (c + 1) as nat);
    let cols_g = |c: int| sum_fn_r(|r: int| g(r, c), (c + 1) as nat);
    assert forall|c: int| 0 <= c < dim implies #[trigger] cols_f(c) == cols_g(c) by {
        lemma_sum_fn_ext(|r: int| f(r, c), |r: int| g(r, c), (c + 1) as nat);
    }
    lemma_sum_fn_ext(cols_f, cols_g, dim);
}

/// Triangle sum of a pointwise sum.
pub proof fn lemma_tri_sum_add(f: spec_fn(int, int) -> real, g: spec_fn(int, int) -> real, h: spec_fn(int, int) -> real, dim: nat)
    requires
        forall|r: int, c: int| 0 <= r <= c < dim ==> #[trigger] h(r, c) == f(r, c) + g(r, c),
    ensures
        tri_sum_r(h, dim) == tri_sum_r(f, dim) + tri_sum_r(g, dim),
{
    let cols_f = |c: int| sum_fn_r(|r: int| f(r, c), (c + 1) as nat);
    let cols_g = |c: int| sum_fn_r(|r: int| g(r, c), (c + 1) as nat);
    let cols_h = |c: int| sum_fn_r(|r: int| h(r, c), (c + 1) as nat);
    assert forall|c: int| 0 <= c < dim implies #[trigger] cols_h(c) == cols_f(c) + cols_g(c) by {
        lemma_sum_fn_add(|r: int| f(r, c), |r: int| g(r, c), |r: int| h(r, c), (c + 1) as nat);
    }
    lemma_sum_fn_add(cols_f, cols_g, cols_h, dim);
}

/// Scaling every term of a triangle sum scales the sum.
pub proof fn lemma_tri_sum_scale(f: spec_fn(int, int) -> real, g: spec_fn(int, int) -> real, s: real, dim: nat)
    requires
        forall|r: int, c: int| 0 <= r <= c < dim ==> #[trigger] g(r, c) == s * f(r, c),
    ensures
        tri_sum_r(g, dim) == s * tri_sum_r(f, dim),
{
    let cols_f = |c: int| sum_fn_r(|r: int| f(r, c), (c + 1) as nat);
    let cols_g = |c: int| sum_fn_r(|r: int| g(r, c), (c + 1) as nat);
    assert forall|c: int| 0 <= c < dim implies #[trigger] cols_g(c) == s * cols_f(c) by {
        lemma_sum_fn_scale(|r: int| f(r, c), |r: int| g(r, c), s, (c + 1) as nat);
    }
    lemma_sum_fn_scale(cols_f, cols_g, s, dim);
}

/// A triangle sum with a single nonzero term (r0, c0) equals that term.
pub proof fn lemma_tri_sum_single(f: spec_fn(int, int) -> real, dim: nat, r0: int, c0: int)
    requires
        0 <= r0 <= c0 < dim,
        forall|r: int, c: int| 0 <= r <= c < dim && !(r == r0 && c == c0) ==> #[trigger] f(r, c) == 0.0real,
    ensures
        tri_sum_r(f, dim) == f(r0, c0),
{
    let cols = |c: int| sum_fn_r(|r: int| f(r, c), (c + 1) as nat);
    assert forall|c: int| 0 <= c < dim && c != c0 implies #[trigger] cols(c) == 0.0real by {
        lemma_sum_fn_zero((c + 1) as nat);
        lemma_sum_fn_ext(|r: int| f(r, c), |r: int| 0.0real, (c + 1) as nat);
    }
    lemma_sum_fn_single(|r: int| f(r, c0), (c0 + 1) as nat, r0);
    lemma_sum_fn_single(cols, dim, c0);
}

/// A triangle sum of sums is the sum of the triangle sums (exchange of summation).
pub proof fn lemma_tri_sum_swap(f: spec_fn(int, int, int) -> real, dim: nat, n: nat)
    ensures
        tri_sum_r(|r: int, c: int| sum_fn_r(|t: int| f(r, c, t), n), dim)
            == sum_fn_r(|t: int| tri_sum_r(|r: int, c: int| f(r, c, t), dim), n),
    decreases n,
{
    let lhs_n = |r: int, c: int| sum_fn_r(|t: int| f(r, c, t), n);
    if n == 0 {
        assert forall|r: int, c: int| 0 <= r <= c < dim implies #[trigger] lhs_n(r, c) == 0.0real * lhs_n(r, c) by {}
        lemma_tri_sum_scale(lhs_n, lhs_n, 0.0real, dim);
    } else {
        let n1 = (n - 1) as nat;
        lemma_tri_sum_swap(f, dim, n1);
        let lhs_n1 = |r: int, c: int| sum_fn_r(|t: int| f(r, c, t), n1);
        let last = |r: int, c: int| f(r, c, n1 as int);
        assert forall|r: int, c: int| 0 <= r <= c < dim implies #[trigger] lhs_n(r, c) == lhs_n1(r, c) + last(r, c) by {}
        lemma_tri_sum_add(lhs_n1, last, lhs_n, dim);
    }
}

/// A flat sum over the packed indices is the triangle sum of the decoded summand.
pub proof fn lemma_tri_sum_packed(g: spec_fn(int) -> real, dim: nat)
    ensures
        pk_len(dim as int) >= 0,
        sum_fn_r(g, pk_len(dim as int) as nat) == tri_sum_r(|r: int, c: int| g(pk_idx(r, c)), dim),
    decreases dim,
{
    lemma_pk_col_start_zero();
    lemma_pk_col_start_monotone(0, dim as int);
    let tri = |r: int, c: int| g(pk_idx(r, c));
    if dim > 0 {
        let d1 = (dim - 1) as nat;
        lemma_tri_sum_packed(g, d1);
        lemma_pk_col_start_step(d1 as int);
        lemma_sum_fn_split(g, pk_len(d1 as int) as nat, dim);
        let shifted = |i: int| g(pk_len(d1 as int) + i);
        let last_col = |r: int| tri(r, d1 as int);
        assert forall|i: int| 0 <= i < dim implies #[trigger] shifted(i) == last_col(i) by {}
        lemma_sum_fn_ext(shifted, last_col, dim);
        assert(tri_sum_r(tri, dim) == tri_sum_r(tri, d1) + sum_fn_r(last_col, dim));
    }
}

/// The Frobenius product as a weighted triangle sum (what the packed loops compute).
pub proof fn lemma_frob_tri(a: Seq<real>, b: Seq<real>, dim: nat)
    ensures
        frob_r(a, b, dim as int) == tri_sum_r(|r: int, c: int| pk_wt(r, c) * (a[pk_idx(r, c)] * b[pk_idx(r, c)]), dim),
    decreases dim,
{
    if dim > 0 {
        let d1 = (dim - 1) as nat;
        lemma_frob_tri(a, b, d1);
        lemma_frob_step(a, b, d1 as int);
        let f = |r: int, c: int| pk_wt(r, c) * (a[pk_idx(r, c)] * b[pk_idx(r, c)]);
        let col = |r: int| f(r, d1 as int);
        let prod = |r: int| a[pk_idx(r, d1 as int)] * b[pk_idx(r, d1 as int)];
        assert forall|r: int| 0 <= r < d1 implies #[trigger] col(r) == 2.0real * prod(r) by {}
        lemma_sum_fn_scale(prod, col, 2.0real, d1);
        assert(tri_sum_r(f, dim) == tri_sum_r(f, d1) + sum_fn_r(col, dim));
        assert(sum_fn_r(col, dim) == sum_fn_r(col, d1) + col(d1 as int));
        assert(col(d1 as int) == a[pk_idx(d1 as int, d1 as int)] * b[pk_idx(d1 as int, d1 as int)]);
    }
}

/* ---------- the constraint operator ---------- */

/// Entry (i, n) of the stored constraint matrix: entry n of the packed matrix M_i.
pub open spec fn constr_entry(m: Seq<real>, row_dim: int, i: int, n: int) -> real {
    cm_entry(m, row_dim, i, n)
}

/// Number of rows of a constraint matrix with `len` entries and `col_dim` columns.
pub open spec fn rows_of(len: int, col_dim: int) -> int {
    len / col_dim
}

/// When col_dim divides len, the row count times col_dim is len.
pub proof fn lemma_rows_of_exact(len: int, col_dim: int)
    requires
        col_dim > 0,
        len % col_dim == 0,
    ensures
        rows_of(len, col_dim) * col_dim == len,
{
    lemma_fundamental_div_mod(len, col_dim);
    assert(col_dim * (len / col_dim) == (len / col_dim) * col_dim) by (nonlinear_arith);
}

/// The zero-pattern list addresses strictly upper-triangular entries of a dim x dim matrix.
pub open spec fn valid_zero_list(list: Seq<(usize, usize)>, dim: int) -> bool {
    forall|k: int| 0 <= k < list.len() ==> (#[trigger] list[k]).0 < list[k].1 < dim
}

/// The zero-pattern list fits a packed matrix of `len` entries (dim is determined by len).
pub open spec fn zero_list_fits(list: Seq<(usize, usize)>, len: int) -> bool {
    exists|dim: int| dim_fits(dim) && #[trigger] pk_len(dim) == len && valid_zero_list(list, dim)
}

/// Packed index of the k-th zero-pattern entry.
pub open spec fn zero_idx(list: Seq<(usize, usize)>, k: int) -> int {
    pk_idx(list[k].0 as int, list[k].1 as int)
}

/// <M_i, x>_packed: the plain dot product of row i with the packed vector x.
pub open spec fn row_dot_r(m: Seq<real>, row_dim: int, i: int, x: Seq<real>) -> real {
    sum_fn_r(|n: int| constr_entry(m, row_dim, i, n) * x[n], x.len())
}

/// Entry i of A(x): a dense row for i < row_dim, a zero-pattern entry of x after that.
pub open spec fn action_entry_r(m: Seq<real>, row_dim: int, list: Seq<(usize, usize)>, x: Seq<real>, i: int) -> real {
    if i < row_dim { row_dot_r(m, row_dim, i, x) } else { x[zero_idx(list, i - row_dim)] }
}

/// A(x): the dense rows applied to x, followed by the zero-pattern entries of x.
pub open spec fn constraint_action_r(m: Seq<real>, row_dim: int, list: Seq<(usize, usize)>, x: Seq<real>) -> Seq<real> {
    Seq::new((row_dim + list.len()) as nat, |i: int| action_entry_r(m, row_dim, list, x, i))
}

/// sum_i M_i[n] y_i: entry n of the plain linear combination of the dense rows.
pub open spec fn col_comb_r(m: Seq<real>, row_dim: int, y: Seq<real>, n: int) -> real {
    sum_fn_r(|i: int| constr_entry(m, row_dim, i, n) * y[i], row_dim as nat)
}

/// The multiplier y[row_dim + k] of the k-th zero-pattern constraint if it addresses (r, c), else 0.
pub open spec fn zero_term_r(list: Seq<(usize, usize)>, row_dim: int, y: Seq<real>, r: int, c: int, k: int) -> real {
    if list[k].0 == r && list[k].1 == c { y[row_dim + k] } else { 0.0real }
}

/// Sum of the multipliers over the first `len` zero-pattern constraints addressing (r, c).
pub open spec fn zero_comb_upto(list: Seq<(usize, usize)>, row_dim: int, y: Seq<real>, r: int, c: int, len: nat) -> real {
    sum_fn_r(|k: int| zero_term_r(list, row_dim, y, r, c, k), len)
}

/// Total multiplier of the zero-pattern constraints on entry (r, c).
pub open spec fn zero_comb_r(list: Seq<(usize, usize)>, row_dim: int, y: Seq<real>, r: int, c: int) -> real {
    zero_comb_upto(list, row_dim, y, r, c, list.len())
}

/// Entry (r, c), r <= c, of the adjoint A*(y) in packed form.
pub open spec fn adjoint_entry_r(m: Seq<real>, row_dim: int, list: Seq<(usize, usize)>, y: Seq<real>, r: int, c: int) -> real {
    adj_wt(r, c) * col_comb_r(m, row_dim, y, pk_idx(r, c)) + 0.5real * zero_comb_r(list, row_dim, y, r, c)
}

/// a is the packed dim x dim adjoint A*(y).
pub open spec fn is_constraint_adjoint(a: Seq<real>, dim: int, m: Seq<real>, row_dim: int, list: Seq<(usize, usize)>, y: Seq<real>) -> bool {
    &&& a.len() == pk_len(dim)
    &&& forall|r: int, c: int| 0 <= r <= c < dim
            ==> #[trigger] a[pk_idx(r, c)] == adjoint_entry_r(m, row_dim, list, y, r, c)
}

/// Entry (a, b) of the Gram matrix A A* of the dense rows (zero-pattern rows excluded):
/// sum over the packed triangle of adj_wt * M_a[n] * M_b[n].
pub open spec fn gram_entry_r(m: Seq<real>, row_dim: int, dim: int, a: int, b: int) -> real {
    tri_sum_r(|r: int, c: int| adj_wt(r, c) * (constr_entry(m, row_dim, a, pk_idx(r, c)) * constr_entry(m, row_dim, b, pk_idx(r, c))), dim as nat)
}

/// g is the packed Gram matrix of the dense rows.
pub open spec fn is_constraint_gram(g: Seq<real>, m: Seq<real>, row_dim: int, dim: int) -> bool {
    &&& g.len() == pk_len(row_dim)
    &&& forall|a: int, b: int| 0 <= a <= b < row_dim
            ==> #[trigger] g[pk_idx(a, b)] == gram_entry_r(m, row_dim, dim, a, b)
}

/* ---------- lemmas ---------- */


/// The packed dot product of column n of the constraint matrix (the slice starting at lo = n * row_dim)
/// with the leading row_dim multipliers is entry n of the linear combination of the rows.
pub proof fn lemma_col_slice_dot(m: Seq<f64>, row_dim: int, y: Seq<f64>, n: int, lo: int)
    requires
        0 <= n,
        0 <= row_dim,
        lo == n * row_dim,
        lo + row_dim <= m.len(),
        row_dim <= y.len(),
    ensures
        dot_r(rvs(m.subrange(lo, lo + row_dim)), rvs(y.subrange(0, row_dim)))
            == col_comb_r(rvs(m), row_dim, rvs(y), n),
{
    let col = rvs(m.subrange(lo, lo + row_dim));
    let yy = rvs(y.subrange(0, row_dim));
    assert(lo >= 0);
    lemma_dot_sum_fn(col, yy);
    let f = |i: int| col[i] * yy[i];
    let g = |i: int| constr_entry(rvs(m), row_dim, i, n) * rvs(y)[i];
    assert forall|i: int| 0 <= i < row_dim implies #[trigger] f(i) == g(i) by {
        assert(col[i] == rv(m[lo + i]));
        assert(cm_idx(row_dim, i, n) == lo + i);
    }
    lemma_sum_fn_ext(f, g, row_dim as nat);
}

/// Diagonal positions receive no zero-pattern multiplier (the list is strictly upper triangular).
pub proof fn lemma_zero_comb_diag(list: Seq<(usize, usize)>, dim: int, row_dim: int, y: Seq<real>, r: int)
    requires
        valid_zero_list(list, dim),
    ensures
        zero_comb_r(list, row_dim, y, r, r) == 0.0real,
{
    let f = |k: int| zero_term_r(list, row_dim, y, r, r, k);
    assert forall|k: int| 0 <= k < list.len() implies #[trigger] f(k) == 0.0real by {
        assert(list[k].0 < list[k].1);
    }
    lemma_sum_fn_all_zero(f, list.len());
}

/// The adjoint identity <A(x), y> = <x, A*(y)>_F, with the factors 1/2 of the
/// off-diagonal packed entries of A*(y) cancelling the factors 2 of the Frobenius product.
pub proof fn lemma_constraint_adjoint(m: Seq<real>, row_dim: int, dim: int, list: Seq<(usize, usize)>, x: Seq<real>, y: Seq<real>, a: Seq<real>)
    requires
        row_dim >= 0,
        dim >= 0,
        valid_zero_list(list, dim),
        x.len() == pk_len(dim),
        y.len() == row_dim + list.len(),
        is_constraint_adjoint(a, dim, m, row_dim, list, y),
    ensures
        dot_r(constraint_action_r(m, row_dim, list, x), y) == frob_r(x, a, dim),
{
    let d = dim as nat;
    let nl = list.len();
    let ax = constraint_action_r(m, row_dim, list, x);
    lemma_pk_col_start_monotone(0, dim);
    let np = pk_len(dim) as nat;

    // the Frobenius product as a triangle sum; each term splits into a dense and a zero-pattern part
    lemma_frob_tri(x, a, d);
    let fr = |r: int, c: int| pk_wt(r, c) * (x[pk_idx(r, c)] * a[pk_idx(r, c)]);
    let dense = |r: int, c: int| x[pk_idx(r, c)] * col_comb_r(m, row_dim, y, pk_idx(r, c));
    let zero = |r: int, c: int| x[pk_idx(r, c)] * zero_comb_r(list, row_dim, y, r, c);
    assert forall|r: int, c: int| 0 <= r <= c < dim implies #[trigger] fr(r, c) == dense(r, c) + zero(r, c) by {
        let xn = x[pk_idx(r, c)];
        let cc = col_comb_r(m, row_dim, y, pk_idx(r, c));
        let zc = zero_comb_r(list, row_dim, y, r, c);
        assert(a[pk_idx(r, c)] == adj_wt(r, c) * cc + 0.5real * zc);
        if r == c {
            lemma_zero_comb_diag(list, dim, row_dim, y, r);
            assert(fr(r, c) == 1.0real * (xn * (1.0real * cc + 0.5real * 0.0real)));
            assert(1.0real * (xn * (1.0real * cc + 0.5real * 0.0real)) == xn * cc + xn * 0.0real);
        } else {
            assert(fr(r, c) == 2.0real * (xn * (0.5real * cc + 0.5real * zc)));
            assert(2.0real * (xn * (0.5real * cc + 0.5real * zc)) == xn * cc + xn * zc) by (nonlinear_arith);
        }
    }
    lemma_tri_sum_add(dense, zero, fr, d);
    assert(frob_r(x, a, dim) == tri_sum_r(dense, d) + tri_sum_r(zero, d));

    // dense part: sum_n x_n sum_i M_i[n] y_i = sum_i y_i <M_i, x> = sum_{i < row_dim} A(x)_i y_i
    let dense_flat = |n: int| x[n] * col_comb_r(m, row_dim, y, n);
    lemma_tri_sum_packed(dense_flat, d);
    let dense_tri = |r: int, c: int| dense_flat(pk_idx(r, c));
    assert forall|r: int, c: int| 0 <= r <= c < dim implies #[trigger] dense_tri(r, c) == dense(r, c) by {}
    lemma_tri_sum_ext(dense_tri, dense, d);
    let dbl = |i: int, j: int| constr_entry(m, row_dim, j, i) * x[i] * y[j];
    let dense_rows = |i: int| sum_fn_r(|j: int| dbl(i, j), row_dim as nat);
    assert forall|n: int| 0 <= n < np implies #[trigger] dense_flat(n) == dense_rows(n) by {
        let g = |i: int| constr_entry(m, row_dim, i, n) * y[i];
        let h = |j: int| dbl(n, j);
        let xn = x[n];
        assert forall|i: int| 0 <= i < row_dim implies #[trigger] h(i) == xn * g(i) by {
            let (u, v) = (constr_entry(m, row_dim, i, n), y[i]);
            assert(u * xn * v == xn * (u * v)) by (nonlinear_arith);
        }
        lemma_sum_fn_scale(g, h, xn, row_dim as nat);
    }
    lemma_sum_fn_ext(dense_flat, dense_rows, np);
    lemma_sum_fn_swap(dbl, np, row_dim as nat);
    let dense_cols = |j: int| sum_fn_r(|i: int| dbl(i, j), np);
    let axy = |i: int| ax[i] * y[i];
    assert forall|j: int| 0 <= j < row_dim implies #[trigger] axy(j) == dense_cols(j) by {
        let g = |n: int| constr_entry(m, row_dim, j, n) * x[n];
        let h = |i: int| dbl(i, j);
        let yj = y[j];
        assert forall|n: int| 0 <= n < np implies #[trigger] h(n) == yj * g(n) by {
            let (u, v) = (constr_entry(m, row_dim, j, n), x[n]);
            assert(u * v * yj == yj * (u * v));
        }
        lemma_sum_fn_scale(g, h, yj, np);
        assert(ax[j] == row_dot_r(m, row_dim, j, x));
        assert(row_dot_r(m, row_dim, j, x) == sum_fn_r(g, np));
        let s = sum_fn_r(g, np);
        assert(s * yj == yj * s);
    }
    lemma_sum_fn_ext(axy, dense_cols, row_dim as nat);
    assert(tri_sum_r(dense, d) == sum_fn_r(axy, row_dim as nat));

    // zero-pattern part: sum_(r,c) x_rc sum_k [list_k = (r,c)] y_(row_dim+k) = sum_k A(x)_(row_dim+k) y_(row_dim+k)
    let trip = |r: int, c: int, t: int| x[pk_idx(r, c)] * zero_term_r(list, row_dim, y, r, c, t);
    let zero_sums = |r: int, c: int| sum_fn_r(|t: int| trip(r, c, t), nl);
    assert forall|r: int, c: int| 0 <= r <= c < dim implies #[trigger] zero(r, c) == zero_sums(r, c) by {
        let g = |k: int| zero_term_r(list, row_dim, y, r, c, k);
        let h = |t: int| trip(r, c, t);
        let xn = x[pk_idx(r, c)];
        assert forall|k: int| 0 <= k < nl implies #[trigger] h(k) == xn * g(k) by {}
        lemma_sum_fn_scale(g, h, xn, nl);
    }
    lemma_tri_sum_ext(zero, zero_sums, d);
    lemma_tri_sum_swap(trip, d, nl);
    let zero_terms = |t: int| tri_sum_r(|r: int, c: int| trip(r, c, t), d);
    let tail = |i: int| axy(row_dim + i);
    assert forall|t: int| 0 <= t < nl implies #[trigger] zero_terms(t) == tail(t) by {
        let (r0, c0) = (list[t].0 as int, list[t].1 as int);
        assert(list[t].0 < list[t].1 < dim);
        let f = |r: int, c: int| trip(r, c, t);
        assert forall|r: int, c: int| 0 <= r <= c < dim && !(r == r0 && c == c0) implies #[trigger] f(r, c) == 0.0real by {
            let xn = x[pk_idx(r, c)];
            assert(zero_term_r(list, row_dim, y, r, c, t) == 0.0real);
            assert(xn * 0.0real == 0.0real);
        }
        lemma_tri_sum_single(f, d, r0, c0);
        assert(ax[row_dim + t] == x[zero_idx(list, t)]);
        assert(f(r0, c0) == x[pk_idx(r0, c0)] * y[row_dim + t]);
    }
    lemma_sum_fn_ext(zero_terms, tail, nl);
    assert(tri_sum_r(zero, d) == sum_fn_r(tail, nl));

    // assemble: <A(x), y> splits into the dense rows and the zero-pattern rows
    lemma_dot_sum_fn(ax, y);
    lemma_sum_fn_split(axy, row_dim as nat, nl);
}

/// The Gram entries are symmetric.
pub proof fn lemma_gram_entry_sym(m: Seq<real>, row_dim: int, dim: int, a: int, b: int)
    requires
        dim >= 0,
    ensures
        gram_entry_r(m, row_dim, dim, a, b) == gram_entry_r(m, row_dim, dim, b, a),
{
    let f = |r: int, c: int| adj_wt(r, c) * (constr_entry(m, row_dim, a, pk_idx(r, c)) * constr_entry(m, row_dim, b, pk_idx(r, c)));
    let g = |r: int, c: int| adj_wt(r, c) * (constr_entry(m, row_dim, b, pk_idx(r, c)) * constr_entry(m, row_dim, a, pk_idx(r, c)));
    assert forall|r: int, c: int| 0 <= r <= c < dim implies #[trigger] f(r, c) == g(r, c) by {
        let (u, v) = (constr_entry(m, row_dim, a, pk_idx(r, c)), constr_entry(m, row_dim, b, pk_idx(r, c)));
        assert(u * v == v * u);
    }
    lemma_tri_sum_ext(f, g, dim as nat);
}

/// The Gram matrix acts as A A* on the dense rows: for the adjoint a_star = A*(y) formed
/// without zero-pattern constraints, (G y)_i = <M_i, A*(y)>_packed = A(A*(y))_i.
pub proof fn lemma_gram_action(g: Seq<real>, m: Seq<real>, row_dim: int, dim: int, y: Seq<real>, a_star: Seq<real>)
    requires
        row_dim >= 0,
        dim >= 0,
        is_constraint_gram(g, m, row_dim, dim),
        y.len() == row_dim,
        is_constraint_adjoint(a_star, dim, m, row_dim, Seq::empty(), y),
    ensures
        sym_mat_vec_r(g, row_dim, y) == constraint_action_r(m, row_dim, Seq::empty(), a_star),
{
    let d = dim as nat;
    let nr = row_dim as nat;
    let gy = sym_mat_vec_r(g, row_dim, y);
    let ax = constraint_action_r(m, row_dim, Seq::empty(), a_star);
    assert forall|i: int| 0 <= i < row_dim implies gy[i] == ax[i] by {
        // (G y)_i = sum_t G_it y_t, with G_it the triangle sum of adj_wt M_i[n] M_t[n]
        lemma_sym_mat_vec_entry(g, row_dim, y, i);
        let gterm = |j: int| sym_entry(g, i, j) * y[j];
        let trip = |r: int, c: int, t: int| adj_wt(r, c) * (constr_entry(m, row_dim, i, pk_idx(r, c)) * constr_entry(m, row_dim, t, pk_idx(r, c))) * y[t];
        let gterm_tri = |t: int| tri_sum_r(|r: int, c: int| trip(r, c, t), d);
        assert forall|t: int| 0 <= t < row_dim implies #[trigger] gterm(t) == gterm_tri(t) by {
            if i <= t {
                assert(sym_entry(g, i, t) == gram_entry_r(m, row_dim, dim, i, t));
            } else {
                assert(sym_entry(g, i, t) == gram_entry_r(m, row_dim, dim, t, i));
                lemma_gram_entry_sym(m, row_dim, dim, t, i);
            }
            let f = |r: int, c: int| adj_wt(r, c) * (constr_entry(m, row_dim, i, pk_idx(r, c)) * constr_entry(m, row_dim, t, pk_idx(r, c)));
            let ft = |r: int, c: int| trip(r, c, t);
            let yt = y[t];
            assert forall|r: int, c: int| 0 <= r <= c < dim implies #[trigger] ft(r, c) == yt * f(r, c) by {
                let v = f(r, c);
                assert(ft(r, c) == v * yt);
                assert(v * yt == yt * v);
            }
            lemma_tri_sum_scale(f, ft, yt, d);
            assert(gram_entry_r(m, row_dim, dim, i, t) == tri_sum_r(f, d));
            let s = tri_sum_r(f, d);
            assert(s * yt == yt * s);
        }
        lemma_sum_fn_ext(gterm, gterm_tri, nr);
        lemma_tri_sum_swap(trip, d, nr);
        // the inner sums over t give M_i[n] * A*(y)_n, a flat sum over the packed indices
        let inner = |r: int, c: int| sum_fn_r(|t: int| trip(r, c, t), nr);
        let flat = |n: int| constr_entry(m, row_dim, i, n) * a_star[n];
        let prod = |r: int, c: int| flat(pk_idx(r, c));
        assert forall|r: int, c: int| 0 <= r <= c < dim implies #[trigger] inner(r, c) == prod(r, c) by {
            let n = pk_idx(r, c);
            let (w, u) = (adj_wt(r, c), constr_entry(m, row_dim, i, n));
            let cb = |i2: int| constr_entry(m, row_dim, i2, n) * y[i2];
            let ht = |t: int| trip(r, c, t);
            assert forall|t: int| 0 <= t < row_dim implies #[trigger] ht(t) == (w * u) * cb(t) by {
                let (v, yt) = (constr_entry(m, row_dim, t, n), y[t]);
                assert(ht(t) == w * (u * v) * yt);
                assert(w * (u * v) * yt == (w * u) * (v * yt)) by (nonlinear_arith);
            }
            lemma_sum_fn_scale(cb, ht, w * u, nr);
            let cc = col_comb_r(m, row_dim, y, n);
            assert(inner(r, c) == (w * u) * cc);
            assert(zero_comb_r(Seq::<(usize, usize)>::empty(), row_dim, y, r, c) == 0.0real);
            assert(a_star[n] == w * cc + 0.5real * 0.0real);
            assert((w * u) * cc == u * (w * cc + 0.5real * 0.0real)) by (nonlinear_arith);
        }
        lemma_tri_sum_ext(inner, prod, d);
        lemma_tri_sum_packed(flat, d);
        assert(ax[i] == row_dot_r(m, row_dim, i, a_star));
        assert(row_dot_r(m, row_dim, i, a_star) == sum_fn_r(flat, pk_len(dim) as nat));
    }
    assert(gy =~= ax);
}


/* ---------- the Gram matrix of the full operator ---------- */

/// A*(y) as a packed vector of order dim.
pub open spec fn adjoint_r(m: Seq<real>, row_dim: int, list: Seq<(usize, usize)>, y: Seq<real>, dim: int) -> Seq<real> {
    Seq::new(pk_len(dim) as nat, |t: int| adjoint_entry_r(m, row_dim, list, y, pk_unpack(t).0, pk_unpack(t).1))
}

/// adjoint_r satisfies the adjoint characterization.
pub proof fn lemma_adjoint_r_ok(m: Seq<real>, row_dim: int, list: Seq<(usize, usize)>, y: Seq<real>, dim: int)
    requires
        dim >= 0,
    ensures
        is_constraint_adjoint(adjoint_r(m, row_dim, list, y, dim), dim, m, row_dim, list, y),
{
    let a = adjoint_r(m, row_dim, list, y, dim);
    lemma_pk_col_start_monotone(0, dim);
    assert forall|r: int, c: int| 0 <= r <= c < dim
        implies #[trigger] a[pk_idx(r, c)] == adjoint_entry_r(m, row_dim, list, y, r, c) by {
        lemma_pk_idx_bound(r, c, dim);
        lemma_pk_unpack_idx(r, c);
    }
}

/// Entry (k, l) of the Gram matrix A A* of the full operator (the dense rows followed by the
/// zero-pattern rows): A(A*(e_l))_k.
pub open spec fn full_gram_entry_r(m: Seq<real>, row_dim: int, list: Seq<(usize, usize)>, dim: int, k: int, l: int) -> real {
    let n = (row_dim + list.len()) as nat;
    action_entry_r(m, row_dim, list, adjoint_r(m, row_dim, list, std_basis_r(l, n), dim), k)
}

/// The Gram matrix A A* of the full operator as an index function.
pub open spec fn full_gram_view(m: Seq<real>, row_dim: int, list: Seq<(usize, usize)>, dim: int) -> MatF {
    |k: int, l: int| full_gram_entry_r(m, row_dim, list, dim, k, l)
}

/// The dense rows carry no coefficient on the zero pattern: M_a[r, c] = 0 for every listed (r, c).
pub open spec fn rows_vanish_on_zeros(m: Seq<real>, row_dim: int, list: Seq<(usize, usize)>) -> bool {
    forall|a: int, q: int| 0 <= a < row_dim && 0 <= q < list.len()
        ==> #[trigger] constr_entry(m, row_dim, a, zero_idx(list, q)) == 0.0real
}

/// The zero-pattern positions are pairwise distinct.
pub open spec fn distinct_zero_list(list: Seq<(usize, usize)>) -> bool {
    forall|q1: int, q2: int| 0 <= q1 < list.len() && 0 <= q2 < list.len() && q1 != q2
        ==> #[trigger] list[q1] != #[trigger] list[q2]
}

/// The dense part of the adjoint of a unit multiplier: entry n of the combination of the dense
/// rows with weights e_l is M_l[n] when l is a dense row and 0 otherwise.
pub proof fn lemma_col_comb_basis(m: Seq<real>, row_dim: int, l: int, n: int, len: nat)
    requires
        0 <= row_dim <= len,
    ensures
        col_comb_r(m, row_dim, std_basis_r(l, len), n) == if 0 <= l < row_dim { constr_entry(m, row_dim, l, n) } else { 0.0real },
{
    let y = std_basis_r(l, len);
    let f = |i: int| constr_entry(m, row_dim, i, n) * y[i];
    if 0 <= l < row_dim {
        assert forall|i: int| 0 <= i < row_dim && i != l implies #[trigger] f(i) == 0.0real by {
            assert(y[i] == 0.0real);
        }
        lemma_sum_fn_single(f, row_dim as nat, l);
        assert(f(l) == constr_entry(m, row_dim, l, n) * 1.0real);
    } else {
        assert forall|i: int| 0 <= i < row_dim implies #[trigger] f(i) == 0.0real by {
            assert(y[i] == 0.0real);
        }
        lemma_sum_fn_all_zero(f, row_dim as nat);
    }
}

/// The zero-pattern part of the adjoint of a unit multiplier: position (r, c) receives 1 when
/// e_l selects a zero constraint addressing (r, c), else 0.
pub proof fn lemma_zero_comb_basis(list: Seq<(usize, usize)>, row_dim: int, l: int, r: int, c: int)
    requires
        row_dim >= 0,
    ensures
        zero_comb_r(list, row_dim, std_basis_r(l, (row_dim + list.len()) as nat), r, c)
            == if row_dim <= l < row_dim + list.len() && list[l - row_dim].0 == r && list[l - row_dim].1 == c { 1.0real } else { 0.0real },
{
    let nz = list.len();
    let y = std_basis_r(l, (row_dim + nz) as nat);
    let f = |q: int| zero_term_r(list, row_dim, y, r, c, q);
    if row_dim <= l < row_dim + nz && list[l - row_dim].0 == r && list[l - row_dim].1 == c {
        let q0 = l - row_dim;
        assert forall|q: int| 0 <= q < nz && q != q0 implies #[trigger] f(q) == 0.0real by {
            if list[q].0 == r && list[q].1 == c {
                assert(y[row_dim + q] == 0.0real);
            }
        }
        lemma_sum_fn_single(f, nz, q0);
        assert(f(q0) == y[l]);
    } else {
        assert forall|q: int| 0 <= q < nz implies #[trigger] f(q) == 0.0real by {
            if list[q].0 == r && list[q].1 == c {
                assert(row_dim + q != l);
                assert(y[row_dim + q] == 0.0real);
            }
        }
        lemma_sum_fn_all_zero(f, nz);
    }
}

/// The adjoint of a unit zero-constraint multiplier is half the unit packed vector at the
/// constraint's position: A*(e_l)[t] = 1/2 if t is that position, else 0.
pub proof fn lemma_adjoint_zero_basis(m: Seq<real>, row_dim: int, list: Seq<(usize, usize)>, dim: int, l: int, t: int)
    requires
        row_dim >= 0,
        dim >= 0,
        valid_zero_list(list, dim),
        row_dim <= l < row_dim + list.len(),
        0 <= t < pk_len(dim),
    ensures
        adjoint_r(m, row_dim, list, std_basis_r(l, (row_dim + list.len()) as nat), dim)[t]
            == if t == zero_idx(list, l - row_dim) { 0.5real } else { 0.0real },
{
    let n = (row_dim + list.len()) as nat;
    let y = std_basis_r(l, n);
    let a = adjoint_r(m, row_dim, list, y, dim);
    let (r, c) = (pk_unpack(t).0, pk_unpack(t).1);
    lemma_pk_unpack(t);
    let (r0, c0) = (list[l - row_dim].0 as int, list[l - row_dim].1 as int);
    assert(a[t] == adj_wt(r, c) * col_comb_r(m, row_dim, y, pk_idx(r, c)) + 0.5real * zero_comb_r(list, row_dim, y, r, c));
    lemma_col_comb_basis(m, row_dim, l, pk_idx(r, c), n);
    lemma_zero_comb_basis(list, row_dim, l, r, c);
    lemma_pk_idx_injective(r, c, r0, c0);
    assert(adj_wt(r, c) * 0.0real == 0.0real);
}

/// When the zero positions are distinct (distinct_zero_list) and every dense row vanishes on
/// them (rows_vanish_on_zeros), A A* is the block matrix (G, I/2): the dense Gram block, and the
/// zero constraints, each of squared Frobenius norm 1/2, orthogonal to the dense rows and to each other.
pub proof fn lemma_full_gram_block(g: Seq<real>, m: Seq<real>, row_dim: int, list: Seq<(usize, usize)>, dim: int)
    requires
        row_dim >= 0,
        dim >= 0,
        m.len() == row_dim * pk_len(dim),
        is_constraint_gram(g, m, row_dim, dim),
        valid_zero_list(list, dim),
        distinct_zero_list(list),
        rows_vanish_on_zeros(m, row_dim, list),
    ensures
        meq(full_gram_view(m, row_dim, list, dim), block_half_view(sym_view(g), row_dim), row_dim + list.len(), row_dim + list.len()),
{
    let nz = list.len() as int;
    let n = (row_dim + nz) as nat;
    let np = pk_len(dim) as nat;
    lemma_pk_col_start_monotone(0, dim);
    let full = full_gram_view(m, row_dim, list, dim);
    let blk = block_half_view(sym_view(g), row_dim);
    assert forall|k: int, l: int| #![trigger full(k, l)] #![trigger blk(k, l)]
        0 <= k < n && 0 <= l < n implies full(k, l) == blk(k, l) by {
        let y = std_basis_r(l, n);
        let a = adjoint_r(m, row_dim, list, y, dim);
        lemma_adjoint_r_ok(m, row_dim, list, y, dim);
        if l < row_dim {
            // the adjoint of a dense unit multiplier is the adjoint without zero constraints
            let yd = std_basis_r(l, row_dim as nat);
            assert forall|r: int, c: int| 0 <= r <= c < dim
                implies #[trigger] a[pk_idx(r, c)] == adjoint_entry_r(m, row_dim, Seq::<(usize, usize)>::empty(), yd, r, c) by {
                let f = |i: int| constr_entry(m, row_dim, i, pk_idx(r, c)) * y[i];
                let fd = |i: int| constr_entry(m, row_dim, i, pk_idx(r, c)) * yd[i];
                assert forall|i: int| 0 <= i < row_dim implies #[trigger] f(i) == fd(i) by {
                    assert(y[i] == yd[i]);
                }
                lemma_sum_fn_ext(f, fd, row_dim as nat);
                lemma_zero_comb_basis(list, row_dim, l, r, c);
                assert(zero_comb_r(Seq::<(usize, usize)>::empty(), row_dim, yd, r, c) == 0.0real);
            }
            assert(is_constraint_adjoint(a, dim, m, row_dim, Seq::<(usize, usize)>::empty(), yd));
            lemma_gram_action(g, m, row_dim, dim, yd, a);
            if k < row_dim {
                // (G e_l)_k = G_kl
                let gy = sym_mat_vec_r(g, row_dim, yd);
                lemma_sym_mat_vec_entry(g, row_dim, yd, k);
                let gt = |j: int| sym_entry(g, k, j) * yd[j];
                assert forall|j: int| 0 <= j < row_dim && j != l implies #[trigger] gt(j) == 0.0real by {
                    assert(yd[j] == 0.0real);
                }
                lemma_sum_fn_single(gt, row_dim as nat, l);
                assert(gy[k] == sym_entry(g, k, l) * 1.0real);
                assert(full(k, l) == row_dot_r(m, row_dim, k, a));
                assert(constraint_action_r(m, row_dim, Seq::<(usize, usize)>::empty(), a)[k] == row_dot_r(m, row_dim, k, a));
            } else {
                // a zero row reads an entry of A*(e_l), which vanishes on the zero pattern
                let q = k - row_dim;
                let (r, c) = (list[q].0 as int, list[q].1 as int);
                assert(full(k, l) == a[pk_idx(r, c)]);
                lemma_col_comb_basis(m, row_dim, l, pk_idx(r, c), n);
                lemma_zero_comb_basis(list, row_dim, l, r, c);
                assert(constr_entry(m, row_dim, l, zero_idx(list, q)) == 0.0real);
                assert(a[pk_idx(r, c)] == adj_wt(r, c) * 0.0real + 0.5real * 0.0real);
            }
        } else {
            let q0 = l - row_dim;
            let t0 = zero_idx(list, q0);
            lemma_pk_idx_bound(list[q0].0 as int, list[q0].1 as int, dim);
            if k < row_dim {
                // a dense row applied to half the unit vector at the zero position: M_k[t0] / 2 = 0
                let f = |t: int| constr_entry(m, row_dim, k, t) * a[t];
                assert forall|t: int| 0 <= t < np && t != t0 implies #[trigger] f(t) == 0.0real by {
                    lemma_adjoint_zero_basis(m, row_dim, list, dim, l, t);
                }
                lemma_sum_fn_single(f, np, t0);
                lemma_adjoint_zero_basis(m, row_dim, list, dim, l, t0);
                assert(constr_entry(m, row_dim, k, t0) == 0.0real);
                assert(full(k, l) == row_dot_r(m, row_dim, k, a));
                assert(row_dot_r(m, row_dim, k, a) == sum_fn_r(f, np));
            } else {
                // two zero rows: 1/2 on the diagonal, 0 off it (distinct positions)
                let q = k - row_dim;
                let t = zero_idx(list, q);
                lemma_pk_idx_bound(list[q].0 as int, list[q].1 as int, dim);
                assert(full(k, l) == a[t]);
                lemma_adjoint_zero_basis(m, row_dim, list, dim, l, t);
                if k != l {
                    assert(list[q] != list[q0]);
                    lemma_pk_idx_injective(list[q].0 as int, list[q].1 as int, list[q0].0 as int, list[q0].1 as int);
                }
            }
        }
    }
}

} // verus!

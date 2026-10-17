// operations on symmetric matrices
// a symmetric matrix is represented by a flat vector containing its upper triangular part in column-major style
//
// The functions inside the `verus!` block are verified against exact-real specifications
// (see src/lin_alg/sym_spec.rs for the packed symmetric view and src/trusted.rs for the
// exact-real abstraction of f64).

use vstd::prelude::*;
pub use crate::lin_alg::matrix::*;
use crate::lin_alg::comparison::{pivot_below_positive_threshold, scaled_near_zero};
#[cfg(verus_keep_ghost)]
use crate::trusted::rv;
#[cfg(verus_keep_ghost)]
use crate::real_spec::{abs_r, max_r};
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::constr_spec::*;
#[cfg(verus_keep_ghost)]
use vstd::arithmetic::div_mod::lemma_fundamental_div_mod;
use vstd::slice::slice_to_vec;

pub type SymMatrix = Vector;

verus! {

broadcast use crate::trusted::group_real_model;

// max diagonal of a symmetric matrix
// (for dim == 0 the result is |mat[0]| if mat is nonempty, else 0; only nonnegativity is claimed then)
pub fn max_diag_sym(mat: &SymMatrix, dim: usize) -> (max: f64)
    requires
        dim_fits(dim as int),
        mat@.len() >= pk_len(dim as int),
    ensures
        dim >= 1 ==> rv(max) == max_abs_diag_r(rvs(mat@), dim as int),
        rv(max) >= 0.0real,
{
    let ghost a = rvs(mat@);
    let ghost f = |i: int| abs_r(sym_diag(a, i));
    let mut max: f64 = mat.first().unwrap_or(&0.0).abs();
    proof {
        if dim >= 1 {
            lemma_pk_idx_bound(0, 0, dim as int);
            assert(max0_fn_r(f, 0) == 0.0real);
            assert(max0_fn_r(f, 1) == max_r(0.0real, f(0)));
        }
    }
    for i in 1..dim
        invariant
            a == rvs(mat@),
            f == (|i: int| abs_r(sym_diag(a, i))),
            dim_fits(dim as int),
            mat@.len() >= pk_len(dim as int),
            dim >= 1 ==> rv(max) == max0_fn_r(f, i as nat),
            rv(max) >= 0.0real,
    {
        proof {
            lemma_dim_fits_idx(dim as int, i as int);
            lemma_pk_diag_idx(i as int);
            lemma_pk_idx_bound(i as int, i as int, dim as int);
        }
        let diag = mat[(i * i + 3 * i) / 2].abs();
        if max < diag {
            max = diag
        }
    }
    max
}

// Frobenius inner product of two symmetric matrices
pub fn frob_prod_sym(mat1: &SymMatrix, mat2: &SymMatrix, dim: usize) -> (sum: f64)
    requires
        dim_fits(dim as int),
        mat1@.len() >= pk_len(dim as int),
        mat2@.len() >= pk_len(dim as int),
    ensures
        rv(sum) == frob_r(rvs(mat1@), rvs(mat2@), dim as int),
{
    let ghost a = rvs(mat1@);
    let ghost b = rvs(mat2@);
    let (mut sum, mut start) = (0.0, 0);
    let mut idx;
    proof {
        lemma_pk_col_start_zero();
    }
    for n in 0..dim
        invariant
            a == rvs(mat1@),
            b == rvs(mat2@),
            dim_fits(dim as int),
            mat1@.len() >= pk_len(dim as int),
            mat2@.len() >= pk_len(dim as int),
            start + n == pk_col_start(n as int),
            rv(sum) == frob_r(a, b, n as int),
    {
        proof {
            lemma_dim_fits_idx(dim as int, n as int);
            lemma_pk_col_start_step(n as int);
        }
        start += n;
        for i in 0..n
            invariant
                a == rvs(mat1@),
                b == rvs(mat2@),
                n < dim,
                dim_fits(dim as int),
                mat1@.len() >= pk_len(dim as int),
                mat2@.len() >= pk_len(dim as int),
                start == pk_col_start(n as int),
                rv(sum) == frob_r(a, b, n as int)
                    + 2.0real * sum_fn_r(|r: int| a[pk_idx(r, n as int)] * b[pk_idx(r, n as int)], i as nat),
        {
            proof {
                lemma_dim_fits_idx(dim as int, n as int);
                lemma_pk_idx_bound(i as int, n as int, dim as int);
            }
            sum = sum + mat1[start + i] * mat2[start + i] * 2.0;
        }
        proof {
            lemma_pk_idx_bound(n as int, n as int, dim as int);
        }
        idx = start + n;
        sum = mat1[idx].mul_add(mat2[idx], sum);
        proof {
            lemma_frob_step(a, b, n as int);
        }
    }
    sum
}

// Frobenius norm of a symmetric matrix
#[inline]
pub fn frob_norm_sym(mat: &SymMatrix, dim: usize) -> (norm: f64)
    requires
        dim_fits(dim as int),
        mat@.len() >= pk_len(dim as int),
    ensures
        rv(norm) == frob_norm_r(rvs(mat@), dim as int),
{
    proof {
        lemma_frob_self_nonneg(rvs(mat@), dim as int);
    }
    frob_prod_sym(mat, mat, dim).sqrt()
}

// Frobenius distance between two symmetric matrices
pub fn frob_dist_sym(mat1: &SymMatrix, mat2: &SymMatrix, dim: usize) -> (dist: f64)
    requires
        dim_fits(dim as int),
        mat1@.len() == mat2@.len(),
        mat1@.len() >= pk_len(dim as int),
    ensures
        rv(dist) == frob_norm_r(vsub_r(rvs(mat1@), rvs(mat2@)), dim as int),
{
    frob_norm_sym(&vect_subt(mat1, mat2), dim)
}

// 1-norm of a symmetric matrix
pub fn sym_mat_one_norm(mat: &SymMatrix, dim: usize) -> (norm: f64)
    requires
        dim_fits(dim as int),
        mat@.len() >= pk_len(dim as int),
    ensures
        rv(norm) == sym_one_norm_r(rvs(mat@), dim as int),
{
    let ghost a = rvs(mat@);
    // term accumulated into row k from entry (k, l)
    let ghost t = |k: int, l: int| abs_r(sym_entry(a, k, l));
    let mut row_sums = vec![0.0; dim];
    let mut idx = 0;
    proof {
        lemma_pk_col_start_zero();
        assert forall|k: int| 0 <= k < dim implies rv(#[trigger] row_sums@[k]) == partial_row(t, dim as int, k, 0, 0) by {
            lemma_partial_row_init(t, dim as int, k);
        }
    }
    for i in 0..dim
        invariant
            a == rvs(mat@),
            t == (|k: int, l: int| abs_r(sym_entry(a, k, l))),
            dim_fits(dim as int),
            mat@.len() >= pk_len(dim as int),
            row_sums@.len() == dim,
            idx == pk_col_start(i as int),
            forall|k: int| 0 <= k < dim ==> rv(#[trigger] row_sums@[k]) == partial_row(t, dim as int, k, i as int, 0),
    {
        for j in 0..=i
            invariant
                a == rvs(mat@),
                t == (|k: int, l: int| abs_r(sym_entry(a, k, l))),
                i < dim,
                dim_fits(dim as int),
                mat@.len() >= pk_len(dim as int),
                row_sums@.len() == dim,
                idx == pk_col_start(i as int) + j,
                forall|k: int| 0 <= k < dim ==> rv(#[trigger] row_sums@[k]) == partial_row(t, dim as int, k, i as int, j as int),
        {
            proof {
                lemma_dim_fits_idx(dim as int, i as int);
                lemma_pk_idx_bound(j as int, i as int, dim as int);
            }
            if j == i {
                row_sums[j] = row_sums[j] + mat[idx].abs();
                proof {
                    lemma_partial_row_diag(t, dim as int, i as int);
                }
            } else {
                row_sums[i] = row_sums[i] + mat[idx].abs();
                row_sums[j] = row_sums[j] + mat[idx].abs();
                proof {
                    lemma_partial_row_offdiag(t, dim as int, i as int, j as int);
                }
            }
            idx += 1;
        }
        proof {
            lemma_pk_col_start_step(i as int);
            assert forall|k: int| 0 <= k < dim
                implies rv(#[trigger] row_sums@[k]) == partial_row(t, dim as int, k, i as int + 1, 0) by {
                lemma_partial_row_next_col(t, dim as int, k, i as int);
            }
        }
    }
    let ghost row = |k: int| sum_fn_r(|j: int| abs_r(sym_entry(a, k, j)), dim as nat);
    proof {
        assert forall|k: int| 0 <= k < dim implies rv(#[trigger] row_sums@[k]) == row(k) by {
            lemma_partial_row_done(t, dim as int, k);
            lemma_sum_fn_ext(|l: int| t(k, l), |j: int| abs_r(sym_entry(a, k, j)), dim as nat);
        }
    }
    let mut norm = 0.0;
    for k in 0..dim
        invariant
            row_sums@.len() == dim,
            forall|k: int| 0 <= k < dim ==> rv(#[trigger] row_sums@[k]) == row(k),
            rv(norm) == max0_fn_r(row, k as nat),
    {
        if row_sums[k] > norm {
            norm = row_sums[k];
        }
    }
    norm
}

// identity symmetric matrix
pub fn ident_sym_mat(dim: usize) -> (result: SymMatrix)
    requires
        dim_fits(dim as int),
    ensures
        result@.len() == pk_len(dim as int),
        is_sym_ident_r(rvs(result@), dim as int),
{
    proof {
        lemma_dim_fits_idx(dim as int, dim as int);
    }
    let mut result: SymMatrix = Vec::with_capacity((dim * (dim + 1)) / 2);
    proof {
        lemma_pk_col_start_zero();
    }
    for i in 0..dim
        invariant
            result@.len() == pk_col_start(i as int),
            forall|r: int, c: int| 0 <= r <= c < i
                ==> rv(#[trigger] result@[pk_idx(r, c)]) == (if r == c { 1.0real } else { 0.0real }),
    {
        for j in 0..=i
            invariant
                i < dim,
                result@.len() == pk_col_start(i as int) + j,
                forall|r: int, c: int| 0 <= r <= c < i
                    ==> rv(#[trigger] result@[pk_idx(r, c)]) == (if r == c { 1.0real } else { 0.0real }),
                forall|r: int| 0 <= r < j
                    ==> rv(#[trigger] result@[pk_idx(r, i as int)]) == (if r == i { 1.0real } else { 0.0real }),
        {
            if j < i {
                result.push(0.0);
            } else {
                result.push(1.0);
            }
            proof {
                assert forall|r: int, c: int| 0 <= r <= c < i
                    implies rv(#[trigger] result@[pk_idx(r, c)]) == (if r == c { 1.0real } else { 0.0real }) by {
                    lemma_pk_idx_bound(r, c, i as int);
                }
            }
        }
        proof {
            lemma_pk_col_start_step(i as int);
        }
    }
    proof {
        assert forall|r: int, c: int| 0 <= r <= c < dim
            implies #[trigger] sym_entry(rvs(result@), r, c) == (if r == c { 1.0real } else { 0.0real }) by {
            lemma_pk_idx_bound(r, c, dim as int);
        }
    }
    result
}

// action of symmetric matrix on vector
pub fn sym_mat_act(mat: &SymMatrix, v: &[f64]) -> (result: Vector)
    requires
        dim_fits(v@.len() as int),
        mat@.len() >= pk_len(v@.len() as int),
    ensures
        result@.len() == v@.len(),
        rvs(result@) == sym_mat_vec_r(rvs(mat@), v@.len() as int, rvs(v@)),
{
    let ghost a = rvs(mat@);
    let ghost x = rvs(v@);
    // term accumulated into row k from entry (k, l)
    let ghost t = |k: int, l: int| sym_entry(a, k, l) * x[l];
    let dim = v.len();
    let mut result = vec![0.0; dim];
    let mut start;
    proof {
        assert forall|k: int| 0 <= k < dim implies rv(#[trigger] result@[k]) == partial_row(t, dim as int, k, 0, 0) by {
            lemma_partial_row_init(t, dim as int, k);
        }
    }
    for j in 0..dim
        invariant
            a == rvs(mat@),
            x == rvs(v@),
            t == (|k: int, l: int| sym_entry(a, k, l) * x[l]),
            dim == v@.len(),
            dim_fits(dim as int),
            mat@.len() >= pk_len(dim as int),
            result@.len() == dim,
            forall|k: int| 0 <= k < dim ==> rv(#[trigger] result@[k]) == partial_row(t, dim as int, k, j as int, 0),
    {
        proof {
            lemma_dim_fits_idx(dim as int, j as int);
        }
        start = j * (j + 1) / 2;
        for i in 0..j
            invariant
                a == rvs(mat@),
                x == rvs(v@),
                t == (|k: int, l: int| sym_entry(a, k, l) * x[l]),
                dim == v@.len(),
                j < dim,
                dim_fits(dim as int),
                mat@.len() >= pk_len(dim as int),
                result@.len() == dim,
                start == pk_col_start(j as int),
                forall|k: int| 0 <= k < dim ==> rv(#[trigger] result@[k]) == partial_row(t, dim as int, k, j as int, i as int),
        {
            proof {
                lemma_dim_fits_idx(dim as int, j as int);
                lemma_pk_idx_bound(i as int, j as int, dim as int);
            }
            let val = mat[start + i];
            result[i] = result[i] + val * v[j];
            result[j] = result[j] + val * v[i];
            proof {
                lemma_partial_row_offdiag(t, dim as int, j as int, i as int);
            }
        }
        proof {
            lemma_pk_idx_bound(j as int, j as int, dim as int);
        }
        result[j] = result[j] + mat[start + j] * v[j];
        proof {
            lemma_partial_row_diag(t, dim as int, j as int);
            assert forall|k: int| 0 <= k < dim
                implies rv(#[trigger] result@[k]) == partial_row(t, dim as int, k, j as int + 1, 0) by {
                lemma_partial_row_next_col(t, dim as int, k, j as int);
            }
        }
    }
    proof {
        assert forall|k: int| 0 <= k < dim
            implies rv(#[trigger] result@[k]) == sum_fn_r(|l: int| sym_entry(a, k, l) * x[l], dim as nat) by {
            lemma_partial_row_done(t, dim as int, k);
            lemma_sum_fn_ext(|l: int| t(k, l), |l: int| sym_entry(a, k, l) * x[l], dim as nat);
        }
        assert(rvs(result@) =~= sym_mat_vec_r(a, dim as int, x));
    }
    result
}

// scaling all off-diagonal elements of symmetric matrix
// (in place; the returned reference is `mat` itself, so `result@` is the scaled matrix and
// whatever the caller does through the returned reference is what `mat` finally holds)
#[allow(unused_variables)]  // the row index r is only used in the loop invariants
pub fn scale_off_diag(s: f64, dim: usize, mat: &mut SymMatrix) -> (result: &mut SymMatrix)
    requires
        dim_fits(dim as int),
        old(mat)@.len() >= pk_len(dim as int),
    ensures
        result@.len() == old(mat)@.len(),
        forall|r: int, c: int| 0 <= r <= c < dim
            ==> rv(#[trigger] result@[pk_idx(r, c)])
                == (if r == c { rv(old(mat)@[pk_idx(r, c)]) } else { rv(s) * rv(old(mat)@[pk_idx(r, c)]) }),
        forall|k: int| pk_len(dim as int) <= k < old(mat)@.len() ==> #[trigger] result@[k] == old(mat)@[k],
        final(mat)@ == final(result)@,
{
    let mut idx = 0;
    proof {
        lemma_pk_col_start_zero();
    }
    for c in 0..dim
        invariant
            dim_fits(dim as int),
            mat@.len() == old(mat)@.len(),
            old(mat)@.len() >= pk_len(dim as int),
            idx == pk_col_start(c as int),
            forall|r: int, c2: int| 0 <= r <= c2 < c
                ==> rv(#[trigger] mat@[pk_idx(r, c2)])
                    == (if r == c2 { rv(old(mat)@[pk_idx(r, c2)]) } else { rv(s) * rv(old(mat)@[pk_idx(r, c2)]) }),
            forall|k: int| idx <= k < old(mat)@.len() ==> #[trigger] mat@[k] == old(mat)@[k],
    {
        for r in 0..c
            invariant
                c < dim,
                dim_fits(dim as int),
                mat@.len() == old(mat)@.len(),
                old(mat)@.len() >= pk_len(dim as int),
                idx == pk_col_start(c as int) + r,
                forall|r2: int, c2: int| 0 <= r2 <= c2 < c
                    ==> rv(#[trigger] mat@[pk_idx(r2, c2)])
                        == (if r2 == c2 { rv(old(mat)@[pk_idx(r2, c2)]) } else { rv(s) * rv(old(mat)@[pk_idx(r2, c2)]) }),
                forall|r2: int| 0 <= r2 < r
                    ==> rv(#[trigger] mat@[pk_idx(r2, c as int)]) == rv(s) * rv(old(mat)@[pk_idx(r2, c as int)]),
                forall|k: int| idx <= k < old(mat)@.len() ==> #[trigger] mat@[k] == old(mat)@[k],
        {
            proof {
                lemma_dim_fits_idx(dim as int, c as int);
                lemma_pk_idx_bound(r as int, c as int, dim as int);
            }
            mat[idx] = mat[idx] * s;
            idx += 1;
            proof {
                assert forall|r2: int, c2: int| 0 <= r2 <= c2 < c
                    implies rv(#[trigger] mat@[pk_idx(r2, c2)])
                        == (if r2 == c2 { rv(old(mat)@[pk_idx(r2, c2)]) } else { rv(s) * rv(old(mat)@[pk_idx(r2, c2)]) }) by {
                    lemma_pk_idx_bound(r2, c2, c as int);
                }
                let ghost e = rv(old(mat)@[pk_idx(r as int, c as int)]);
                assert(rv(s) * e == e * rv(s)) by (nonlinear_arith);
            }
        }
        proof {
            lemma_dim_fits_idx(dim as int, c as int);
            lemma_pk_idx_bound(c as int, c as int, dim as int);
            lemma_pk_col_start_step(c as int);
        }
        idx += 1;
    }
    proof {
        lemma_pk_col_start_monotone(0, dim as int);
    }
    mat
}

// returns the upper half of an arbitrary outer product
pub fn outer_prod_upper(vect1: &Vector, vect2: &Vector) -> (result: SymMatrix)
    requires
        vect1@.len() == vect2@.len(),
    ensures
        result@.len() == pk_len(vect1@.len() as int),
        forall|r: int, c: int| 0 <= r <= c < vect1@.len()
            ==> rv(#[trigger] result@[pk_idx(r, c)]) == rv(vect1@[r]) * rv(vect2@[c]),
{
    let mut result = Vec::new();
    let dim = vect1.len();
    proof {
        lemma_pk_col_start_zero();
    }
    for c in 0..dim
        invariant
            dim == vect1@.len() == vect2@.len(),
            result@.len() == pk_col_start(c as int),
            forall|r: int, c2: int| 0 <= r <= c2 < c
                ==> rv(#[trigger] result@[pk_idx(r, c2)]) == rv(vect1@[r]) * rv(vect2@[c2]),
    {
        for r in 0..=c
            invariant
                c < dim == vect1@.len() == vect2@.len(),
                result@.len() == pk_col_start(c as int) + r,
                forall|r2: int, c2: int| 0 <= r2 <= c2 < c
                    ==> rv(#[trigger] result@[pk_idx(r2, c2)]) == rv(vect1@[r2]) * rv(vect2@[c2]),
                forall|r2: int| 0 <= r2 < r
                    ==> rv(#[trigger] result@[pk_idx(r2, c as int)]) == rv(vect1@[r2]) * rv(vect2@[c as int]),
        {
            result.push(vect1[r] * vect2[c]);
            proof {
                assert forall|r2: int, c2: int| 0 <= r2 <= c2 < c
                    implies rv(#[trigger] result@[pk_idx(r2, c2)]) == rv(vect1@[r2]) * rv(vect2@[c2]) by {
                    lemma_pk_idx_bound(r2, c2, c as int);
                }
            }
        }
        proof {
            lemma_pk_col_start_step(c as int);
        }
    }
    result
}

// returns outer product of a vector with itself
pub fn outer_prod_single(vect: &Vector) -> (result: SymMatrix)
    ensures
        result@.len() == pk_len(vect@.len() as int),
        forall|r: int, c: int| 0 <= r <= c < vect@.len()
            ==> rv(#[trigger] result@[pk_idx(r, c)]) == rv(vect@[r]) * rv(vect@[c]),
{
    let mut result = Vec::new();
    let dim = vect.len();
    proof {
        lemma_pk_col_start_zero();
    }
    for c in 0..dim
        invariant
            dim == vect@.len(),
            result@.len() == pk_col_start(c as int),
            forall|r: int, c2: int| 0 <= r <= c2 < c
                ==> rv(#[trigger] result@[pk_idx(r, c2)]) == rv(vect@[r]) * rv(vect@[c2]),
    {
        for r in 0..=c
            invariant
                c < dim == vect@.len(),
                result@.len() == pk_col_start(c as int) + r,
                forall|r2: int, c2: int| 0 <= r2 <= c2 < c
                    ==> rv(#[trigger] result@[pk_idx(r2, c2)]) == rv(vect@[r2]) * rv(vect@[c2]),
                forall|r2: int| 0 <= r2 < r
                    ==> rv(#[trigger] result@[pk_idx(r2, c as int)]) == rv(vect@[r2]) * rv(vect@[c as int]),
        {
            result.push(vect[r] * vect[c]);
            proof {
                assert forall|r2: int, c2: int| 0 <= r2 <= c2 < c
                    implies rv(#[trigger] result@[pk_idx(r2, c2)]) == rv(vect@[r2]) * rv(vect@[c2]) by {
                    lemma_pk_idx_bound(r2, c2, c as int);
                }
            }
        }
        proof {
            lemma_pk_col_start_step(c as int);
        }
    }
    result
}

// returns outer product of a scaled vector sv with v
pub fn outer_prod_single_scale(s: f64, vect: &Vector) -> (result: SymMatrix)
    ensures
        result@.len() == pk_len(vect@.len() as int),
        forall|r: int, c: int| 0 <= r <= c < vect@.len()
            ==> rv(#[trigger] result@[pk_idx(r, c)]) == rv(s) * rv(vect@[r]) * rv(vect@[c]),
{
    let mut result = Vec::new();
    let dim = vect.len();
    proof {
        lemma_pk_col_start_zero();
    }
    for c in 0..dim
        invariant
            dim == vect@.len(),
            result@.len() == pk_col_start(c as int),
            forall|r: int, c2: int| 0 <= r <= c2 < c
                ==> rv(#[trigger] result@[pk_idx(r, c2)]) == rv(s) * rv(vect@[r]) * rv(vect@[c2]),
    {
        for r in 0..=c
            invariant
                c < dim == vect@.len(),
                result@.len() == pk_col_start(c as int) + r,
                forall|r2: int, c2: int| 0 <= r2 <= c2 < c
                    ==> rv(#[trigger] result@[pk_idx(r2, c2)]) == rv(s) * rv(vect@[r2]) * rv(vect@[c2]),
                forall|r2: int| 0 <= r2 < r
                    ==> rv(#[trigger] result@[pk_idx(r2, c as int)]) == rv(s) * rv(vect@[r2]) * rv(vect@[c as int]),
        {
            result.push(s * vect[r] * vect[c]);
            proof {
                assert forall|r2: int, c2: int| 0 <= r2 <= c2 < c
                    implies rv(#[trigger] result@[pk_idx(r2, c2)]) == rv(s) * rv(vect@[r2]) * rv(vect@[c2]) by {
                    lemma_pk_idx_bound(r2, c2, c as int);
                }
            }
        }
        proof {
            lemma_pk_col_start_step(c as int);
        }
    }
    result
}

// returns diagonal vector of n-dimensional symmetric matrix
pub fn diag_of_sym_mat(mat: &SymMatrix, dim: usize) -> (result: Vector)
    requires
        dim_fits(dim as int),
        mat@.len() >= pk_len(dim as int),
    ensures
        result@.len() == dim,
        rvs(result@) == sym_diag_r(rvs(mat@), dim as int),
{
    let mut result = Vec::with_capacity(dim);
    for j in 0..dim
        invariant
            dim_fits(dim as int),
            mat@.len() >= pk_len(dim as int),
            result@.len() == j,
            forall|k: int| 0 <= k < j ==> rv(#[trigger] result@[k]) == sym_diag(rvs(mat@), k),
    {
        proof {
            lemma_dim_fits_idx(dim as int, j as int);
            lemma_pk_diag_idx(j as int);
            lemma_pk_idx_bound(j as int, j as int, dim as int);
        }
        result.push(mat[(j * j + 3 * j) / 2]);
    }
    proof {
        assert(rvs(result@) =~= sym_diag_r(rvs(mat@), dim as int));
    }
    result
}

// returns linear map computing diagonal block of symmetric matrix specified by starting position and dimension of block
// returns, in increasing row dimension, the column entries of the map that equal one (the rest being zero)
pub fn sym_matrix_diag_block_map(start: usize, dim: usize) -> (result: Vec<usize>)
    requires
        dim_fits((start + dim) as int),
    ensures
        result@.len() == pk_len(dim as int),
        forall|r: int, c: int| 0 <= r <= c < dim
            ==> #[trigger] result@[pk_idx(r, c)] == pk_idx(start + r, start + c),
{
    proof {
        lemma_dim_fits_idx((start + dim) as int, dim as int);
    }
    let mut result = Vec::with_capacity((dim * (dim + 1)) / 2);
    let mut k = start;
    let mut first;
    proof {
        lemma_pk_col_start_zero();
    }
    for i in 0..dim
        invariant
            dim_fits((start + dim) as int),
            k == start + i,
            result@.len() == pk_col_start(i as int),
            forall|r: int, c: int| 0 <= r <= c < i
                ==> #[trigger] result@[pk_idx(r, c)] == pk_idx(start + r, start + c),
    {
        proof {
            lemma_dim_fits_idx((start + dim) as int, k as int);
            lemma_pk_idx_bound(start as int, k as int, k as int + 1);
        }
        first = (k * (k + 1)) / 2 + start;
        for j in 0..=i
            invariant
                i < dim,
                k == start + i,
                first == pk_idx(start as int, k as int),
                pk_col_start(k as int + 1) <= usize::MAX,
                result@.len() == pk_col_start(i as int) + j,
                forall|r: int, c: int| 0 <= r <= c < i
                    ==> #[trigger] result@[pk_idx(r, c)] == pk_idx(start + r, start + c),
                forall|r: int| 0 <= r < j
                    ==> #[trigger] result@[pk_idx(r, i as int)] == pk_idx(start + r, start + i),
        {
            proof {
                lemma_pk_idx_bound((start + j) as int, k as int, k as int + 1);
            }
            result.push(first + j);
            proof {
                assert forall|r: int, c: int| 0 <= r <= c < i
                    implies #[trigger] result@[pk_idx(r, c)] == pk_idx(start + r, start + c) by {
                    lemma_pk_idx_bound(r, c, i as int);
                }
            }
        }
        proof {
            lemma_pk_col_start_step(i as int);
        }
        k += 1;
    }
    result
}

// returns the actual diagonal block of symmetric matrix
pub fn sym_matrix_diag_block(start: usize, dim: usize, mat: &SymMatrix) -> (result: SymMatrix)
    requires
        dim_fits((start + dim) as int),
        mat@.len() >= pk_len((start + dim) as int),
    ensures
        result@.len() == pk_len(dim as int),
        forall|r: int, c: int| 0 <= r <= c < dim
            ==> rv(#[trigger] result@[pk_idx(r, c)]) == rv(mat@[pk_idx(start + r, start + c)]),
{
    proof {
        lemma_dim_fits_idx((start + dim) as int, dim as int);
    }
    let mut result: SymMatrix = Vec::with_capacity((dim * (dim + 1)) / 2);
    let mut k = start;
    let mut first;
    proof {
        lemma_pk_col_start_zero();
    }
    for i in 0..dim
        invariant
            dim_fits((start + dim) as int),
            mat@.len() >= pk_len((start + dim) as int),
            k == start + i,
            result@.len() == pk_col_start(i as int),
            forall|r: int, c: int| 0 <= r <= c < i
                ==> rv(#[trigger] result@[pk_idx(r, c)]) == rv(mat@[pk_idx(start + r, start + c)]),
    {
        proof {
            lemma_dim_fits_idx((start + dim) as int, k as int);
            lemma_pk_idx_bound(start as int, k as int, k as int + 1);
        }
        first = (k * (k + 1)) / 2 + start;
        for j in 0..=i
            invariant
                i < dim,
                k == start + i,
                mat@.len() >= pk_len((start + dim) as int),
                first == pk_idx(start as int, k as int),
                pk_col_start(k as int + 1) <= usize::MAX,
                result@.len() == pk_col_start(i as int) + j,
                forall|r: int, c: int| 0 <= r <= c < i
                    ==> rv(#[trigger] result@[pk_idx(r, c)]) == rv(mat@[pk_idx(start + r, start + c)]),
                forall|r: int| 0 <= r < j
                    ==> rv(#[trigger] result@[pk_idx(r, i as int)]) == rv(mat@[pk_idx(start + r, start + i)]),
        {
            proof {
                lemma_pk_idx_bound((start + j) as int, k as int, (start + dim) as int);
            }
            result.push(mat[first + j]);
            proof {
                assert forall|r: int, c: int| 0 <= r <= c < i
                    implies rv(#[trigger] result@[pk_idx(r, c)]) == rv(mat@[pk_idx(start + r, start + c)]) by {
                    lemma_pk_idx_bound(r, c, i as int);
                }
            }
        }
        proof {
            lemma_pk_col_start_step(i as int);
        }
        k += 1;
    }
    result
}

// returns linear map computing diagonal of a symmetric matrix
// (a dim x pk_len(dim) column-major matrix with a 1 in row i, column pk_idx(i, i))
pub fn sym_matrix_diagonal(dim: usize) -> (result: Matrix)
    requires
        dim_fits(dim as int),
        pk_len(dim as int) * dim <= usize::MAX,
    ensures
        result@.len() == pk_len(dim as int) * dim,
        forall|r: int, c: int| 0 <= r < dim && 0 <= c < pk_len(dim as int)
            ==> #[trigger] cm_entry(rvs(result@), dim as int, r, c) == (if c == pk_idx(r, r) { 1.0real } else { 0.0real }),
{
    proof {
        lemma_dim_fits_idx(dim as int, dim as int);
    }
    let size = (dim * (dim + 1)) / 2;
    let mut result = zero_mat(size, dim);
    proof {
        assert forall|r: int, c: int| 0 <= r < dim && 0 <= c < size
            implies #[trigger] cm_entry(rvs(result@), dim as int, r, c) == 0.0real by {
            assert(0 <= cm_idx(dim as int, r, c) < size * dim) by (nonlinear_arith)
                requires 0 <= r < dim, 0 <= c < size;
        }
    }
    for i in 0..dim
        invariant
            dim_fits(dim as int),
            size == pk_len(dim as int),
            size * dim <= usize::MAX,
            result@.len() == size * dim,
            forall|r: int, c: int| 0 <= r < dim && 0 <= c < size
                ==> #[trigger] cm_entry(rvs(result@), dim as int, r, c)
                    == (if c == pk_idx(r, r) && r < i { 1.0real } else { 0.0real }),
    {
        let ghost prev = result@;
        proof {
            lemma_dim_fits_idx(dim as int, i as int);
            lemma_pk_diag_idx(i as int);
            lemma_pk_idx_bound(i as int, i as int, dim as int);
            assert(pk_idx(i as int, i as int) * dim + i < size * dim) by (nonlinear_arith)
                requires pk_idx(i as int, i as int) + 1 <= size, i < dim;
        }
        result[((i * i + 3 * i) / 2) * dim + i] = 1.0;
        proof {
            assert forall|r: int, c: int| 0 <= r < dim && 0 <= c < size
                implies #[trigger] cm_entry(rvs(result@), dim as int, r, c)
                    == (if c == pk_idx(r, r) && r < i + 1 { 1.0real } else { 0.0real }) by {
                lemma_cm_idx_injective(dim as int, r, c, i as int, pk_idx(i as int, i as int));
                assert(0 <= cm_idx(dim as int, r, c) < size * dim) by (nonlinear_arith)
                    requires 0 <= r < dim, 0 <= c < size;
                if r == i && c == pk_idx(i as int, i as int) {
                } else {
                    assert(cm_entry(rvs(prev), dim as int, r, c)
                        == (if c == pk_idx(r, r) && r < i { 1.0real } else { 0.0real }));
                }
            }
        }
    }
    result
}

} // verus!

// composition with diagonal block map
pub fn matrix_diagonal_mult_sym(sym_dim : usize, mat : &Matrix, start : usize, dim : usize) -> Matrix {
    let mut result : Matrix = Vec::new();
    let ones = sym_matrix_diag_block_map(start, dim);
    if ones.is_empty() {
	return result;
    }
    let row_dim = mat.len() / ones.len();
    let (mut offset, mut i) = (0, 0);
    for c in ones {
	for _ in offset..c {
	    result.extend(vec![0.0; row_dim]);
	}
	result.extend_from_slice(&mat[i * row_dim..(i + 1) * row_dim]);
	offset = c + 1;
	i += 1;
    }
    for _ in offset..((sym_dim * (sym_dim + 1)) / 2) {
	result.extend(vec![0.0; row_dim]);
    }
    result
}

// composition of real-valued linear map on symmetric matrices with diagonal block map
pub fn mat_sym_diagonal_mult_sym(sym_dim : usize, mat : &SymMatrix, start : usize, dim : usize) -> SymMatrix {
    let mut result : SymMatrix = Vec::new();
    let ones = sym_matrix_diag_block_map(start, dim);
    let (mut offset, mut i) = (0, 0);
    for c in ones {
	for _ in offset..c {
	    result.push(0.0);
	}
	result.push(mat[i]);
	offset = c + 1;
	i += 1;
    }
    for _ in offset..((sym_dim * (sym_dim + 1)) / 2) {
	result.push(0.0);
    }
    result
}

// removes redundant constraints from an augmented matrix [A | b] where rows of A represent symmetric matrices
// also preconditions resulting constraint matrix by normalizing rows
use std::f64::consts::FRAC_1_SQRT_2;
pub fn linear_remove_redundant_sym(mat : &mut Matrix, dim : usize) -> Matrix {
    let col_dim = (dim * (dim + 1)) / 2 + 1;
    let row_dim = mat.len() / col_dim;
    let (echelon, rank, solvable) = gauss_elim(mat, col_dim, row_dim);
    if !solvable {
        panic!("linear_remove_redundant_sym: your system of constraints is inconsistent");
    }
    let mut result : Matrix = vec![0.0; col_dim * rank];
    let (mut start, mut scale) : (usize, f64);
    for i in 0..rank {
	scale = 0.0;
	scale = scale.hypot(echelon[i]);
	start = 1;
	for k in 1..dim {
	    let diag = (k * k + 3 * k) / 2;
	    for col in start..diag {
		scale = scale.hypot(echelon[col * row_dim + i] * FRAC_1_SQRT_2);
	    }
	    scale = scale.hypot(echelon[diag * row_dim + i]);
	    start += k;
	}
	assert!(scale.is_finite(), "linear_remove_redundant_sym: row has non-finite packed norm");
	for j in 0..col_dim {
	    result[j * rank + i] = echelon[j * row_dim + i] / scale;
	}
    }
    result
}


// the constraint matrix for block-diagonal decision variable Z is represented implicitly by a matrix M and a vector V
// each row of M is a symmetric matrix
// V is a list of pairs (row, column), with column > row, representing entries of Z that must be zero
// the full constraint matrix is formed by stacking M on top of V
//
// The four functions below are verified against the operator specifications in
// src/lin_alg/constr_spec.rs.  When M is empty both constraint_action and constraint_lin_comb
// ignore V (as the original code does); the specifications state exactly that.

verus! {

// action of constraint matrix on decision variable
pub fn constraint_action(constr: &(&Matrix, &Vec<(usize, usize)>), var: &SymMatrix) -> (result: Vector)
    requires
        constr.0@.len() > 0 ==> var@.len() > 0 && constr.0@.len() as int % var@.len() as int == 0,
        zero_list_fits(constr.1@, var@.len() as int),
    ensures
        constr.0@.len() == 0 ==> result@.len() == 0,
        constr.0@.len() > 0 ==> {
            let row_dim = rows_of(constr.0@.len() as int, var@.len() as int);
            &&& result@.len() == row_dim + constr.1@.len()
            &&& rvs(result@) == constraint_action_r(rvs(constr.0@), row_dim, constr.1@, rvs(var@))
        },
{
    let (mat, list) = constr;
    if mat.is_empty() {
        return vec![]
    }
    let ghost m = rvs(mat@);
    let ghost x = rvs(var@);
    let col_dim = var.len();
    let mat_len = mat.len();
    let row_dim = mat_len / col_dim;
    proof {
        lemma_fundamental_div_mod(mat@.len() as int, col_dim as int);
        assert(row_dim * col_dim == mat@.len()) by (nonlinear_arith)
            requires mat@.len() as int == col_dim as int * (mat@.len() as int / col_dim as int), row_dim as int == mat@.len() as int / col_dim as int;
    }
    let col0 = slice_to_vec(&mat[..row_dim]);
    let mut result = scal_vect(var[0], &col0);
    proof {
        assert forall|i: int| 0 <= i < row_dim
            implies rv(#[trigger] result@[i]) == sum_fn_r(|n: int| constr_entry(m, row_dim as int, i, n) * x[n], 1) by {
            let f = |n: int| constr_entry(m, row_dim as int, i, n) * x[n];
            assert(sum_fn_r(f, 1) == sum_fn_r(f, 0) + f(0));
            assert(cm_idx(row_dim as int, i, 0) == i) by (nonlinear_arith);
            assert(rvs(result@)[i] == rv(var@[0]) * rvs(col0@)[i]);
            assert(rvs(col0@)[i] == rv(mat@[i]));
            assert(x[0] * m[i] == m[i] * x[0]) by (nonlinear_arith);
        }
    }
    for n in 1..col_dim
        invariant
            m == rvs(mat@),
            x == rvs(var@),
            col_dim == var@.len(),
            mat_len == mat@.len() == row_dim * col_dim,
            result@.len() == row_dim,
            forall|i: int| 0 <= i < row_dim
                ==> rv(#[trigger] result@[i]) == sum_fn_r(|t: int| constr_entry(m, row_dim as int, i, t) * x[t], n as nat),
    {
        for i in 0..row_dim
            invariant
                m == rvs(mat@),
                x == rvs(var@),
                col_dim == var@.len(),
                n < col_dim,
                mat_len == mat@.len() == row_dim * col_dim,
                result@.len() == row_dim,
                forall|i2: int| 0 <= i2 < i
                    ==> rv(#[trigger] result@[i2]) == sum_fn_r(|t: int| constr_entry(m, row_dim as int, i2, t) * x[t], (n + 1) as nat),
                forall|i2: int| i <= i2 < row_dim
                    ==> rv(#[trigger] result@[i2]) == sum_fn_r(|t: int| constr_entry(m, row_dim as int, i2, t) * x[t], n as nat),
        {
            proof {
                assert(n * row_dim + i < mat_len) by (nonlinear_arith)
                    requires n < col_dim, i < row_dim, mat_len == row_dim * col_dim;
            }
            result[i] = result[i] + var[n] * mat[n * row_dim + i];
            proof {
                let f = |t: int| constr_entry(m, row_dim as int, i as int, t) * x[t];
                assert(sum_fn_r(f, (n + 1) as nat) == sum_fn_r(f, n as nat) + f(n as int));
                assert(cm_idx(row_dim as int, i as int, n as int) == n * row_dim + i);
                let (u, v) = (m[n * row_dim + i], x[n as int]);
                assert(v * u == u * v) by (nonlinear_arith);
            }
        }
    }
    let ghost dim = choose|dim: int| dim_fits(dim) && #[trigger] pk_len(dim) == var@.len() && valid_zero_list(list@, dim);
    for k in 0..list.len()
        invariant
            m == rvs(mat@),
            x == rvs(var@),
            col_dim == var@.len(),
            mat@.len() == row_dim * col_dim,
            dim_fits(dim) && pk_len(dim) == var@.len() && valid_zero_list(list@, dim),
            result@.len() == row_dim + k,
            forall|i: int| 0 <= i < row_dim
                ==> rv(#[trigger] result@[i]) == sum_fn_r(|t: int| constr_entry(m, row_dim as int, i, t) * x[t], col_dim as nat),
            forall|t: int| 0 <= t < k ==> rv(#[trigger] result@[row_dim + t]) == x[zero_idx(list@, t)],
    {
        let (r, c) = list[k];
        proof {
            assert(list@[k as int].0 < list@[k as int].1 < dim);
            lemma_dim_fits_idx(dim, c as int);
            lemma_pk_idx_bound(r as int, c as int, dim);
        }
        result.push(var[(c * (c + 1)) / 2 + r]);
    }
    proof {
        assert forall|i: int| 0 <= i < row_dim + list@.len()
            implies rv(#[trigger] result@[i]) == action_entry_r(m, row_dim as int, list@, x, i) by {
            if i < row_dim {
                assert(rv(result@[i]) == sum_fn_r(|t: int| constr_entry(m, row_dim as int, i, t) * x[t], col_dim as nat));
            } else {
                assert(rv(result@[row_dim + (i - row_dim)]) == x[zero_idx(list@, i - row_dim)]);
            }
        }
        assert(rvs(result@) =~= constraint_action_r(m, row_dim as int, list@, x));
    }
    result
}

// linear combination of the symmetric matrices making up the constraint matrix
// forms the adjoint operator to constraint_action (above) under the Frobenius inner product
#[allow(unused_variables)]  // vect_len is only used in the loop invariants
pub fn constraint_lin_comb(dim: usize, constr: &(&Matrix, &Vec<(usize, usize)>), vect: &Vector) -> (result: SymMatrix)
    requires
        dim_fits(dim as int),
        valid_zero_list(constr.1@, dim as int),
        constr.0@.len() > 0 ==> dim >= 1 && constr.0@.len() as int % pk_len(dim as int) == 0
            && vect@.len() >= rows_of(constr.0@.len() as int, pk_len(dim as int)) + constr.1@.len(),
    ensures
        result@.len() == pk_len(dim as int),
        constr.0@.len() == 0 ==> forall|n: int| 0 <= n < result@.len() ==> rv(#[trigger] result@[n]) == 0.0real,
        constr.0@.len() > 0 ==> is_constraint_adjoint(rvs(result@), dim as int, rvs(constr.0@),
            rows_of(constr.0@.len() as int, pk_len(dim as int)), constr.1@, rvs(vect@)),
{
    let (mat, list) = constr;
    proof {
        lemma_dim_fits_idx(dim as int, dim as int);
    }
    if mat.is_empty() {
        return vec![0.0; dim * (dim + 1) / 2]
    }
    let ghost m = rvs(mat@);
    let ghost y = rvs(vect@);
    let col_dim = dim * (dim + 1) / 2;
    let mat_len = mat.len();
    let vect_len = vect.len();
    proof {
        lemma_pk_col_start_monotone(1, dim as int);
        assert(pk_col_start(1) == 1);
    }
    let row_dim = mat_len / col_dim;
    proof {
        lemma_fundamental_div_mod(mat@.len() as int, col_dim as int);
        assert(row_dim * col_dim == mat@.len()) by (nonlinear_arith)
            requires mat@.len() as int == col_dim as int * (mat@.len() as int / col_dim as int), row_dim as int == mat@.len() as int / col_dim as int;
        lemma_pk_col_start_zero();
    }
    let mut result = Vec::new();
    let mut start = 0;
    let mut diag;
    for i in 0..dim
        invariant
            m == rvs(mat@),
            y == rvs(vect@),
            dim_fits(dim as int),
            col_dim == pk_len(dim as int),
            mat_len == mat@.len() == row_dim * col_dim,
            vect_len == vect@.len() >= row_dim,
            start == pk_col_start(i as int),
            result@.len() == start,
            forall|r: int, c: int| 0 <= r <= c < i
                ==> rv(#[trigger] result@[pk_idx(r, c)]) == adj_wt(r, c) * col_comb_r(m, row_dim as int, y, pk_idx(r, c)),
    {
        proof {
            lemma_dim_fits_idx(dim as int, i as int);
            lemma_pk_diag_idx(i as int);
            lemma_pk_idx_bound(i as int, i as int, dim as int);
        }
        diag = (i * i + 3 * i) / 2;
        for j in start..diag
            invariant
                m == rvs(mat@),
                y == rvs(vect@),
                i < dim,
                dim_fits(dim as int),
                col_dim == pk_len(dim as int),
                mat_len == mat@.len() == row_dim * col_dim,
                vect_len == vect@.len() >= row_dim,
                start == pk_col_start(i as int),
                diag == pk_idx(i as int, i as int) < col_dim,
                result@.len() == j,
                forall|r: int, c: int| 0 <= r <= c < i
                    ==> rv(#[trigger] result@[pk_idx(r, c)]) == adj_wt(r, c) * col_comb_r(m, row_dim as int, y, pk_idx(r, c)),
                forall|r: int| 0 <= r < j - start
                    ==> rv(#[trigger] result@[pk_idx(r, i as int)]) == adj_wt(r, i as int) * col_comb_r(m, row_dim as int, y, pk_idx(r, i as int)),
        {
            proof {
                assert(j * row_dim + row_dim == (j + 1) * row_dim) by (nonlinear_arith);
                assert((j + 1) * row_dim <= col_dim * row_dim) by (nonlinear_arith) requires j + 1 <= col_dim;
                assert(row_dim * col_dim == col_dim * row_dim) by (nonlinear_arith);
            }
            let lo = j * row_dim;
            proof {
                lemma_col_slice_dot(mat@, row_dim as int, vect@, j as int, lo as int);
            }
            result.push(0.5 * dot_prod(&mat[lo..lo + row_dim], &vect[..row_dim]));
            proof {
                assert forall|r: int, c: int| 0 <= r <= c < i
                    implies rv(#[trigger] result@[pk_idx(r, c)]) == adj_wt(r, c) * col_comb_r(m, row_dim as int, y, pk_idx(r, c)) by {
                    lemma_pk_idx_bound(r, c, i as int);
                }
                assert forall|r: int| 0 <= r < j + 1 - start
                    implies rv(#[trigger] result@[pk_idx(r, i as int)]) == adj_wt(r, i as int) * col_comb_r(m, row_dim as int, y, pk_idx(r, i as int)) by {
                    if r == j - start {
                        assert(pk_idx(r, i as int) == j);
                    }
                }
            }
        }
        proof {
            assert(diag * row_dim + row_dim == (diag + 1) * row_dim) by (nonlinear_arith);
            assert((diag + 1) * row_dim <= col_dim * row_dim) by (nonlinear_arith) requires diag + 1 <= col_dim;
            assert(row_dim * col_dim == col_dim * row_dim) by (nonlinear_arith);
        }
        let lo = diag * row_dim;
        proof {
            lemma_col_slice_dot(mat@, row_dim as int, vect@, diag as int, lo as int);
        }
        result.push(dot_prod(&mat[lo..lo + row_dim], &vect[..row_dim]));
        proof {
            lemma_pk_col_start_step(i as int);
            assert forall|r: int, c: int| 0 <= r <= c < i + 1
                implies rv(#[trigger] result@[pk_idx(r, c)]) == adj_wt(r, c) * col_comb_r(m, row_dim as int, y, pk_idx(r, c)) by {
                if c < i {
                    lemma_pk_idx_bound(r, c, i as int);
                } else if r < i {
                    lemma_pk_idx_bound(r, c, dim as int);
                } else {
                    assert(pk_idx(r, c) == diag);
                    assert(1.0real * col_comb_r(m, row_dim as int, y, pk_idx(r, c)) == col_comb_r(m, row_dim as int, y, pk_idx(r, c)));
                }
            }
        }
        start += i + 1;
    }
    for k in 0..list.len()
        invariant
            m == rvs(mat@),
            y == rvs(vect@),
            dim_fits(dim as int),
            valid_zero_list(list@, dim as int),
            vect_len == vect@.len() >= row_dim + list@.len(),
            result@.len() == pk_len(dim as int),
            forall|r: int, c: int| 0 <= r <= c < dim
                ==> rv(#[trigger] result@[pk_idx(r, c)]) == adj_wt(r, c) * col_comb_r(m, row_dim as int, y, pk_idx(r, c))
                    + 0.5real * zero_comb_upto(list@, row_dim as int, y, r, c, k as nat),
    {
        let (r, c) = list[k];
        proof {
            assert(list@[k as int].0 < list@[k as int].1 < dim);
            lemma_dim_fits_idx(dim as int, c as int);
            lemma_pk_idx_bound(r as int, c as int, dim as int);
        }
        let idx = (c * (c + 1)) / 2 + r;
        result[idx] = result[idx] + 0.5 * vect[row_dim + k];
        proof {
            assert forall|r2: int, c2: int| 0 <= r2 <= c2 < dim
                implies rv(#[trigger] result@[pk_idx(r2, c2)]) == adj_wt(r2, c2) * col_comb_r(m, row_dim as int, y, pk_idx(r2, c2))
                    + 0.5real * zero_comb_upto(list@, row_dim as int, y, r2, c2, (k + 1) as nat) by {
                lemma_pk_idx_bound(r2, c2, dim as int);
                let f = |t: int| zero_term_r(list@, row_dim as int, y, r2, c2, t);
                assert(zero_comb_upto(list@, row_dim as int, y, r2, c2, (k + 1) as nat)
                    == zero_comb_upto(list@, row_dim as int, y, r2, c2, k as nat) + f(k as int));
                lemma_pk_idx_injective(r2, c2, r as int, c as int);
            }
        }
    }
    proof {
        assert forall|r: int, c: int| 0 <= r <= c < dim
            implies #[trigger] rvs(result@)[pk_idx(r, c)] == adjoint_entry_r(m, row_dim as int, list@, y, r, c) by {
            lemma_pk_idx_bound(r, c, dim as int);
        }
    }
    result
}

// computation of Gram matrix of a constraint matrix (i.e., its composite with its adjoint)
// we ignore the off-diagonal zero constraints because we only need the upper left block of the Gram matrix
pub fn constraint_gram(dim: usize, mat: &Matrix) -> (result: SymMatrix)
    requires
        dim_fits(dim as int),
        mat@.len() > 0 ==> dim >= 1 && mat@.len() as int % pk_len(dim as int) == 0
            && dim_fits(rows_of(mat@.len() as int, pk_len(dim as int))),
    ensures
        mat@.len() == 0 ==> result@.len() == 0,
        mat@.len() > 0 ==> is_constraint_gram(rvs(result@), rvs(mat@), rows_of(mat@.len() as int, pk_len(dim as int)), dim as int),
{
    if mat.is_empty() {
        return vec![];
    }
    let ghost m = rvs(mat@);
    proof {
        lemma_dim_fits_idx(dim as int, dim as int);
        lemma_pk_col_start_monotone(1, dim as int);
        assert(pk_col_start(1) == 1);
    }
    let col_dim = dim * (dim + 1) / 2;
    let mat_len = mat.len();
    let gram_dim = mat_len / col_dim;
    proof {
        lemma_fundamental_div_mod(mat@.len() as int, col_dim as int);
        assert(gram_dim * col_dim == mat@.len()) by (nonlinear_arith)
            requires mat@.len() as int == col_dim as int * (mat@.len() as int / col_dim as int), gram_dim as int == mat@.len() as int / col_dim as int;
        lemma_dim_fits_idx(gram_dim as int, gram_dim as int);
        lemma_pk_col_start_zero();
    }
    // term (r, c) of the triangle sum defining entry (a, b) of the Gram matrix
    let ghost term = |a: int, b: int, r: int, c: int|
        adj_wt(r, c) * (constr_entry(m, gram_dim as int, a, pk_idx(r, c)) * constr_entry(m, gram_dim as int, b, pk_idx(r, c)));
    let gram_len = gram_dim * (gram_dim + 1) / 2;
    let mut result: SymMatrix = vec![0.0; gram_len];
    proof {
        assert forall|a: int, b: int| 0 <= a <= b < gram_dim
            implies rv(#[trigger] result@[pk_idx(a, b)]) == tri_sum_r(|r: int, c: int| term(a, b, r, c), 0) by {
            lemma_pk_idx_bound(a, b, gram_dim as int);
        }
    }
    let mut start = 0;
    let mut diag;
    for i in 0..dim
        invariant
            m == rvs(mat@),
            term == (|a: int, b: int, r: int, c: int|
                adj_wt(r, c) * (constr_entry(m, gram_dim as int, a, pk_idx(r, c)) * constr_entry(m, gram_dim as int, b, pk_idx(r, c)))),
            dim_fits(dim as int),
            dim_fits(gram_dim as int),
            col_dim == pk_len(dim as int),
            mat_len == mat@.len() == gram_dim * col_dim,
            gram_len == pk_len(gram_dim as int),
            start == pk_col_start(i as int),
            result@.len() == gram_len,
            forall|a: int, b: int| 0 <= a <= b < gram_dim
                ==> rv(#[trigger] result@[pk_idx(a, b)]) == tri_sum_r(|r: int, c: int| term(a, b, r, c), i as nat),
    {
        proof {
            lemma_dim_fits_idx(dim as int, i as int);
            lemma_pk_diag_idx(i as int);
            lemma_pk_idx_bound(i as int, i as int, dim as int);
        }
        diag = (i * i + 3 * i) / 2;
        for j in start..diag
            invariant
                m == rvs(mat@),
                term == (|a: int, b: int, r: int, c: int|
                    adj_wt(r, c) * (constr_entry(m, gram_dim as int, a, pk_idx(r, c)) * constr_entry(m, gram_dim as int, b, pk_idx(r, c)))),
                i < dim,
                dim_fits(dim as int),
                dim_fits(gram_dim as int),
                col_dim == pk_len(dim as int),
                mat_len == mat@.len() == gram_dim * col_dim,
                gram_len == pk_len(gram_dim as int),
                start == pk_col_start(i as int),
                diag == pk_idx(i as int, i as int) < col_dim,
                result@.len() == gram_len,
                forall|a: int, b: int| 0 <= a <= b < gram_dim
                    ==> rv(#[trigger] result@[pk_idx(a, b)]) == tri_sum_r(|r: int, c: int| term(a, b, r, c), i as nat)
                        + sum_fn_r(|r: int| term(a, b, r, i as int), (j - start) as nat),
        {
            proof {
                assert(j * gram_dim + gram_dim == (j + 1) * gram_dim) by (nonlinear_arith);
                assert((j + 1) * gram_dim <= col_dim * gram_dim) by (nonlinear_arith) requires j + 1 <= col_dim;
                assert(gram_dim * col_dim == col_dim * gram_dim) by (nonlinear_arith);
            }
            let lo = j * gram_dim;
            let col = slice_to_vec(&mat[lo..lo + gram_dim]);
            let op = outer_prod_single(&col);
            let ghost prev = result@;
            proof {
                assert(col@ == mat@.subrange(lo as int, lo + gram_dim));
                assert(col@.len() == gram_dim);
            }
            for t in 0..gram_len
                invariant
                    gram_len == pk_len(gram_dim as int),
                    result@.len() == gram_len,
                    prev.len() == gram_len,
                    op@.len() == gram_len,
                    forall|t2: int| 0 <= t2 < t ==> rv(#[trigger] result@[t2]) == rv(prev[t2]) + 0.5real * rv(op@[t2]),
                    forall|t2: int| t <= t2 < gram_len ==> #[trigger] result@[t2] == prev[t2],
            {
                result[t] = result[t] + 0.5 * op[t];
            }
            proof {
                assert forall|a: int, b: int| 0 <= a <= b < gram_dim
                    implies rv(#[trigger] result@[pk_idx(a, b)]) == tri_sum_r(|r: int, c: int| term(a, b, r, c), i as nat)
                        + sum_fn_r(|r: int| term(a, b, r, i as int), (j + 1 - start) as nat) by {
                    lemma_pk_idx_bound(a, b, gram_dim as int);
                    let r0 = (j - start) as int;
                    let f = |r: int| term(a, b, r, i as int);
                    assert(sum_fn_r(f, (j + 1 - start) as nat) == sum_fn_r(f, (j - start) as nat) + f(r0));
                    assert(pk_idx(r0, i as int) == j);
                    assert(col@[a] == mat@[lo + a]);
                    assert(col@[b] == mat@[lo + b]);
                    assert(cm_idx(gram_dim as int, a, j as int) == lo + a);
                    assert(cm_idx(gram_dim as int, b, j as int) == lo + b);
                    assert(constr_entry(m, gram_dim as int, a, j as int) == m[lo + a]);
                    assert(constr_entry(m, gram_dim as int, b, j as int) == m[lo + b]);
                    assert(rv(op@[pk_idx(a, b)]) == rv(col@[a]) * rv(col@[b]));
                    assert(rv(result@[pk_idx(a, b)]) == rv(prev[pk_idx(a, b)]) + 0.5real * rv(op@[pk_idx(a, b)]));
                    assert(f(r0) == 0.5real * (m[lo + a] * m[lo + b]));
                }
            }
        }
        proof {
            assert(diag * gram_dim + gram_dim == (diag + 1) * gram_dim) by (nonlinear_arith);
            assert((diag + 1) * gram_dim <= col_dim * gram_dim) by (nonlinear_arith) requires diag + 1 <= col_dim;
            assert(gram_dim * col_dim == col_dim * gram_dim) by (nonlinear_arith);
        }
        let lo = diag * gram_dim;
        let col = slice_to_vec(&mat[lo..lo + gram_dim]);
        let op = outer_prod_single(&col);
        let ghost prev = result@;
        proof {
            assert(col@ == mat@.subrange(lo as int, lo + gram_dim));
            assert(col@.len() == gram_dim);
        }
        for t in 0..gram_len
            invariant
                gram_len == pk_len(gram_dim as int),
                result@.len() == gram_len,
                prev.len() == gram_len,
                op@.len() == gram_len,
                forall|t2: int| 0 <= t2 < t ==> rv(#[trigger] result@[t2]) == rv(prev[t2]) + rv(op@[t2]),
                forall|t2: int| t <= t2 < gram_len ==> #[trigger] result@[t2] == prev[t2],
        {
            result[t] = result[t] + op[t];
        }
        proof {
            lemma_pk_col_start_step(i as int);
            assert forall|a: int, b: int| 0 <= a <= b < gram_dim
                implies rv(#[trigger] result@[pk_idx(a, b)]) == tri_sum_r(|r: int, c: int| term(a, b, r, c), (i + 1) as nat) by {
                lemma_pk_idx_bound(a, b, gram_dim as int);
                let tri = |r: int, c: int| term(a, b, r, c);
                let f = |r: int| tri(r, i as int);
                let g = |r: int| term(a, b, r, i as int);
                assert(tri_sum_r(tri, (i + 1) as nat) == tri_sum_r(tri, i as nat) + sum_fn_r(f, (i + 1) as nat));
                assert(sum_fn_r(f, (i + 1) as nat) == sum_fn_r(f, i as nat) + f(i as int));
                assert forall|r: int| 0 <= r < i implies #[trigger] g(r) == f(r) by {}
                lemma_sum_fn_ext(g, f, i as nat);
                assert(col@[a] == mat@[lo + a]);
                assert(col@[b] == mat@[lo + b]);
                assert(cm_idx(gram_dim as int, a, diag as int) == lo + a);
                assert(cm_idx(gram_dim as int, b, diag as int) == lo + b);
                assert(constr_entry(m, gram_dim as int, a, diag as int) == m[lo + a]);
                assert(constr_entry(m, gram_dim as int, b, diag as int) == m[lo + b]);
                assert(rv(op@[pk_idx(a, b)]) == rv(col@[a]) * rv(col@[b]));
                assert(rv(result@[pk_idx(a, b)]) == rv(prev[pk_idx(a, b)]) + rv(op@[pk_idx(a, b)]));
                assert(f(i as int) == 1.0real * (m[lo + a] * m[lo + b]));
            }
        }
        start += i + 1;
    }
    proof {
        assert forall|a: int, b: int| 0 <= a <= b < gram_dim
            implies #[trigger] rvs(result@)[pk_idx(a, b)] == gram_entry_r(m, gram_dim as int, dim as int, a, b) by {
            lemma_pk_idx_bound(a, b, gram_dim as int);
            let tri = |r: int, c: int| term(a, b, r, c);
            let g = |r: int, c: int| adj_wt(r, c) * (constr_entry(m, gram_dim as int, a, pk_idx(r, c)) * constr_entry(m, gram_dim as int, b, pk_idx(r, c)));
            assert forall|r: int, c: int| 0 <= r <= c < dim implies #[trigger] tri(r, c) == g(r, c) by {}
            lemma_tri_sum_ext(tri, g, dim as nat);
        }
    }
    result
}

// computation of QDQ^T where Q is any matrix and D is diagonal
pub fn diag_scale_gram(d: &Vector, q: &Matrix) -> (result: SymMatrix)
    requires
        d@.len() > 0 ==> q@.len() as int % d@.len() as int == 0 && dim_fits(rows_of(q@.len() as int, d@.len() as int)),
    ensures
        d@.len() == 0 ==> result@.len() == 0,
        d@.len() > 0 ==> {
            let len = rows_of(q@.len() as int, d@.len() as int);
            let qv = cm_view(rvs(q@), len);
            &&& result@.len() == pk_len(len)
            &&& meq(sym_view(rvs(result@)), mmul(mmul(qv, diag_view(rvs(d@)), d@.len() as int), mtr(qv), d@.len() as int), len, len)
        },
{
    if d.is_empty() {
        return vec![];
    }
    let ghost dv = rvs(d@);
    let ghost qs = rvs(q@);
    let m = d.len();
    let q_len = q.len();
    let len = q_len / m;
    proof {
        lemma_fundamental_div_mod(q@.len() as int, m as int);
        assert(len * m == q@.len()) by (nonlinear_arith)
            requires q@.len() as int == m as int * (q@.len() as int / m as int), len as int == q@.len() as int / m as int;
        lemma_dim_fits_idx(len as int, len as int);
    }
    let ghost qv = cm_view(qs, len as int);
    let ghost term = |a: int, b: int, k: int| dv[k] * qv(a, k) * qv(b, k);
    let pk = len * (len + 1) / 2;
    let mut result = vec![0.0; pk];
    proof {
        assert forall|a: int, b: int| 0 <= a <= b < len
            implies rv(#[trigger] result@[pk_idx(a, b)]) == sum_fn_r(|k: int| term(a, b, k), 0) by {
            lemma_pk_idx_bound(a, b, len as int);
        }
    }
    for k in 0..m
        invariant
            dv == rvs(d@),
            qs == rvs(q@),
            qv == cm_view(qs, len as int),
            term == (|a: int, b: int, k: int| dv[k] * qv(a, k) * qv(b, k)),
            m == d@.len(),
            q_len == q@.len() == len * m,
            dim_fits(len as int),
            pk == pk_len(len as int),
            result@.len() == pk,
            forall|a: int, b: int| 0 <= a <= b < len
                ==> rv(#[trigger] result@[pk_idx(a, b)]) == sum_fn_r(|k: int| term(a, b, k), k as nat),
    {
        proof {
            assert(k * len + len == (k + 1) * len) by (nonlinear_arith);
            assert((k + 1) * len <= m * len) by (nonlinear_arith) requires k + 1 <= m;
            assert(m * len == len * m) by (nonlinear_arith);
        }
        let lo = k * len;
        let col = slice_to_vec(&q[lo..lo + len]);
        let op = outer_prod_single_scale(d[k], &col);
        let ghost prev = result@;
        proof {
            assert(col@ == q@.subrange(lo as int, lo + len));
            assert(col@.len() == len);
        }
        for t in 0..pk
            invariant
                pk == pk_len(len as int),
                result@.len() == pk,
                prev.len() == pk,
                op@.len() == pk,
                forall|t2: int| 0 <= t2 < t ==> rv(#[trigger] result@[t2]) == rv(prev[t2]) + rv(op@[t2]),
                forall|t2: int| t <= t2 < pk ==> #[trigger] result@[t2] == prev[t2],
        {
            result[t] = result[t] + op[t];
        }
        proof {
            assert forall|a: int, b: int| 0 <= a <= b < len
                implies rv(#[trigger] result@[pk_idx(a, b)]) == sum_fn_r(|k2: int| term(a, b, k2), (k + 1) as nat) by {
                lemma_pk_idx_bound(a, b, len as int);
                let f = |k2: int| term(a, b, k2);
                assert(sum_fn_r(f, (k + 1) as nat) == sum_fn_r(f, k as nat) + f(k as int));
                assert(col@[a] == q@[lo + a]);
                assert(col@[b] == q@[lo + b]);
                assert(cm_idx(len as int, a, k as int) == lo + a);
                assert(cm_idx(len as int, b, k as int) == lo + b);
                assert(qv(a, k as int) == qs[lo + a]);
                assert(qv(b, k as int) == qs[lo + b]);
                assert(rv(op@[pk_idx(a, b)]) == rv(d@[k as int]) * rv(col@[a]) * rv(col@[b]));
                assert(rv(result@[pk_idx(a, b)]) == rv(prev[pk_idx(a, b)]) + rv(op@[pk_idx(a, b)]));
            }
        }
    }
    proof {
        let prod = mmul(mmul(qv, diag_view(dv), m as int), mtr(qv), m as int);
        assert forall|a: int, b: int| 0 <= a < len && 0 <= b < len
            implies #[trigger] sym_view(rvs(result@))(a, b) == prod(a, b) by {
            lemma_mul_diag_transpose(qv, dv, m as int, a, b);
            let f = |k: int| dv[k] * qv(a, k) * qv(b, k);
            if a <= b {
                lemma_pk_idx_bound(a, b, len as int);
                assert(sym_view(rvs(result@))(a, b) == rv(result@[pk_idx(a, b)]));
                let g = |k: int| term(a, b, k);
                assert(rv(result@[pk_idx(a, b)]) == sum_fn_r(g, m as nat));
                assert forall|k: int| 0 <= k < m implies #[trigger] g(k) == f(k) by {}
                lemma_sum_fn_ext(g, f, m as nat);
            } else {
                lemma_pk_idx_bound(b, a, len as int);
                assert(sym_view(rvs(result@))(a, b) == rv(result@[pk_idx(b, a)]));
                let g = |k: int| term(b, a, k);
                assert(rv(result@[pk_idx(b, a)]) == sum_fn_r(g, m as nat));
                assert forall|k: int| 0 <= k < m implies #[trigger] g(k) == f(k) by {
                    let (u, v, w) = (dv[k], qv(a, k), qv(b, k));
                    assert(g(k) == u * w * v);
                    assert(u * w * v == u * v * w) by (nonlinear_arith);
                }
                lemma_sum_fn_ext(g, f, m as nat);
            }
        }
    }
    result
}

} // verus!

// LDLT decomposition of a symmetric positive definite matrix
pub fn ldlt_decomp(mat : &SymMatrix, dim : usize) -> Option<(LowTriMatrix, Vector)> {
    let (mut l, mut d) : (LowTriMatrix, Vector) = (ident_mat_low_tri(dim), Vec::with_capacity(dim));
    let mut sum;
    for i in 0..dim {
	l[(i * (2 * dim - i + 1)) / 2] = 1.0;
    }
    let mut ind;
    for i in 0..dim {
	for j in 0..i {
	    sum = 0.0;
	    for k in 0..j {
		sum = (l[k * (2 * dim - k + 1) / 2 + (i - k)] * l[k * (2 * dim - k + 1) / 2 + (j - k)]).mul_add(d[k], sum);
	    }
	    l[j * (2 * dim - j + 1) / 2 + (i - j)] = (mat[(i * (i + 1)) / 2 + j] - sum) / d[j]
	}
	
	sum = 0.0;
	for k in 0..i {
	    ind = k * (2 * dim - k + 1) / 2 + (i - k);
	    sum = (l[ind] * l[ind]).mul_add(d[k], sum);
	}
	d.push(mat[(i * i + 3 * i) / 2] - sum);

	if pivot_below_positive_threshold(d[i], max_diag_sym(mat, dim).max(1.0), TOL) {
	    return None;
	}
    }
    Some((l, d))
}

// check whether a block-diagonal symmetric matrix S is positive semidefinite by running LDLT on S + eps * I
pub fn psd_block_check(mat : &SymMatrix, blocks : &Vec<(usize, usize)>) -> bool {
    let (mut eps, mut idx) : (f64, usize);
    for (start, dim) in blocks {
	let block = sym_matrix_diag_block(*start, *dim, mat);
	eps = 1e-9 * max_diag_sym(&block, *dim).max(1.0);
	let mut block_add_scal : SymMatrix = Vec::new();
	idx = 0;
	for c in 0..*dim {
	    for _ in 0..c {
		block_add_scal.push(block[idx]);
		idx += 1;
	    }
	    block_add_scal.push(block[idx] + eps);
	    idx += 1;
	}
	if ldlt_decomp(&block_add_scal, *dim).is_none() {
	    return false
	}
    }
    true
}

// solver for diagonal linear system Dx = b
pub fn diag_solver(diag : &Vector, vect : &Vector) -> Vector {
    let mut solution = Vec::new();
    let scale = diag.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1.0);
    for (v, d) in vect.iter().zip(diag.iter()) {
        if scaled_near_zero(*d, scale, TOL) {
            panic!("diag_solver: division by zero");
        }
        solution.push(v / d);
    }
    solution
}

// substitution solvers for full-rank triangular linear systems Ax = b

pub fn forw_subst(mat : &LowTriMatrix, vect : &mut [f64]) -> Vector {
    let dim = vect.len();
    let mut solution = Vec::with_capacity(dim);
    let mut d : f64;
    let scale = mat.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1.0);
    for j in 0..dim {
	d = mat[(j * (2 * dim - j + 1)) / 2];
	if scaled_near_zero(d, scale, TOL) {
	    panic!("forw_subst: division by zero");
	}
	solution.push(vect[j] / d);
	for i in j + 1..dim {
	    vect[i] -= solution[j] * mat[j * (2 * dim - j + 1) / 2 + (i - j)];
	}
    }
    solution
}

// here we implicitly treat a given lower triangular matrix as its (upper triangular) transpose
pub fn back_subst(mat : &LowTriMatrix, vect : &mut [f64]) -> Vector {
    let dim = vect.len();
    let mut solution = vec![0.0; dim];
    let mut d : f64;
    let scale = mat.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1.0);
    for j in (0..dim).rev() {
	d = mat[(j * (2 * dim - j + 1)) / 2];
	if scaled_near_zero(d, scale, TOL) {
	    panic!("back_subst: division by zero");
	}
	solution[j] = vect[j] / d;
	for i in 0..j {
	    vect[i] -= solution[j] * mat[i * (2 * dim - i + 1) / 2 + (j - i)];
	}
    }
    solution
}

// substitution solver for symmetric positive definite matrix
pub fn pos_def_solver(mat : &SymMatrix, vect : &mut [f64]) -> Vector {
    let Some((l, d)) = ldlt_decomp(mat, vect.len()) else { panic!("ldlt_decomp: given matrix is not positive definite"); };
    let sol1 = forw_subst(&l, vect);
    let mut sol2 = diag_solver(&d, &sol1);
    back_subst(&l, &mut sol2)
}

// solver for block-diagonal symmetric positive definite matrix (A, I)
pub fn pos_def_solver_upperleft(mat : &SymMatrix, vect : &mut Vector, id_len : usize) -> Vector {
    let ul_len = vect.len() - id_len;
    let mut result = pos_def_solver(mat, &mut vect[..ul_len]);
    for v in &vect[ul_len..] {
	result.push(*v);
    }
    result
}

/* some functions producing real-valued linear maps represented as partitioned vectors that act on n-dimensional symmetric matrices */

pub fn mk_symmap_real(dim : usize, vect : &Vector) -> Vec<Vector> {
    let mut result = Vec::with_capacity(dim);
    let mut offset = 0;
    for i in 0..dim {
	result.push(vect[offset..=offset + i].to_vec());
	offset += i + 1;
    }
    result
}

pub fn sym_matrix_trace(dim : usize) -> Vec<Vector> {
    let mut result = Vec::with_capacity(dim);
    for i in 0..dim {
	let mut v = vec![0.0; i + 1];
	v[i] = 1.0;
	result.push(v);
    }
    result
}

pub fn sym_matrix_scale(s : f64, vect : &Vec<Vector>) -> Vec<Vector> {
    let mut result = Vec::new();
    for v in vect {
	result.push(scal_vect(s, v));
    }
    result
}

pub fn sym_matrix_zero(dim : usize) -> Vec<Vector> {
    let mut result = Vec::with_capacity(dim);
    for i in 0..dim {
	result.push(vec![0.0; i + 1])
    }
    result
}

pub fn sym_matrix_proj(dim : usize, row : usize, col : usize) -> Vec<Vector> {
    if row > col {
	panic!("sym_matrix_proj: make sure that row <= column");
    }
    let mut result = Vec::with_capacity(dim);
    for i in 0..col {
	result.push(vec![0.0; i + 1]);
    }
    result.push(standard_basis(row, col + 1));
    for i in col + 1..dim {
	result.push(vec![0.0; i + 1]);
    }
    result
}

pub fn sym_matrix_add(vect1 : &[Vector], vect2 : &[Vector]) -> Vec<Vector> {
    let mut result = Vec::new();
    for (v1, v2) in vect1.iter().zip(vect2.iter()) {
	result.push(vect_add(v1, v2));
    }
    result
}

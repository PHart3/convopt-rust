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
use crate::lin_alg::solve_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::ldlt_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::reduce_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::gauss_spec::*;
#[cfg(verus_keep_ghost)]
use crate::trusted::sqrt_r;
#[cfg(verus_keep_ghost)]
use crate::real_spec::lemma_sqrt_unique;
use crate::trusted::abort;
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

verus! {

// composition with diagonal block map
// (mat is a row_dim x pk_len(dim) column-major matrix acting on dim x dim symmetric matrices;
// the result is the row_dim x pk_len(sym_dim) matrix of the composition with the embedding
// of such a matrix as the diagonal block at `start` of a sym_dim x sym_dim matrix: column
// pk_idx(start + a, start + b) is column pk_idx(a, b) of mat, every other column is zero)
pub fn matrix_diagonal_mult_sym(sym_dim: usize, mat: &Matrix, start: usize, dim: usize) -> (result: Matrix)
    requires
        dim_fits(sym_dim as int),
        start + dim <= sym_dim,
        dim >= 1 ==> (mat@.len() as int) % pk_len(dim as int) == 0,
        dim >= 1 ==> ((mat@.len() as int) / pk_len(dim as int)) * pk_len(sym_dim as int) <= usize::MAX,
    ensures
        dim == 0 ==> result@.len() == 0,
        dim >= 1 ==> {
            let row_dim = (mat@.len() as int) / pk_len(dim as int);
            &&& result@.len() == row_dim * pk_len(sym_dim as int)
            &&& forall|r: int, a: int, b: int| 0 <= r < row_dim && 0 <= a <= b < dim
                    ==> rv(#[trigger] result@[cm_idx(row_dim, r, pk_idx(start + a, start + b))]) == rv(mat@[cm_idx(row_dim, r, pk_idx(a, b))])
            &&& forall|r: int, t: int| 0 <= r < row_dim && 0 <= t < pk_len(sym_dim as int) && !is_block_idx(t, start as int, dim as int)
                    ==> rv(#[trigger] result@[cm_idx(row_dim, r, t)]) == 0.0real
        },
{
    let mut result: Matrix = Vec::new();
    proof {
        lemma_dim_fits_monotone((start + dim) as int, sym_dim as int);
        lemma_dim_fits_idx(sym_dim as int, sym_dim as int);
        lemma_pk_col_start_zero();
    }
    let ones = sym_matrix_diag_block_map(start, dim);
    if ones.is_empty() {
        proof {
            if dim >= 1 {
                lemma_pk_idx_bound(0, 0, dim as int);
            }
        }
        return result;
    }
    let row_dim = mat.len() / ones.len();
    let ghost (sd, st, d, rd) = (sym_dim as int, start as int, dim as int, row_dim as int);
    proof {
        if dim == 0 {
            lemma_pk_col_start_zero();
        }
        lemma_fundamental_div_mod(mat@.len() as int, pk_len(d));
        assert(pk_len(d) * rd == rd * pk_len(d)) by (nonlinear_arith);
        assert(mat@.len() == rd * pk_len(d));
    }
    let mut offset = 0;
    let mut i = 0;
    let ghost mut ga: int = 0;
    let ghost mut gb: int = 0;
    while i < ones.len()
        invariant
            sd == sym_dim as int,
            st == start as int,
            d == dim as int,
            rd == row_dim as int,
            dim >= 1,
            st + d <= sd,
            dim_fits(sd),
            pk_len(sd) <= usize::MAX,
            rd * pk_len(sd) <= usize::MAX,
            mat@.len() == rd * pk_len(d),
            ones@.len() == pk_len(d),
            forall|a: int, b: int| 0 <= a <= b < d ==> #[trigger] ones@[pk_idx(a, b)] == pk_idx(st + a, st + b),
            i <= ones@.len(),
            0 <= ga <= gb,
            i == pk_idx(ga, gb),
            i < ones@.len() ==> gb < d,
            i == ones@.len() ==> ga == 0 && gb == d,
            offset == block_prev_end(st, ga, gb),
            offset <= pk_len(sd),
            result@.len() == offset * rd,
            forall|r: int, a: int, b: int| 0 <= r < rd && 0 <= a <= b < d && pk_idx(a, b) < i
                ==> rv(#[trigger] result@[cm_idx(rd, r, pk_idx(st + a, st + b))]) == rv(mat@[cm_idx(rd, r, pk_idx(a, b))]),
            forall|r: int, t: int| 0 <= r < rd && 0 <= t < offset && !is_block_idx(t, st, d)
                ==> rv(#[trigger] result@[cm_idx(rd, r, t)]) == 0.0real,
        decreases ones@.len() - i,
    {
        let c = ones[i];
        proof {
            assert(ones@[pk_idx(ga, gb)] == pk_idx(st + ga, st + gb));
            lemma_pk_idx_bound(st + ga, st + gb, sd);
            lemma_block_prev_end_le(st, ga, gb);
            assert(mat.len() == rd * pk_len(d));
        }
        // zero columns between the previous block entry and this one
        let ghost before = result@;
        for _z in offset..c
            invariant
                rd == row_dim as int,
                st == start as int,
                d == dim as int,
                0 <= ga <= gb < d,
                offset <= _z <= c,
                c == pk_idx(st + ga, st + gb),
                offset == block_prev_end(st, ga, gb),
                c < pk_len(sd),
                rd * pk_len(sd) <= usize::MAX,
                result@.len() == _z * rd,
                before.len() == offset * rd,
                forall|q: int| 0 <= q < before.len() ==> #[trigger] result@[q] == before[q],
                forall|r: int, t: int| 0 <= r < rd && offset <= t < _z ==> rv(#[trigger] result@[cm_idx(rd, r, t)]) == 0.0real,
        {
            proof {
                assert((_z + 1) * rd == _z * rd + rd) by (nonlinear_arith);
                assert(offset * rd <= _z * rd) by (nonlinear_arith)
                    requires offset <= _z, rd >= 0;
            }
            let mut zeros = vec![0.0; row_dim];
            let ghost prev = result@;
            result.append(&mut zeros);
            proof {
                assert forall|r: int, t: int| 0 <= r < rd && offset <= t < _z + 1
                    implies rv(#[trigger] result@[cm_idx(rd, r, t)]) == 0.0real by {
                    lemma_cm_idx_bound(rd, r, t, _z as int + 1);
                    if t < _z {
                        lemma_cm_idx_bound(rd, r, t, _z as int);
                        assert(result@[cm_idx(rd, r, t)] == prev[cm_idx(rd, r, t)]);
                    }
                }
                assert forall|q: int| 0 <= q < before.len() implies #[trigger] result@[q] == before[q] by {
                    assert(result@[q] == prev[q]);
                }
            }
        }
        // column pk_idx(a, b) of mat becomes column c
        proof {
            assert((i + 1) * rd == i * rd + rd) by (nonlinear_arith);
            assert(i * rd + rd <= pk_len(d) * rd) by (nonlinear_arith)
                requires i + 1 <= pk_len(d), rd >= 0;
            assert(rd * pk_len(d) == pk_len(d) * rd) by (nonlinear_arith);
            assert((c + 1) * rd == c * rd + rd) by (nonlinear_arith);
            assert(offset * rd <= c * rd) by (nonlinear_arith)
                requires offset <= c, rd >= 0;
        }
        let ghost mid = result@;
        result.extend_from_slice(&mat[i * row_dim..(i + 1) * row_dim]);
        proof {
            let cur = result@;
            let ii = i as int;
            // the new column
            assert forall|r: int| 0 <= r < rd
                implies rv(#[trigger] cur[cm_idx(rd, r, c as int)]) == rv(mat@[cm_idx(rd, r, ii)]) by {
                assert(cur[c * rd + r] == mat@.subrange(ii * rd, (ii + 1) * rd)[r]);
            }
            // the gap columns are outside the block
            assert forall|t: int| offset <= t < c implies !is_block_idx(t, st, d) by {
                lemma_block_idx_gap(st, d, ga, gb, t);
            }
            // earlier block columns and zero columns are unchanged
            assert forall|r: int, a: int, b: int| 0 <= r < rd && 0 <= a <= b < d && pk_idx(a, b) < i + 1
                implies rv(#[trigger] cur[cm_idx(rd, r, pk_idx(st + a, st + b))]) == rv(mat@[cm_idx(rd, r, pk_idx(a, b))]) by {
                if pk_idx(a, b) < ii {
                    lemma_block_prev_end_bound(st, a, b, ga, gb);
                    lemma_cm_idx_bound(rd, r, pk_idx(st + a, st + b), offset as int);
                    let q = cm_idx(rd, r, pk_idx(st + a, st + b));
                    assert(cur[q] == mid[q]);
                    assert(mid[q] == before[q]);
                } else {
                    assert(pk_idx(a, b) == ii);
                    lemma_pk_idx_injective(a, b, ga, gb);
                }
            }
            assert forall|r: int, t: int| 0 <= r < rd && 0 <= t < c + 1 && !is_block_idx(t, st, d)
                implies rv(#[trigger] cur[cm_idx(rd, r, t)]) == 0.0real by {
                lemma_cm_idx_bound(rd, r, t, c as int);
                let q = cm_idx(rd, r, t);
                assert(cur[q] == mid[q]);
                if t < offset {
                    lemma_cm_idx_bound(rd, r, t, offset as int);
                    assert(mid[q] == before[q]);
                }
            }
            // the block entry (ga, gb) is followed by (ga + 1, gb) or (0, gb + 1)
            lemma_pk_col_start_step(gb);
            if ga < gb {
                ga = ga + 1;
            } else {
                ga = 0;
                gb = gb + 1;
            }
            if gb < d {
                lemma_pk_idx_bound(ga, gb, d);
            }
        }
        offset = c + 1;
        i += 1;
    }
    // zero columns after the block
    let ghost before = result@;
    proof {
        lemma_dim_fits_idx(sd, sd);
        lemma_block_prev_end_le(st, 0, d);
        lemma_pk_idx_bound(st + d - 1, st + d - 1, sd);
    }
    for _z in offset..((sym_dim * (sym_dim + 1)) / 2)
        invariant
            rd == row_dim as int,
            st == start as int,
            d == dim as int,
            dim >= 1,
            sd == sym_dim as int,
            dim_fits(sd),
            pk_len(sd) <= usize::MAX,
            rd * pk_len(sd) <= usize::MAX,
            offset == block_prev_end(st, 0, d),
            offset <= _z <= pk_len(sd),
            result@.len() == _z * rd,
            before.len() == offset * rd,
            forall|q: int| 0 <= q < before.len() ==> #[trigger] result@[q] == before[q],
            forall|r: int, t: int| 0 <= r < rd && offset <= t < _z ==> rv(#[trigger] result@[cm_idx(rd, r, t)]) == 0.0real,
    {
        proof {
            assert((_z + 1) * rd == _z * rd + rd) by (nonlinear_arith);
            assert(offset * rd <= _z * rd) by (nonlinear_arith)
                requires offset <= _z, rd >= 0;
        }
        let mut zeros = vec![0.0; row_dim];
        let ghost prev = result@;
        result.append(&mut zeros);
        proof {
            assert forall|r: int, t: int| 0 <= r < rd && offset <= t < _z + 1
                implies rv(#[trigger] result@[cm_idx(rd, r, t)]) == 0.0real by {
                lemma_cm_idx_bound(rd, r, t, _z as int + 1);
                if t < _z {
                    lemma_cm_idx_bound(rd, r, t, _z as int);
                    assert(result@[cm_idx(rd, r, t)] == prev[cm_idx(rd, r, t)]);
                }
            }
            assert forall|q: int| 0 <= q < before.len() implies #[trigger] result@[q] == before[q] by {
                assert(result@[q] == prev[q]);
            }
        }
    }
    proof {
        let cur = result@;
        assert forall|r: int, a: int, b: int| 0 <= r < rd && 0 <= a <= b < d
            implies rv(#[trigger] result@[cm_idx(rd, r, pk_idx(st + a, st + b))]) == rv(mat@[cm_idx(rd, r, pk_idx(a, b))]) by {
            lemma_pk_idx_bound(a, b, d);
            lemma_block_prev_end_bound(st, a, b, 0, d);
            lemma_cm_idx_bound(rd, r, pk_idx(st + a, st + b), offset as int);
            let q = cm_idx(rd, r, pk_idx(st + a, st + b));
            assert(cur[q] == before[q]);
            assert(rv(before[q]) == rv(mat@[cm_idx(rd, r, pk_idx(a, b))]));
        }
        assert forall|r: int, t: int| 0 <= r < rd && 0 <= t < pk_len(sd) && !is_block_idx(t, st, d)
            implies rv(#[trigger] result@[cm_idx(rd, r, t)]) == 0.0real by {
            if t < offset {
                lemma_cm_idx_bound(rd, r, t, offset as int);
                let q = cm_idx(rd, r, t);
                assert(cur[q] == before[q]);
                assert(rv(before[q]) == 0.0real);
            }
        }
        // the postcondition's row count is the integer quotient
        let rdp = (mat@.len() as int) / pk_len(dim as int);
        assert(rdp == rd);
        assert(pk_len(sd) * rd == rd * pk_len(sd)) by (nonlinear_arith);
        assert(result@.len() == rdp * pk_len(sym_dim as int));
        assert forall|r: int, a: int, b: int| 0 <= r < rdp && 0 <= a <= b < dim
            implies rv(#[trigger] result@[cm_idx(rdp, r, pk_idx(start + a, start + b))]) == rv(mat@[cm_idx(rdp, r, pk_idx(a, b))]) by {
            assert(rv(result@[cm_idx(rd, r, pk_idx(st + a, st + b))]) == rv(mat@[cm_idx(rd, r, pk_idx(a, b))]));
        }
        assert forall|r: int, t: int| 0 <= r < rdp && 0 <= t < pk_len(sym_dim as int) && !is_block_idx(t, start as int, dim as int)
            implies rv(#[trigger] result@[cm_idx(rdp, r, t)]) == 0.0real by {
            assert(rv(result@[cm_idx(rd, r, t)]) == 0.0real);
        }
    }
    result
}

// composition of real-valued linear map on symmetric matrices with diagonal block map
// (the packed dim x dim matrix mat embedded as the diagonal block at `start` of a packed
// sym_dim x sym_dim matrix: entry (start + a, start + b) is entry (a, b) of mat, the rest is zero)
pub fn mat_sym_diagonal_mult_sym(sym_dim: usize, mat: &SymMatrix, start: usize, dim: usize) -> (result: SymMatrix)
    requires
        dim_fits(sym_dim as int),
        start + dim <= sym_dim,
        mat@.len() >= pk_len(dim as int),
    ensures
        result@.len() == pk_len(sym_dim as int),
        forall|a: int, b: int| 0 <= a <= b < dim
            ==> rv(#[trigger] result@[pk_idx(start + a, start + b)]) == rv(mat@[pk_idx(a, b)]),
        forall|t: int| 0 <= t < pk_len(sym_dim as int) && !is_block_idx(t, start as int, dim as int)
            ==> rv(#[trigger] result@[t]) == 0.0real,
{
    let mut result: SymMatrix = Vec::new();
    proof {
        lemma_dim_fits_monotone((start + dim) as int, sym_dim as int);
        lemma_dim_fits_idx(sym_dim as int, sym_dim as int);
        lemma_pk_col_start_zero();
    }
    let ones = sym_matrix_diag_block_map(start, dim);
    let ghost (sd, st, d) = (sym_dim as int, start as int, dim as int);
    proof {
        if dim >= 1 {
            lemma_pk_idx_bound(0, 0, d);
        }
    }
    let mut offset = 0;
    let mut i = 0;
    let ghost mut ga: int = 0;
    let ghost mut gb: int = 0;
    while i < ones.len()
        invariant
            sd == sym_dim as int,
            st == start as int,
            d == dim as int,
            st + d <= sd,
            dim_fits(sd),
            pk_len(sd) <= usize::MAX,
            mat@.len() >= pk_len(d),
            ones@.len() == pk_len(d),
            forall|a: int, b: int| 0 <= a <= b < d ==> #[trigger] ones@[pk_idx(a, b)] == pk_idx(st + a, st + b),
            i <= ones@.len(),
            0 <= ga <= gb,
            i == pk_idx(ga, gb),
            i < ones@.len() ==> gb < d,
            i == ones@.len() ==> ga == 0 && gb == d,
            offset == block_prev_end(st, ga, gb),
            offset <= pk_len(sd),
            result@.len() == offset,
            forall|a: int, b: int| 0 <= a <= b < d && pk_idx(a, b) < i
                ==> rv(#[trigger] result@[pk_idx(st + a, st + b)]) == rv(mat@[pk_idx(a, b)]),
            forall|t: int| 0 <= t < offset && !is_block_idx(t, st, d) ==> rv(#[trigger] result@[t]) == 0.0real,
        decreases ones@.len() - i,
    {
        let c = ones[i];
        proof {
            assert(ones@[pk_idx(ga, gb)] == pk_idx(st + ga, st + gb));
            lemma_pk_idx_bound(st + ga, st + gb, sd);
            lemma_pk_idx_bound(ga, gb, d);
            lemma_block_prev_end_le(st, ga, gb);
        }
        let ghost before = result@;
        for _z in offset..c
            invariant
                offset <= _z <= c,
                result@.len() == _z,
                before.len() == offset,
                forall|q: int| 0 <= q < offset ==> #[trigger] result@[q] == before[q],
                forall|t: int| offset <= t < _z ==> rv(#[trigger] result@[t]) == 0.0real,
        {
            result.push(0.0);
        }
        let ghost mid = result@;
        result.push(mat[i]);
        proof {
            let cur = result@;
            assert forall|t: int| offset <= t < c implies !is_block_idx(t, st, d) by {
                lemma_block_idx_gap(st, d, ga, gb, t);
            }
            assert forall|a: int, b: int| 0 <= a <= b < d && pk_idx(a, b) < i + 1
                implies rv(#[trigger] cur[pk_idx(st + a, st + b)]) == rv(mat@[pk_idx(a, b)]) by {
                if pk_idx(a, b) < i {
                    lemma_block_prev_end_bound(st, a, b, ga, gb);
                    let q = pk_idx(st + a, st + b);
                    assert(cur[q] == mid[q]);
                    assert(mid[q] == before[q]);
                } else {
                    lemma_pk_idx_injective(a, b, ga, gb);
                }
            }
            assert forall|t: int| 0 <= t < c + 1 && !is_block_idx(t, st, d)
                implies rv(#[trigger] cur[t]) == 0.0real by {
                assert(cur[t] == mid[t]);
                if t < offset {
                    assert(mid[t] == before[t]);
                }
            }
            lemma_pk_col_start_step(gb);
            if ga < gb {
                ga = ga + 1;
            } else {
                ga = 0;
                gb = gb + 1;
            }
            if gb < d {
                lemma_pk_idx_bound(ga, gb, d);
            }
        }
        offset = c + 1;
        i += 1;
    }
    let ghost before = result@;
    proof {
        lemma_block_prev_end_le(st, 0, d);
        if dim >= 1 {
            lemma_pk_idx_bound(st + d - 1, st + d - 1, sd);
        }
        lemma_dim_fits_idx(sd, sd);
    }
    for _z in offset..((sym_dim * (sym_dim + 1)) / 2)
        invariant
            st == start as int,
            d == dim as int,
            sd == sym_dim as int,
            dim_fits(sd),
            offset == block_prev_end(st, 0, d),
            offset <= _z <= pk_len(sd),
            result@.len() == _z,
            before.len() == offset,
            forall|q: int| 0 <= q < offset ==> #[trigger] result@[q] == before[q],
            forall|t: int| offset <= t < _z ==> rv(#[trigger] result@[t]) == 0.0real,
    {
        result.push(0.0);
    }
    proof {
        assert forall|a: int, b: int| 0 <= a <= b < d
            implies rv(#[trigger] result@[pk_idx(st + a, st + b)]) == rv(mat@[pk_idx(a, b)]) by {
            lemma_pk_idx_bound(a, b, d);
            lemma_block_prev_end_bound(st, a, b, 0, d);
            let q = pk_idx(st + a, st + b);
            assert(result@[q] == before[q]);
            assert(rv(before[q]) == rv(mat@[pk_idx(a, b)]));
        }
        assert forall|t: int| 0 <= t < pk_len(sd) && !is_block_idx(t, st, d)
            implies rv(#[trigger] result@[t]) == 0.0real by {
            if t < offset {
                assert(result@[t] == before[t]);
                assert(rv(before[t]) == 0.0real);
            }
        }
    }
    result
}

} // verus!

// removes redundant constraints from an augmented matrix [A | b] where rows of A represent symmetric matrices
// also preconditions resulting constraint matrix by normalizing rows
//
// Verified against src/lin_alg/reduce_spec.rs: the result consists of the first `rank` rows
// of the echelon form produced by gauss_elim, each divided by the Frobenius norm of the
// symmetric matrix it represents, sqrt(sum_{r <= c} pk_wt(r, c) M_rc^2) (off-diagonal packed
// entries count twice); an inconsistent system aborts.
verus! {

pub fn linear_remove_redundant_sym(mat: &mut Matrix, dim: usize) -> (result: Matrix)
    requires
        dim >= 1,
        dim_fits(dim as int),
        old(mat)@.len() as int % red_col_dim(dim as int) == 0,
        abs_max_r(rvs(old(mat)@)) > 0.0real,
    ensures
        final(mat)@.len() == old(mat)@.len(),
        reduced_system(rvs(old(mat)@), rvs(final(mat)@), rvs(result@), dim as int),
{
    let ghost a0 = rvs(old(mat)@);
    let ghost d = dim as int;
    let ghost cd = red_col_dim(d);
    let ghost eps = gauss_eps_r(a0);
    let ghost len = old(mat)@.len() as int;
    proof {
        lemma_dim_fits_idx(d, d);
        lemma_pk_col_start_monotone(0, d);
        assert(d * (d + 1) / 2 + 1 <= (d + 2) * (d + 2)) by (nonlinear_arith)
            requires d >= 0;
    }
    let col_dim = (dim * (dim + 1)) / 2 + 1;
    let row_dim = mat.len() / col_dim;
    let ghost rd = row_dim as int;
    proof {
        assert(col_dim as int == cd);
        lemma_fundamental_div_mod(len, cd);
        assert(cd * rd == rd * cd) by (nonlinear_arith);
        assert(old(mat)@.len() == row_dim * col_dim);
        assert(len <= usize::MAX);
    }
    let (echelon, rank, solvable) = gauss_elim(mat, col_dim, row_dim);
    let ghost e = rvs(echelon@);
    let ghost pivots = choose|p: Seq<int>| {
        &&& #[trigger] echelon_form(rvs(echelon@), row_dim as int, col_dim as int, p, rv(TOL) * abs_max_r(rvs(old(mat)@)))
        &&& skipped_bounded(rvs(echelon@), row_dim as int, col_dim as int, p, rv(TOL) * abs_max_r(rvs(old(mat)@)), col_dim as int)
        &&& p.len() == rank
        &&& solvable == (p.len() == 0 || p.last() < col_dim - 1)
    };
    if !solvable {
        abort("linear_remove_redundant_sym: your system of constraints is inconsistent");
    }
    let ghost rk = rank as int;
    proof {
        assert(echelon_form(e, rd, cd, pivots, eps));
        // every pivot column lies in the packed part (the right-hand side holds no pivot)
        assert forall|l: int| 0 <= l < rk implies #[trigger] pivots[l] < cd - 1 by {
            if l < rk - 1 {
                assert(pivots[l] < pivots[rk - 1]);
            }
        }
        assert(rank <= row_dim);
        assert(col_dim * rank <= col_dim * row_dim) by (nonlinear_arith)
            requires rank <= row_dim;
        assert(cd * rk == rk * cd) by (nonlinear_arith);
        assert(rv(TOL) == 0.0000000001real);
        assert(eps > 0.0real) by (nonlinear_arith)
            requires rv(TOL) > 0.0real, abs_max_r(a0) > 0.0real, eps == rv(TOL) * abs_max_r(a0);
        lemma_pk_col_start_zero();
        assert(0 * rd == 0) by (nonlinear_arith);
    }
    let mut result: Matrix = vec![0.0; col_dim * rank];
    let f = 2.0_f64.sqrt();
    proof {
        assert(rv(f) * rv(f) == 2.0real);
    }
    for i in 0..rank
        invariant
            dim >= 1,
            d == dim as int,
            dim_fits(d),
            cd == red_col_dim(d),
            col_dim as int == cd,
            (dim * (dim + 1)) / 2 + 1 == col_dim,
            rd == row_dim as int,
            rk == rank as int,
            rank <= row_dim,
            len == rd * cd,
            len == cd * rd,
            len <= usize::MAX,
            col_dim * rank <= len,
            e == rvs(echelon@),
            echelon@.len() == len,
            echelon_form(e, rd, cd, pivots, eps),
            pivots.len() == rk,
            eps > 0.0real,
            forall|l: int| 0 <= l < rk ==> #[trigger] pivots[l] < cd - 1,
            rv(f) * rv(f) == 2.0real,
            result@.len() == col_dim * rank,
            cd * rk == rk * cd,
            forall|i2: int, j: int| 0 <= i2 < i && 0 <= j < cd
                ==> #[trigger] cm_entry(rvs(result@), rk, i2, j) == cm_entry(e, rd, i2, j) / row_scale_r(e, rd, d, i2),
    {
        let ghost ii = i as int;
        proof {
            lemma_cm_idx_bound(rd, ii, 0, cd);
            lemma_pk_col_start_zero();
            assert(0 * rd == 0) by (nonlinear_arith);
            assert(cm_idx(rd, ii, 0) == ii);
        }
        let mut scale: f64 = 0.0;
        scale = scale.hypot(echelon[i]);
        proof {
            let x = rv(echelon@[ii]);
            assert(x * x >= 0.0real) by (nonlinear_arith);
            assert(rv(scale) == sqrt_r(0.0real * 0.0real + x * x));
            assert(constr_entry(e, rd, ii, pk_idx(0, 0)) == x);
            assert(row_sq_term_r(e, rd, ii, 0, 0) == 1.0real * (x * x));
            lemma_row_sq_upto_zero(e, rd, ii);
            lemma_row_sq_upto_step(e, rd, ii, 0);
            lemma_col_sq_zero(e, rd, ii, 0);
            lemma_col_sq_step(e, rd, ii, 0, 0);
            assert(rv(scale) * rv(scale) == row_sq_upto_r(e, rd, ii, 1));
        }
        let mut start = 1;
        for k in 1..dim
            invariant
                dim >= 1,
                d == dim as int,
                dim_fits(d),
                cd == red_col_dim(d),
                rd == row_dim as int,
                len == rd * cd,
                len == cd * rd,
                len <= usize::MAX,
                e == rvs(echelon@),
                echelon@.len() == len,
                ii == i as int,
                i < rank,
                rank <= row_dim,
                rv(f) * rv(f) == 2.0real,
                1 <= k <= dim,
                start as int == pk_col_start(k as int),
                rv(scale) >= 0.0real,
                rv(scale) * rv(scale) == row_sq_upto_r(e, rd, ii, k as int),
        {
            let ghost kk = k as int;
            proof {
                lemma_dim_fits_idx(d, kk);
                lemma_pk_diag_idx(kk);
                lemma_pk_idx_bound(kk, kk, d);
                lemma_pk_col_start_step(kk);
                lemma_col_sq_zero(e, rd, ii, kk);
            }
            let diag = (k * k + 3 * k) / 2;
            proof {
                assert(diag as int == pk_idx(kk, kk));
                assert(diag == start + k);
            }
            for col in start..diag
                invariant
                    d == dim as int,
                    dim_fits(d),
                    cd == red_col_dim(d),
                    rd == row_dim as int,
                    len == rd * cd,
                    len == cd * rd,
                    len <= usize::MAX,
                    e == rvs(echelon@),
                    echelon@.len() == len,
                    ii == i as int,
                    i < rank,
                    rank <= row_dim,
                    rv(f) * rv(f) == 2.0real,
                    kk == k as int,
                    1 <= k < dim,
                    start as int == pk_col_start(kk),
                    diag as int == pk_idx(kk, kk),
                    start <= col <= diag,
                    rv(scale) >= 0.0real,
                    rv(scale) * rv(scale) == row_sq_upto_r(e, rd, ii, kk) + col_sq_r(e, rd, ii, kk, col as int - start as int),
            {
                let ghost r = col as int - start as int;
                proof {
                    assert(col as int == pk_idx(r, kk));
                    lemma_pk_idx_bound(r, kk, d);
                    lemma_cm_idx_bound(rd, ii, col as int, cd);
                }
                let v = echelon[col * row_dim + i];
                let y = v * f;
                let ghost prev = rv(scale);
                scale = scale.hypot(y);
                proof {
                    let (xv, fv) = (rv(v), rv(f));
                    assert(rv(y) * rv(y) == (fv * fv) * (xv * xv)) by (nonlinear_arith)
                        requires rv(y) == xv * fv;
                    assert(prev * prev >= 0.0real) by (nonlinear_arith);
                    assert(xv * xv >= 0.0real) by (nonlinear_arith);
                    assert(rv(scale) == sqrt_r(prev * prev + rv(y) * rv(y)));
                    assert(constr_entry(e, rd, ii, pk_idx(r, kk)) == xv);
                    assert(row_sq_term_r(e, rd, ii, r, kk) == 2.0real * (xv * xv));
                    lemma_col_sq_step(e, rd, ii, kk, r);
                }
            }
            proof {
                lemma_cm_idx_bound(rd, ii, diag as int, cd);
            }
            let v = echelon[diag * row_dim + i];
            let ghost prev = rv(scale);
            scale = scale.hypot(v);
            proof {
                let xv = rv(v);
                assert(prev * prev >= 0.0real) by (nonlinear_arith);
                assert(xv * xv >= 0.0real) by (nonlinear_arith);
                assert(rv(scale) == sqrt_r(prev * prev + xv * xv));
                assert(constr_entry(e, rd, ii, pk_idx(kk, kk)) == xv);
                assert(row_sq_term_r(e, rd, ii, kk, kk) == 1.0real * (xv * xv));
                lemma_col_sq_step(e, rd, ii, kk, kk);
                lemma_row_sq_upto_step(e, rd, ii, kk);
            }
            start = diag + 1;
        }
        if !scale.is_finite() {
            abort("linear_remove_redundant_sym: row has non-finite packed norm");
        }
        proof {
            lemma_row_scale_pos(e, rd, d, pivots, eps, ii);
            lemma_sqrt_unique(row_sq_r(e, rd, d, ii), rv(scale));
            assert(rv(scale) == row_scale_r(e, rd, d, ii));
        }
        let ghost before = rvs(result@);
        for j in 0..col_dim
            invariant
                d == dim as int,
                cd == red_col_dim(d),
                col_dim as int == cd,
                rd == row_dim as int,
                rk == rank as int,
                len == rd * cd,
                len == cd * rd,
                len <= usize::MAX,
                e == rvs(echelon@),
                echelon@.len() == len,
                ii == i as int,
                i < rank,
                rank <= row_dim,
                rv(scale) == row_scale_r(e, rd, d, ii),
                rv(scale) > 0.0real,
                col_dim * rank <= len,
                result@.len() == col_dim * rank,
                cd * rk == rk * cd,
                forall|i2: int, j2: int| 0 <= i2 < rk && 0 <= j2 < cd && !(i2 == ii && j2 < j)
                    ==> #[trigger] cm_entry(rvs(result@), rk, i2, j2) == cm_entry(before, rk, i2, j2),
                forall|j2: int| 0 <= j2 < j
                    ==> #[trigger] cm_entry(rvs(result@), rk, ii, j2) == cm_entry(e, rd, ii, j2) / row_scale_r(e, rd, d, ii),
        {
            let ghost jj = j as int;
            proof {
                lemma_cm_idx_bound(rk, ii, jj, cd);
                lemma_cm_idx_bound(rd, ii, jj, cd);
            }
            let ghost prev = rvs(result@);
            result[j * rank + i] = echelon[j * row_dim + i] / scale;
            proof {
                let cur = rvs(result@);
                assert(cm_entry(cur, rk, ii, jj) == cm_entry(e, rd, ii, jj) / rv(scale));
                assert forall|i2: int, j2: int| 0 <= i2 < rk && 0 <= j2 < cd && !(i2 == ii && j2 < jj + 1)
                    implies #[trigger] cm_entry(cur, rk, i2, j2) == cm_entry(before, rk, i2, j2) by {
                    lemma_cm_entry_write_other(prev, cur, rk, cd, ii, jj, i2, j2);
                }
                assert forall|j2: int| 0 <= j2 < jj + 1
                    implies #[trigger] cm_entry(cur, rk, ii, j2) == cm_entry(e, rd, ii, j2) / row_scale_r(e, rd, d, ii) by {
                    if j2 < jj {
                        lemma_cm_entry_write_other(prev, cur, rk, cd, ii, jj, ii, j2);
                    }
                }
            }
        }
    }
    proof {
        assert forall|i: int| 0 <= i < rk implies #[trigger] row_scale_r(e, rd, d, i) > 0.0real by {
            lemma_row_scale_pos(e, rd, d, pivots, eps, i);
        }
        assert(reduced_with(a0, e, rvs(result@), d, pivots));
        assert(pivots.len() == rk);
    }
    result
}

} // verus!

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

verus! {

// Pivot scale of the solvers: max(max_t |mat[t]|, 1) over the whole stored matrix.
#[inline]
fn pivot_scale(mat: &[f64]) -> (scale: f64)
    ensures
        rv(scale) == pivot_scale_r(rvs(mat@)),
{
    let mut max: f64 = 0.0;
    for t in 0..mat.len()
        invariant
            rv(max) == abs_max_prefix_r(rvs(mat@), t as nat),
    {
        proof {
            lemma_abs_max_prefix_step(rvs(mat@), t as nat);
        }
        max = max.max(mat[t].abs());
    }
    max.max(1.0)
}

// solver for diagonal linear system Dx = b
// (aborts when some diagonal entry is below TOL times the pivot scale)
pub fn diag_solver(diag: &Vector, vect: &Vector) -> (solution: Vector)
    requires
        diag@.len() == vect@.len(),
    ensures
        solution@.len() == vect@.len(),
        forall|i: int| 0 <= i < vect@.len()
            ==> abs_r(rv(#[trigger] diag@[i])) >= rv(TOL) * pivot_scale_r(rvs(diag@)),
        forall|i: int| 0 <= i < vect@.len()
            ==> rv(#[trigger] solution@[i]) == rv(vect@[i]) / rv(diag@[i]),
        mat_vec(diag_view(rvs(diag@)), vect@.len() as int, rvs(solution@)) == rvs(vect@),
{
    let mut solution = Vec::new();
    let scale = pivot_scale(diag);
    for i in 0..vect.len()
        invariant
            diag@.len() == vect@.len(),
            rv(scale) == pivot_scale_r(rvs(diag@)),
            solution@.len() == i,
            forall|k: int| 0 <= k < i ==> abs_r(rv(#[trigger] diag@[k])) >= rv(TOL) * rv(scale),
            forall|k: int| 0 <= k < i ==> rv(#[trigger] solution@[k]) == rv(vect@[k]) / rv(diag@[k]),
    {
        if scaled_near_zero(diag[i], scale, TOL) {
            abort("diag_solver: division by zero");
        }
        solution.push(vect[i] / diag[i]);
    }
    proof {
        let (d, x, b) = (rvs(diag@), rvs(solution@), rvs(vect@));
        let dim = vect@.len() as int;
        assert forall|i: int| 0 <= i < dim implies #[trigger] mat_vec(diag_view(d), dim, x)[i] == b[i] by {
            lemma_diag_mat_vec(d, dim, x, i);
            let (u, v) = (d[i], b[i]);
            assert(u * (v / u) == v) by (nonlinear_arith)
                requires u != 0.0real;
        }
        assert(mat_vec(diag_view(d), dim, x) =~= b);
    }
    solution
}

// substitution solvers for full-rank triangular linear systems Ax = b
// (each aborts when a diagonal entry is below TOL times the pivot scale;
// on return, every diagonal entry satisfies the pivot condition and the
// solution solves the system exactly in the real model)

pub fn forw_subst(mat: &LowTriMatrix, vect: &mut [f64]) -> (solution: Vector)
    requires
        dim_fits(old(vect)@.len() as int),
        mat@.len() >= old(vect)@.len() * (old(vect)@.len() + 1) / 2,
    ensures
        solution@.len() == old(vect)@.len(),
        final(vect)@.len() == old(vect)@.len(),
        forall|j: int| 0 <= j < old(vect)@.len()
            ==> abs_r(#[trigger] lt_entry(rvs(mat@), old(vect)@.len() as int, j, j)) >= rv(TOL) * pivot_scale_r(rvs(mat@)),
        mat_vec(lt_view(rvs(mat@), old(vect)@.len() as int), old(vect)@.len() as int, rvs(solution@)) == rvs(old(vect)@),
{
    let dim = vect.len();
    let ghost l = lt_view(rvs(mat@), dim as int);
    let ghost b = rvs(old(vect)@);
    let mut solution = Vec::with_capacity(dim);
    let mut d : f64;
    let scale = pivot_scale(mat);
    proof {
        lemma_lt_col_start_end(dim as int);
    }
    for j in 0..dim
        invariant
            dim == old(vect)@.len(),
            vect@.len() == dim,
            dim_fits(dim as int),
            mat@.len() >= lt_col_start(dim as int, dim as int),
            l == lt_view(rvs(mat@), dim as int),
            b == rvs(old(vect)@),
            rv(scale) == pivot_scale_r(rvs(mat@)),
            solution@.len() == j,
            forall|i: int| 0 <= i < j
                ==> abs_r(#[trigger] lt_entry(rvs(mat@), dim as int, i, i)) >= rv(TOL) * rv(scale),
            forall|i: int| 0 <= i < j ==> #[trigger] row_prefix_r(l, rvs(solution@), i, (i + 1) as nat) == b[i],
            forall|i: int| j <= i < dim ==> rv(#[trigger] vect@[i]) == b[i] - row_prefix_r(l, rvs(solution@), i, j as nat),
    {
        proof {
            lemma_lt_idx_fits(dim as int, j as int);
            lemma_lt_idx_bound(dim as int, j as int, j as int, dim as int);
        }
        d = mat[(j * (2 * dim - j + 1)) / 2];
        if scaled_near_zero(d, scale, TOL) {
            abort("forw_subst: division by zero");
        }
        let ghost sol = rvs(solution@);
        solution.push(vect[j] / d);
        proof {
            let sol2 = rvs(solution@);
            // the new solution entry solves row j; earlier rows and the pending right-hand sides are unchanged
            assert forall|i: int| 0 <= i < j
                implies #[trigger] row_prefix_r(l, sol2, i, (i + 1) as nat) == b[i] by {
                lemma_row_prefix_ext(l, sol, sol2, i, (i + 1) as nat);
            }
            lemma_row_prefix_ext(l, sol, sol2, j as int, j as nat);
            lemma_row_prefix_step(l, sol2, j as int, j as nat);
            let (u, v) = (rv(d), rv(vect@[j as int]));
            assert(u * (v / u) == v) by (nonlinear_arith)
                requires u != 0.0real;
            assert forall|i: int| j < i < dim
                implies rv(#[trigger] vect@[i]) == b[i] - row_prefix_r(l, sol2, i, j as nat) by {
                lemma_row_prefix_ext(l, sol, sol2, i, j as nat);
            }
        }
        for i in j + 1..dim
            invariant
                j < dim,
                dim == old(vect)@.len(),
                vect@.len() == dim,
                dim_fits(dim as int),
                mat@.len() >= lt_col_start(dim as int, dim as int),
                l == lt_view(rvs(mat@), dim as int),
                b == rvs(old(vect)@),
                solution@.len() == j + 1,
                forall|i2: int| j < i2 < i
                    ==> rv(#[trigger] vect@[i2]) == b[i2] - row_prefix_r(l, rvs(solution@), i2, (j + 1) as nat),
                forall|i2: int| i <= i2 < dim
                    ==> rv(#[trigger] vect@[i2]) == b[i2] - row_prefix_r(l, rvs(solution@), i2, j as nat),
        {
            proof {
                lemma_lt_idx_fits(dim as int, j as int);
                lemma_lt_idx_bound(dim as int, i as int, j as int, dim as int);
                lemma_row_prefix_step(l, rvs(solution@), i as int, j as nat);
            }
            vect[i] = vect[i] - solution[j] * mat[j * (2 * dim - j + 1) / 2 + (i - j)];
            proof {
                let (p, q) = (rv(solution@[j as int]), l(i as int, j as int));
                assert(p * q == q * p) by (nonlinear_arith);
            }
        }
    }
    proof {
        let x = rvs(solution@);
        assert forall|i: int| 0 <= i < dim implies #[trigger] mat_vec(l, dim as int, x)[i] == b[i] by {
            lemma_lt_mat_vec(rvs(mat@), dim as int, x, i);
        }
        assert(mat_vec(l, dim as int, x) =~= b);
    }
    solution
}

// here we implicitly treat a given lower triangular matrix as its (upper triangular) transpose
pub fn back_subst(mat: &LowTriMatrix, vect: &mut [f64]) -> (solution: Vector)
    requires
        dim_fits(old(vect)@.len() as int),
        mat@.len() >= old(vect)@.len() * (old(vect)@.len() + 1) / 2,
    ensures
        solution@.len() == old(vect)@.len(),
        final(vect)@.len() == old(vect)@.len(),
        forall|j: int| 0 <= j < old(vect)@.len()
            ==> abs_r(#[trigger] lt_entry(rvs(mat@), old(vect)@.len() as int, j, j)) >= rv(TOL) * pivot_scale_r(rvs(mat@)),
        mat_vec(mtr(lt_view(rvs(mat@), old(vect)@.len() as int)), old(vect)@.len() as int, rvs(solution@)) == rvs(old(vect)@),
{
    let dim = vect.len();
    let ghost u = mtr(lt_view(rvs(mat@), dim as int));
    let ghost b = rvs(old(vect)@);
    let mut solution = vec![0.0; dim];
    let mut d : f64;
    let scale = pivot_scale(mat);
    proof {
        lemma_lt_col_start_end(dim as int);
    }
    // columns dim - 1, ..., j + 1 have been processed when the loop body runs for j
    for j in it: (0..dim).rev()
        invariant
            dim == old(vect)@.len(),
            vect@.len() == dim,
            dim_fits(dim as int),
            mat@.len() >= lt_col_start(dim as int, dim as int),
            u == mtr(lt_view(rvs(mat@), dim as int)),
            b == rvs(old(vect)@),
            rv(scale) == pivot_scale_r(rvs(mat@)),
            solution@.len() == dim,
            it.seq().len() == dim,
            forall|k: int| 0 <= k < dim ==> #[trigger] it.seq()[k] == dim - 1 - k,
            forall|i: int| dim - it.index() <= i < dim
                ==> abs_r(#[trigger] lt_entry(rvs(mat@), dim as int, i, i)) >= rv(TOL) * rv(scale),
            forall|i: int| dim - it.index() <= i < dim
                ==> #[trigger] row_range_r(u, rvs(solution@), i, i as nat, dim as nat) == b[i],
            forall|i: int| 0 <= i < dim - it.index()
                ==> rv(#[trigger] vect@[i]) == b[i] - row_range_r(u, rvs(solution@), i, (dim - it.index()) as nat, dim as nat),
    {
        proof {
            assert(j == dim - 1 - it.index());
            lemma_lt_idx_fits(dim as int, j as int);
            lemma_lt_idx_bound(dim as int, j as int, j as int, dim as int);
        }
        d = mat[(j * (2 * dim - j + 1)) / 2];
        if scaled_near_zero(d, scale, TOL) {
            abort("back_subst: division by zero");
        }
        let ghost sol = rvs(solution@);
        solution[j] = vect[j] / d;
        proof {
            let sol2 = rvs(solution@);
            // the new solution entry solves row j; later rows and the pending right-hand sides are unchanged
            assert forall|i: int| j < i < dim
                implies #[trigger] row_range_r(u, sol2, i, i as nat, dim as nat) == b[i] by {
                lemma_row_range_ext(u, sol, sol2, i, i as nat, dim as nat);
            }
            lemma_row_range_ext(u, sol, sol2, j as int, (j + 1) as nat, dim as nat);
            lemma_row_range_step(u, sol2, j as int, j as nat, dim as nat);
            let (p, v) = (rv(d), rv(vect@[j as int]));
            assert(p * (v / p) == v) by (nonlinear_arith)
                requires p != 0.0real;
            assert forall|i: int| 0 <= i < j
                implies rv(#[trigger] vect@[i]) == b[i] - row_range_r(u, sol2, i, (j + 1) as nat, dim as nat) by {
                lemma_row_range_ext(u, sol, sol2, i, (j + 1) as nat, dim as nat);
            }
        }
        for i in 0..j
            invariant
                j < dim,
                dim == old(vect)@.len(),
                vect@.len() == dim,
                dim_fits(dim as int),
                mat@.len() >= lt_col_start(dim as int, dim as int),
                u == mtr(lt_view(rvs(mat@), dim as int)),
                b == rvs(old(vect)@),
                solution@.len() == dim,
                forall|i2: int| 0 <= i2 < i
                    ==> rv(#[trigger] vect@[i2]) == b[i2] - row_range_r(u, rvs(solution@), i2, j as nat, dim as nat),
                forall|i2: int| i <= i2 < j
                    ==> rv(#[trigger] vect@[i2]) == b[i2] - row_range_r(u, rvs(solution@), i2, (j + 1) as nat, dim as nat),
        {
            proof {
                lemma_lt_idx_fits(dim as int, i as int);
                lemma_lt_idx_bound(dim as int, j as int, i as int, dim as int);
                lemma_row_range_step(u, rvs(solution@), i as int, j as nat, dim as nat);
            }
            vect[i] = vect[i] - solution[j] * mat[i * (2 * dim - i + 1) / 2 + (j - i)];
            proof {
                let (p, q) = (rv(solution@[j as int]), u(i as int, j as int));
                assert(p * q == q * p) by (nonlinear_arith);
            }
        }
    }
    proof {
        let x = rvs(solution@);
        assert forall|i: int| 0 <= i < dim implies #[trigger] mat_vec(u, dim as int, x)[i] == b[i] by {
            lemma_ut_mat_vec(rvs(mat@), dim as int, x, i);
        }
        assert(mat_vec(u, dim as int, x) =~= b);
    }
    solution
}

// LDLT decomposition of a symmetric positive definite matrix
// (returns None when a pivot falls below TOL times max(max |A_ii|, 1); on Some((l, d)),
// l is unit lower-triangular, every pivot is at least that threshold, and L D L^T = A)
pub fn ldlt_decomp(mat: &SymMatrix, dim: usize) -> (result: Option<(LowTriMatrix, Vector)>)
    requires
        dim_fits(dim as int),
        mat@.len() >= pk_len(dim as int),
    ensures
        result matches Some((l, d)) ==> {
            &&& l@.len() == dim * (dim + 1) / 2
            &&& d@.len() == dim
            &&& ldlt_ok(rvs(mat@), rvs(l@), rvs(d@), dim as int)
            &&& forall|k: int| 0 <= k < dim
                    ==> rv(#[trigger] d@[k]) >= rv(TOL) * max_r(max_abs_diag_r(rvs(mat@), dim as int), 1.0real)
            &&& meq(mmul(mmul(lt_view(rvs(l@), dim as int), diag_view(rvs(d@)), dim as int), mtr(lt_view(rvs(l@), dim as int)), dim as int),
                    sym_view(rvs(mat@)), dim as int, dim as int)
        },
{
    let ghost a = rvs(mat@);
    let ghost scale = max_r(max_abs_diag_r(a, dim as int), 1.0real);
    let (mut l, mut d) : (LowTriMatrix, Vector) = (ident_mat_low_tri(dim), Vec::with_capacity(dim));
    let mut sum;
    proof {
        lemma_lt_col_start_end(dim as int);
    }
    for i in 0..dim
        invariant
            dim_fits(dim as int),
            l@.len() == lt_col_start(dim as int, dim as int),
            is_lt_ident_r(rvs(l@), dim as int),
    {
        proof {
            lemma_lt_idx_fits(dim as int, i as int);
            lemma_lt_idx_bound(dim as int, i as int, i as int, dim as int);
        }
        let ghost prev = rvs(l@);
        l[(i * (2 * dim - i + 1)) / 2] = 1.0;
        proof {
            // the diagonal entry was already 1, so the real view is unchanged
            assert(lt_entry(prev, dim as int, i as int, i as int) == 1.0real);
            assert(lt_idx(dim as int, i as int, i as int) == lt_col_start(dim as int, i as int));
            assert(rvs(l@) =~= prev);
        }
    }
    let mut ind;
    for i in 0..dim
        invariant
            dim_fits(dim as int),
            a == rvs(mat@),
            mat@.len() >= pk_len(dim as int),
            scale == max_r(max_abs_diag_r(a, dim as int), 1.0real),
            l@.len() == lt_col_start(dim as int, dim as int),
            d@.len() == i,
            forall|r: int| 0 <= r < dim ==> #[trigger] lt_entry(rvs(l@), dim as int, r, r) == 1.0real,
            forall|r: int, c: int| 0 <= c <= r < i ==> #[trigger] ldlt_entry_ok(a, rvs(l@), rvs(d@), dim as int, r, c),
            forall|k: int| 0 <= k < i ==> rv(#[trigger] d@[k]) >= rv(TOL) * scale,
    {
        // row i of L: L_ij = (A_ij - sum_{k < j} d_k L_ik L_jk) / d_j for j < i
        for j in 0..i
            invariant
                i < dim,
                dim_fits(dim as int),
                a == rvs(mat@),
                mat@.len() >= pk_len(dim as int),
                scale == max_r(max_abs_diag_r(a, dim as int), 1.0real),
                l@.len() == lt_col_start(dim as int, dim as int),
                d@.len() == i,
                forall|r: int| 0 <= r < dim ==> #[trigger] lt_entry(rvs(l@), dim as int, r, r) == 1.0real,
                forall|r: int, c: int| 0 <= c <= r < i ==> #[trigger] ldlt_entry_ok(a, rvs(l@), rvs(d@), dim as int, r, c),
                forall|c: int| 0 <= c < j ==> #[trigger] ldlt_entry_ok(a, rvs(l@), rvs(d@), dim as int, i as int, c),
                forall|k: int| 0 <= k < i ==> rv(#[trigger] d@[k]) >= rv(TOL) * scale,
        {
            sum = 0.0;
            for k in 0..j
                invariant
                    j < i < dim,
                    dim_fits(dim as int),
                    l@.len() == lt_col_start(dim as int, dim as int),
                    d@.len() == i,
                    rv(sum) == ldlt_sum_r(rvs(l@), rvs(d@), dim as int, i as int, j as int, k as nat),
            {
                proof {
                    lemma_lt_idx_fits(dim as int, k as int);
                    lemma_lt_idx_bound(dim as int, i as int, k as int, dim as int);
                    lemma_lt_idx_bound(dim as int, j as int, k as int, dim as int);
                    lemma_ldlt_sum_step(rvs(l@), rvs(d@), dim as int, i as int, j as int, k as nat);
                }
                sum = (l[k * (2 * dim - k + 1) / 2 + (i - k)] * l[k * (2 * dim - k + 1) / 2 + (j - k)]).mul_add(d[k], sum);
                proof {
                    let (u, v, w) = (lt_entry(rvs(l@), dim as int, i as int, k as int), lt_entry(rvs(l@), dim as int, j as int, k as int), rv(d@[k as int]));
                    assert(u * v * w == w * u * v) by (nonlinear_arith);
                }
            }
            proof {
                lemma_lt_idx_fits(dim as int, j as int);
                lemma_lt_idx_bound(dim as int, i as int, j as int, dim as int);
                lemma_dim_fits_idx(dim as int, i as int);
                lemma_pk_idx_bound(j as int, i as int, dim as int);
                assert(rv(TOL) * scale >= rv(TOL)) by (nonlinear_arith)
                    requires scale >= 1.0real, rv(TOL) > 0.0real;
            }
            let ghost prev = rvs(l@);
            l[j * (2 * dim - j + 1) / 2 + (i - j)] = (mat[(i * (i + 1)) / 2 + j] - sum) / d[j];
            proof {
                let (cur, dv) = (rvs(l@), rvs(d@));
                assert(forall|t: int| 0 <= t < prev.len() && t != lt_idx(dim as int, i as int, j as int) ==> cur[t] == prev[t]);
                // entries other than (i, j) keep their values and their recurrences
                assert forall|r: int| 0 <= r < dim implies #[trigger] lt_entry(cur, dim as int, r, r) == 1.0real by {
                    lemma_lt_entry_write_other(prev, cur, dim as int, i as int, j as int, r, r);
                }
                assert forall|r: int, c: int| 0 <= c <= r < i implies #[trigger] ldlt_entry_ok(a, cur, dv, dim as int, r, c) by {
                    lemma_ldlt_entry_ok_write(a, prev, cur, dv, dim as int, i as int, j as int, r, c);
                }
                assert forall|c: int| 0 <= c < j implies #[trigger] ldlt_entry_ok(a, cur, dv, dim as int, i as int, c) by {
                    lemma_ldlt_entry_ok_write(a, prev, cur, dv, dim as int, i as int, j as int, i as int, c);
                }
                // the new entry satisfies its recurrence: d_j L_ij = A_ij - sum
                assert forall|k: int| 0 <= k < j implies #[trigger] lt_entry(prev, dim as int, i as int, k) == lt_entry(cur, dim as int, i as int, k) by {
                    lemma_lt_entry_write_other(prev, cur, dim as int, i as int, j as int, i as int, k);
                }
                assert forall|k: int| 0 <= k < j implies #[trigger] lt_entry(prev, dim as int, j as int, k) == lt_entry(cur, dim as int, j as int, k) by {
                    lemma_lt_entry_write_other(prev, cur, dim as int, i as int, j as int, j as int, k);
                }
                lemma_ldlt_sum_ext(prev, cur, dv, dv, dim as int, i as int, j as int, j as nat);
                let (p, q, s) = (dv[j as int], sym_entry(a, i as int, j as int), rv(sum));
                assert(lt_entry(cur, dim as int, i as int, j as int) == (q - s) / p);
                assert(p * ((q - s) / p) == q - s) by (nonlinear_arith)
                    requires p != 0.0real;
            }
        }
        // pivot d_i = A_ii - sum_{k < i} d_k L_ik L_ik
        sum = 0.0;
        for k in 0..i
            invariant
                i < dim,
                dim_fits(dim as int),
                l@.len() == lt_col_start(dim as int, dim as int),
                d@.len() == i,
                rv(sum) == ldlt_sum_r(rvs(l@), rvs(d@), dim as int, i as int, i as int, k as nat),
        {
            proof {
                lemma_lt_idx_fits(dim as int, k as int);
                lemma_lt_idx_bound(dim as int, i as int, k as int, dim as int);
                lemma_ldlt_sum_step(rvs(l@), rvs(d@), dim as int, i as int, i as int, k as nat);
            }
            ind = k * (2 * dim - k + 1) / 2 + (i - k);
            sum = (l[ind] * l[ind]).mul_add(d[k], sum);
            proof {
                let (u, w) = (lt_entry(rvs(l@), dim as int, i as int, k as int), rv(d@[k as int]));
                assert(u * u * w == w * u * u) by (nonlinear_arith);
            }
        }
        proof {
            lemma_dim_fits_idx(dim as int, i as int);
            lemma_pk_diag_idx(i as int);
            lemma_pk_idx_bound(i as int, i as int, dim as int);
        }
        let ghost prev_d = rvs(d@);
        d.push(mat[(i * i + 3 * i) / 2] - sum);
        proof {
            let (lv, dv) = (rvs(l@), rvs(d@));
            // rows 0..i and the off-diagonal part of row i only read the previous pivots
            assert forall|r: int, c: int| 0 <= c <= r <= i && c < i implies #[trigger] ldlt_entry_ok(a, lv, dv, dim as int, r, c) by {
                lemma_ldlt_entry_ok_push(a, lv, prev_d, dv, dim as int, r, c);
            }
            lemma_ldlt_sum_ext(lv, lv, prev_d, dv, dim as int, i as int, i as int, i as nat);
            let (u, v) = (dv[i as int], lt_entry(lv, dim as int, i as int, i as int));
            assert(u * v == u) by (nonlinear_arith)
                requires v == 1.0real;
            assert(ldlt_entry_ok(a, lv, dv, dim as int, i as int, i as int));
        }

        if pivot_below_positive_threshold(d[i], max_diag_sym(mat, dim).max(1.0), TOL) {
            return None;
        }
    }
    proof {
        lemma_ldlt_product(a, rvs(l@), rvs(d@), dim as int);
    }
    Some((l, d))
}

// substitution solver for symmetric positive definite matrix
// (aborts when the LDLT factorization fails; on return A x = b exactly in the real model)
pub fn pos_def_solver(mat: &SymMatrix, vect: &mut [f64]) -> (solution: Vector)
    requires
        dim_fits(old(vect)@.len() as int),
        mat@.len() >= pk_len(old(vect)@.len() as int),
    ensures
        solution@.len() == old(vect)@.len(),
        final(vect)@.len() == old(vect)@.len(),
        mat_vec(sym_view(rvs(mat@)), old(vect)@.len() as int, rvs(solution@)) == rvs(old(vect)@),
{
    let dim = vect.len();
    let Some((l, d)) = ldlt_decomp(mat, dim) else { abort("ldlt_decomp: given matrix is not positive definite") };
    let sol1 = forw_subst(&l, vect);
    let mut sol2 = diag_solver(&d, &sol1);
    let solution = back_subst(&l, &mut sol2);
    proof {
        // A x = (L D L^T) x = L (D (L^T x)) = L (D sol2) = L sol1 = b
        let n = dim as int;
        let (lv, dv, x) = (lt_view(rvs(l@), n), diag_view(rvs(d@)), rvs(solution@));
        let ldlt = mmul(mmul(lv, dv, n), mtr(lv), n);
        lemma_mat_vec_meq(sym_view(rvs(mat@)), ldlt, n, x);
        lemma_mul_assoc_meq(lv, dv, mtr(lv), n, n, n, n);
        lemma_mat_vec_meq(ldlt, mmul(lv, mmul(dv, mtr(lv), n), n), n, x);
        lemma_mat_vec_mul(lv, mmul(dv, mtr(lv), n), n, x);
        lemma_mat_vec_mul(dv, mtr(lv), n, x);
    }
    solution
}

// solver for block-diagonal symmetric positive definite matrix (A, I)
pub fn pos_def_solver_upperleft(mat: &SymMatrix, vect: &mut Vector, id_len: usize) -> (result: Vector)
    requires
        id_len <= old(vect)@.len(),
        dim_fits((old(vect)@.len() - id_len) as int),
        mat@.len() >= pk_len((old(vect)@.len() - id_len) as int),
    ensures
        result@.len() == old(vect)@.len(),
        final(vect)@.len() == old(vect)@.len(),
        mat_vec(block_ident_view(sym_view(rvs(mat@)), (old(vect)@.len() - id_len) as int), old(vect)@.len() as int, rvs(result@))
            == rvs(old(vect)@),
{
    let ghost b = rvs(old(vect)@);
    let ul_len = vect.len() - id_len;
    let mut result = pos_def_solver(mat, &mut vect.as_mut_slice()[..ul_len]);
    let ghost head = rvs(result@);
    for t in ul_len..vect.len()
        invariant
            ul_len <= vect@.len(),
            vect@.len() == b.len(),
            result@.len() == t,
            head.len() == ul_len,
            forall|s: int| 0 <= s < ul_len ==> rv(#[trigger] result@[s]) == head[s],
            forall|s: int| ul_len <= s < t ==> rv(#[trigger] result@[s]) == b[s],
            forall|s: int| ul_len <= s < vect@.len() ==> rv(#[trigger] vect@[s]) == b[s],
    {
        result.push(vect[t]);
    }
    proof {
        let x = rvs(result@);
        assert(x.subrange(0, ul_len as int) =~= head);
        assert(b.subrange(0, ul_len as int) =~= rvs(old(vect)@.subrange(0, ul_len as int)));
        lemma_block_ident_solve(sym_view(rvs(mat@)), ul_len as int, b.len() as int, x, b);
    }
    result
}

// check whether a block-diagonal symmetric matrix S is positive semidefinite by running LDLT on S + eps * I
// (a successful check establishes B >= -eps I for every listed diagonal block B, with
// eps = 1e-9 * max(max_i |B_ii|, 1), because the factorization B + eps I = L D L^T has
// positive pivots; nothing is claimed when the check fails)
pub fn psd_block_check(mat: &SymMatrix, blocks: &Vec<(usize, usize)>) -> (result: bool)
    requires
        blocks_fit(blocks@, mat@.len()),
    ensures
        result ==> forall|b: int| 0 <= b < blocks@.len()
            ==> #[trigger] block_psd_ok(rvs(mat@), blocks@[b].0 as int, blocks@[b].1 as int),
{
    let ghost a = rvs(mat@);
    for b in 0..blocks.len()
        invariant
            a == rvs(mat@),
            blocks_fit(blocks@, mat@.len()),
            forall|t: int| 0 <= t < b ==> #[trigger] block_psd_ok(a, blocks@[t].0 as int, blocks@[t].1 as int),
    {
        let (start, dim) = blocks[b];
        proof {
            let n = start as int + dim as int;
            assert(n + 2 <= (n + 2) * (n + 2)) by (nonlinear_arith)
                requires n >= 0;
            lemma_dim_fits_monotone(dim as int, n);
        }
        let block = sym_matrix_diag_block(start, dim, mat);
        let eps = 1e-9 * max_diag_sym(&block, dim).max(1.0);
        // block + eps * I in packed form
        let mut block_add_scal: SymMatrix = Vec::new();
        let mut idx = 0;
        proof {
            lemma_pk_col_start_zero();
        }
        for c in 0..dim
            invariant
                dim_fits(dim as int),
                block@.len() == pk_len(dim as int),
                idx == pk_col_start(c as int),
                block_add_scal@.len() == idx,
                forall|r: int, c2: int| 0 <= r < c2 < c
                    ==> rv(#[trigger] block_add_scal@[pk_idx(r, c2)]) == rv(block@[pk_idx(r, c2)]),
                forall|c2: int| 0 <= c2 < c
                    ==> rv(#[trigger] block_add_scal@[pk_idx(c2, c2)]) == rv(block@[pk_idx(c2, c2)]) + rv(eps),
        {
            proof {
                lemma_dim_fits_idx(dim as int, c as int);
            }
            for _r in 0..c
                invariant
                    c < dim,
                    dim_fits(dim as int),
                    pk_col_start(c as int + 1) <= usize::MAX,
                    block@.len() == pk_len(dim as int),
                    idx == pk_col_start(c as int) + _r,
                    block_add_scal@.len() == idx,
                    forall|r2: int, c2: int| 0 <= r2 < c2 < c
                        ==> rv(#[trigger] block_add_scal@[pk_idx(r2, c2)]) == rv(block@[pk_idx(r2, c2)]),
                    forall|c2: int| 0 <= c2 < c
                        ==> rv(#[trigger] block_add_scal@[pk_idx(c2, c2)]) == rv(block@[pk_idx(c2, c2)]) + rv(eps),
                    forall|r2: int| 0 <= r2 < _r
                        ==> rv(#[trigger] block_add_scal@[pk_idx(r2, c as int)]) == rv(block@[pk_idx(r2, c as int)]),
            {
                proof {
                    lemma_pk_idx_bound(_r as int, c as int, dim as int);
                }
                block_add_scal.push(block[idx]);
                idx += 1;
                proof {
                    assert forall|r2: int, c2: int| 0 <= r2 < c2 < c
                        implies rv(#[trigger] block_add_scal@[pk_idx(r2, c2)]) == rv(block@[pk_idx(r2, c2)]) by {
                        lemma_pk_idx_bound(r2, c2, c as int);
                    }
                    assert forall|c2: int| 0 <= c2 < c
                        implies rv(#[trigger] block_add_scal@[pk_idx(c2, c2)]) == rv(block@[pk_idx(c2, c2)]) + rv(eps) by {
                        lemma_pk_idx_bound(c2, c2, c as int);
                    }
                }
            }
            proof {
                lemma_pk_idx_bound(c as int, c as int, dim as int);
            }
            block_add_scal.push(block[idx] + eps);
            idx += 1;
            proof {
                lemma_pk_col_start_step(c as int);
                assert forall|r2: int, c2: int| 0 <= r2 < c2 < c
                    implies rv(#[trigger] block_add_scal@[pk_idx(r2, c2)]) == rv(block@[pk_idx(r2, c2)]) by {
                    lemma_pk_idx_bound(r2, c2, c as int);
                }
                assert forall|c2: int| 0 <= c2 < c
                    implies rv(#[trigger] block_add_scal@[pk_idx(c2, c2)]) == rv(block@[pk_idx(c2, c2)]) + rv(eps) by {
                    lemma_pk_idx_bound(c2, c2, c as int);
                }
            }
        }
        let fact = ldlt_decomp(&block_add_scal, dim);
        if fact.is_none() {
            return false;
        }
        proof {
            let n = dim as int;
            let (l, d) = fact->Some_0;
            let (bv, sv, dv) = (rvs(block@), rvs(block_add_scal@), rvs(d@));
            let shifted = madd(sym_view(bv), mscale(rv(eps), mident()));
            // the stored shifted block is B + eps I, and B is the block of the full matrix
            assert forall|r: int, c: int| 0 <= r < n && 0 <= c < n
                implies #[trigger] sym_view(sv)(r, c) == shifted(r, c) by {
                let (lo, hi) = if r <= c { (r, c) } else { (c, r) };
                lemma_pk_idx_bound(lo, hi, n);
                assert(sym_entry(sv, r, c) == rv(block_add_scal@[pk_idx(lo, hi)]));
                assert(sym_entry(bv, r, c) == rv(block@[pk_idx(lo, hi)]));
                if r == c {
                    assert(shifted(r, c) == sym_entry(bv, r, c) + rv(eps) * 1.0real);
                } else {
                    assert(shifted(r, c) == sym_entry(bv, r, c) + rv(eps) * 0.0real);
                }
            }
            assert forall|r: int, c: int| 0 <= r < n && 0 <= c < n
                implies #[trigger] sym_view(bv)(r, c) == block_view(a, start as int)(r, c) by {
                let (lo, hi) = if r <= c { (r, c) } else { (c, r) };
                lemma_pk_idx_bound(lo, hi, n);
                lemma_pk_idx_bound(start + lo, start + hi, start as int + n);
                assert(sym_entry(bv, r, c) == rv(block@[pk_idx(lo, hi)]));
                assert(sym_entry(a, start + r, start + c) == rv(mat@[pk_idx(start + lo, start + hi)]));
            }
            // positive pivots: d_k >= TOL * max(.., 1) >= TOL > 0
            assert forall|k: int| 0 <= k < n implies #[trigger] dv[k] >= 0.0real by {
                let scale = max_r(max_abs_diag_r(sv, n), 1.0real);
                assert(rv(TOL) * scale >= rv(TOL)) by (nonlinear_arith)
                    requires scale >= 1.0real, rv(TOL) > 0.0real;
            }
            lemma_meq_trans(mmul(mmul(lt_view(rvs(l@), n), diag_view(dv), n), mtr(lt_view(rvs(l@), n)), n), sym_view(sv), shifted, n, n);
            lemma_shifted_factorization_psd(sym_view(bv), lt_view(rvs(l@), n), dv, n, rv(eps));
            lemma_psd_shifted_meq(sym_view(bv), block_view(a, start as int), n, rv(eps));
            if dim >= 1 {
                assert forall|i: int| 0 <= i < n
                    implies #[trigger] sym_diag(bv, i) == sym_entry(a, start + i, start + i) by {
                    lemma_pk_idx_bound(i, i, n);
                    lemma_pk_idx_bound(start + i, start + i, start as int + n);
                }
                lemma_block_max_abs_diag(bv, a, start as int, n);
            } else {
                lemma_psd_shifted_zero_dim(block_view(a, start as int), psd_check_eps_r(a, start as int, n));
            }
            assert(block_psd_ok(a, start as int, dim as int));
        }
    }
    true
}

} // verus!

/* some functions producing real-valued linear maps represented as partitioned vectors that act on n-dimensional symmetric matrices
   (a dim x dim symmetric matrix partitioned by the columns of its upper triangle: part_shape in sym_spec.rs) */

verus! {

// the packed symmetric matrix vect in partitioned form
pub fn mk_symmap_real(dim: usize, vect: &Vector) -> (result: Vec<Vector>)
    requires
        dim_fits(dim as int),
        vect@.len() >= pk_len(dim as int),
    ensures
        part_shape(result@, dim as int),
        forall|c: int, r: int| 0 <= r <= c < dim ==> rv(#[trigger] result@[c]@[r]) == rv(vect@[pk_idx(r, c)]),
{
    let mut result = Vec::with_capacity(dim);
    let mut offset = 0;
    proof {
        lemma_pk_col_start_zero();
    }
    for i in 0..dim
        invariant
            dim_fits(dim as int),
            vect@.len() >= pk_len(dim as int),
            offset == pk_col_start(i as int),
            part_shape(result@, i as int),
            forall|c: int, r: int| 0 <= r <= c < i ==> rv(#[trigger] result@[c]@[r]) == rv(vect@[pk_idx(r, c)]),
    {
        proof {
            lemma_dim_fits_idx(dim as int, i as int);
            lemma_pk_idx_bound(i as int, i as int, dim as int);
            lemma_pk_col_start_step(i as int);
        }
        result.push(slice_to_vec(&vect[offset..=offset + i]));
        offset += i + 1;
    }
    result
}

// the identity matrix in partitioned form (the trace functional)
pub fn sym_matrix_trace(dim: usize) -> (result: Vec<Vector>)
    ensures
        part_shape(result@, dim as int),
        forall|c: int, r: int| 0 <= r <= c < dim ==> rv(#[trigger] result@[c]@[r]) == (if r == c { 1.0real } else { 0.0real }),
{
    let mut result = Vec::with_capacity(dim);
    for i in 0..dim
        invariant
            part_shape(result@, i as int),
            forall|c: int, r: int| 0 <= r <= c < i ==> rv(#[trigger] result@[c]@[r]) == (if r == c { 1.0real } else { 0.0real }),
    {
        let mut v = vec![0.0; i + 1];
        v[i] = 1.0;
        result.push(v);
    }
    result
}

// scalar multiple of a partitioned matrix
pub fn sym_matrix_scale(s: f64, vect: &Vec<Vector>) -> (result: Vec<Vector>)
    ensures
        result@.len() == vect@.len(),
        forall|c: int| 0 <= c < vect@.len()
            ==> (#[trigger] result@[c])@.len() == vect@[c]@.len() && rvs(result@[c]@) == vscale_r(rv(s), rvs(vect@[c]@)),
{
    let mut result: Vec<Vector> = Vec::new();
    for c in 0..vect.len()
        invariant
            result@.len() == c,
            forall|k: int| 0 <= k < c
                ==> (#[trigger] result@[k])@.len() == vect@[k]@.len() && rvs(result@[k]@) == vscale_r(rv(s), rvs(vect@[k]@)),
    {
        result.push(scal_vect(s, &vect[c]));
    }
    result
}

// the zero matrix in partitioned form
pub fn sym_matrix_zero(dim: usize) -> (result: Vec<Vector>)
    ensures
        part_shape(result@, dim as int),
        forall|c: int, r: int| 0 <= r <= c < dim ==> rv(#[trigger] result@[c]@[r]) == 0.0real,
{
    let mut result = Vec::with_capacity(dim);
    for i in 0..dim
        invariant
            part_shape(result@, i as int),
            forall|c: int, r: int| 0 <= r <= c < i ==> rv(#[trigger] result@[c]@[r]) == 0.0real,
    {
        result.push(vec![0.0; i + 1])
    }
    result
}

// the matrix with a single 1 at (row, col), row <= col < dim, in partitioned form
// (aborts when row > col)
pub fn sym_matrix_proj(dim: usize, row: usize, col: usize) -> (result: Vec<Vector>)
    requires
        col < dim,
    ensures
        row <= col,
        part_shape(result@, dim as int),
        forall|c: int, r: int| 0 <= r <= c < dim
            ==> rv(#[trigger] result@[c]@[r]) == (if c == col && r == row { 1.0real } else { 0.0real }),
{
    if row > col {
        abort("sym_matrix_proj: make sure that row <= column");
    }
    let mut result = Vec::with_capacity(dim);
    for i in 0..col
        invariant
            part_shape(result@, i as int),
            forall|c: int, r: int| 0 <= r <= c < i ==> rv(#[trigger] result@[c]@[r]) == 0.0real,
    {
        result.push(vec![0.0; i + 1]);
    }
    let basis = standard_basis(row, col + 1);
    result.push(basis);
    proof {
        assert forall|c: int, r: int| 0 <= r <= c < col + 1
            implies rv(#[trigger] result@[c]@[r]) == (if c == col && r == row { 1.0real } else { 0.0real }) by {
            if c == col {
                assert(rv(result@[c]@[r]) == std_basis_r(row as int, (col + 1) as nat)[r]);
            }
        }
    }
    for i in col + 1..dim
        invariant
            row <= col < i,
            part_shape(result@, i as int),
            forall|c: int, r: int| 0 <= r <= c < i
                ==> rv(#[trigger] result@[c]@[r]) == (if c == col && r == row { 1.0real } else { 0.0real }),
    {
        result.push(vec![0.0; i + 1]);
    }
    result
}

// sum of two partitioned matrices
pub fn sym_matrix_add(vect1: &[Vector], vect2: &[Vector]) -> (result: Vec<Vector>)
    requires
        vect1@.len() == vect2@.len(),
        forall|c: int| 0 <= c < vect1@.len() ==> (#[trigger] vect1@[c])@.len() == vect2@[c]@.len(),
    ensures
        result@.len() == vect1@.len(),
        forall|c: int| 0 <= c < vect1@.len()
            ==> (#[trigger] result@[c])@.len() == vect1@[c]@.len() && rvs(result@[c]@) == vadd_r(rvs(vect1@[c]@), rvs(vect2@[c]@)),
{
    let mut result: Vec<Vector> = Vec::new();
    for c in 0..vect1.len()
        invariant
            vect1@.len() == vect2@.len(),
            forall|k: int| 0 <= k < vect1@.len() ==> (#[trigger] vect1@[k])@.len() == vect2@[k]@.len(),
            result@.len() == c,
            forall|k: int| 0 <= k < c
                ==> (#[trigger] result@[k])@.len() == vect1@[k]@.len() && rvs(result@[k]@) == vadd_r(rvs(vect1@[k]@), rvs(vect2@[k]@)),
    {
        result.push(vect_add(&vect1[c], &vect2[c]));
    }
    result
}

} // verus!

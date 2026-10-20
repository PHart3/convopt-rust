// various basic matrix operations
//
// The executable functions below are verified against exact-real specifications
// (see src/lin_alg/vec_spec.rs for the spec vocabulary and src/trusted.rs for the
// exact-real abstraction of f64).

use vstd::prelude::*;
pub use crate::lin_alg::comparison::float_equality;
use crate::lin_alg::comparison::scaled_near_zero;
#[cfg(verus_keep_ghost)]
use crate::trusted::rv;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::comparison::approx_equal;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::real_spec::{abs_r, lemma_abs_quotient_le_one};
#[cfg(verus_keep_ghost)]
use crate::lin_alg::solve_spec::{abs_max_prefix_r, abs_max_r, lemma_abs_max_prefix_step, lemma_abs_max_attained};
#[cfg(verus_keep_ghost)]
use vstd::arithmetic::div_mod::{lemma_fundamental_div_mod, lemma_mod_bound};
#[cfg(verus_keep_ghost)]
use crate::lin_alg::gauss_spec::*;

// column vector
pub type Vector = Vec<f64>;
// column-major matrix stored as a flat vector
pub type Matrix = Vector;
// column-major lower triangular matrix stored as a flat vector
pub type LowTriMatrix = Vector;

verus! {

broadcast use crate::trusted::group_real_model;

// default tolerance level
pub const TOL: f64 = 1e-10;

// standard basis vectors
pub fn standard_basis(i: usize, dim: usize) -> (result: Vector)
    ensures
        result@.len() == dim,
        rvs(result@) == std_basis_r(i as int, dim as nat),
{
    let mut result: Vector = Vec::with_capacity(dim);
    for j in 0..dim
        invariant
            result@.len() == j,
            forall|k: int| 0 <= k < j ==> rv(#[trigger] result@[k]) == (if k == i { 1.0real } else { 0.0real }),
    {
        if j == i {
            result.push(1.0);
        } else {
            result.push(0.0);
        }
    }
    proof {
        assert(rvs(result@) =~= std_basis_r(i as int, dim as nat));
    }
    result
}

// identity matrix
pub fn ident_mat(dim: usize) -> (result: Matrix)
    requires
        dim * dim <= usize::MAX,
    ensures
        result@.len() == dim * dim,
        is_ident_r(rvs(result@), dim as int),
{
    let mut result: Matrix = Vec::with_capacity(dim * dim);
    proof {
        assert(0 * dim == 0) by (nonlinear_arith);
    }
    for i in 0..dim
        invariant
            result@.len() == i * dim,
            forall|r: int, c: int| 0 <= r < dim && 0 <= c < i
                ==> #[trigger] cm_entry(rvs(result@), dim as int, r, c) == (if r == c { 1.0real } else { 0.0real }),
    {
        let ghost prev = result@;
        let mut col = standard_basis(i, dim);
        let ghost col_view = col@;
        result.append(&mut col);
        proof {
            assert((i + 1) * dim == i * dim + dim) by (nonlinear_arith);
            assert forall|r: int, c: int| 0 <= r < dim && 0 <= c < i + 1
                implies #[trigger] cm_entry(rvs(result@), dim as int, r, c) == (if r == c { 1.0real } else { 0.0real }) by {
                let idx = cm_idx(dim as int, r, c);
                if c < i {
                    // entry already placed by an earlier column; append leaves it unchanged
                    assert(0 <= c * dim + r < i * dim) by (nonlinear_arith)
                        requires 0 <= r < dim, 0 <= c < i;
                    assert(result@[idx] == prev[idx]);
                    assert(cm_entry(rvs(prev), dim as int, r, c) == (if r == c { 1.0real } else { 0.0real }));
                } else {
                    // entry (r, i) is entry r of the appended basis vector e_i
                    assert(idx == i * dim + r);
                    assert(result@[idx] == col_view[r]);
                    assert(rvs(col_view)[r] == std_basis_r(i as int, dim as nat)[r]);
                }
            }
        }
    }
    result
}

// zero matrix
pub fn zero_mat(dim_c: usize, dim_r: usize) -> (result: Matrix)
    requires
        dim_c * dim_r <= usize::MAX,
    ensures
        result@.len() == dim_c * dim_r,
        rvs(result@) == const_r((dim_c * dim_r) as nat, 0.0real),
{
    let result = vec![0.0; dim_c * dim_r];
    proof {
        assert(rvs(result@) =~= const_r((dim_c * dim_r) as nat, 0.0real));
    }
    result
}

// standard inner product
#[inline]
pub fn dot_prod(vect1: &[f64], vect2: &[f64]) -> (prod: f64)
    requires
        vect1@.len() == vect2@.len(),
    ensures
        rv(prod) == dot_r(rvs(vect1@), rvs(vect2@)),
{
    let mut prod = 0.0;
    let n = vect1.len();
    for i in 0..n
        invariant
            n == vect1@.len() == vect2@.len(),
            rv(prod) == sum_r(vmul_r(rvs(vect1@), rvs(vect2@)).take(i as int)),
    {
        prod = vect1[i].mul_add(vect2[i], prod);
        proof {
            lemma_sum_r_take_succ(vmul_r(rvs(vect1@), rvs(vect2@)), i as int);
        }
    }
    proof {
        lemma_take_all(vmul_r(rvs(vect1@), rvs(vect2@)));
    }
    prod
}

// Euclidean norm
#[inline]
pub fn euclid_norm(vect: &[f64]) -> (norm: f64)
    ensures
        rv(norm) == norm_r(rvs(vect@)),
{
    proof {
        lemma_dot_self_nonneg(rvs(vect@));
    }
    dot_prod(vect, vect).sqrt()
}

// Euclidean distance
pub fn euclid_distance(vect1: &[f64], vect2: &[f64]) -> (dist: f64)
    requires
        vect1@.len() == vect2@.len(),
    ensures
        rv(dist) == dist_r(rvs(vect1@), rvs(vect2@)),
{
    let ghost diff_r = vsub_r(rvs(vect1@), rvs(vect2@));
    let mut sum = 0.0;
    let mut diff;
    let n = vect1.len();
    for i in 0..n
        invariant
            n == vect1@.len() == vect2@.len(),
            diff_r == vsub_r(rvs(vect1@), rvs(vect2@)),
            rv(sum) == sum_r(vmul_r(diff_r, diff_r).take(i as int)),
    {
        diff = vect1[i] - vect2[i];
        sum = diff.mul_add(diff, sum);
        proof {
            lemma_sum_r_take_succ(vmul_r(diff_r, diff_r), i as int);
        }
    }
    proof {
        lemma_take_all(vmul_r(diff_r, diff_r));
        lemma_dot_self_nonneg(diff_r);
    }
    sum.sqrt()
}

// vector addition
pub fn vect_add(vect1: &Vector, vect2: &Vector) -> (sum: Vector)
    requires
        vect1@.len() == vect2@.len(),
    ensures
        sum@.len() == vect1@.len(),
        rvs(sum@) == vadd_r(rvs(vect1@), rvs(vect2@)),
{
    let mut sum: Vector = Vec::new();
    let n = vect1.len();
    for i in 0..n
        invariant
            n == vect1@.len() == vect2@.len(),
            sum@.len() == i,
            forall|k: int| 0 <= k < i ==> rv(#[trigger] sum@[k]) == rv(vect1@[k]) + rv(vect2@[k]),
    {
        sum.push(vect1[i] + vect2[i]);
    }
    proof {
        assert(rvs(sum@) =~= vadd_r(rvs(vect1@), rvs(vect2@)));
    }
    sum
}

// vector subtraction
pub fn vect_subt(vect1: &Vector, vect2: &Vector) -> (diff: Vector)
    requires
        vect1@.len() == vect2@.len(),
    ensures
        diff@.len() == vect1@.len(),
        rvs(diff@) == vsub_r(rvs(vect1@), rvs(vect2@)),
{
    let mut diff: Vector = Vec::new();
    let n = vect1.len();
    for i in 0..n
        invariant
            n == vect1@.len() == vect2@.len(),
            diff@.len() == i,
            forall|k: int| 0 <= k < i ==> rv(#[trigger] diff@[k]) == rv(vect1@[k]) - rv(vect2@[k]),
    {
        diff.push(vect1[i] - vect2[i]);
    }
    proof {
        assert(rvs(diff@) =~= vsub_r(rvs(vect1@), rvs(vect2@)));
    }
    diff
}

// scalar action on vector
pub fn scal_vect(s: f64, vect: &Vector) -> (result: Vector)
    ensures
        result@.len() == vect@.len(),
        rvs(result@) == vscale_r(rv(s), rvs(vect@)),
{
    let mut result: Vector = Vec::new();
    let n = vect.len();
    for i in 0..n
        invariant
            n == vect@.len(),
            result@.len() == i,
            forall|k: int| 0 <= k < i ==> rv(#[trigger] result@[k]) == rv(s) * rv(vect@[k]),
    {
        result.push(s * vect[i]);
    }
    proof {
        assert(rvs(result@) =~= vscale_r(rv(s), rvs(vect@)));
    }
    result
}

// inverse scalar action on vector
pub fn negscal_vect(s: f64, vect: &Vector) -> (result: Vector)
    ensures
        result@.len() == vect@.len(),
        rvs(result@) == vscale_r(-rv(s), rvs(vect@)),
{
    let mut result: Vector = Vec::new();
    let n = vect.len();
    for i in 0..n
        invariant
            n == vect@.len(),
            result@.len() == i,
            forall|k: int| 0 <= k < i ==> rv(#[trigger] result@[k]) == -rv(s) * rv(vect@[k]),
    {
        result.push(crate::trusted::neg(s * vect[i]));
        proof {
            assert(-rv(s) * rv(vect@[i as int]) == -(rv(s) * rv(vect@[i as int]))) by (nonlinear_arith);
        }
    }
    proof {
        assert(rvs(result@) =~= vscale_r(-rv(s), rvs(vect@)));
    }
    result
}

// computing vector of form vect1 + s * vect2
pub fn vect_add_scaled(vect1: &Vector, s: f64, vect2: &Vector) -> (result: Vector)
    requires
        vect1@.len() == vect2@.len(),
    ensures
        result@.len() == vect1@.len(),
        rvs(result@) == vadd_r(rvs(vect1@), vscale_r(rv(s), rvs(vect2@))),
{
    let mut result: Vector = Vec::new();
    let n = vect1.len();
    for i in 0..n
        invariant
            n == vect1@.len() == vect2@.len(),
            result@.len() == i,
            forall|k: int| 0 <= k < i ==> rv(#[trigger] result@[k]) == rv(vect1@[k]) + rv(s) * rv(vect2@[k]),
    {
        result.push(s.mul_add(vect2[i], vect1[i]));
    }
    proof {
        assert(rvs(result@) =~= vadd_r(rvs(vect1@), vscale_r(rv(s), rvs(vect2@))));
    }
    result
}

// evaluates the matrix expression A + (B - C)
pub fn matrix_add_diff(mat1: &Matrix, mat2: &Matrix, mat3: &Matrix) -> (sum: Matrix)
    requires
        mat1@.len() == mat2@.len() == mat3@.len(),
    ensures
        sum@.len() == mat1@.len(),
        rvs(sum@) == vadd_r(rvs(mat1@), vsub_r(rvs(mat2@), rvs(mat3@))),
{
    let mut sum: Matrix = Vec::with_capacity(mat1.len());
    let n = mat1.len();
    for i in 0..n
        invariant
            n == mat1@.len() == mat2@.len() == mat3@.len(),
            sum@.len() == i,
            forall|k: int| 0 <= k < i ==> rv(#[trigger] sum@[k]) == rv(mat1@[k]) + (rv(mat2@[k]) - rv(mat3@[k])),
    {
        sum.push(mat1[i] + (mat2[i] - mat3[i]));
    }
    proof {
        assert(rvs(sum@) =~= vadd_r(rvs(mat1@), vsub_r(rvs(mat2@), rvs(mat3@))));
    }
    sum
}

// lower triangular identity matrix
pub fn ident_mat_low_tri(dim: usize) -> (result: LowTriMatrix)
    ensures
        result@.len() == dim * (dim + 1) / 2,
        is_lt_ident_r(rvs(result@), dim as int),
{
    let mut result: LowTriMatrix = Vec::new();
    proof {
        lemma_lt_col_start_zero(dim as int);
    }
    for i in 0..dim
        invariant
            result@.len() == lt_col_start(dim as int, i as int),
            forall|r: int, c: int| 0 <= c <= r < dim && c < i
                ==> #[trigger] lt_entry(rvs(result@), dim as int, r, c) == (if r == c { 1.0real } else { 0.0real }),
    {
        let ghost prev = result@;
        let ghost start = lt_col_start(dim as int, i as int);
        // column i: a 1 on the diagonal followed by dim - i - 1 zeros
        result.push(1.0);
        for _k in 1..dim - i
            invariant
                result@.len() == start + _k,
                forall|j: int| 0 <= j < start ==> #[trigger] result@[j] == prev[j],
                rv(result@[start]) == 1.0real,
                forall|j: int| start < j < start + _k ==> rv(#[trigger] result@[j]) == 0.0real,
        {
            result.push(0.0);
        }
        proof {
            lemma_lt_col_start_step(dim as int, i as int);
            assert forall|r: int, c: int| 0 <= c <= r < dim && c < i + 1
                implies #[trigger] lt_entry(rvs(result@), dim as int, r, c) == (if r == c { 1.0real } else { 0.0real }) by {
                if c < i {
                    lemma_lt_idx_bound(dim as int, r, c, i as int);
                    assert(lt_entry(rvs(prev), dim as int, r, c) == (if r == c { 1.0real } else { 0.0real }));
                } else {
                    assert(lt_idx(dim as int, r, c) == start + (r - i));
                }
            }
        }
    }
    proof {
        lemma_lt_col_start_end(dim as int);
    }
    result
}

// testing if two vectors are the same
pub fn vector_equality(vect1: &Vector, vect2: &Vector, tol: f64) -> (equal: bool)
    ensures
        equal == (vect1@.len() == vect2@.len()
            && forall|k: int| 0 <= k < vect1@.len() ==> approx_equal(rv(#[trigger] vect1@[k]), rv(vect2@[k]), rv(tol))),
{
    if vect1.len() != vect2.len() {
        return false;
    } else {
        let n = vect1.len();
        for i in 0..n
            invariant
                n == vect1@.len() == vect2@.len(),
                forall|k: int| 0 <= k < i ==> approx_equal(rv(#[trigger] vect1@[k]), rv(vect2@[k]), rv(tol)),
        {
            if !(float_equality(vect1[i], vect2[i], tol)) {
                return false;
            }
        }
    }
    true
}

} // verus!

verus! {

// Gaussian elimination via partial pivoting
// also computes rank and consistency of given augmented matrix
// (the row operations preserve the solution set of [A | b] and keep zero columns zero; the result is in echelon form
// with pivot columns strictly increasing, pivots of magnitude at least TOL * max |entry|
// of the input and exact zeros below every pivot; the rank is the number of pivots and the
// consistency flag says that no pivot lies in the last column)
pub fn gauss_elim(mat: &mut Matrix, col_dim: usize, row_dim: usize) -> (res: (&mut Matrix, usize, bool))
    requires
        col_dim >= 1,
        old(mat)@.len() == row_dim * col_dim,
    ensures
        res.0@.len() == old(mat)@.len(),
        final(mat)@ == final(res.0)@,
        same_solutions(rvs(old(mat)@), rvs(res.0@), row_dim as int, col_dim as int),
        zero_cols_kept(rvs(old(mat)@), rvs(res.0@), row_dim as int, col_dim as int),
        exists|pivots: Seq<int>| {
            &&& #[trigger] echelon_form(rvs(res.0@), row_dim as int, col_dim as int, pivots, rv(TOL) * abs_max_r(rvs(old(mat)@)))
            &&& skipped_bounded(rvs(res.0@), row_dim as int, col_dim as int, pivots, rv(TOL) * abs_max_r(rvs(old(mat)@)), col_dim as int)
            &&& pivots.len() == res.1
            &&& res.2 == (pivots.len() == 0 || pivots.last() < col_dim - 1)
        },
        abs_max_r(rvs(old(mat)@)) > 0.0real ==> res.1 >= 1,
{
    let ghost a0 = rvs(old(mat)@);
    let ghost (rd, cd) = (row_dim as int, col_dim as int);
    let mut row_search = 0;
    let mut coeff_rank = 0;
    // scale = max |entry| of the input
    let mut scale: f64 = 0.0;
    for t in 0..mat.len()
        invariant
            mat@ == old(mat)@,
            rv(scale) == abs_max_prefix_r(rvs(mat@), t as nat),
    {
        proof {
            lemma_abs_max_prefix_step(rvs(mat@), t as nat);
        }
        scale = scale.max(mat[t].abs());
    }
    let ghost eps = rv(TOL) * abs_max_r(a0);
    let ghost mut pivots: Seq<int> = Seq::empty();
    proof {
        lemma_same_solutions_refl(a0, rd, cd);
        lemma_zero_cols_refl(a0, rd, cd);
    }
    let mut n = 0;
    while n < col_dim && row_search < row_dim
        invariant
            col_dim >= 1,
            n <= col_dim,
            row_search <= row_dim,
            rd == row_dim as int,
            cd == col_dim as int,
            a0 == rvs(old(mat)@),
            mat@.len() == old(mat)@.len(),
            old(mat)@.len() == row_dim * col_dim,
            rv(scale) == abs_max_r(a0),
            eps == rv(TOL) * abs_max_r(a0),
            pivots.len() == row_search,
            row_search == 0 ==> mat@ == old(mat)@,
            forall|i: int| 0 <= i < pivots.len() ==> #[trigger] pivots[i] < n,
            echelon_form(rvs(mat@), rd, cd, pivots, eps),
            skipped_bounded(rvs(mat@), rd, cd, pivots, eps, n as int),
            same_solutions(a0, rvs(mat@), rd, cd),
            zero_cols_kept(a0, rvs(mat@), rd, cd),
            coeff_rank as int == row_search - (if row_search > 0 && pivots.last() == cd - 1 { 1int } else { 0int }),
        decreases col_dim - n,
    {
        proof {
            assert(mat.len() == row_dim * col_dim);
            assert(n * row_dim + row_dim <= row_dim * col_dim) by (nonlinear_arith)
                requires n < col_dim;
        }
        // partial pivoting: a remaining entry of column n of largest magnitude
        let mut pivot_row = row_search;
        for j in row_search + 1..row_dim
            invariant
                row_search <= pivot_row < row_dim,
                row_search < row_dim,
                n < col_dim,
                rd == row_dim as int,
                mat@.len() == row_dim * col_dim,
                row_dim * col_dim <= usize::MAX,
                n * row_dim + row_dim <= row_dim * col_dim,
                forall|i: int| row_search <= i < j
                    ==> abs_r(#[trigger] cm_entry(rvs(mat@), rd, i, n as int)) <= abs_r(cm_entry(rvs(mat@), rd, pivot_row as int, n as int)),
        {
            if mat[n * row_dim + j].abs() > mat[n * row_dim + pivot_row].abs() {
                pivot_row = j;
            }
        }
        let ghost cur0 = rvs(mat@);
        proof {
            // pivot-selection correctness: pivot_row maximizes |A_in| over the remaining rows
            assert(forall|i: int| row_search <= i < rd
                ==> abs_r(#[trigger] cm_entry(cur0, rd, i, n as int)) <= abs_r(cm_entry(cur0, rd, pivot_row as int, n as int)));
        }
        if !scaled_near_zero(mat[n * row_dim + pivot_row], scale, TOL) {
            if pivot_row != row_search {
                let ghost before = rvs(mat@);
                for m3 in 0..col_dim
                    invariant
                        row_search < pivot_row < row_dim,
                        n < col_dim,
                        rd == row_dim as int,
                        cd == col_dim as int,
                        mat@.len() == row_dim * col_dim,
                    row_dim * col_dim <= usize::MAX,
                        row_dim * col_dim <= usize::MAX,
                row_dim * col_dim <= usize::MAX,
                        before.len() == mat@.len(),
                        forall|i: int, c: int| 0 <= i < rd && 0 <= c < cd
                            ==> #[trigger] cm_entry(rvs(mat@), rd, i, c)
                                == cm_entry(before, rd, (if c < m3 { if i == row_search { pivot_row as int } else if i == pivot_row { row_search as int } else { i } } else { i }), c),
                {
                    proof {
                        assert(mat.len() == row_dim * col_dim);
                        assert(m3 * row_dim + row_dim <= row_dim * col_dim) by (nonlinear_arith)
                            requires m3 < col_dim;
                        lemma_cm_idx_injective(rd, row_search as int, m3 as int, pivot_row as int, m3 as int);
                    }
                    let (i1, i2) = (m3 * row_dim + row_search, m3 * row_dim + pivot_row);
                    let tmp = mat[i1];
                    let ghost p0 = rvs(mat@);
                    mat[i1] = mat[i2];
                    let ghost p1 = rvs(mat@);
                    mat[i2] = tmp;
                    proof {
                        let cur = rvs(mat@);
                        let (rs, pr, m3i) = (row_search as int, pivot_row as int, m3 as int);
                        assert(i1 == cm_idx(rd, rs, m3i));
                        assert(i2 == cm_idx(rd, pr, m3i));
                        assert forall|i: int, c: int| 0 <= i < rd && 0 <= c < cd
                            implies #[trigger] cm_entry(cur, rd, i, c)
                                == cm_entry(before, rd, (if c < m3i + 1 { if i == rs { pr } else if i == pr { rs } else { i } } else { i }), c) by {
                            if c == m3i && (i == rs || i == pr) {
                                assert(cm_entry(p0, rd, pr, m3i) == cm_entry(before, rd, pr, m3i));
                                assert(cm_entry(p0, rd, rs, m3i) == cm_entry(before, rd, rs, m3i));
                                assert(cm_entry(cur, rd, rs, m3i) == cm_entry(p1, rd, rs, m3i)) by {
                                    lemma_cm_entry_write_other(p1, cur, rd, cd, pr, m3i, rs, m3i);
                                }
                            } else {
                                lemma_cm_entry_write_other(p0, p1, rd, cd, rs, m3i, i, c);
                                lemma_cm_entry_write_other(p1, cur, rd, cd, pr, m3i, i, c);
                            }
                        }
                    }
                }
                proof {
                    let cur = rvs(mat@);
                    assert(rows_swapped(before, cur, rd, cd, row_search as int, pivot_row as int));
                    lemma_swap_same_solutions(before, cur, rd, cd, row_search as int, pivot_row as int);
                    lemma_same_solutions_trans(a0, before, cur, rd, cd);
                    lemma_swap_zero_cols(before, cur, rd, cd, row_search as int, pivot_row as int);
                    lemma_zero_cols_trans(a0, before, cur, rd, cd);
                    lemma_echelon_swap(before, cur, rd, cd, pivots, eps, row_search as int, pivot_row as int);
                    lemma_skipped_swap(before, cur, rd, cd, pivots, eps, n as int, row_search as int, pivot_row as int);
                }
            }
            // row row_search now holds the pivot: of maximal magnitude among the remaining rows and >= TOL * scale
            let ghost swapped = rvs(mat@);
            proof {
                let (rs, pr, ni) = (row_search as int, pivot_row as int, n as int);
                assert(cm_entry(swapped, rd, rs, ni) == cm_entry(cur0, rd, pr, ni));
                assert forall|i: int| rs <= i < rd
                    implies abs_r(#[trigger] cm_entry(swapped, rd, i, ni)) <= abs_r(cm_entry(swapped, rd, rs, ni)) by {
                    if pr != rs {
                        let src = if i == rs { pr } else if i == pr { rs } else { i };
                        assert(cm_entry(swapped, rd, i, ni) == cm_entry(cur0, rd, src, ni));
                    }
                }
                assert(abs_r(cm_entry(swapped, rd, rs, ni)) >= eps);
            }
            let denom = mat[n * row_dim + row_search];
            for j in row_search + 1..row_dim
                invariant
                    row_search < row_dim,
                    n < col_dim,
                    col_dim >= 1,
                    rd == row_dim as int,
                    cd == col_dim as int,
                    mat@.len() == row_dim * col_dim,
                    row_dim * col_dim <= usize::MAX,
                row_dim * col_dim <= usize::MAX,
                    swapped.len() == mat@.len(),
                    n * row_dim + row_dim <= row_dim * col_dim,
                    rv(denom) == cm_entry(swapped, rd, row_search as int, n as int),
                    abs_r(cm_entry(swapped, rd, row_search as int, n as int)) >= eps,
                    pivots.len() == row_search,
                    forall|i: int, c: int| 0 <= i < rd && 0 <= c < cd && (i == row_search || i >= j)
                        ==> #[trigger] cm_entry(rvs(mat@), rd, i, c) == cm_entry(swapped, rd, i, c),
                    forall|i: int| row_search < i < j ==> #[trigger] cm_entry(rvs(mat@), rd, i, n as int) == 0.0real,
                    forall|i: int| row_search <= i < rd
                        ==> abs_r(#[trigger] cm_entry(swapped, rd, i, n as int)) <= abs_r(cm_entry(swapped, rd, row_search as int, n as int)),
                    echelon_form(rvs(mat@), rd, cd, pivots, eps),
                    same_solutions(a0, rvs(mat@), rd, cd),
                    zero_cols_kept(a0, rvs(mat@), rd, cd),
                    forall|i: int| 0 <= i < pivots.len() ==> #[trigger] pivots[i] < n,
                    forall|c: int, i: int| 0 <= c < n && !is_pivot_col(pivots, c) && pivots_below(pivots, c) <= i < rd
                        ==> abs_r(#[trigger] cm_entry(rvs(mat@), rd, i, c))
                            < skip_bound(pivots, c, eps) + (if row_search < i < j { skip_bound(pivots, c, eps) } else { 0.0real }),
            {
                let piv = mat[n * row_dim + j] / denom;
                let ghost before = rvs(mat@);
                let ghost (rs, ji, ni) = (row_search as int, j as int, n as int);
                for m in 0..col_dim
                    invariant
                        row_search < j < row_dim,
                        n < col_dim,
                        rd == row_dim as int,
                        cd == col_dim as int,
                        mat@.len() == row_dim * col_dim,
                    row_dim * col_dim <= usize::MAX,
                        row_dim * col_dim <= usize::MAX,
                row_dim * col_dim <= usize::MAX,
                        before.len() == mat@.len(),
                        rs == row_search as int,
                        ji == j as int,
                        forall|i: int, c: int| 0 <= i < rd && 0 <= c < cd
                            ==> #[trigger] cm_entry(rvs(mat@), rd, i, c)
                                == (if i == ji && c < m { cm_entry(before, rd, ji, c) - rv(piv) * cm_entry(before, rd, rs, c) } else { cm_entry(before, rd, i, c) }),
                {
                    proof {
                        assert(mat.len() == row_dim * col_dim);
                        assert(m * row_dim + row_dim <= row_dim * col_dim) by (nonlinear_arith)
                            requires m < col_dim;
                    }
                    let ghost p0 = rvs(mat@);
                    mat[m * row_dim + j] = mat[m * row_dim + j] - piv * mat[m * row_dim + row_search];
                    proof {
                        let cur = rvs(mat@);
                        let mi = m as int;
                        assert(m * row_dim + j == cm_idx(rd, ji, mi));
                        assert(m * row_dim + row_search == cm_idx(rd, rs, mi));
                        assert(cm_entry(p0, rd, ji, mi) == cm_entry(before, rd, ji, mi));
                        assert(cm_entry(p0, rd, rs, mi) == cm_entry(before, rd, rs, mi));
                        assert(cm_entry(cur, rd, ji, mi) == cm_entry(before, rd, ji, mi) - rv(piv) * cm_entry(before, rd, rs, mi));
                        assert forall|i: int, c: int| 0 <= i < rd && 0 <= c < cd
                            implies #[trigger] cm_entry(cur, rd, i, c)
                                == (if i == ji && c < mi + 1 { cm_entry(before, rd, ji, c) - rv(piv) * cm_entry(before, rd, rs, c) } else { cm_entry(before, rd, i, c) }) by {
                            if !(i == ji && c == mi) {
                                lemma_cm_entry_write_other(p0, cur, rd, cd, ji, mi, i, c);
                            }
                        }
                    }
                }
                proof {
                    let cur = rvs(mat@);
                    assert(row_subtracted(before, cur, rd, cd, ji, rs, rv(piv)));
                    lemma_row_sub_same_solutions(before, cur, rd, cd, ji, rs, rv(piv));
                    lemma_same_solutions_trans(a0, before, cur, rd, cd);
                    lemma_row_sub_zero_cols(before, cur, rd, cd, ji, rs, rv(piv));
                    lemma_zero_cols_trans(a0, before, cur, rd, cd);
                    lemma_echelon_row_sub(before, cur, rd, cd, pivots, eps, ji, rs, rv(piv));
                    // the entry (j, n) is annihilated: A_jn - (A_jn / A_rn) A_rn = 0
                    let (num, dn) = (cm_entry(before, rd, ji, ni), rv(denom));
                    assert(num == cm_entry(swapped, rd, ji, ni));
                    assert(cm_entry(cur, rd, ji, ni) == num - rv(piv) * dn);
                    if dn != 0.0real {
                        assert((num / dn) * dn == num) by (nonlinear_arith)
                            requires dn != 0.0real;
                    } else {
                        // a zero pivot is the maximum only if the whole remaining column is zero
                        assert(abs_r(num) <= abs_r(dn));
                        assert(rv(piv) * 0.0real == 0.0real);
                    }
                    // skipped columns: row j at most doubles its bound (|piv| <= 1 by partial pivoting)
                    assert forall|c: int, i: int| 0 <= c < n && !is_pivot_col(pivots, c) && pivots_below(pivots, c) <= i < rd
                        implies abs_r(#[trigger] cm_entry(cur, rd, i, c))
                            < skip_bound(pivots, c, eps) + (if rs < i < ji + 1 { skip_bound(pivots, c, eps) } else { 0.0real }) by {
                        let bound = skip_bound(pivots, c, eps);
                        lemma_pivots_below_bounds(pivots, c);
                        if i == ji {
                            assert(abs_r(cm_entry(before, rd, ji, c)) < bound);
                            assert(abs_r(cm_entry(before, rd, rs, c)) < bound);
                            let k = (pivots.len() - pivots_below(pivots, c)) as nat;
                            lemma_pow2_pos(k);
                            assert(eps > 0.0real) by (nonlinear_arith)
                                requires pow2_r(k) > 0.0real, pow2_r(k) * eps > 0.0real;
                            assert(abs_r(dn) >= eps);
                            lemma_abs_quotient_le_one(num, dn);
                            lemma_abs_sub_double(cm_entry(before, rd, ji, c), cm_entry(before, rd, rs, c), rv(piv), bound);
                        } else {
                            assert(cm_entry(cur, rd, i, c) == cm_entry(before, rd, i, c));
                            assert(abs_r(cm_entry(before, rd, i, c))
                                < bound + (if rs < i < ji { bound } else { 0.0real }));
                        }
                    }
                }
            }
            proof {
                let cur = rvs(mat@);
                assert forall|c: int, i: int| 0 <= c < n && !is_pivot_col(pivots, c) && pivots_below(pivots, c) <= i < rd
                    implies abs_r(#[trigger] cm_entry(cur, rd, i, c)) < skip_bound(pivots, c, eps) + skip_bound(pivots, c, eps) by {
                    assert(abs_r(cm_entry(cur, rd, i, c))
                        < skip_bound(pivots, c, eps) + (if row_search < i < rd { skip_bound(pivots, c, eps) } else { 0.0real }));
                }
                lemma_skipped_after_pivot(cur, rd, cd, pivots, eps, n as int);
                lemma_echelon_push(cur, rd, cd, pivots, eps, n as int);
                pivots = pivots.push(n as int);
            }
            row_search += 1;
            if n < col_dim - 1 {
                coeff_rank += 1;
            }
        } else {
            proof {
                // the tolerance-based skip rule: every remaining entry of column n is below TOL * scale
                assert(forall|i: int| row_search <= i < rd ==> abs_r(#[trigger] cm_entry(cur0, rd, i, n as int)) < eps);
                lemma_skipped_skip(cur0, rd, cd, pivots, eps, n as int);
            }
        }
        n += 1;
    }
    proof {
        if n < col_dim {
            lemma_skipped_extend(rvs(mat@), rd, cd, pivots, eps, n as int);
        }
        // a nonzero input has a pivot: without one the matrix is unchanged and every column was
        // skipped, so every entry would be below TOL * max |entry|
        if abs_max_r(a0) > 0.0real && row_search == 0 {
            lemma_abs_max_attained(a0);
            let t = choose|t: int| 0 <= t < a0.len() && abs_r(#[trigger] a0[t]) == abs_max_r(a0);
            assert(rd > 0) by (nonlinear_arith)
                requires a0.len() == rd * cd, a0.len() > 0, rd >= 0;
            assert(n == col_dim);
            lemma_fundamental_div_mod(t, rd);
            lemma_mod_bound(t, rd);
            let (j, c) = (t % rd, t / rd);
            assert(c < cd) by (nonlinear_arith)
                requires t < rd * cd, t == rd * c + j, j >= 0, rd > 0;
            assert(cm_entry(a0, rd, j, c) == a0[t]);
            assert(!is_pivot_col(pivots, c));
            assert(pivots_below(pivots, c) == 0);
            assert(skip_bound(pivots, c, eps) == pow2_r(0) * eps);
            assert(abs_r(a0[t]) < eps);
            assert(rv(TOL) == 0.0000000001real);
            assert(false) by (nonlinear_arith)
                requires abs_r(a0[t]) > 0.0real, abs_r(a0[t]) < 0.0000000001real * abs_r(a0[t]);
        }
    }
    let res = (mat, row_search, row_search == coeff_rank);
    proof {
        // the pivot sequence witnesses the echelon form (the last pivot column is at most cd - 1,
        // so the consistency flag says that it is not the last column)
        assert(echelon_form(rvs(res.0@), row_dim as int, col_dim as int, pivots, rv(TOL) * abs_max_r(rvs(old(mat)@))));
        assert(skipped_bounded(rvs(res.0@), row_dim as int, col_dim as int, pivots, rv(TOL) * abs_max_r(rvs(old(mat)@)), col_dim as int));
        assert(res.2 == (pivots.len() == 0 || pivots.last() < col_dim - 1));
    }
    res
}

} // verus!

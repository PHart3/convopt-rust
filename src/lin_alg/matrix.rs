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

// Gaussian elimination via partial pivoting
// also computes rank and consistency of given augmented matrix
pub fn gauss_elim(mat: &mut Matrix, col_dim: usize, row_dim: usize) -> (&mut Matrix, usize, bool) {
    let (mut row_search, mut coeff_rank, mut pivot_row);
    (row_search, coeff_rank) = (0, 0);
    let scale = mat.iter().map(|x| x.abs()).fold(0.0, f64::max);
    for n in 0..col_dim {
        if row_search >= row_dim {
            break;
        }
        pivot_row = row_search;
        for j in row_search + 1..row_dim {
            if mat[n * row_dim + j].abs() > mat[n * row_dim + pivot_row].abs() {
                pivot_row = j;
            }
        }
        if scaled_near_zero(mat[n * row_dim + pivot_row], scale, TOL) {
            continue;
        }
        if pivot_row != row_search {
            for m3 in 0..col_dim {
                mat.swap(m3 * row_dim + row_search, m3 * row_dim + pivot_row);
            }
        }
        let denom = mat[n * row_dim + row_search];
        for j in row_search + 1..row_dim {
            let piv = mat[n * row_dim + j] / denom;
            for m in 0..col_dim {
                mat[m * row_dim + j] -= piv * mat[m * row_dim + row_search];
            }
        }
        row_search += 1;
        if n < col_dim - 1 {
            coeff_rank += 1;
        }
    }
    (mat, row_search, row_search == coeff_rank)
}


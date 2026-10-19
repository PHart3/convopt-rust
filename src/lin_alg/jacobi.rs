// Two variants of Jacobi eigenvalue algorithm, one using a cache to find new pivot, the other using column-cyclic ordering
// Boths Jacobi methods compute spectral decomposition (L, Q) of real symmetric matrix via Jacobi rotations
// and store eigenvalues in L and corresponding eigenvectors in Q.

use vstd::prelude::*;
use crate::lin_alg::sym_matrix::*;
use crate::lin_alg::comparison::{is_significantly_negative, scaled_near_zero};
use crate::trusted::{abort, neg_infinity};
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
use crate::lin_alg::jacobi_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::eigen_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::constr_spec::lemma_pk_idx_injective;

verus! {

broadcast use crate::trusted::group_real_model;

/// Which symmetric entries hold their rotated value once rows m < done have been swept:
/// the 2 x 2 pivot block, and the entries (j, m), (k, m) (in either index order) for m < done.
pub open spec fn rot_done(j: int, k: int, done: int, r: int, l: int) -> bool {
    let (rp, lp) = (r == j || r == k, l == j || l == k);
    (rp && lp) || (rp && !lp && l < done) || (lp && !rp && r < done)
}

// Jacobi rotation subroutine (where orth stores the eventual eigenvectors)
// (pivot entry (j, k) with j < k, nonzero: for J = rot_mat(j, k, c, s) with c^2 + s^2 = 1 the
// packed matrix becomes J^T A J and the column-major matrix Q J (jacobi_step); the pivot
// entry is annihilated and |A_jk| is returned)
fn jacobi_rot<'a>(mat: &'a mut SymMatrix, orth: &'a mut Matrix, dim: usize, k: usize, j: usize)
    -> (res: (&'a mut SymMatrix, &'a mut Matrix, f64))
    requires
        dim_fits(dim as int),
        j < k < dim,
        old(mat)@.len() >= pk_len(dim as int),
        old(orth)@.len() == dim * dim,
        rv(old(mat)@[pk_idx(j as int, k as int)]) != 0.0real,
    ensures
        res.0@.len() == old(mat)@.len(),
        res.1@.len() == old(orth)@.len(),
        final(mat)@ == final(res.0)@,
        final(orth)@ == final(res.1)@,
        exists|c: real, s: real|
            jacobi_step(rvs(old(mat)@), rvs(old(orth)@), rvs(res.0@), rvs(res.1@), dim as int, j as int, k as int, c, s),
        sym_entry(rvs(res.0@), j as int, k as int) == 0.0real,
        rv(res.2) == abs_r(rv(old(mat)@[pk_idx(j as int, k as int)])),
{
    let ghost n = dim as int;
    let ghost (ji, ki) = (j as int, k as int);
    let ghost a = rvs(old(mat)@);
    let ghost q = rvs(old(orth)@);
    proof {
        lemma_dim_fits_idx(n, ki);
        lemma_dim_fits_idx(n, ji);
        lemma_pk_diag_idx(ji);
        lemma_pk_diag_idx(ki);
        lemma_pk_idx_bound(ji, ki, n);
        lemma_pk_idx_bound(ji, ji, n);
        lemma_pk_idx_bound(ki, ki, n);
        lemma_pk_idx_injective(ji, ki, ji, ji);
        lemma_pk_idx_injective(ji, ki, ki, ki);
        lemma_pk_idx_injective(ji, ji, ki, ki);
    }
    let ind = (k * (k + 1)) / 2 + j;
    let val_old = mat[ind];
    let (ind_j, ind_k) = ((j * j + 3 * j) / 2, (k * k + 3 * k) / 2);
    let tau = (mat[ind_k] - mat[ind_j]) / (2.0 * val_old);
    let sigma = tau.signum();
    let sq = (1.0 + tau * tau).sqrt();
    let t = sigma / (tau.abs() + sq);
    let sq2 = (1.0 + t * t).sqrt();
    let c = sq2.recip();
    let s = c * t;
    let ghost (ajj, akk, ajk) = (sym_entry(a, ji, ji), sym_entry(a, ki, ki), sym_entry(a, ji, ki));
    let ghost (tr, cr, sr) = (rv(t), rv(c), rv(s));
    proof {
        // tau = (a_kk - a_jj) / (2 a_jk); t solves t^2 + 2 tau t - 1 = 0; c^2 + s^2 = 1
        let (taur, sg, sqr, sq2r) = (rv(tau), rv(sigma), rv(sq), rv(sq2));
        let (den, denom) = (2.0real * ajk, abs_r(taur) + sqr);
        assert(den * ((akk - ajj) / den) == akk - ajj) by (nonlinear_arith)
            requires den != 0.0real;
        assert(taur * taur >= 0.0real) by (nonlinear_arith);
        assert(sg * sg == 1.0real);
        assert(sg * taur == abs_r(taur));
        assert(sqr > 0.0real) by (nonlinear_arith)
            requires sqr >= 0.0real, sqr * sqr == 1.0real + taur * taur, taur * taur >= 0.0real;
        assert(denom * (sg / denom) == sg) by (nonlinear_arith)
            requires denom != 0.0real;
        lemma_jacobi_t(taur, sg, sqr, tr);
        assert(tr * tr >= 0.0real) by (nonlinear_arith);
        assert(sq2r > 0.0real) by (nonlinear_arith)
            requires sq2r >= 0.0real, sq2r * sq2r == 1.0real + tr * tr, tr * tr >= 0.0real;
        assert((1.0real / sq2r) * sq2r == 1.0real) by (nonlinear_arith)
            requires sq2r != 0.0real;
        lemma_jacobi_cs(tr, sq2r, cr, sr);
        // the pivot equation times a_jk: a_jk t^2 + (a_kk - a_jj) t - a_jk = 0
        assert(ajk * (tr * tr + 2.0real * taur * tr - 1.0real) == ajk * tr * tr + (den * taur) * tr - ajk) by (nonlinear_arith)
            requires den == 2.0real * ajk;
        assert(ajk * tr * tr + (akk - ajj) * tr - ajk == 0.0real);
        lemma_jacobi_scalars(ajj, akk, ajk, tr, cr, sr);
    }
    let ghost target = rot_target(sym_view(a), ji, ki, cr, sr);
    let ghost prev0 = rvs(mat@);
    mat[ind_k] = mat[ind_k] + t * val_old;
    let ghost mid0 = rvs(mat@);
    mat[ind_j] = mat[ind_j] - t * val_old;
    let ghost mid1 = rvs(mat@);
    mat[ind] = 0.0;
    proof {
        // the pivot block holds its rotated values, everything else is unchanged
        let cur = rvs(mat@);
        assert(sym_entry(cur, ki, ki) == akk + tr * ajk);
        assert(sym_entry(cur, ji, ji) == ajj - tr * ajk);
        assert(sym_entry(cur, ji, ki) == 0.0real);
        assert(sym_view(a)(ki, ji) == ajk);
        assert forall|r: int, l: int| 0 <= r < n && 0 <= l < n
            implies #[trigger] sym_entry(cur, r, l) == (if rot_done(ji, ki, 0, r, l) { target(r, l) } else { sym_entry(a, r, l) }) by {
            if !rot_done(ji, ki, 0, r, l) {
                lemma_sym_entry_write_other(prev0, mid0, n, ki, ki, r, l);
                lemma_sym_entry_write_other(mid0, mid1, n, ji, ji, r, l);
                lemma_sym_entry_write_other(mid1, cur, n, ji, ki, r, l);
            }
        }
    }
    // rows and columns j, k outside the pivot block
    for m in 0..dim
        invariant
            j < k < dim,
            n == dim as int,
            ji == j as int,
            ki == k as int,
            dim_fits(n),
            a == rvs(old(mat)@),
            old(mat)@.len() >= pk_len(n),
            mat@.len() == old(mat)@.len(),
            cr == rv(c),
            sr == rv(s),
            target == rot_target(sym_view(a), ji, ki, cr, sr),
            forall|r: int, l: int| 0 <= r < n && 0 <= l < n
                ==> #[trigger] sym_entry(rvs(mat@), r, l) == (if rot_done(ji, ki, m as int, r, l) { target(r, l) } else { sym_entry(a, r, l) }),
    {
        let ghost mi = m as int;
        if m != j && m != k {
            let ghost (lo_k, hi_k) = if mi < ki { (mi, ki) } else { (ki, mi) };
            let ghost (lo_j, hi_j) = if mi < ji { (mi, ji) } else { (ji, mi) };
            proof {
                lemma_dim_fits_idx(n, mi);
                lemma_pk_idx_bound(lo_k, hi_k, n);
                lemma_pk_idx_bound(lo_j, hi_j, n);
                lemma_pk_idx_injective(lo_k, hi_k, lo_j, hi_j);
            }
            let ind_mk = if m > k { (m * (m + 1)) / 2 + k } else { (k * (k + 1)) / 2 + m };
            let ind_mj = if m > j { (m * (m + 1)) / 2 + j } else { (j * (j + 1)) / 2 + m };
            let val_old_k = mat[ind_mk];
            let val_old_j = mat[ind_mj];
            let ghost prev = rvs(mat@);
            proof {
                assert(ind_mk == pk_idx(lo_k, hi_k));
                assert(ind_mj == pk_idx(lo_j, hi_j));
                // the entries (m, k) and (m, j) still hold their original values
                assert(sym_entry(prev, mi, ki) == sym_entry(a, mi, ki));
                assert(sym_entry(prev, mi, ji) == sym_entry(a, mi, ji));
                assert(rv(val_old_k) == sym_entry(prev, mi, ki));
                assert(rv(val_old_j) == sym_entry(prev, mi, ji));
            }
            mat[ind_mk] = c * val_old_k + s * val_old_j;
            let ghost mid = rvs(mat@);
            mat[ind_mj] = c * val_old_j - s * val_old_k;
            proof {
                let cur = rvs(mat@);
                assert(sym_entry(cur, mi, ki) == cr * sym_entry(a, mi, ki) + sr * sym_entry(a, mi, ji));
                assert(sym_entry(cur, mi, ji) == cr * sym_entry(a, mi, ji) - sr * sym_entry(a, mi, ki));
                assert forall|r: int, l: int| 0 <= r < n && 0 <= l < n
                    implies #[trigger] sym_entry(cur, r, l) == (if rot_done(ji, ki, mi + 1, r, l) { target(r, l) } else { sym_entry(a, r, l) }) by {
                    if (r == mi && (l == ji || l == ki)) || (l == mi && (r == ji || r == ki)) {
                        assert(sym_view(a)(ji, mi) == sym_entry(a, mi, ji));
                        assert(sym_view(a)(ki, mi) == sym_entry(a, mi, ki));
                    } else {
                        lemma_sym_entry_write_other(prev, mid, n, lo_k, hi_k, r, l);
                        lemma_sym_entry_write_other(mid, cur, n, lo_j, hi_j, r, l);
                    }
                }
            }
        }
    }
    // columns j and k of the eigenvector matrix: Q J
    proof {
        assert(orth.len() == dim * dim);
        assert(j * dim + dim <= dim * dim) by (nonlinear_arith)
            requires j < dim;
        assert(k * dim + dim <= dim * dim) by (nonlinear_arith)
            requires k < dim;
    }
    let (base_j, base_k) = (j * dim, k * dim);
    let ghost qtarget = rot_cols(cm_view(q, n), ji, ki, cr, sr);
    for m in 0..dim
        invariant
            j < k < dim,
            n == dim as int,
            ji == j as int,
            ki == k as int,
            q == rvs(old(orth)@),
            orth@.len() == old(orth)@.len(),
            old(orth)@.len() == dim * dim,
            dim * dim <= usize::MAX,
            base_j == j * dim,
            base_k == k * dim,
            j * dim + dim <= dim * dim,
            k * dim + dim <= dim * dim,
            cr == rv(c),
            sr == rv(s),
            qtarget == rot_cols(cm_view(q, n), ji, ki, cr, sr),
            forall|r: int, l: int| 0 <= r < n && 0 <= l < n
                ==> #[trigger] cm_entry(rvs(orth@), n, r, l)
                    == (if (l == ji || l == ki) && r < m { qtarget(r, l) } else { cm_entry(q, n, r, l) }),
    {
        let ghost mi = m as int;
        proof {
            lemma_cm_idx_injective(n, mi, ji, mi, ki);
        }
        let old_j = orth[base_j + m];
        let old_k = orth[base_k + m];
        let ghost prev = rvs(orth@);
        proof {
            assert(base_j + m == cm_idx(n, mi, ji));
            assert(base_k + m == cm_idx(n, mi, ki));
            // row m of columns j and k still holds its original values
            assert(cm_entry(prev, n, mi, ji) == cm_entry(q, n, mi, ji));
            assert(cm_entry(prev, n, mi, ki) == cm_entry(q, n, mi, ki));
            assert(rv(old_j) == cm_entry(prev, n, mi, ji));
            assert(rv(old_k) == cm_entry(prev, n, mi, ki));
        }
        orth[base_j + m] = c * old_j - s * old_k;
        let ghost mid = rvs(orth@);
        orth[base_k + m] = s * old_j + c * old_k;
        proof {
            let cur = rvs(orth@);
            assert(cm_entry(cur, n, mi, ji) == cr * cm_entry(q, n, mi, ji) - sr * cm_entry(q, n, mi, ki));
            assert(cm_entry(cur, n, mi, ki) == sr * cm_entry(q, n, mi, ji) + cr * cm_entry(q, n, mi, ki));
            assert forall|r: int, l: int| 0 <= r < n && 0 <= l < n
                implies #[trigger] cm_entry(cur, n, r, l)
                    == (if (l == ji || l == ki) && r < mi + 1 { qtarget(r, l) } else { cm_entry(q, n, r, l) }) by {
                if !(r == mi && (l == ji || l == ki)) {
                    lemma_cm_entry_write_other(prev, mid, n, n, mi, ji, r, l);
                    lemma_cm_entry_write_other(mid, cur, n, n, mi, ki, r, l);
                }
            }
        }
    }
    proof {
        let (b2, q2) = (rvs(mat@), rvs(orth@));
        let jm = rot_mat(ji, ki, cr, sr);
        // A' = J^T A J: the sweep produced rot_target, which is J^T A J
        let jtaj = mmul(mmul(mtr(jm), sym_view(a), n), jm, n);
        assert forall|r: int, l: int| 0 <= r < n && 0 <= l < n implies #[trigger] sym_view(b2)(r, l) == target(r, l) by {
            assert(sym_entry(b2, r, l) == (if rot_done(ji, ki, n, r, l) { target(r, l) } else { sym_entry(a, r, l) }));
        }
        lemma_rot_congruence(sym_view(a), n, ji, ki, cr, sr);
        lemma_meq_sym(jtaj, target, n, n);
        lemma_meq_trans(sym_view(b2), target, jtaj, n, n);
        // Q' = Q J
        let qj = mmul(cm_view(q, n), jm, n);
        assert forall|r: int, l: int| 0 <= r < n && 0 <= l < n implies #[trigger] cm_view(q2, n)(r, l) == qtarget(r, l) by {
            assert(cm_entry(q2, n, r, l) == (if (l == ji || l == ki) && r < n { qtarget(r, l) } else { cm_entry(q, n, r, l) }));
        }
        lemma_rot_cols(cm_view(q, n), n, ji, ki, cr, sr);
        lemma_meq_sym(qj, qtarget, n, n);
        lemma_meq_trans(cm_view(q2, n), qtarget, qj, n, n);
        assert(jacobi_step(rvs(old(mat)@), rvs(old(orth)@), rvs(mat@), rvs(orth@), dim as int, j as int, k as int, cr, sr));
        assert(exists|c: real, s: real|
            jacobi_step(rvs(old(mat)@), rvs(old(orth)@), rvs(mat@), rvs(orth@), dim as int, j as int, k as int, c, s));
        // the pivot entry is annihilated
        assert(sym_entry(b2, ji, ki) == target(ji, ki));
    }
    (mat, orth, val_old.abs())
}

/* ---------- cached Jacobi ---------- */

// first Jacobi variant, with each sweep taking cubic time on average by caching maximal element of each row

/// Row i holds a maximal off-diagonal magnitude of column c (over the rows r < c).
pub open spec fn col_max_at(b: Seq<real>, c: int, i: int) -> bool {
    &&& 0 <= i < c
    &&& forall|r: int| 0 <= r < c ==> abs_r(#[trigger] sym_entry(b, r, c)) <= abs_r(sym_entry(b, i, c))
}

/// The cache of `jacobi_eigen_cach`: maxima[c - 1] is a maximal row of column c for 1 <= c < dim.
pub open spec fn cache_ok(b: Seq<real>, maxima: Seq<usize>, dim: int) -> bool {
    &&& maxima.len() == dim - 1
    &&& forall|t: int| 0 <= t < dim - 1 ==> col_max_at(b, t + 1, #[trigger] maxima[t] as int)
}

/// After a rotation at (j, k) with old pivot magnitude big, a column m outside {j, k} keeps its
/// entries in the unrotated rows, and those are bounded by the old cached entry (i, m) and by big.
proof fn lemma_col_unrotated(prev: Seq<real>, cur: Seq<real>, dim: int, j: int, k: int, m: int, i: int, big: real)
    requires
        offdiag_max_at(prev, dim, j, k),
        big == abs_r(sym_entry(prev, j, k)),
        same_outside(prev, cur, dim, j, k),
        col_max_at(prev, m, i),
        1 <= m < dim,
        m != j,
        m != k,
    ensures
        abs_r(sym_entry(prev, i, m)) <= big,
        i != j && i != k ==> sym_entry(cur, i, m) == sym_entry(prev, i, m),
        forall|r: int| 0 <= r < m && r != j && r != k
            ==> abs_r(#[trigger] sym_entry(cur, r, m)) <= abs_r(sym_entry(prev, i, m)),
{
    assert forall|r: int| 0 <= r < m && r != j && r != k
        implies abs_r(#[trigger] sym_entry(cur, r, m)) <= abs_r(sym_entry(prev, i, m)) by {
        assert(sym_entry(cur, r, m) == sym_entry(prev, r, m));
    }
}

// given a column in a symmetric matrix, find index of offdiagonal element with largest magnitude
// (the first row attaining the maximum magnitude among the rows 0..col)
fn index_max_in_col(a: &SymMatrix, col: usize) -> (index_max: usize)
    requires
        col >= 1,
        dim_fits(col as int),
        a@.len() >= pk_len(col as int + 1),
    ensures
        col_max_at(rvs(a@), col as int, index_max as int),
{
    if col == 0 {
        abort("index_max_in_col : column position must be positive");
    }
    let ghost b = rvs(a@);
    let ghost ci = col as int;
    proof {
        lemma_dim_fits_idx(ci, ci);
        lemma_pk_idx_bound(0, ci, ci + 1);
    }
    let start = (col * (col + 1)) / 2;
    let mut max = a[start];
    let mut index_max = 0;
    for i in 0..col
        invariant
            b == rvs(a@),
            ci == col as int,
            col >= 1,
            dim_fits(ci),
            a@.len() >= pk_len(ci + 1),
            start == pk_col_start(ci),
            index_max < col,
            rv(max) == sym_entry(b, index_max as int, ci),
            forall|r: int| 0 <= r < i ==> abs_r(#[trigger] sym_entry(b, r, ci)) <= abs_r(rv(max)),
    {
        proof {
            assert(a.len() >= pk_len(ci + 1));
            lemma_pk_idx_bound(i as int, ci, ci + 1);
        }
        let index = start + i;
        if a[index].abs() > max.abs() {
            index_max = i;
            max = a[index];
        }
    }
    index_max
}

/// The first column k >= 1 whose cached entry has the largest magnitude; with a valid cache,
/// (maxima[k - 1], k) is then a position of maximal off-diagonal magnitude.
fn cached_pivot_col(a: &SymMatrix, maxima: &Vec<usize>, dim: usize) -> (k: usize)
    requires
        dim >= 2,
        dim_fits(dim as int),
        a@.len() >= pk_len(dim as int),
        cache_ok(rvs(a@), maxima@, dim as int),
    ensures
        1 <= k < dim,
        offdiag_max_at(rvs(a@), dim as int, maxima@[k - 1] as int, k as int),
{
    let ghost b = rvs(a@);
    let ghost n = dim as int;
    proof {
        assert(col_max_at(b, 1, maxima@[0] as int));
        lemma_pk_idx_bound(maxima@[0] as int, 1, n);
    }
    let mut max = a[1 + maxima[0]];
    let mut k = 1;
    for m in 2..dim
        invariant
            b == rvs(a@),
            n == dim as int,
            dim >= 2,
            dim_fits(n),
            a@.len() >= pk_len(n),
            cache_ok(b, maxima@, n),
            1 <= k < m,
            rv(max) == sym_entry(b, maxima@[k - 1] as int, k as int),
            forall|t: int| 0 <= t < m - 1
                ==> abs_r(sym_entry(b, #[trigger] maxima@[t] as int, t + 1)) <= abs_r(rv(max)),
    {
        let ghost mi = m as int;
        let i = maxima[m - 1];
        proof {
            assert(col_max_at(b, mi, i as int));
            lemma_dim_fits_idx(n, mi);
            lemma_pk_idx_bound(i as int, mi, n);
        }
        let ind = (m * (m + 1)) / 2 + i;
        if a[ind].abs() > max.abs() {
            max = a[ind];
            k = m;
        }
    }
    proof {
        let (ji, ki) = (maxima@[k - 1] as int, k as int);
        assert forall|r: int, c: int| 0 <= r < c < n
            implies abs_r(#[trigger] sym_entry(b, r, c)) <= abs_r(sym_entry(b, ji, ki)) by {
            assert(col_max_at(b, c, maxima@[c - 1] as int));
            assert(abs_r(sym_entry(b, maxima@[c - 1] as int, (c - 1) + 1)) <= abs_r(rv(max)));
        }
    }
    k
}

const MAX_SWEEPS: usize = 50;

// (for symmetric A of order dim, returns (diag(B), Q) with A = Q B Q^T, Q orthogonal and the
// off-diagonal part of B below the stopping threshold; see eigen_spec.rs. Each rotation pivots
// on the position of maximal off-diagonal magnitude, found through the per-column cache
// `maxima`; aborts if MAX_SWEEPS sweeps worth of rotations do not reach the threshold)
pub fn jacobi_eigen_cach(a: &mut SymMatrix, dim: usize) -> (res: (Vector, Matrix))
    requires
        dim >= 1,
        dim_fits(dim as int),
        old(a)@.len() >= pk_len(dim as int),
    ensures
        final(a)@.len() == old(a)@.len(),
        res.0@.len() == dim,
        res.1@.len() == dim * dim,
        eigensolver_ok(rvs(old(a)@), rvs(final(a)@), rvs(res.1@), rvs(res.0@), dim as int),
{
    let ghost a0 = rvs(old(a)@);
    let ghost n = dim as int;
    let ghost f = |i: int| abs_r(sym_diag(a0, i));
    proof {
        lemma_dim_fits_idx(n, n);
        assert(dim * dim <= dim * (dim + 1)) by (nonlinear_arith);
        lemma_pk_idx_bound(0, 0, n);
    }
    if dim == 1 {
        let res = (vec![a[0]], vec![1.0]);
        proof {
            // Q = I and B = A: an exact eigendecomposition without off-diagonal entries
            assert(cm_entry(rvs(res.1@), 1, 0, 0) == 1.0real);
            assert(is_ident_r(rvs(res.1@), 1));
            lemma_eigen_inv_init(a0, rvs(res.1@), 1);
            assert(rvs(res.0@) =~= sym_diag_r(a0, 1));
            lemma_converged_approx(a0, rvs(res.1@), a0, 1);
        }
        return res;
    }
    // the cache and the diagonal scale of the input
    let mut maxima: Vec<usize> = Vec::with_capacity(dim - 1);
    let mut max_diag = a[0].abs();
    proof {
        assert(max0_fn_r(f, 0) == 0.0real);
        assert(max0_fn_r(f, 1) == max_r(0.0real, f(0)));
    }
    for m in 1..dim
        invariant
            a0 == rvs(a@),
            n == dim as int,
            dim >= 2,
            dim_fits(n),
            a@.len() >= pk_len(n),
            f == (|i: int| abs_r(sym_diag(a0, i))),
            maxima@.len() == m - 1,
            forall|t: int| 0 <= t < m - 1 ==> col_max_at(a0, t + 1, #[trigger] maxima@[t] as int),
            rv(max_diag) == max0_fn_r(f, m as nat),
    {
        let ghost mi = m as int;
        proof {
            lemma_dim_fits_idx(n, mi);
            lemma_pk_diag_idx(mi);
            lemma_pk_idx_bound(mi, mi, n);
            lemma_dim_fits_monotone(mi, n);
            lemma_pk_col_start_monotone(mi + 1, n);
        }
        let diag = a[(m * m + 3 * m) / 2].abs();
        if diag > max_diag {
            max_diag = diag;
        }
        maxima.push(index_max_in_col(a, m));
    }
    let mut q0 = ident_mat(dim);
    let mut vals = a;
    let mut vects = &mut q0;
    let ghost qf = final(vects)@;
    // magnitude of the pivot of the last rotation (unread before the first rotation)
    let mut biggest = neg_infinity();
    // pivot position (j, k)
    let mut k = cached_pivot_col(vals, &maxima, dim);
    let mut j = maxima[k - 1];
    let (mut count, mut first1, mut first2) = (0, true, true);
    proof {
        assert(dim * (dim - 1) <= dim * (dim + 1)) by (nonlinear_arith);
    }
    let max_rotations = (dim * (dim - 1) / 2).saturating_mul(MAX_SWEEPS);
    // the matrix before the last rotation
    let ghost mut prev = rvs(vals@);
    proof {
        lemma_eigen_inv_init(a0, rvs(vects@), n);
    }
    loop
        invariant_except_break
            first1 == first2,
            // before the first rotation: cache, pivot and diagonal scale describe the current matrix
            first1 ==> cache_ok(rvs(vals@), maxima@, n)
                && offdiag_max_at(rvs(vals@), n, j as int, k as int)
                && rv(max_diag) == max_abs_diag_r(rvs(vals@), n),
            // afterwards: they describe the matrix before the last rotation at (j, k)
            !first1 ==> cache_ok(prev, maxima@, n)
                && offdiag_max_at(prev, n, j as int, k as int)
                && rv(biggest) == abs_r(sym_entry(prev, j as int, k as int))
                && same_outside(prev, rvs(vals@), n, j as int, k as int),
        invariant
            n == dim as int,
            dim >= 2,
            dim_fits(n),
            a0 == rvs(old(a)@),
            old(a)@.len() >= pk_len(n),
            vals@.len() == old(a)@.len(),
            vects@.len() == dim * dim,
            final(a)@ == final(vals)@,
            final(vects)@ == qf,
            eigen_decomp_inv(a0, rvs(vects@), rvs(vals@), n),
            maxima@.len() == dim - 1,
            count <= max_rotations,
            j < k < dim,
        ensures
            vals@.len() == old(a)@.len(),
            vects@.len() == dim * dim,
            final(a)@ == final(vals)@,
            final(vects)@ == qf,
            eigen_decomp_inv(a0, rvs(vects@), rvs(vals@), n),
            jacobi_converged(rvs(vals@), n),
        decreases max_rotations - count,
    {
        if count == max_rotations {
            abort("jacobi_eigen_cach failed to converge after 50 sweeps");
        }
        count += 1;
        let ghost cur = rvs(vals@);
        let ghost fc = |i: int| abs_r(sym_diag(cur, i));
        if first1 {
            first1 = false;
        } else {
            // update the cache and the diagonal scale after the rotation at (j, k)
            let ghost mx0 = maxima@;
            proof {
                lemma_pk_idx_bound(0, 0, n);
            }
            max_diag = vals[0].abs();
            proof {
                assert(max0_fn_r(fc, 0) == 0.0real);
                assert(max0_fn_r(fc, 1) == max_r(0.0real, fc(0)));
            }
            for m in 1..dim
                invariant
                    n == dim as int,
                    dim >= 2,
                    dim_fits(n),
                    cur == rvs(vals@),
                    vals@.len() >= pk_len(n),
                    fc == (|i: int| abs_r(sym_diag(cur, i))),
                    mx0.len() == dim - 1,
                    maxima@.len() == dim - 1,
                    j < k < dim,
                    cache_ok(prev, mx0, n),
                    offdiag_max_at(prev, n, j as int, k as int),
                    rv(biggest) == abs_r(sym_entry(prev, j as int, k as int)),
                    same_outside(prev, cur, n, j as int, k as int),
                    forall|t: int| 0 <= t < m - 1 ==> col_max_at(cur, t + 1, #[trigger] maxima@[t] as int),
                    forall|t: int| m - 1 <= t < dim - 1 ==> #[trigger] maxima@[t] == mx0[t],
                    rv(max_diag) == max0_fn_r(fc, m as nat),
            {
                let ghost (mi, ji, ki) = (m as int, j as int, k as int);
                proof {
                    lemma_dim_fits_idx(n, mi);
                    lemma_pk_diag_idx(mi);
                    lemma_pk_idx_bound(mi, mi, n);
                    lemma_dim_fits_monotone(mi, n);
                    lemma_pk_col_start_monotone(mi + 1, n);
                }
                let diag = vals[(m * m + 3 * m) / 2].abs();
                if diag > max_diag {
                    max_diag = diag;
                }
                let i = maxima[m - 1];
                let ghost ii = i as int;
                proof {
                    assert(col_max_at(prev, mi, ii));
                    lemma_pk_idx_bound(ii, mi, n);
                    if j < m {
                        lemma_pk_idx_bound(ji, mi, n);
                    }
                    if k < m {
                        lemma_pk_idx_bound(ki, mi, n);
                    }
                    if m != j && m != k {
                        lemma_col_unrotated(prev, cur, n, ji, ki, mi, ii, rv(biggest));
                    }
                }
                let col = (m * (m + 1)) / 2;
                let ind = col + i;
                if m == k || m == j {
                    // the whole column was rotated
                    maxima[m - 1] = index_max_in_col(vals, m);
                } else if (i != k && k < m) && (i != j && j < m) {
                    // the cached entry is unrotated and still dominates the other unrotated
                    // entries; the rotated rows j and k are the only other candidates
                    let (val_j, val_k) = (vals[col + j].abs(), vals[col + k].abs());
                    let val_ind = vals[ind].abs();
                    if val_j > val_ind {
                        if val_k > val_j {
                            maxima[m - 1] = k;
                        } else {
                            maxima[m - 1] = j;
                        }
                    } else if val_k > val_ind {
                        maxima[m - 1] = k;
                    }
                    proof {
                        let ni = maxima@[mi - 1] as int;
                        assert forall|r: int| 0 <= r < mi
                            implies abs_r(#[trigger] sym_entry(cur, r, mi)) <= abs_r(sym_entry(cur, ni, mi)) by {
                            if r != ji && r != ki {
                                assert(abs_r(sym_entry(cur, r, mi)) <= abs_r(sym_entry(cur, ii, mi)));
                            }
                        }
                    }
                } else if vals[ind].abs() < biggest {
                    // the cached entry was rotated and shrank below the old maximum: rescan
                    maxima[m - 1] = index_max_in_col(vals, m);
                } else if k < m {
                    // j < k < m and the cached entry (in row j or k) is at least the old maximum,
                    // which bounds every unrotated entry: the maximum is in row j or k
                    if vals[col + j].abs() > vals[col + k].abs() {
                        maxima[m - 1] = j;
                    } else {
                        maxima[m - 1] = k;
                    }
                    proof {
                        let ni = maxima@[mi - 1] as int;
                        assert forall|r: int| 0 <= r < mi
                            implies abs_r(#[trigger] sym_entry(cur, r, mi)) <= abs_r(sym_entry(cur, ni, mi)) by {
                            if r != ji && r != ki {
                                assert(abs_r(sym_entry(cur, r, mi)) <= abs_r(sym_entry(prev, ii, mi)));
                            }
                        }
                    }
                } else if j < m && vals[col + j].abs() > vals[ind].abs() {
                    // j < m < k: only row j of this column was rotated, and it now exceeds the
                    // cached entry, which is at least the old maximum
                    maxima[m - 1] = j;
                    proof {
                        assert forall|r: int| 0 <= r < mi
                            implies abs_r(#[trigger] sym_entry(cur, r, mi)) <= abs_r(sym_entry(cur, ji, mi)) by {
                            if r != ji {
                                assert(abs_r(sym_entry(cur, r, mi)) <= abs_r(sym_entry(prev, ii, mi)));
                            }
                        }
                    }
                } else {
                    // either j < m < k and the cached entry (at least the old maximum) still
                    // dominates the rotated row j, or m < j < k and the column is unrotated
                    proof {
                        assert forall|r: int| 0 <= r < mi
                            implies abs_r(#[trigger] sym_entry(cur, r, mi)) <= abs_r(sym_entry(cur, ii, mi)) by {
                            if r != ji && r != ki {
                                assert(abs_r(sym_entry(cur, r, mi)) <= abs_r(sym_entry(prev, ii, mi)));
                            }
                        }
                    }
                }
            }
        }
        if first2 {
            first2 = false;
        } else {
            k = cached_pivot_col(vals, &maxima, dim);
            j = maxima[k - 1];
        }
        proof {
            lemma_dim_fits_idx(n, k as int);
            lemma_pk_idx_bound(j as int, k as int, n);
            assert(vals.len() >= pk_len(n));
        }
        let piv = vals[(k * (k + 1)) / 2 + j];
        if scaled_near_zero(piv, 1.0_f64.max(max_diag), 1e-12) {
            proof {
                lemma_offdiag_max_below(cur, n, j as int, k as int, jacobi_thr_r(cur, n));
            }
            break;
        }
        // the pivot is nonzero: its magnitude is at least 1e-12 * max(1, max_diag)
        let ghost qprev = rvs(vects@);
        proof {
            prev = cur;
        }
        let (v, q, b) = jacobi_rot(vals, vects, dim, k, j);
        vals = v;
        vects = q;
        biggest = b;
        proof {
            lemma_eigen_inv_step(a0, qprev, prev, rvs(vects@), rvs(vals@), n, j as int, k as int);
            lemma_jacobi_step_same_outside(prev, qprev, rvs(vals@), rvs(vects@), n, j as int, k as int);
        }
    }
    let eigenvals = diag_of_sym_mat(vals, dim);
    proof {
        lemma_converged_approx(a0, rvs(vects@), rvs(vals@), n);
    }
    (eigenvals, q0)
}

/* ---------- column-cyclic Jacobi ---------- */

// second Jacobi variant, with each sweep taking cubic worst-case time by cyclically moving through columns
// (same contract as `jacobi_eigen_cach`; each sweep visits the pairs (j, k), j < k, in
// column-cyclic order (`visits_each_pair_once`) and rotates at every nonzero one; the stopping
// test scans all off-diagonal entries for the maximal magnitude)
pub fn jacobi_eigen_cyc(a: &mut SymMatrix, dim: usize) -> (res: (Vector, Matrix))
    requires
        dim >= 1,
        dim_fits(dim as int),
        old(a)@.len() >= pk_len(dim as int),
    ensures
        final(a)@.len() == old(a)@.len(),
        res.0@.len() == dim,
        res.1@.len() == dim * dim,
        eigensolver_ok(rvs(old(a)@), rvs(final(a)@), rvs(res.1@), rvs(res.0@), dim as int),
{
    let ghost a0 = rvs(old(a)@);
    let ghost n = dim as int;
    proof {
        lemma_dim_fits_idx(n, n);
        assert(dim * dim <= dim * (dim + 1)) by (nonlinear_arith);
        lemma_pk_idx_bound(0, 0, n);
    }
    if dim == 1 {
        let res = (vec![a[0]], vec![1.0]);
        proof {
            // Q = I and B = A: an exact eigendecomposition without off-diagonal entries
            assert(cm_entry(rvs(res.1@), 1, 0, 0) == 1.0real);
            assert(is_ident_r(rvs(res.1@), 1));
            lemma_eigen_inv_init(a0, rvs(res.1@), 1);
            assert(rvs(res.0@) =~= sym_diag_r(a0, 1));
            lemma_converged_approx(a0, rvs(res.1@), a0, 1);
        }
        return res;
    }
    let mut q0 = ident_mat(dim);
    let mut vals = a;
    let mut vects = &mut q0;
    let ghost qf = final(vects)@;
    let mut count = 0;
    proof {
        lemma_eigen_inv_init(a0, rvs(vects@), n);
    }
    loop
        invariant
            n == dim as int,
            dim >= 2,
            dim_fits(n),
            a0 == rvs(old(a)@),
            old(a)@.len() >= pk_len(n),
            vals@.len() == old(a)@.len(),
            vects@.len() == dim * dim,
            final(a)@ == final(vals)@,
            final(vects)@ == qf,
            eigen_decomp_inv(a0, rvs(vects@), rvs(vals@), n),
            count <= MAX_SWEEPS,
        ensures
            vals@.len() == old(a)@.len(),
            vects@.len() == dim * dim,
            final(a)@ == final(vals)@,
            final(vects)@ == qf,
            eigen_decomp_inv(a0, rvs(vects@), rvs(vals@), n),
            jacobi_converged(rvs(vals@), n),
        decreases MAX_SWEEPS - count,
    {
        if count == MAX_SWEEPS {
            abort("jacobi_eigen_cyc failed to converge after 50 sweeps");
        }
        count += 1;
        // column-cyclic ordering of pivot positions
        let ghost mut visited: Seq<(int, int)> = Seq::empty();
        for k in 1..dim
            invariant
                n == dim as int,
                dim >= 2,
                dim_fits(n),
                a0 == rvs(old(a)@),
                old(a)@.len() >= pk_len(n),
                vals@.len() == old(a)@.len(),
                vects@.len() == dim * dim,
                final(a)@ == final(vals)@,
                final(vects)@ == qf,
                eigen_decomp_inv(a0, rvs(vects@), rvs(vals@), n),
                visited == cyc_order(k as int),
        {
            let ghost ki = k as int;
            proof {
                lemma_dim_fits_idx(n, ki);
            }
            for j in 0..k
                invariant
                    n == dim as int,
                    dim >= 2,
                    dim_fits(n),
                    a0 == rvs(old(a)@),
                    old(a)@.len() >= pk_len(n),
                    vals@.len() == old(a)@.len(),
                    vects@.len() == dim * dim,
                    final(a)@ == final(vals)@,
                    final(vects)@ == qf,
                    eigen_decomp_inv(a0, rvs(vects@), rvs(vals@), n),
                    ki == k as int,
                    1 <= k < dim,
                    visited == cyc_order(ki) + Seq::new(j as nat, |t: int| (t, ki)),
            {
                let ghost ji = j as int;
                proof {
                    lemma_dim_fits_idx(n, ki);
                    lemma_pk_idx_bound(ji, ki, n);
                    assert(vals.len() >= pk_len(n));
                    visited = visited.push((ji, ki));
                    assert(visited =~= cyc_order(ki) + Seq::new((j + 1) as nat, |t: int| (t, ki)));
                }
                if vals[(k * (k + 1)) / 2 + j] != 0.0 {
                    let ghost (prev, qprev) = (rvs(vals@), rvs(vects@));
                    let (v, q, _b) = jacobi_rot(vals, vects, dim, k, j);
                    vals = v;
                    vects = q;
                    proof {
                        lemma_eigen_inv_step(a0, qprev, prev, rvs(vects@), rvs(vals@), n, ji, ki);
                    }
                }
            }
            proof {
                assert(cyc_order(ki + 1) == cyc_order(ki) + Seq::new(ki as nat, |j: int| (j, ki + 1 - 1)));
                assert(visited =~= cyc_order(ki + 1));
            }
        }
        proof {
            // the sweep visited every pair j < k < dim exactly once, in column-cyclic order
            assert(visited == cyc_order(n));
            lemma_cyc_order_visits_each_pair_once(n);
            assert(visits_each_pair_once(visited, n));
        }
        // stopping test: the maximal off-diagonal magnitude against the diagonal scale
        let ghost cur = rvs(vals@);
        let ghost fc = |i: int| abs_r(sym_diag(cur, i));
        proof {
            lemma_pk_idx_bound(0, 0, n);
        }
        let (mut max_diag, mut max_offdiag) = (vals[0].abs(), 0.0);
        proof {
            assert(max0_fn_r(fc, 0) == 0.0real);
        }
        for c in 0..dim
            invariant
                n == dim as int,
                dim >= 2,
                dim_fits(n),
                cur == rvs(vals@),
                vals@.len() >= pk_len(n),
                fc == (|i: int| abs_r(sym_diag(cur, i))),
                rv(max_offdiag) >= 0.0real,
                forall|r: int, l: int| 0 <= r < l < c ==> abs_r(#[trigger] sym_entry(cur, r, l)) <= rv(max_offdiag),
                rv(max_diag) == max_r(fc(0), max0_fn_r(fc, c as nat)),
        {
            let ghost ci = c as int;
            proof {
                lemma_dim_fits_idx(n, ci);
                lemma_pk_diag_idx(ci);
                lemma_pk_idx_bound(ci, ci, n);
            }
            for r in 0..c
                invariant
                    n == dim as int,
                    dim_fits(n),
                    cur == rvs(vals@),
                    vals@.len() >= pk_len(n),
                    ci == c as int,
                    c < dim,
                    rv(max_offdiag) >= 0.0real,
                    forall|r: int, l: int| 0 <= r < l < c ==> abs_r(#[trigger] sym_entry(cur, r, l)) <= rv(max_offdiag),
                    forall|r2: int| 0 <= r2 < r ==> abs_r(#[trigger] sym_entry(cur, r2, ci)) <= rv(max_offdiag),
            {
                proof {
                    lemma_dim_fits_idx(n, ci);
                    lemma_pk_idx_bound(r as int, ci, n);
                    assert(vals.len() >= pk_len(n));
                }
                let off = vals[(c * (c + 1)) / 2 + r].abs();
                if off > max_offdiag {
                    max_offdiag = off;
                }
            }
            let diag = vals[(c * c + 3 * c) / 2].abs();
            if diag > max_diag {
                max_diag = diag;
            }
        }
        proof {
            lemma_max0_fn_bound(fc, dim as nat);
            assert(rv(max_diag) == max_abs_diag_r(cur, n));
        }
        if scaled_near_zero(max_offdiag, 1.0_f64.max(max_diag), 1e-12) {
            proof {
                assert forall|r: int, l: int| 0 <= r < n && 0 <= l < n && r != l
                    implies #[trigger] abs_r(sym_entry(cur, r, l)) < jacobi_thr_r(cur, n) by {
                    if r > l {
                        assert(sym_entry(cur, r, l) == sym_entry(cur, l, r));
                    }
                }
            }
            break;
        }
    }
    let eigenvals = diag_of_sym_mat(vals, dim);
    proof {
        lemma_converged_approx(a0, rvs(vects@), rvs(vals@), n);
    }
    (eigenvals, q0)
}

} // verus!

pub enum JacobiVariant {
    Cached,
    Cyclic,
}
use JacobiVariant::*;

// extracting the positive part of the above spectral decomposition
pub fn poseigendecomp(a : &mut SymMatrix, dim : usize, var : &JacobiVariant) -> (Vector, Matrix) {
    let (mut pos_eigenvals, mut pos_eigenvects) : (Vector, Matrix) = (Vec::new(), Vec::new());
    let (eigenvals, eigenvects) =
	match var { Cached => jacobi_eigen_cach(a, dim), Cyclic => jacobi_eigen_cyc(a, dim) };
    for (val, vect) in eigenvals.iter().zip(eigenvects.chunks_exact(dim)) {
	if *val > 0.0 {
	    pos_eigenvals.push(*val);
	    pos_eigenvects.extend_from_slice(vect);
	}
    }
    (pos_eigenvals, pos_eigenvects)
}

// extracting the negative part of the above spectral decomposition
pub fn negeigendecomp(a : &mut SymMatrix, dim : usize, var : &JacobiVariant) -> (Vector, Matrix) {
    let (mut neg_eigenvals, mut neg_eigenvects) : (Vector, Matrix) = (Vec::new(), Vec::new());
    let scale = a.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1.0);
    let (eigenvals, eigenvects) =
        match var { Cached => jacobi_eigen_cach(a, dim), Cyclic => jacobi_eigen_cyc(a, dim) };
    for (val, vect) in eigenvals.iter().zip(eigenvects.chunks_exact(dim)) {
	if is_significantly_negative(*val, scale, 1e-8) {
	    neg_eigenvals.push(*val);
	    neg_eigenvects.extend_from_slice(vect);
	}
    }
    (neg_eigenvals, neg_eigenvects)
}

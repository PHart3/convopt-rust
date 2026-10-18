// Two variants of Jacobi eigenvalue algorithm, one using a cache to find new pivot, the other using column-cyclic ordering
// Boths Jacobi methods compute spectral decomposition (L, Q) of real symmetric matrix via Jacobi rotations
// and store eigenvalues in L and corresponding eigenvectors in Q.

use vstd::prelude::*;
use crate::lin_alg::sym_matrix::*;
use crate::lin_alg::comparison::{is_significantly_negative, scaled_near_zero};
#[cfg(verus_keep_ghost)]
use crate::trusted::rv;
#[cfg(verus_keep_ghost)]
use crate::real_spec::abs_r;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::vec_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::sym_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::mat_spec::*;
#[cfg(verus_keep_ghost)]
use crate::lin_alg::jacobi_spec::*;
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

} // verus!

// first Jacobi variant, with each sweep taking cubic time on average by caching maximal element of each row

// given a column in a symmetric matrix, find index of offdiagonal element with largest magnitude
fn index_max_in_col(a : &SymMatrix, col : usize) -> usize {
    if col == 0 {
	panic!("index_max_in_col : column position must be positive");
    }
    let mut max = a[(col * (col + 1)) / 2];
    let mut index : usize;
    let mut index_max = 0;
    for i in 0..col {
	index = (col * (col + 1)) / 2 + i;
	if a[index].abs() > max.abs() {
	    index_max = i;
	    max = a[index]; 
	}
    }
    index_max
}

const MAX_SWEEPS: usize = 50;

pub fn jacobi_eigen_cach(a : &mut SymMatrix, dim : usize) -> (Vector, Matrix) {
    if dim == 1 {
	return (vec![a[0]], vec![1.0]);
    }
    let mut maxima = Vec::with_capacity(dim - 1);
    let mut max_diag = a[0].abs();
    for n in 1..dim {
	if a[(n * n + 3 * n) / 2].abs() > max_diag {
	    max_diag = a[(n * n + 3 * n) / 2].abs();
	}
	maxima.push(index_max_in_col(a, n));
    }
    let (mut eigenvals_new, mut eigenvects_new) = (a, &mut ident_mat(dim));
    let (mut eigenvals, mut eigenvects) : (&mut SymMatrix, &mut Matrix);
    let mut biggest = f64::NEG_INFINITY;
    let mut max : (usize, f64) = (1, eigenvals_new[1 + maxima[0]]);
    let mut m1;
    let mut m2 = 2;
    let mut ind : usize;
    for i in &maxima[1..] {
	ind = (m2 * (m2 + 1)) / 2 + i;
	if eigenvals_new[ind].abs() > max.1.abs() {
	    max = (m2, eigenvals_new[ind]);
	}
	m2 += 1;
    }
    // pivot position (j, k)
    let mut k = max.0;
    let mut j = maxima[k - 1];
    ind = (k * (k + 1)) / 2 + j;
    let mut biggest_new = eigenvals_new[ind].abs();
    let (mut count, mut first1, mut first2) = (0, true, true);
    let max_rotations = MAX_SWEEPS * dim * (dim - 1) / 2;
    loop {
	if count == max_rotations {
	    panic!("jacobi_eigen_cach failed to converge after {} sweeps", MAX_SWEEPS);
	}
	count += 1;
	if first1 {
	    first1 = false;
	} else {
	    m1 = 1;
	    max_diag = eigenvals_new[0].abs();
	    for i in &mut maxima {
		if eigenvals_new[(m1 * m1 + 3 * m1) / 2].abs() > max_diag {
		    max_diag = eigenvals_new[(m1 * m1 + 3 * m1) / 2].abs();
		}
		ind = (m1 * (m1 + 1)) / 2 + *i;
		if m1 == k || m1 == j {
		    *i = index_max_in_col(eigenvals_new, m1);
		} else if (*i != k && k < m1) && (*i != j && j < m1) {
		    let (val_j, val_k) =
			(eigenvals_new[(m1 * (m1 + 1)) / 2 + j].abs(), eigenvals_new[(m1 * (m1 + 1)) / 2 + k].abs());
		    let val_ind = eigenvals_new[ind].abs();
		    if val_j > val_ind {
			if val_k > val_j {
			    *i = k;
			} else {
			    *i = j;
			}
		    } else if val_k > val_ind {
			*i = k;
		    }
		} else if eigenvals_new[ind].abs() < biggest {
		    *i = index_max_in_col(eigenvals_new, m1);
		} else if eigenvals_new[(m1 * (m1 + 1)) / 2 + j].abs() > eigenvals_new[(m1 * (m1 + 1)) / 2 + k].abs() {
		    *i = j;
		} else {
		    *i = k;
		}
		m1 += 1;
	    }
	}
	if first2 {
	    first2 = false
	} else {
	    m2 = 1;
	    for i in &maxima {
		ind = (m2 * (m2 + 1)) / 2 + *i;
		if m2 == 1 || eigenvals_new[ind].abs() > (max.1).abs() {
		    max = (m2, eigenvals_new[ind]);
		}
		m2 += 1;
	    }
	    k = max.0;
	    j = maxima[k - 1];
	}
	if scaled_near_zero(eigenvals_new[(k * (k + 1)) / 2 + j], 1.0_f64.max(max_diag), 1e-12) {
	    return (diag_of_sym_mat(eigenvals_new, dim), eigenvects_new.to_vec())
	} else {
	    (eigenvals, eigenvects, biggest) = (eigenvals_new, eigenvects_new, biggest_new);
	    (eigenvals_new, eigenvects_new, biggest_new) = jacobi_rot(eigenvals, eigenvects, dim, k, j);
	}
    }
}

// second Jacobi variant, with each sweep taking cubic worst-case time by cyclically moving through columns
pub fn jacobi_eigen_cyc(a : &mut SymMatrix, dim : usize) -> (Vector, Matrix) {
    if dim == 1 {
	return (vec![a[0]], vec![1.0]);
    }
    let (mut eigenvals_new, mut eigenvects_new) = (a, &mut ident_mat(dim));
    let (mut eigenvals, mut eigenvects) : (&mut SymMatrix, &mut Matrix);
    let mut count = 0;
    loop {
	if count == MAX_SWEEPS {
	    panic!("jacobi_eigen_cyc failed to converge after {} sweeps", MAX_SWEEPS);
	}
	count += 1;
	// column-cyclic ordering of pivot positions
	for k in 1..dim {
	    for j in 0..k {
		if eigenvals_new[(k * (k + 1)) / 2 + j] == 0.0 {
		    continue;
		}
		(eigenvals, eigenvects) = (eigenvals_new, eigenvects_new);
		(eigenvals_new, eigenvects_new, _) = jacobi_rot(eigenvals, eigenvects, dim, k, j)
	    }
	}
	let (mut max_diag, mut max_offdiag) = (eigenvals_new[0].abs(), 0.0);
	for n in 0..dim {
	    for m in 0..n {
		if eigenvals_new[(n * (n + 1)) / 2 + m].abs() > max_offdiag {
		    max_offdiag = eigenvals_new[(n * (n + 1)) / 2 + m].abs();
		}
	    }
	    if eigenvals_new[(n * n + 3 * n) / 2].abs() > max_diag {
		max_diag = eigenvals_new[(n * n + 3 * n) / 2].abs();
	    }
	} 
	if scaled_near_zero(max_offdiag, 1.0_f64.max(max_diag), 1e-12) {
	    return (diag_of_sym_mat(eigenvals_new, dim), eigenvects_new.to_vec())
	}
    }
}

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

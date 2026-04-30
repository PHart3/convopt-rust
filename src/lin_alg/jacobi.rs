// Jacobi eigenvalue algorithm with each sweep taking cubic time on average

use crate::lin_alg::sym_matrix::*;

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

// Jacobi rotation subroutine (where orth stores the eventual eigenvectors)
fn jacobi_rot<'a>(mat : &'a mut SymMatrix, orth : &'a mut Matrix, k : usize, j : usize) -> (&'a mut SymMatrix, &'a mut Matrix, f64) {
    let dim = orth.len();
    let ind = (k * (k + 1)) / 2 + j;
    let val_old = mat[ind];
    let (ind_j, ind_k) = ((j * j + 3 * j) / 2, (k * k + 3 * k) / 2);
    let tau = (mat[ind_k] - mat[ind_j]) / (2.0 * val_old);
    let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
    let c = (1.0 + t * t).sqrt().recip();
    let s = c * t;
    mat[ind_k] += t * val_old; // mat[ind_j] * s.powi(2) + mat[ind_k] * c.powi(2) - 2.0 * s * c * val_old;
    mat[ind_j] -= t * val_old; // mat[ind_j] * c.powi(2) + mat[ind_k] * s.powi(2) + 2.0 * s * c * val_old;
    mat[ind] = 0.0;
    
    let (mut val_old_k, mut val_old_j, mut ind_mj, mut ind_mk);
    for m in 0..dim {
	if m == j || m == k { continue;	}
	ind_mk = if m > k { (m * (m + 1)) / 2 + k } else { (k * (k + 1)) / 2 + m };
	ind_mj = if m > j { (m * (m + 1)) / 2 + j } else { (j * (j + 1)) / 2 + m };
	val_old_k = mat[ind_mk];
	val_old_j = mat[ind_mj];
	mat[ind_mk] = c * val_old_k + s * val_old_j;
	mat[ind_mj] = c * val_old_j - s * val_old_k;
    }
    
    let tmp_col = orth[j].clone();
    orth[j] = vect_subt(&scal_vect(c, &orth[j]), &scal_vect(s, &orth[k]));
    orth[k] = vect_add(&scal_vect(s, &tmp_col), &scal_vect(c, &orth[k]));
    (mat, orth, val_old.abs())
}

const MAX_SWEEPS: usize = 50;

// Jacobi method computing spectral decomposition (L, Q) of real symmetric matrix via Jacobi rotations
// stores eigenvalues in L and corresponding eigenvectors in Q
pub fn jacobi_eigen(a : &mut SymMatrix, dim : usize) -> (Vector, Matrix) {
    if dim == 1 {
	return (vec![a[0]], vec![vec![1.0]]);
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
    let mut k = max.0;
    let mut j = maxima[k - 1];
    ind = (k * (k + 1)) / 2 + j;
    let mut biggest_new = eigenvals_new[ind].abs();
    let (mut count, mut first1, mut first2) = (0, true, true);
    let max_rotations = MAX_SWEEPS * dim * (dim - 1) / 2;
    loop {
	if count == max_rotations {
	    panic!("jacobi_eigen failed to converge after {} sweeps", MAX_SWEEPS);
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
		    *i = index_max_in_col(&mut eigenvals_new, m1);
		} else if (*i != k && k < m1) && (*i != j && j < m1) {
		    let (val_j, val_k) = (eigenvals_new[(m1 * (m1 + 1)) / 2 + j].abs(), eigenvals_new[(m1 * (m1 + 1)) / 2 + k].abs());
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
		    *i = index_max_in_col(&mut eigenvals_new, m1);
		} else {
		    if eigenvals_new[(m1 * (m1 + 1)) / 2 + j].abs() > eigenvals_new[(m1 * (m1 + 1)) / 2 + k].abs() {
			*i = j;
		    } else {
			*i = k;
		    }
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
	if eigenvals_new[(k * (k + 1)) / 2 + j].abs() < 1e-12 * 1.0_f64.max(max_diag) {
	    return (diag_of_sym_mat(eigenvals_new, dim), eigenvects_new.to_vec())
	} else {
	    (eigenvals, eigenvects, biggest) = (eigenvals_new, eigenvects_new, biggest_new);
	    (eigenvals_new, eigenvects_new, biggest_new) = jacobi_rot(eigenvals, eigenvects, k, j);
	}
    }
}

// partitioning the above spectral decomposition into nonnegative and negative parts
pub fn eigendecomp(a : &mut SymMatrix, dim : usize) -> ((Vector, Matrix), (Vector, Matrix)) {
    
    let (mut nonneg_eigenvals, mut nonneg_eigenvects) : (Vector, Matrix) = (Vec::new(), Vec::new());
    let (mut neg_eigenvals, mut neg_eigenvects) : (Vector, Matrix) = (Vec::new(), Vec::new());
    
    let (eigenvals, eigenvects) = jacobi_eigen(a, dim);

    for (val, vect) in eigenvals.iter().zip(eigenvects.into_iter()) {
	if val.is_sign_negative() {
	    neg_eigenvals.push(*val);
	    neg_eigenvects.push(vect);
	} else {
	    nonneg_eigenvals.push(*val);
	    nonneg_eigenvects.push(vect);
	}
    }
    ((nonneg_eigenvals, nonneg_eigenvects), (neg_eigenvals, neg_eigenvects))
}

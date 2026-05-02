// operations on symmetric matrices
// a symmetric matrix is represented by a flat vector containing its upper triangular part in column-major style
pub use crate::lin_alg::matrix::*;
pub type SymMatrix = Vector;

// Frobenius inner product of two symmetric matrices
pub fn frob_prod_sym(mat1 : &SymMatrix, mat2 : &SymMatrix, dim : usize) -> f64 {
    let (mut sum, mut start) = (0.0, 0);
    for n in 0..dim {
	start += n;
	for i in 0..n {
	    sum += mat1[start + i] * mat2[start + i] * 2.0;
	}
	sum += mat1[start + n] * mat2[start + n];
    }
    sum
}

// Frobenius norm of a symmetric matrix
pub fn frob_norm_sym(mat : &SymMatrix, dim : usize) -> f64 {
    frob_prod_sym(mat, mat, dim).sqrt()
}

// Frobenius distance between two symmetric matrices
pub fn frob_dist_sym(mat1 : &SymMatrix, mat2 : &SymMatrix, dim : usize) -> f64 {
    frob_norm_sym(&vect_subt(mat1, mat2), dim)
}

// 1-norm of a symmetric matrix
pub fn sym_mat_one_norm(mat : &SymMatrix, dim : usize) -> f64 {
    let mut row_sums = vec![0.0; dim];
    let mut idx = 0;
    for i in 0..dim {
	for j in 0..=i {
	    if j == i {
		row_sums[j] += mat[idx].abs();
	    } else {
		row_sums[i] += mat[idx].abs();
		row_sums[j] += mat[idx].abs();
	    }
	    idx += 1;
	}
    }
    row_sums.iter().fold(0.0, |acc, v| { if *v > acc { *v } else { acc } })
}

// identity symmetric matrix
pub fn ident_sym_mat(dim : usize) -> SymMatrix {
    let mut result : SymMatrix = Vec::with_capacity((dim * (dim + 1)) / 2);
    for i in 0..dim {
	for j in 0..=i {
	    if j < i {
		result.push(0.0);
	    } else {
		result.push(1.0);
	    }
	}
    }
    result
}

// scaling all off-diagonal elements of symmetric matrix
pub fn scale_off_diag(s : f64, dim : usize, mat : &mut SymMatrix) -> &mut SymMatrix {
    let mut idx = 0;
    for c in 0..dim {
	for _ in 0..c {
	    mat[idx] *= s;
	    idx += 1;
	}
	idx += 1;
    }
    mat
}

// returns the upper half of an arbitrary outer product
pub fn outer_prod_upper(vect1 : &Vector, vect2 : &Vector) -> SymMatrix {
    let mut result = Vec::new();
    let dim = vect1.len();
    for c in 0..dim {
	for r in 0..=c {
	    result.push(vect1[r] * vect2[c]);
	}
    }
    result
}

// returns outer product of a vector with itself
pub fn outer_prod_single(vect : &Vector) -> SymMatrix {
    let mut result = Vec::new();
    let dim = vect.len();
    for c in 0..dim {
	for r in 0..=c {
	    result.push(vect[r] * vect[c]);
	}
    }
    result
}

// returns outer product of a scaled vector sv with v
pub fn outer_prod_single_scale(s : f64, vect : &Vector) -> SymMatrix {
    let mut result = Vec::new();
    let dim = vect.len();
    for c in 0..dim {
	for r in 0..=c {
	    result.push(s * vect[r] * vect[c]);
	}
    }
    result
}

// returns diagonal vector of n-dimensional symmetric matrix
pub fn diag_of_sym_mat(mat : &SymMatrix, dim : usize) -> Vector {
    let mut result = Vec::with_capacity(dim);
    for j in 0..dim {
	result.push(mat[(j * j + 3 * j) / 2]);
    }
    result
}

// returns linear map computing diagonal block of symmetric matrix specified by starting position and dimension of block
// returns, in increasing row dimension, the column entries of the map that equal one (the rest being zero)
pub fn sym_matrix_diag_block_map(start : usize, dim : usize) -> Vec<usize> {
    let mut result = Vec::with_capacity((dim * (dim + 1)) / 2);
    let mut k = start;
    let mut first;
    for i in 0..dim {
	first = (k * (k + 1)) / 2 + start;
	for j in 0..=i {
	    result.push(first + j);
	}
	k += 1;
    }
    result
}

// returns the actual diagonal block of symmetric matrix
pub fn sym_matrix_diag_block(start : usize, dim : usize, mat : &SymMatrix) -> SymMatrix {
    let mut result : SymMatrix = Vec::with_capacity((dim * (dim + 1)) / 2);
    let mut k = start;
    let mut first;
    for i in 0..dim {
	first = (k * (k + 1)) / 2 + start;
	for j in 0..=i {
	    result.push(mat[first + j]);
	}
	k += 1;
    }
    result
}

// returns linear map computing diagonal of a symmetric matrix
pub fn sym_matrix_diagonal(dim : usize) -> Matrix {
    let size = (dim * (dim + 1)) / 2;
    let mut result = zero_mat(size, dim);
    for i in 0..dim {
	result[(i * i + 3 * i) / 2][i] = 1.0;
    }
    result
}

// composition with diagonal block map
pub fn matrix_diagonal_mult_sym(sym_dim : usize, mat : &Matrix, start : usize, dim : usize) -> Matrix {
    let mut result : Matrix = Vec::new();
    let row_dim = mat[0].len();
    let ones = sym_matrix_diag_block_map(start, dim);
    let (mut offset, mut i) = (0, 0);
    for c in ones {
	for _ in offset..c {
	    result.push(vec![0.0; row_dim]);
	}
	result.push(mat[i].clone());
	offset = c + 1;
	i += 1;
    }
    for _ in offset..((sym_dim * (sym_dim + 1)) / 2) {
	result.push(vec![0.0; row_dim]);
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

// the constraint matrix for block-diagonal decision variable Z is represented implicitly by a matrix M and a vector V
// each row of M is a symmetric matrix
// V is a list of pairs (row, column), with column >= row, representing entries of Z that must be zero
// the full constraint matrix is formed by stacking M on top of V

// action of constraint matrix on decision variable
pub fn constraint_action((mat, list) : &(&Matrix, &Vec<(usize, usize)>), var : &SymMatrix) -> Vector {
    if mat.is_empty() {
	return vec![]
    }
    let mut result = scal_vect(var[0], &mat[0]);
    for (n, col) in mat.iter().enumerate().skip(1) {
	result = vect_add(&result, &scal_vect(var[n], col));
    }
    for (r, c) in *list {
	result.push(var[(c * (c + 1)) / 2 + r]);
    }
    result
}

// linear combination of the symmetric matrices making up the constraint matrix
// forms the adjoint operator to constraint_action (above) under the Frobenius inner product
pub fn constraint_lin_comb(dim : usize, (mat, list) : &(&Matrix, &Vec<(usize, usize)>), vect : &Vector) -> SymMatrix {
    if mat.is_empty() {
	return vec![0.0; dim * (dim + 1) / 2]
    }
    let mut result = Vec::new();
    let mut start = 0;
    let mut diag;
    for i in 0..dim {
	diag = (i * i + 3 * i) / 2;
	for j in start..diag {
	    result.push(0.5 * dot_prod(&mat[j], vect));
	}
	result.push(dot_prod(&mat[diag], vect));
	start += i + 1;
    }
    let k = mat[0].len();
    for (start, &(r, c)) in list.iter().enumerate() {
	result[(c * (c + 1)) / 2 + r] += 0.5 * vect[k + start];
    }
    result
}

// computation of Gram matrix of a constraint matrix (i.e., its composite with its adjoint)
// we ignore the off-diagonal zero constraints because we only need the upper left block of the Gram matrix
pub fn constraint_gram(dim : usize, mat : &Matrix) -> SymMatrix {
    if mat.is_empty() {
	return vec![];
    }
    let gram_dim = mat[0].len();
    let mut result : SymMatrix = vec![0.0; gram_dim * (gram_dim + 1) / 2];
    let mut start = 0;
    let mut diag;
    for i in 0..dim {
	diag = (i * i + 3 * i) / 2;
	for j in start..diag {
	    result = vect_add_scaled(&result, 0.5, &outer_prod_single(&mat[j]));
	}
	result = vect_add(&result, &outer_prod_single(&mat[diag]));
	start += i + 1;
    }
    result
}

// optimized computation of QDQ^T where Q is any matrix and D is diagonal
pub fn diag_scale_gram(d : &Vector, q : &Matrix) -> SymMatrix {
    let len = q[0].len();
    let mut result = vec![0.0; len * (len + 1) / 2]; 
    for (s, c) in d.iter().zip(q) {
	result = vect_add(&result, &outer_prod_single_scale(*s, c));
    }
    result
}

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
		sum += l[k * (2 * dim - k + 1) / 2 + (i - k)] * l[k * (2 * dim - k + 1) / 2 + (j - k)] * d[k];
	    }
	    l[j * (2 * dim - j + 1) / 2 + (i - j)] = (mat[(i * (i + 1)) / 2 + j] - sum) / d[j]
	}
	
	sum = 0.0;
	for k in 0..i {
	    ind = k * (2 * dim - k + 1) / 2 + (i - k);
	    sum += l[ind] * l[ind] * d[k];
	}
	d.push(mat[(i * i + 3 * i) / 2] - sum);

	if d[i] < TOL * sym_mat_one_norm(mat, dim).max(1.0) {
	    return None;
	}
    }
    Some((l, d))
}

// check whether a block-diagonal symmetric matrix S is positive definite by running LDLT on S + eps * I
pub fn pos_def_block_check(mat : &SymMatrix, blocks : &Vec<(usize, usize)>) -> bool {
    let (mut eps, mut idx) : (f64, usize);
    for (start, dim) in blocks {
	let block = &sym_matrix_diag_block(*start, *dim, mat);
	eps = 1e-8 * sym_mat_one_norm(block, *dim).max(1.0);
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

// diagonal linear system solver
pub fn diag_solver(diag : &Vector, vect : &Vector) -> Vector {
    let mut solution = Vec::new();
    for (v, d) in vect.iter().zip(diag.iter()) {
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
	if d.abs() < TOL * scale {
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
	if d.abs() < TOL * scale {
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

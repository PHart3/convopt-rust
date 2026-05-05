// various basic matrix operations

// column vector
pub type Vector = Vec<f64>;
// column-major matrix
pub type Matrix = Vec<Vector>;
// default tolerance level
pub const TOL: f64 = 1e-10;

// creating a column-major square matrix of a specified dimension
pub fn mk_sqmatrix(dim : usize, entries : Vector) -> Matrix {
    let mut result = Vec::with_capacity(dim);
    let mut start : usize;
    for i in 0..dim {
	start = dim * i;
	result.push(entries[start..start + dim].to_vec());	
    }
    result
}

// turning matrix into vector
pub fn matrix_to_vector(mat : &Matrix) -> Vector {
    let mut result : Vector = Vec::new();
    for col in mat {
	result.extend(col);
    }
    result
}

// turning additive inverse of matrix into vector
pub fn matrix_to_vector_inv(mat : &Matrix) -> Vector {
    let mut result : Vector = Vec::new();
    for col in mat {
	for v in col {
	    result.push(-v);
	}
    }
    result
}

// standard basis vectors
pub fn standard_basis(i : usize, dim : usize) -> Vector {
    let mut result : Vector = Vec::with_capacity(dim);
    for j in 0..dim {
	if j == i {
	    result.push(1.0);
	} else {
	    result.push(0.0);
	}
    }
    result
}

// identity matrix
pub fn ident_mat(dim : usize) -> Matrix {
    let mut result : Matrix = Vec::with_capacity(dim);
    for i in 0..dim {
	result.push(standard_basis(i, dim));
    }
    result
}

// zero matrix
pub fn zero_mat(dim_c : usize, dim_r : usize) -> Matrix {
    let mut result : Matrix = Vec::with_capacity(dim_c);
    for _ in 0..dim_c {
	result.push(vec![0.0; dim_r]);
    }
    result
}

// standard inner product
pub fn dot_prod(vect1 : &Vector, vect2 : &Vector) -> f64 {
    let mut prod = 0.0;
    for (e1, e2) in vect1.iter().zip(vect2.iter()) {
	prod = (*e1).mul_add(*e2, prod);
    }
    prod
}

// Euclidean norm
pub fn euclid_norm(vect : &Vector) -> f64 {
    dot_prod(vect, vect).sqrt()
}

// Euclidean distance
pub fn euclid_distance(vect1 : &Vector, vect2 : &Vector) -> f64 {
    let mut sum = 0.0;
    let mut diff;
    for (v1, v2) in vect1.iter().zip(vect2.iter()) {
	diff = v1 - v2;
	sum = diff.mul_add(diff, sum);
    }
    sum.sqrt()
}

// vector addition
pub fn vect_add(vect1 : &Vector, vect2 : &Vector) -> Vector {
    let mut sum : Vector = Vec::new(); 
    for (v1, v2) in vect1.iter().zip(vect2.iter()) {
	sum.push(v1 + v2);
    }
    sum
}

// vector subtraction
pub fn vect_subt(vect1 : &Vector, vect2 : &Vector) -> Vector {
    let mut sum : Vector = Vec::new(); 
    for (v1, v2) in vect1.iter().zip(vect2.iter()) {
	sum.push(v1 - v2);
    }
    sum
}

// scalar action on vector
pub fn scal_vect(s : f64, vect : &Vector) -> Vector {
    let mut result : Vector = Vec::new();
    for v in vect {
	result.push(s * v);
    }
    result
}

// inverse scalar action on vector
pub fn negscal_vect(s : f64, vect : &Vector) -> Vector {
    let mut result : Vector = Vec::new();
    for v in vect {
	result.push(-(s * v));
    }
    result
}

// computing vector of form vect1 + s * vect2
pub fn vect_add_scaled(vect1 : &Vector, s : f64, vect2 : &Vector) -> Vector {
    let mut result : Vector = Vec::new(); 
    for (v1, v2) in vect1.iter().zip(vect2.iter()) {
	result.push(s.mul_add(*v2, *v1));
    }
    result
}

// matrix addition
pub fn matrix_add(mat1 : &Matrix, mat2 : &Matrix) -> Matrix {
    let (dim_c, dim_r) = (mat1.len(), mat1[0].len());
    let mut sum : Matrix = zero_mat(dim_c, dim_r);
    for (n, (col1, col2)) in mat1.iter().zip(mat2.iter()).enumerate() {
	for (m, (v1, v2)) in col1.iter().zip(col2.iter()).enumerate() {
            sum[n][m] = v1 + v2;
	}
    }
    sum
}

// matrix subtraction
pub fn matrix_subt(mat1 : &Matrix, mat2 : &Matrix) -> Matrix {
    let (dim_c, dim_r) = (mat1.len(), mat1[0].len());
    let mut diff : Matrix = zero_mat(dim_c, dim_r);
    for (n, (col1, col2)) in mat1.iter().zip(mat2.iter()).enumerate() {
	for (m, (v1, v2)) in col1.iter().zip(col2.iter()).enumerate() {
            diff[n][m] = v1 - v2;
	}
    }
    diff
}

// Gaussian elimination via partial pivoting
// also computes rank and consistency of given augmented matrix
pub fn gauss_elim(mat : &mut Matrix) -> (&mut Matrix, usize, bool) {
    let (col_dim, row_dim) = (mat.len(), mat[0].len());
    let (mut row_search, mut coeff_rank, mut pivot_row);
    (row_search, coeff_rank) = (0, 0);
    let scale = mat.iter().flat_map(|col| col.iter()).map(|x| x.abs()).fold(0.0, f64::max);
    for n in 0..col_dim {
	if row_search >= row_dim {
	    break;
	}
	pivot_row = row_search;
	for j in row_search + 1..row_dim {
	    if mat[n][j].abs() > mat[n][pivot_row].abs() {
		pivot_row = j;
	    }
	}
	if mat[n][pivot_row].abs() < TOL * scale {
	    continue;
	}
	if pivot_row != row_search {
	    for m3 in 0..col_dim {
		mat[m3].swap(row_search, pivot_row);
	    }
	}
	let denom = mat[n][row_search];
	for j in row_search + 1..row_dim {
	    let piv = mat[n][j] / denom;
	    for m in 0..col_dim {
		mat[m][j] -= piv * mat[m][row_search];	
	    }
	}
	row_search += 1;
	if n < col_dim - 1 {
	    coeff_rank += 1;
	}
    }
    (mat, row_search, row_search == coeff_rank)
}

// column-major lower triangular matrix stored as a flat vector
pub type LowTriMatrix = Vector;

// lower triangular identity matrix
pub fn ident_mat_low_tri(dim : usize) -> LowTriMatrix {
    let mut result = Vec::new();
    for i in 0..dim {
	result.push(1.0);
	for _ in 1..dim - i {
	    result.push(0.0);
	}
    }
    result
}

pub fn float_equality(f1: f64, f2: f64, tol: f64) -> bool {
    let diff = (f1 - f2).abs();
    diff <= tol || diff <= tol * f2.abs().max(f1.abs())
}

// testing if two vectors are the same
pub fn vector_equality(vect1 : &Vector, vect2 : &Vector, tol : f64) -> bool {
    if vect1.len() != vect2.len() {
	return false;
    } else {
	for (v1, v2) in vect1.iter().zip(vect2.iter()) {
	    if ! (float_equality(*v1, *v2, tol)) {
		return false;
	    }
	}
    }
    true
}

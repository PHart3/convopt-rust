// various basic matrix operations

// column vector
pub type Vector = Vec<f64>;
// column-major matrix stored as a flat vector
pub type Matrix = Vector;
// default tolerance level
pub const TOL: f64 = 1e-10;

// standard basis vectors
pub fn standard_basis(i: usize, dim: usize) -> Vector {
    let mut result: Vector = Vec::with_capacity(dim);
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
pub fn ident_mat(dim: usize) -> Matrix {
    let mut result: Matrix = Vec::with_capacity(dim * dim);
    for i in 0..dim {
        result.extend(standard_basis(i, dim));
    }
    result
}

// zero matrix
pub fn zero_mat(dim_c: usize, dim_r: usize) -> Matrix {
    vec![0.0; dim_c * dim_r]
}

// standard inner product
pub fn dot_prod(vect1: &[f64], vect2: &[f64]) -> f64 {
    let mut prod = 0.0;
    for (e1, e2) in vect1.iter().zip(vect2.iter()) {
        prod = (*e1).mul_add(*e2, prod);
    }
    prod
}

// Euclidean norm
pub fn euclid_norm(vect: &[f64]) -> f64 {
    dot_prod(vect, vect).sqrt()
}

// Euclidean distance
pub fn euclid_distance(vect1: &[f64], vect2: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut diff;
    for (v1, v2) in vect1.iter().zip(vect2.iter()) {
        diff = v1 - v2;
        sum = diff.mul_add(diff, sum);
    }
    sum.sqrt()
}

// vector addition
pub fn vect_add(vect1: &Vector, vect2: &Vector) -> Vector {
    let mut sum: Vector = Vec::new();
    for (v1, v2) in vect1.iter().zip(vect2.iter()) {
        sum.push(v1 + v2);
    }
    sum
}

// vector subtraction
pub fn vect_subt(vect1: &Vector, vect2: &Vector) -> Vector {
    let mut sum: Vector = Vec::new();
    for (v1, v2) in vect1.iter().zip(vect2.iter()) {
        sum.push(v1 - v2);
    }
    sum
}

// scalar action on vector
pub fn scal_vect(s: f64, vect: &Vector) -> Vector {
    let mut result: Vector = Vec::new();
    for v in vect {
        result.push(s * v);
    }
    result
}

// inverse scalar action on vector
pub fn negscal_vect(s: f64, vect: &Vector) -> Vector {
    let mut result: Vector = Vec::new();
    for v in vect {
        result.push(-(s * v));
    }
    result
}

// computing vector of form vect1 + s * vect2
pub fn vect_add_scaled(vect1: &Vector, s: f64, vect2: &Vector) -> Vector {
    let mut result: Vector = Vec::new();
    for (v1, v2) in vect1.iter().zip(vect2.iter()) {
        result.push(s.mul_add(*v2, *v1));
    }
    result
}

// evaluates the matrix expression A + (B - C)
pub fn matrix_add_diff(mat1: &Matrix, mat2: &Matrix, mat3: &Matrix) -> Matrix {
    let mut sum: Matrix = Vec::with_capacity(mat1.len());
    for (v1, (v2, v3)) in mat1.iter().zip(mat2.iter().zip(mat3.iter())) {
        sum.push(v1 + (v2 - v3));
    }
    sum
}

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
        if mat[n * row_dim + pivot_row].abs() < TOL * scale {
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

// column-major lower triangular matrix stored as a flat vector
pub type LowTriMatrix = Vector;

// lower triangular identity matrix
pub fn ident_mat_low_tri(dim: usize) -> LowTriMatrix {
    let mut result = Vec::new();
    for i in 0..dim {
        result.push(1.0);
        for _ in 1..dim - i {
            result.push(0.0);
        }
    }
    result
}

// testing if two vectors are the same

pub fn float_equality(f1: f64, f2: f64, tol: f64) -> bool {
    let diff = (f1 - f2).abs();
    diff <= tol || diff <= tol * f2.abs().max(f1.abs())
}

pub fn vector_equality(vect1: &Vector, vect2: &Vector, tol: f64) -> bool {
    if vect1.len() != vect2.len() {
        return false;
    } else {
        for (v1, v2) in vect1.iter().zip(vect2.iter()) {
            if !(float_equality(*v1, *v2, tol)) {
                return false;
            }
        }
    }
    true
}

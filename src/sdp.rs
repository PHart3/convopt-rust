pub use crate::lin_alg::sym_matrix::*;

// linear map on symmetric matrices that is designed for user input
type MatrixSym = Vec<Vec<Vector>>;

// semidefinite programming problem
pub struct SDP {
    objective : MatrixSym,
    constraint : (Vec<MatrixSym>, Vector),
    lmi : Vec<(Vec<MatrixSym>, Matrix)>
}

impl SDP {   
    pub fn new(
	objective : MatrixSym,
	constraint : (Vec<MatrixSym>, Vector),
	lmi : Vec<(Vec<MatrixSym>, Matrix)>
    ) -> Self {
	Self { objective, constraint, lmi }
    }
    pub fn objective(&self) -> &MatrixSym {
        &self.objective
    }
    pub fn constraint(&self) -> &(Vec<MatrixSym>, Vector) {
        &self.constraint
    }
    pub fn lmi(&self) -> &Vec<(Vec<MatrixSym>, Matrix)> {
        &self.lmi
    }
}

// flatten the rows of a MatrixSym
fn matrixsym_flatten(mat : &MatrixSym) -> Matrix {
    let dim_c = mat.first().unwrap_or(&vec![]).len();
    let mut result : Matrix = Vec::with_capacity(mat.len() * (dim_c * (dim_c + 1)) / 2);
    for i in 0..dim_c {
	for j in 0..=i {
	    for row in mat {
		result.push(row[i][j]);
	    }
	}
    }
    result
}

// converting a user-given SDP to one in standard form (c, A, b)
// also stores the dimension of the final decision variable,
// the dimensions of any smaller input decision variables,
// and the dimensions of the block constraints

// takes a boolean flag "check_constraints" to decide whether to
// run Gaussian elimination on dense constraint matrix
pub fn sdp_to_standard(sdp : &SDP, check_constraints : bool) ->
    (SymMatrix, (Matrix, Vec<(usize, usize)>), Vector, usize, Vec<usize>, Vec<usize>)
{
    if sdp.lmi.is_empty() {
	assert!(sdp.objective.len() == 1,
		"you must supply exactly one objective map since you have exactly one decision variable");
	if sdp.constraint.0.is_empty() {
	    let obj = &sdp.objective[0];
	    (obj.concat(), (vec![], vec![]), vec![], obj.last().expect("you have not provided an objective function").len(),
	     vec![], vec![])
	} else if sdp.constraint.0.len() == 1 {
	    let obj = &sdp.objective[0];
	    let total_dim = obj.last().expect("you have not provided an objective function").len();
	    if check_constraints {
		// make augmented constraint matrix full rank
		let mut constraints = matrixsym_flatten(&(sdp.constraint.0)[0]);
		constraints.extend_from_slice(&sdp.constraint.1);
		let mut constraints_red = linear_remove_redundant_sym(&mut constraints, total_dim);
		let rank = constraints_red.len() / (((total_dim * (total_dim + 1)) / 2) + 1);
		let point = constraints_red.split_off(((total_dim * (total_dim + 1)) / 2) * rank);

		(obj.concat(), (constraints_red, vec![]), point, total_dim, vec![], vec![])
	    } else {
		let (constraints, point) = (matrixsym_flatten(&(sdp.constraint.0)[0]), sdp.constraint.1.clone());
		(obj.concat(), (constraints, vec![]), point, total_dim, vec![], vec![])
	    }
	} else {
	    panic!("you have one decision variable but have supplied constraint maps for more than one decision variable");
	}
    } else {
	let lmis = &sdp.lmi;
	let (mut block_dims, mut symm_dims) = (Vec::new(), Vec::new());
	let mut first = true;
	let mut sequence_len = 0;
	let (mut block_size, mut symm_size);
	let (mut block_size_sum, mut symm_size_sum) = (0, 0);
	let mut map_len_old : Option<usize> = None;
	for (maps, mat) in lmis {
	    block_size = mat.len().isqrt();
	    assert!(block_size > 0 && block_size * block_size == mat.len(),
		    "the constant matrix for each LMI must be nonempty and square");
	    sequence_len += 1;
	    block_dims.push(block_size);
	    block_size_sum += block_size;
	    if first {
		map_len_old = Some(maps.len());
		for map in maps {		    
		    symm_size = map[0].last().unwrap_or_else(|| {
			panic!("first row of some matrix in affine combination {} is empty", sequence_len) }).len();
		    symm_dims.push(symm_size);
		    symm_size_sum += symm_size; 
		}
	    } else {
		assert_eq!(maps.len(), map_len_old.expect("never initialized"),
			   "each LMI must have the same number of decision variables");
		map_len_old = Some(maps.len());
	    }
	    first = false;
	}
	let total_dim = block_size_sum + 2 * symm_size_sum;
	let total_size = (total_dim * (total_dim + 1)) / 2;
	
	let mut blocks : Vec<(Matrix, Vector)> = Vec::with_capacity(sequence_len + 1);
	let mut sum : Matrix;
	let (mut symm_offset, mut dim);
	let (mut map_flatten, mut block_current);
	let (mut block_offset, mut b) = (0, 0);
	let mut row;
	for (maps, mat) in lmis {
	    symm_offset = 0;
	    block_size = block_dims[b];
	    sum = zero_mat(total_size, block_size * block_size);
	    for (n, map) in maps.iter().enumerate() {
		if block_size * block_size != map.len() {
		    println!("block size = {} and rows = {}", block_size, map.len());
		    panic!("the size of map {} in LMI {} does not match the size of the constant matrix", n, b);
		}
		dim = symm_dims[n];
		map_flatten = matrixsym_flatten(map);
		sum = matrix_add_diff(&sum,
				      &matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset, dim),
				      &matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset + dim, dim));
		symm_offset += 2 * dim;
	    }
	    block_current = sym_matrix_diag_block_map(block_offset, block_size);
	    
	    row = 0;
	    for c in 0..block_size {
		for r in 0..block_size {
		    let packed = if r <= c {
			block_current[(c * (c + 1)) / 2 + r]
		    } else {
			block_current[(r * (r + 1)) / 2 + c]
		    };
		    sum[packed * block_size * block_size + row] -= 1.0;
		    row += 1;
		}
	    }
	    blocks.push((sum, mat.clone()));
	    block_offset += block_size;
	    b += 1;
	}

	let constr_len = sdp.constraint.1.len();
	let mut obj_sum : SymMatrix = vec![0.0; total_size];
	let mut constr_sum : Matrix;
	let mut obj_flatten : SymMatrix;
	symm_offset = 0;
	if sdp.constraint.0.is_empty() {
	    assert!(sdp.constraint.1.is_empty(), "empty constraints but nonempty vector");
	    constr_sum = vec![];
	    for (n, obj) in sdp.objective.iter().enumerate() {
		dim = symm_dims[n];

		obj_flatten = obj.concat();
		let db1 = mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset, dim);
		let db2 = mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset + dim, dim);
		for ((s, v1), v2) in obj_sum.iter_mut().zip(db1.iter()).zip(db2.iter()) {
		    *s += *v1 - *v2;
		}
		symm_offset += 2 * dim;
	    }
	} else {
	    constr_sum = zero_mat(total_size, constr_len);
	    for (n, (obj, constr)) in (sdp.objective).iter().zip(sdp.constraint.0.iter()).enumerate() {
		dim = symm_dims[n];

		obj_flatten = obj.concat();
		let db1 = mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset, dim);
		let db2 = mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset + dim, dim);
		for ((s, v1), v2) in obj_sum.iter_mut().zip(db1.iter()).zip(db2.iter()) {
		    *s += *v1 - *v2;
		}

		map_flatten = matrixsym_flatten(constr);
		constr_sum = matrix_add_diff(&constr_sum,
					     &matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset, dim),
					     &matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset + dim, dim));

		symm_offset += 2 * dim;
	    }
	}
	blocks.push((constr_sum, sdp.constraint.1.clone()));

	// compute zero regions of constraint matrix
	let block_total = block_dims.iter().fold(0, |acc, k| acc + (k * (k + 1)) / 2);
	let symm_total = symm_dims.iter().fold(0, |acc, k| acc + (k * (k + 1)) / 2);
	let zeros_len = total_size - (block_total + 2 * symm_total);
	let mut zeros : Vec<(usize, usize)> = Vec::with_capacity(zeros_len);
	let (mut start, mut count) = (0, 0);
	for k in block_dims.iter().chain(symm_dims.iter()) {
	    for _ in 0..2 {
		for c in start + k..total_dim {
		    for r in start..start + k {
			zeros.push((r, c));
		    }
		}
		start += k;
		if count < sequence_len {
		    break;
		}
	    }
	    count += 1;
	}
	assert_eq!(start, total_dim);
	assert_eq!(zeros.len(), zeros_len);

	// compute augmented constraint matrix by stacking blocks
	let constraint_rows = blocks.iter().map(|(_, v)| v.len()).sum();
	let mut constraints : (Matrix, Vector) =
	    (Vec::with_capacity(total_size * constraint_rows), Vec::with_capacity(constraint_rows));
	for col in 0..total_size {
	    for (mat, vect) in &blocks {
		let row_dim = vect.len();
		constraints.0.extend_from_slice(&mat[col * row_dim..(col + 1) * row_dim]);
	    }
	}
	for (_, vect) in &blocks {
	    constraints.1.extend_from_slice(vect);
	}

	if check_constraints {
	    // make augmented constraint matrix full rank
	    constraints.0.extend_from_slice(&constraints.1);
	    let mut constraints_red = linear_remove_redundant_sym(&mut constraints.0, total_dim);
	    let rank = constraints_red.len() / (total_size + 1);
	    let mut point = constraints_red.split_off(total_size * rank);
	    point.append(&mut vec![0.0; zeros.len()]);

	    (obj_sum, (constraints_red, zeros), point, total_dim, symm_dims, block_dims)
	} else {
	    constraints.1.append(&mut vec![0.0; zeros.len()]);

	    (obj_sum, (constraints.0, zeros), constraints.1, total_dim, symm_dims, block_dims)
	}
    }
}

// make an instance of MatrixSym with specified input dimension and size of output vector (i.e, number of constraints)
pub fn mk_symmap(sym_dim : usize, output_size : usize, entries : Vector) -> MatrixSym {
    let mut result = Vec::with_capacity(output_size);
    let mut start = 0;
    let (mut diff, mut offset);
    let total = (sym_dim * (sym_dim + 1)) / 2;
    for _ in 0..output_size {
	let mut v = Vec::with_capacity(sym_dim);
	diff = 0;
	for n in 0..sym_dim {
	    offset = start + diff;
	    v.push(entries[offset..=offset + n].to_vec());
	    diff += n + 1;
	}
	result.push(v);
	start += total;
    }
    result
}

// zero MatrixSym
pub fn zero_symmap(sym_dim : usize, output_size : usize) -> MatrixSym {
    let mut result = Vec::with_capacity(output_size);
    for _ in 0..output_size {
	let mut v = Vec::with_capacity(sym_dim);
	for n in 0..sym_dim {
	    v.push(vec![0.0; n + 1]);
	}
	result.push(v);
    }
    result
}

// identity MatrixSym
pub fn ident_symmap(dim : usize) -> MatrixSym {
    let mut result = Vec::new();
    for n in 0..dim {
	for i in 0..n {
	    result.push(sym_matrix_proj(dim, i, n));
	}
	for i in n..dim {
	    result.push(sym_matrix_proj(dim, n, i));
	}
    }
    result
}

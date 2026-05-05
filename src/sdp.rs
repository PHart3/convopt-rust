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
    let mut result : Matrix = Vec::with_capacity((dim_c * (dim_c + 1)) / 2);
    let mut concat : Vector;
    for i in 0..dim_c {
	for j in 0..=i {
	    concat = mat.iter().fold(Vec::new(), |mut acc, row| {
		acc.push(row[i][j]);
		acc
	    });
	    result.push(concat);
	}
    }
    result
}

// converting a user-given SDP to one in standard form (c, A, b)
// also stores the dimension of the final decision variable,
// the dimensions of any smaller input decision variables,
// and the dimensions of the block constraints
pub fn sdp_to_standard(sdp : &SDP) -> (SymMatrix, (Matrix, Vec<(usize, usize)>), Vector, usize, Vec<usize>, Vec<usize>) {
    if sdp.lmi.is_empty() {
	assert!(sdp.objective.len() == 1,
		"you must supply exactly one objective map since you have exactly one decision variable");
	if sdp.constraint.0.is_empty() {
	    let obj = &sdp.objective[0];
	    (obj.concat(), (vec![], vec![]), vec![], obj.last().expect("you have not provided an objective function").len(),
	     vec![], vec![])
	} else if sdp.constraint.0.len() == 1 {
	    let obj = &sdp.objective[0];
	    // make augmented constraint matrix full rank
	    let mut constraints = matrixsym_flatten(&(sdp.constraint.0)[0]);
	    constraints.push(sdp.constraint.1.clone());
	    let total_dim = obj.last().expect("you have not provided an objective function").len();
	    let mut constraints_red = linear_remove_redundant_sym(&mut constraints, total_dim);
	    let point = constraints_red.pop().unwrap_or(vec![]);

	    (obj.concat(), (constraints_red, vec![]), point, total_dim, vec![], vec![])	    
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
	    assert_eq!(mat.len(), mat.last().expect("constant matrix in LMI must be nonempty").len(),
		       "the constant matrix for each LMI must be square");
	    sequence_len += 1;
	    block_size = mat.len();
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
		assert_eq!(maps.len(), map_len_old.expect("safely initialized"),
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
	let (mut map_flatten, mut block_diff, mut block_current);
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
		block_diff = matrix_subt(
		    &matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset, dim),
		    &matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset + dim, dim));
		sum = matrix_add(&sum, &block_diff);
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
		    sum[packed][row] -= 1.0;
		    row += 1;
		}
	    }
	    blocks.push((sum, matrix_to_vector(mat)));
	    block_offset += block_size;
	    b += 1;
	}

	let constr_len = sdp.constraint.1.len();
	let mut obj_sum : SymMatrix = vec![0.0; total_size];
	let mut constr_sum : Matrix;
	let (mut obj_flatten, mut obj_diff) : (SymMatrix, SymMatrix);
	symm_offset = 0;
	if sdp.constraint.0.is_empty() {
	    assert!(sdp.constraint.1.is_empty(), "empty constraints but nonempty vector");
	    constr_sum = vec![];
	    for (n, obj) in sdp.objective.iter().enumerate() {
		dim = symm_dims[n];

		obj_flatten = obj.concat();
		obj_diff = vect_subt(
		    &mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset, dim),
		    &mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset + dim, dim));
		obj_sum = vect_add(&obj_sum, &obj_diff);

		symm_offset += 2 * dim;
	    }
	} else {
	    constr_sum = zero_mat(total_size, constr_len);
	    for (n, (obj, constr)) in (sdp.objective).iter().zip(sdp.constraint.0.iter()).enumerate() {
		dim = symm_dims[n];

		obj_flatten = obj.concat();
		obj_diff = vect_subt(
		    &mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset, dim),
		    &mat_sym_diagonal_mult_sym(total_dim, &obj_flatten, block_size_sum + symm_offset + dim, dim));
		obj_sum = vect_add(&obj_sum, &obj_diff);

		map_flatten = matrixsym_flatten(constr);
		block_diff = matrix_subt(
		    &matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset, dim),
		    &matrix_diagonal_mult_sym(total_dim, &map_flatten, block_size_sum + symm_offset + dim, dim));
		constr_sum = matrix_add(&constr_sum, &block_diff);

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
	let mut constraints : (Matrix, Vector) =
	    blocks.iter_mut().fold((vec![vec![]; total_size], Vec::new()), |mut acc, (m, v)| {
		for (c1, c2) in (acc.0.iter_mut()).zip(m.iter_mut()) {
		    c1.append(c2);
		}
		(acc.1).append(v);
		acc
	    });

	// make augmented constraint matrix full rank
	constraints.0.push(constraints.1);
	let mut constraints_red = linear_remove_redundant_sym(&mut constraints.0, total_dim);
	let mut point = constraints_red.pop().unwrap_or(vec![]);
	point.extend(vec![0.0; zeros.len()]);

	(obj_sum, (constraints_red, zeros), point, total_dim, symm_dims, block_dims)
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
